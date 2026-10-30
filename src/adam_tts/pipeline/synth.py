"""synth pipeline (design §6): per-scene synthesis -> forced alignment -> per-line wavs + manifest."""

from __future__ import annotations

import dataclasses
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from adam_tts import audio, config as config_mod, manifest as mf
from adam_tts.align import split as splitmod
from adam_tts.align.srt import strip_display, to_srt
from adam_tts.backends.base import SynthRequest, validate_request
from adam_tts.errors import (EXIT_PARTIAL, AdamTTSError, AlignError, BackendError, UsageError)
from adam_tts.localmodels import language_name
from adam_tts.output import log
from adam_tts.textsplit import chunk_text
from adam_tts.tsv import Line, group_scenes, read_lines

PUNCT = "。！？；：，、…—」』）.!?;:,"
CHUNK_GAP_S = 0.15


@dataclass
class SynthOptions:
    scenes: list[str] | None = None
    only: list[str] | None = None
    force: bool = False
    no_align: bool = False
    allow_fallback: bool = False
    scene_gap: float = 0.5
    dry_run: bool = False


@dataclass
class Engine:
    """A backend + its fully-resolved params (primary or fallback)."""
    backend: object
    params: config_mod.ResolvedParams
    voice: str
    label: str = "primary"
    pdict: dict = field(default_factory=dict)

    def request(self, text: str) -> SynthRequest:
        p = self.params
        return SynthRequest(text=text, voice=self.voice, style=p.style, prefix=p.prefix, speed=p.speed,
                            pitch=p.pitch, volume=p.volume, lang=p.lang)


def make_engine(ctx, rp: config_mod.ResolvedParams, label: str) -> Engine:
    be = ctx.backend(rp.backend)
    if not getattr(be, "implemented", True):
        raise be._err()
    voice = rp.voice or be.default_voice
    eng = Engine(be, rp, voice, label)
    validate_request(rp.backend, be.capabilities(voice), eng.request("校验。"))
    eng.pdict = {**rp.as_dict(), "voice": voice, "model": getattr(be, "model_id", None) or getattr(be, "model", None),
                 "adapter_version": be.version}
    eng.pdict["model"] = str(eng.pdict["model"]) if eng.pdict["model"] is not None else None
    return eng


def join_text(texts: list[str]) -> str:
    out = ""
    for t in texts:
        t = t.strip()
        if out and out[-1] not in PUNCT:
            out += "。"
        out += t
    return out


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)


def plan_chunks(lines: list[Line], max_chars: int | None, max_bytes: int | None,
                prefix: str | None) -> list[list[tuple[str, str]]]:
    """Chunks of (line_id, piece_text). Prefer line boundaries; split only overlong lines at sentences."""
    plen = len(prefix or "")
    mc = (max_chars - plen) if max_chars else None
    mb = (max_bytes - len((prefix or "").encode())) if max_bytes else None

    def fits(s):
        return (mc is None or len(s) <= mc) and (mb is None or len(s.encode()) <= mb)

    chunks: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []

    def add(pc):
        nonlocal cur
        if cur and not fits(join_text([t for _, t in cur + [pc]])):
            chunks.append(cur)
            cur = []
        cur.append(pc)

    for ln in lines:
        if fits(ln.text):
            add((ln.id, ln.text))
            continue
        if cur:  # overlong line: start it on a fresh chunk so line boundaries stay chunk boundaries
            chunks.append(cur)
            cur = []
        for part in chunk_text(ln.text, mc, mb):
            add((ln.id, part))
    if cur:
        chunks.append(cur)
    return chunks


class SynthRun:
    def __init__(self, ctx, tsv: Path, outdir: Path, rp: config_mod.ResolvedParams, opts: SynthOptions):
        self.ctx, self.tsv, self.outdir, self.opts = ctx, Path(tsv), Path(outdir).expanduser(), opts
        self.lines = read_lines(self.tsv)
        self.scenes = group_scenes(self.lines)
        self.scene_index = {s: i for i, s in enumerate(self.scenes, 1)}
        self.primary = make_engine(ctx, rp, "primary")
        self.fallback: Engine | None = None
        if opts.allow_fallback:
            fname = ctx.cfg.data.get("fallback_preset")
            if not fname:
                raise UsageError("--allow-fallback 需要配置 fallback_preset")
            frp = config_mod.resolve_params(ctx.cfg, preset=fname, backend=None, cli={})
            if frp.backend in config_mod.EXPLICIT_ONLY_BACKENDS:
                raise UsageError(f"降级不能使用 {frp.backend}")
            self.fallback = make_engine(ctx, frp, "fallback")
        self.aligner = None
        self.lang = language_name(rp.lang or ctx.cfg.data.get("aligner", {}).get("language"))
        self.selected = self._select()
        self.m = mf.load(self.outdir) or mf.new(self.tsv)
        self.cache_dir = self.outdir / ".cache"

    # ---------- selection ----------
    def _select(self) -> list[str]:
        sel = []
        if self.opts.scenes:
            for s in self.opts.scenes:
                if s not in self.scenes:
                    raise UsageError(f"TSV 中没有场景 {s}", hint="可选: " + ", ".join(self.scenes))
                sel.append(s)
        if self.opts.only:
            ids = {ln.id: ln.scene for ln in self.lines}
            for i in self.opts.only:
                if i not in ids:
                    raise UsageError(f"TSV 中没有行 {i}")
                sel.append(ids[i])
        if not sel:
            return list(self.scenes)
        return [s for s in self.scenes if s in set(sel)]

    # ---------- manifest helpers ----------
    def _line_entry(self, lid: str) -> dict | None:
        return next((x for x in self.m["lines"] if x["id"] == lid), None)

    def _scene_entry(self, scene: str) -> dict | None:
        return next((x for x in self.m["scenes"] if x["scene"] == scene), None)

    def _put(self, key: str, entry: dict, idkey: str) -> None:
        lst = self.m[key]
        for i, x in enumerate(lst):
            if x[idkey] == entry[idkey]:
                lst[i] = entry
                return
        lst.append(entry)

    def _fresh(self, ln: Line, phash: str) -> bool:
        e = self._line_entry(ln.id)
        return bool(e and e.get("status") == "done" and e.get("text_hash") == mf.text_hash(ln.text)
                    and e.get("params_hash") == phash and (self.outdir / e["path"]).is_file())

    # ---------- caching ----------
    def _tts_key(self, eng: Engine, text: str) -> str:
        return mf.sha({**eng.pdict, "text": text}, 24)

    def _synth_chunk(self, eng: Engine, text: str) -> tuple[Path, bool, dict]:
        key = self._tts_key(eng, text)
        p = self.cache_dir / "tts" / f"{key}.wav"
        if p.is_file() and not self.opts.force:
            return p, True, {"cache_key": key, "cache_hit": True, "chars": len(text)}
        res = eng.backend.synthesize(eng.request(text), p)
        return p, False, {"cache_key": key, "cache_hit": False, "chars": len(text), "elapsed_s": res.elapsed_s}

    def _get_aligner(self):
        if self.aligner is None:
            from adam_tts.align import qwen_aligner

            self.aligner = qwen_aligner.create(self.ctx.cfg, self.ctx.secrets)
        return self.aligner

    def _align_chunk(self, wav_path: Path, key: str, texts: list[str]) -> tuple[list, list[str]]:
        al = self._get_aligner()
        ck = self.cache_dir / "align" / f"{key}-{mf.sha([al.model_id, self.lang, texts], 12)}.json"
        if ck.is_file() and not self.opts.force:
            d = json.loads(ck.read_text(encoding="utf-8"))
            return [tuple(x) if x else None for x in d["spans"]], d["notes"]
        tokens = al.align(wav_path, join_text(texts), self.lang)
        counts = al.unit_counts(texts, self.lang)
        spans, notes = splitmod.piece_spans(tokens, counts)
        audio.atomic_write_bytes(ck, json.dumps({"spans": spans, "notes": notes}, ensure_ascii=False).encode())
        return spans, notes

    # ---------- one scene ----------
    def _run_scene(self, scene: str, eng: Engine, phash: str) -> dict:
        lines = self.scenes[scene]
        caps = eng.backend.capabilities(eng.voice)
        chunks = plan_chunks(lines, caps.max_text_chars, caps.max_text_bytes, eng.params.prefix)
        chunk_meta, wavs, offsets, t = [], [], [], 0.0
        for ci, ch in enumerate(chunks):
            text = join_text([x for _, x in ch])
            p, hit, meta = self._synth_chunk(eng, text)
            w = audio.read_audio(p)
            if ci:
                t += CHUNK_GAP_S
            offsets.append(t)
            t += len(w) / audio.SAMPLE_RATE
            wavs.append(w)
            chunk_meta.append(meta)
            log(f"    chunk {ci + 1}/{len(chunks)} chars={len(text)} {'cache' if hit else 'synth'} "
                f"{len(w) / audio.SAMPLE_RATE:.2f}s")
        scene_wav = audio.concat(wavs, CHUNK_GAP_S)
        idx = self.scene_index[scene]
        seg_rel = f"segments/seg_{idx:02d}_{safe_name(scene)}.wav"
        audio.write_wav(self.outdir / seg_rel, scene_wav)
        dur = len(scene_wav) / audio.SAMPLE_RATE
        entry = {"scene": scene, "index": idx, "path": seg_rel, "duration_s": round(dur, 3), "status": "done",
                 "aligned": False, "backend": eng.params.backend, "voice": eng.voice,
                 "fallback": eng.label == "fallback", "params_hash": phash, "chunks": chunk_meta,
                 "notes": [], "error": None}
        if self.opts.no_align:
            return entry
        # --- alignment per chunk -> line spans in scene time ---
        line_spans: dict[str, list] = {ln.id: [] for ln in lines}
        line_chunks: dict[str, list[int]] = {ln.id: [] for ln in lines}
        for ci, ch in enumerate(chunks):
            spans, notes = self._align_chunk(self.cache_dir / "tts" / f"{chunk_meta[ci]['cache_key']}.wav",
                                             chunk_meta[ci]["cache_key"], [x for _, x in ch])
            entry["notes"] += [f"chunk{ci + 1}: {n}" for n in notes]
            for (lid, _), sp in zip(ch, spans):
                line_chunks[lid].append(ci)
                if sp is None:
                    raise AlignError(f"场景 {scene} 行 {lid} 没有对齐结果（{'; '.join(notes) or '无 token'}）")
                line_spans[lid].append((sp[0] + offsets[ci], sp[1] + offsets[ci]))
        speech = {lid: (min(a for a, _ in v), max(b for _, b in v)) for lid, v in line_spans.items()}
        db = splitmod.frame_db(scene_wav, audio.SAMPLE_RATE)
        cuts, quality = [], []
        for a, b in zip(lines, lines[1:]):
            ca, cb = line_chunks[a.id][-1], line_chunks[b.id][0]
            if ca != cb:
                cut, q = offsets[cb] - CHUNK_GAP_S / 2, "chunk"
            else:
                cut, q = splitmod.cut_between(db, speech[a.id][1], speech[b.id][0])
            cuts.append(round(cut, 3))
            quality.append(q)
        bounds = [0.0] + cuts + [round(dur, 3)]
        if any(bounds[i] >= bounds[i + 1] for i in range(len(bounds) - 1)):
            raise AlignError(f"场景 {scene} 切点不单调: {bounds}", hint="可能是对齐失败；试试 --force 重新合成")
        entry["aligned"] = True
        write_ids = None
        if self.opts.only:
            only = set(self.opts.only)
            write_ids = {ln.id for ln in lines if ln.id in only or not self._fresh(ln, phash)}
        cues = []
        for i, ln in enumerate(lines):
            s, e = speech[ln.id]
            cues.append((s, e, strip_display(ln.text)))
            if write_ids is not None and ln.id not in write_ids:
                continue
            a, b = int(bounds[i] * audio.SAMPLE_RATE), int(bounds[i + 1] * audio.SAMPLE_RATE)
            rel = f"lines/{safe_name(ln.id)}.wav"
            audio.write_wav(self.outdir / rel, scene_wav[a:b])
            weak = [q for q in (quality[i - 1] if i else None, quality[i] if i < len(quality) else None)
                    if q == "gap_mid"]
            self._put("lines", {
                "id": ln.id, "scene": scene, "text_hash": mf.text_hash(ln.text), "params_hash": phash,
                "status": "done", "path": rel, "start_s": bounds[i], "end_s": bounds[i + 1],
                "speech_start_s": round(s, 3), "speech_end_s": round(e, 3),
                "cut_after": quality[i] if i < len(quality) else None, "weak_cut": bool(weak), "error": None,
            }, "id")
        seg_srt = seg_rel[:-4] + ".srt"
        audio.atomic_write_bytes(self.outdir / seg_srt, to_srt(cues).encode("utf-8"))
        entry["srt"] = seg_srt
        entry["cuts"] = [{"after": a.id, "t": c, "quality": q} for a, c, q in zip(lines, cuts, quality)]
        return entry

    def _fail_scene(self, scene: str, eng: Engine, phash: str, err: AdamTTSError, keep_audio: bool) -> None:
        prev = self._scene_entry(scene) or {}
        e = {**prev, "scene": scene, "index": self.scene_index[scene], "status": "failed",
             "backend": eng.params.backend, "fallback": eng.label == "fallback", "params_hash": phash,
             "error": err.to_dict()}
        if not keep_audio:
            e.setdefault("path", None)
        self._put("scenes", e, "scene")
        for ln in self.scenes[scene]:
            if self.opts.only and ln.id not in self.opts.only and self._fresh(ln, phash):
                continue
            le = self._line_entry(ln.id) or {"id": ln.id, "scene": scene, "path": None}
            le.update({"status": "failed", "text_hash": mf.text_hash(ln.text), "params_hash": phash,
                       "error": err.to_dict()})
            self._put("lines", le, "id")

    # ---------- full ----------
    def _build_full(self) -> dict | None:
        parts, cues, offset, missing = [], [], 0.0, []
        lines_by_id = {x["id"]: x for x in self.m["lines"]}
        for scene, lines in self.scenes.items():
            se = self._scene_entry(scene)
            if not se or not se.get("path") or not (self.outdir / se["path"]).is_file() or se.get("status") != "done":
                missing.append(scene)
                continue
            w = audio.read_audio(self.outdir / se["path"])
            if parts:
                offset += self.opts.scene_gap
            se["offset_s"] = round(offset, 3)
            for ln in lines:
                le = lines_by_id.get(ln.id)
                if le and le.get("speech_start_s") is not None:
                    cues.append((offset + le["speech_start_s"], offset + le["speech_end_s"], strip_display(ln.text)))
            parts.append(w)
            offset += len(w) / audio.SAMPLE_RATE
        if not parts:
            return None
        full = audio.concat(parts, self.opts.scene_gap)
        audio.write_wav(self.outdir / "full.wav", full)
        audio.atomic_write_bytes(self.outdir / "full.srt", to_srt(cues).encode("utf-8"))
        return {"path": "full.wav", "srt": "full.srt", "duration_s": round(len(full) / audio.SAMPLE_RATE, 3),
                "scene_gap_s": self.opts.scene_gap, "missing_scenes": missing}

    # ---------- main ----------
    def plan(self) -> dict:
        eng = self.primary
        caps = eng.backend.capabilities(eng.voice)
        phash = mf.sha(eng.pdict)
        out = []
        for s in self.selected:
            chunks = plan_chunks(self.scenes[s], caps.max_text_chars, caps.max_text_bytes, eng.params.prefix)
            texts = [join_text([x for _, x in c]) for c in chunks]
            out.append({"scene": s, "lines": [ln.id for ln in self.scenes[s]],
                        "fresh": all(self._fresh(ln, phash) for ln in self.scenes[s]),
                        "chunks": [{"chars": len(t), "cached": (self.cache_dir / "tts" /
                                    f"{self._tts_key(eng, t)}.wav").is_file()} for t in texts]})
        return {"dry_run": True, "outdir": str(self.outdir), "params": eng.pdict, "scenes": out}

    def run(self) -> tuple[dict, int]:
        if self.opts.dry_run:
            return self.plan(), 0
        self.outdir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        self.m["input"] = {"tsv": str(self.tsv), "tsv_sha256": mf.sha(self.tsv.read_text(encoding="utf-8"), 64)}
        self.m["params"] = self.primary.pdict
        valid = {ln.id for ln in self.lines}
        self.m["lines"] = [x for x in self.m["lines"] if x["id"] in valid]
        self.m["scenes"] = [x for x in self.m["scenes"] if x["scene"] in self.scenes]
        phash = mf.sha(self.primary.pdict)
        done, failed, skipped, errors = [], [], [], []
        try:
            for k, scene in enumerate(self.selected, 1):
                lines = self.scenes[scene]
                se = self._scene_entry(scene)
                if (not self.opts.force and se and se.get("status") == "done"
                        and (self.outdir / (se.get("path") or "")).is_file()
                        and (self.opts.no_align or all(self._fresh(ln, phash) for ln in lines))
                        and se.get("params_hash") == phash):
                    skipped.append(scene)
                    log(f"[{k}/{len(self.selected)}] {scene}: up to date, skip")
                    continue
                log(f"[{k}/{len(self.selected)}] {scene}: {len(lines)} lines via {self.primary.params.backend}")
                eng = self.primary
                try:
                    try:
                        entry = self._run_scene(scene, eng, phash)
                    except BackendError as e:
                        if not self.fallback:
                            raise
                        log(f"    {eng.params.backend} failed ({e.message}); fallback -> {self.fallback.params.backend}")
                        eng = self.fallback
                        entry = self._run_scene(scene, eng, mf.sha(eng.pdict))
                        entry["notes"].append(f"fallback after: {e.message}")
                    self._put("scenes", entry, "scene")
                    done.append(scene)
                except (BackendError, AlignError) as e:
                    log(f"    FAILED: {e.message}")
                    self._fail_scene(scene, eng, mf.sha(eng.pdict), e, keep_audio=isinstance(e, AlignError))
                    failed.append(scene)
                    errors.append(e)
                mf.save(self.outdir, self.m)
        finally:
            for x in (self.primary, self.fallback):
                if x and hasattr(x.backend, "close"):
                    x.backend.close()
            if self.aligner:
                self.aligner.close()
        self.m["full"] = self._build_full()
        mf.save(self.outdir, self.m)
        sel_lines = [x for x in self.m["lines"] if x["scene"] in set(self.selected)]
        weak = [{"after": x["id"], "scene": x["scene"]} for x in self.m["lines"]
                if x.get("cut_after") == "gap_mid" and x["scene"] in set(self.selected)]
        result = {
            "outdir": str(self.outdir), "manifest": str(self.outdir / mf.NAME),
            "full": str(self.outdir / "full.wav") if self.m["full"] else None,
            "full_duration_s": (self.m["full"] or {}).get("duration_s"),
            "scenes": {"selected": len(self.selected), "done": done, "skipped": skipped, "failed": failed},
            "lines": {"done": sum(1 for x in sel_lines if x["status"] == "done"),
                      "failed": sum(1 for x in sel_lines if x["status"] == "failed"),
                      "total_selected": sum(len(self.scenes[s]) for s in self.selected)},
            "weak_cuts": weak, "params": self.primary.pdict, "elapsed_s": round(time.monotonic() - t0, 1),
        }
        code = 0
        if errors:
            code = EXIT_PARTIAL if (done or skipped) else errors[0].exit_code
            result["error"] = {**errors[0].to_dict(), "code": code}
        return result, code
