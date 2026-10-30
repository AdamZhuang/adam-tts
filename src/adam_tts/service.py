"""Command implementations, independent of argparse (easy to unit-test)."""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

from adam_tts import audio, config as config_mod
from adam_tts.backends import registry
from adam_tts.backends.base import CheckResult, SynthRequest, validate_request
from adam_tts.errors import EXIT_BACKEND, EXIT_CONFIG, EXIT_DEPENDENCY, UsageError
from adam_tts.secrets import KNOWN_SECRETS, Secrets
from adam_tts.textsplit import chunk_text


class Context:
    def __init__(self, config_path: str | None = None, env_file: str | None = None):
        self.secrets = Secrets(env_file)
        if self.secrets.missing_explicit:
            raise UsageError(f".env 文件不存在: {env_file}")
        self.cfg = config_mod.load_config(config_path, set(registry.names()))

    def backend(self, name: str):
        return registry.create(name, self.cfg, self.secrets)


def say(ctx: Context, *, text: str, out: Path, preset: str | None, backend: str | None, params: dict) -> dict:
    if out.suffix.lower() != ".wav":
        raise UsageError(f"输出文件必须是 .wav: {out}")
    rp = config_mod.resolve_params(ctx.cfg, preset=preset, backend=backend, cli=params)
    be = ctx.backend(rp.backend)
    if not getattr(be, "implemented", True):
        raise be._err()
    voice = rp.voice or be.default_voice
    req = SynthRequest(text=text.strip(), voice=voice, style=rp.style, prefix=rp.prefix,
                       speed=rp.speed, pitch=rp.pitch, volume=rp.volume, lang=rp.lang)
    caps = be.capabilities(voice)
    validate_request(rp.backend, caps, req)  # before any request is sent
    chunks = chunk_text(req.text, caps.max_text_chars, caps.max_text_bytes)
    out = out.expanduser()
    elapsed = 0.0
    extras = []
    try:
        return _say_run(be, req, chunks, out, rp, voice)
    finally:
        if hasattr(be, "close"):
            be.close()


def _say_run(be, req, chunks, out, rp, voice) -> dict:
    elapsed = 0.0
    extras = []
    if len(chunks) == 1:
        res = be.synthesize(req, out)
        elapsed = res.elapsed_s
        extras.append(res.extra)
    else:
        with tempfile.TemporaryDirectory() as d:
            parts = []
            for i, c in enumerate(chunks):
                p = Path(d) / f"chunk_{i:03d}.wav"
                r = be.synthesize(dataclasses.replace(req, text=c), p)
                elapsed += r.elapsed_s
                extras.append(r.extra)
                parts.append(p)
            audio.concat_wavs(parts, out)
    params_out = rp.as_dict()
    params_out["voice"] = voice
    return {
        "out": str(out),
        **params_out,
        "duration_s": round(audio.wav_duration(out), 3),
        "elapsed_s": round(elapsed, 2),
        "chunks": len(chunks),
        "sample_rate": audio.wav_info(out)[0],
    }


def voice_list(ctx: Context, *, backend: str, lang: str | None) -> dict:
    be = ctx.backend(backend)
    voices = be.list_voices(lang)
    return {"backend": backend, "count": len(voices), "voices": [v.to_dict() for v in voices]}


_CODE_PRIORITY = (EXIT_CONFIG, EXIT_DEPENDENCY, EXIT_BACKEND)


def doctor(ctx: Context, *, backend: str | None, deep: bool, aligner: str | None = "optional") -> tuple[dict, int]:
    checks: list[dict] = []

    def add(c: CheckResult, required: bool, backend_name: str | None = None):
        d = c.to_dict()
        d["required"] = required
        d["backend"] = backend_name
        d["failing"] = c.status == "fail" or (required and c.status == "skip")
        checks.append(d)

    ff = audio.ffmpeg_version()
    add(CheckResult("ffmpeg", "ok" if ff else "warn",
                    ff or "not found (optional: only needed for formats soundfile cannot decode)"), False)
    add(CheckResult("config", "ok", str(ctx.cfg.path) if ctx.cfg.path else "built-in defaults (no config.toml)"), False)
    add(CheckResult("env_file", "ok", str(ctx.secrets.env_file) if ctx.secrets.env_file else "none found"), False)

    if backend:
        targets = {backend: True}
    else:
        presets = ctx.cfg.presets
        required = {presets[ctx.cfg.data[k]]["backend"] for k in ("default_preset", "fallback_preset")
                    if ctx.cfg.data.get(k)}
        targets = {n: (n in required) for n in registry.names()}
    for name, req in targets.items():
        for c in ctx.backend(name).check(deep=deep):
            add(c, req, name)

    if aligner:
        try:
            from adam_tts.align import qwen_aligner

            for c in qwen_aligner.create(ctx.cfg, ctx.secrets).check():
                add(c, aligner == "required", "aligner")
        except Exception as e:  # noqa: BLE001
            add(CheckResult("aligner", "fail", str(e), EXIT_DEPENDENCY), aligner == "required", "aligner")

    failing = [c for c in checks if c["failing"] and c["required"]]
    code = 0
    if failing:
        codes = {c["code"] or 1 for c in failing}
        code = next((p for p in _CODE_PRIORITY if p in codes), min(codes))
    result = {
        "healthy": not failing,
        "secrets": ctx.secrets.report(KNOWN_SECRETS),
        "checks": [{k: v for k, v in c.items() if k != "failing"} for c in checks],
    }
    return result, code


# ---------------- M2: synth / status / align ----------------

def synth(ctx: Context, *, tsv: Path, outdir: Path, preset, backend, params: dict, opts) -> tuple[dict, int]:
    from adam_tts.pipeline.synth import SynthRun

    rp = config_mod.resolve_params(ctx.cfg, preset=preset, backend=backend, cli=params)
    return SynthRun(ctx, tsv, outdir, rp, opts).run()


def status(ctx: Context | None, *, outdir: Path, tsv: Path | None) -> tuple[dict, int]:
    from adam_tts import manifest as mf
    from adam_tts.errors import EXIT_PARTIAL
    from adam_tts.tsv import read_lines

    outdir = Path(outdir).expanduser()
    m = mf.load(outdir)
    if m is None:
        raise UsageError(f"{outdir} 下没有 manifest.json", hint="先运行 adam_tts synth")
    tsv = Path(tsv).expanduser() if tsv else (Path(m["input"]["tsv"]) if m.get("input", {}).get("tsv") else None)
    entries = {x["id"]: x for x in m["lines"]}
    phash = mf.sha(m["params"]) if m.get("params") else None
    rows = []
    if tsv and tsv.is_file():
        for ln in read_lines(tsv):
            e = entries.get(ln.id)
            if not e or e.get("status") is None or (e.get("status") == "done" and not e.get("path")):
                st = "missing"
            elif e["status"] == "failed":
                st = "failed"
            elif e.get("text_hash") != mf.text_hash(ln.text) or (phash and e.get("params_hash") != phash
                                                                  and not _is_fallback_scene(m, ln.scene)):
                st = "stale"
            elif not (outdir / e["path"]).is_file():
                st = "missing"
            else:
                st = "done"
            rows.append({"id": ln.id, "scene": ln.scene, "status": st,
                         "weak_cut": bool(e and e.get("cut_after") == "gap_mid")})
    else:
        tsv = None
        for e in m["lines"]:
            st = e["status"] if e["status"] != "done" or (outdir / (e.get("path") or "")).is_file() else "missing"
            rows.append({"id": e["id"], "scene": e["scene"], "status": st, "weak_cut": e.get("cut_after") == "gap_mid"})
    counts = {k: sum(1 for r in rows if r["status"] == k) for k in ("done", "missing", "failed", "stale")}
    bad = [r for r in rows if r["status"] != "done"]
    bad_scenes = list(dict.fromkeys(r["scene"] for r in bad))
    suggest = None
    if bad:
        suggest = f"adam_tts synth {tsv or '<lines.tsv>'} -o {outdir} --scene {','.join(bad_scenes)}"
    fallback = [s["scene"] for s in m["scenes"] if s.get("fallback")]
    result = {"outdir": str(outdir), "tsv": str(tsv) if tsv else None, "params": m.get("params"),
              "counts": counts, "total": len(rows), "complete": not bad,
              "incomplete_scenes": bad_scenes, "fallback_scenes": fallback,
              "weak_cuts": [r["id"] for r in rows if r["weak_cut"]],
              "full": m.get("full"), "lines": rows, "suggested_command": suggest,
              "updated_at": m.get("updated_at")}
    return result, (0 if not bad else EXIT_PARTIAL)


def _is_fallback_scene(m: dict, scene: str) -> bool:
    return any(s["scene"] == scene and s.get("fallback") for s in m["scenes"])


def align(ctx: Context, *, audio_path: Path, outdir: Path, text: str | None, tsv: Path | None,
          scene: str | None, lang: str | None, max_cue_chars: int = 20) -> dict:
    import json

    from adam_tts.align import qwen_aligner, split as splitmod
    from adam_tts.align.srt import cue_texts, strip_display, to_srt
    from adam_tts.localmodels import language_name
    from adam_tts.pipeline.synth import join_text
    from adam_tts.tsv import group_scenes, read_lines

    audio_path = Path(audio_path).expanduser()
    if not audio_path.is_file():
        raise UsageError(f"找不到音频: {audio_path}")
    if tsv:
        if not scene:
            raise UsageError("--tsv 需要同时给 --scene")
        scenes = group_scenes(read_lines(tsv))
        if scene not in scenes:
            raise UsageError(f"TSV 中没有场景 {scene}")
        cues_src = [(ln.id, ln.text) for ln in scenes[scene]]
    else:
        if not text or not text.strip():
            raise UsageError("需要 --text / --text-file / --tsv+--scene 之一")
        cues_src = [(str(i), t) for i, t in enumerate(cue_texts(text, max_cue_chars), 1)]
    outdir = Path(outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    language = language_name(lang or "zh")
    # aligner works on 16k internally; give it a standard 24k mono wav copy
    import tempfile

    al = qwen_aligner.create(ctx.cfg, ctx.secrets)
    try:
        with tempfile.TemporaryDirectory() as d:
            wav = audio.read_audio(audio_path)
            tmp = Path(d) / "in.wav"
            audio.write_wav(tmp, wav)
            texts = [t for _, t in cues_src]
            tokens = al.align(tmp, join_text(texts), language)
            counts = al.unit_counts(texts, language)
    finally:
        al.close()
    spans, notes = splitmod.piece_spans(tokens, counts)
    cues, items = [], []
    for (cid, t), sp in zip(cues_src, spans):
        items.append({"id": cid, "text": t, "start": sp[0] if sp else None, "end": sp[1] if sp else None})
        if sp:
            cues.append((sp[0], sp[1], strip_display(t)))
    srt_path = outdir / f"{audio_path.stem}.srt"
    json_path = outdir / "align.json"
    audio.atomic_write_bytes(srt_path, to_srt(cues).encode("utf-8"))
    audio.atomic_write_bytes(json_path, json.dumps({
        "audio": str(audio_path), "language": language, "model": al.model_id, "notes": notes, "cues": items,
        "tokens": [{"text": x.text, "start": x.start, "end": x.end} for x in tokens]},
        ensure_ascii=False, indent=2).encode("utf-8"))
    return {"audio": str(audio_path), "srt": str(srt_path), "json": str(json_path), "cues": len(cues),
            "unaligned": sum(1 for s in spans if s is None), "tokens": len(tokens), "notes": notes,
            "duration_s": round(len(wav) / audio.SAMPLE_RATE, 3)}
