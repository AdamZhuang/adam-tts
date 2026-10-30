"""synth / status / align pipeline with a layout-aware fake backend + fake aligner (no models)."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from adam_tts import cli
from adam_tts.align import qwen_aligner
from adam_tts.align.base import Token
from adam_tts.backends import registry
from adam_tts.backends.base import Capabilities, CheckResult, SynthResult, VoiceInfo
from adam_tts.errors import BackendError

SR = 24000
CHAR_S, PUNCT_S = 0.05, 0.12
PUNCT = set("。！？；：，、…,.!?;:")


def layout(text):
    """(char, start, end, is_punct) for the fake voice: tone per char, silence per punctuation."""
    t, out = 0.0, []
    for ch in text:
        d = PUNCT_S if ch in PUNCT else CHAR_S
        out.append((ch, t, t + d, ch in PUNCT))
        t += d
    return out, t


class SceneBackend:
    name = "scene"
    version = "1"
    default_voice = "v1"
    model_id = "fake-model"
    calls: list = []
    fail_on: set = set()
    caps = Capabilities(style=True, prefix=True, max_text_chars=40)

    def __init__(self, settings, secrets=None):
        pass

    def capabilities(self, voice=None):
        return type(self).caps

    def list_voices(self, lang=None):
        return [VoiceInfo("v1", "zh")]

    def check(self, deep=False):
        return [CheckResult("scene.ok", "ok", "")]

    def synthesize(self, req, out_wav: Path):
        if any(x in req.text for x in type(self).fail_on):
            raise BackendError("boom")
        type(self).calls.append(req)
        lay, dur = layout(req.text)
        x = np.zeros(int(round(dur * SR)), dtype=np.float32)
        tt = np.arange(len(x)) / SR
        for ch, a, b, p in lay:
            if not p:
                i, j = int(a * SR), int(b * SR)
                x[i:j] = 0.3 * np.sin(2 * np.pi * 220 * tt[i:j])
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_wav, x, SR, subtype="PCM_16")
        return SynthResult(out=out_wav, duration_s=dur, elapsed_s=0.0, bytes=0)


class EdgeLike(SceneBackend):
    name = "edgelike"
    calls: list = []
    fail_on: set = set()


class FakeAligner:
    name = "fake"
    model_id = "fake-aligner"
    calls = 0

    def __init__(self, settings, qwen_settings, secrets=None):
        pass

    def align(self, audio_path, text, language):
        type(self).calls += 1
        lay, _ = layout(text)
        return [Token(c, a, b) for c, a, b, p in lay if not p]

    def unit_counts(self, texts, language):
        return [sum(1 for c in t if c not in PUNCT) for t in texts]

    def check(self):
        return [CheckResult("aligner.fake", "ok", "")]

    def close(self):
        pass


TSV = "id\tscene\ttext\nL1\tA\t你好世界。\nL2\tA\t今天天气很好，我们出门吧。\nL3\tA\t好\nL4\tB\t第二个场景开始了。\nL5\tB\t结束。\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setitem(registry.BACKENDS, "scene", SceneBackend)
    monkeypatch.setitem(registry.BACKENDS, "edgelike", EdgeLike)
    monkeypatch.setitem(qwen_aligner.ALIGNERS, "fake", FakeAligner)
    SceneBackend.calls, SceneBackend.fail_on = [], set()
    EdgeLike.calls, EdgeLike.fail_on = [], set()
    FakeAligner.calls = 0
    cfg = tmp_path / "config.toml"
    cfg.write_text('default_preset = "sc"\nfallback_preset = "el"\n[aligner]\nkind = "fake"\n'
                   '[presets.sc]\nbackend = "scene"\nvoice = "v1"\nstyle = "平静"\n'
                   '[presets.el]\nbackend = "edgelike"\nvoice = "v1"\n', encoding="utf-8")
    monkeypatch.setenv("ADAM_TTS_CONFIG", str(cfg))
    tsv = tmp_path / "lines.tsv"
    tsv.write_text(TSV, encoding="utf-8")
    return tsv, tmp_path / "out"


def run(capsys, *a):
    code = cli.main(list(a))
    o = capsys.readouterr()
    return code, o.out, o.err


def J(s):
    return json.loads(s)


def test_full_run_outputs(capsys, env):
    tsv, out = env
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--json")
    r = J(o)
    assert code == 0 and r["ok"], o
    assert r["scenes"]["done"] == ["A", "B"] and r["lines"]["done"] == 5
    assert r["weak_cuts"] == []
    for i in range(1, 6):
        assert (out / "lines" / f"L{i}.wav").is_file()
    assert (out / "segments" / "seg_01_A.wav").is_file() and (out / "segments" / "seg_02_B.srt").is_file()
    m = J((out / "manifest.json").read_text())
    assert m["schema_version"] == 1 and len(m["lines"]) == 5
    # scene-level: one request per scene; L3 has no punctuation -> joined with 。
    assert [c.text for c in SceneBackend.calls] == ["你好世界。今天天气很好，我们出门吧。好", "第二个场景开始了。结束。"]
    # cuts land in silence (punctuation) and line durations sum to the scene duration
    a = [x for x in m["lines"] if x["scene"] == "A"]
    assert all(x["cut_after"] in ("silence", None) for x in a)
    seg = sf.info(out / "segments" / "seg_01_A.wav").duration
    assert abs(sum(x["end_s"] - x["start_s"] for x in a) - seg) < 0.02
    # full.wav = A + 0.5 gap + B
    full = sf.info(out / "full.wav").duration
    segb = sf.info(out / "segments" / "seg_02_B.wav").duration
    assert abs(full - (seg + 0.5 + segb)) < 0.01
    srt = (out / "full.srt").read_text(encoding="utf-8")
    assert srt.count("-->") == 5 and "今天天气很好，我们出门吧" in srt


def test_rerun_is_cached_and_skipped(capsys, env):
    tsv, out = env
    run(capsys, "synth", str(tsv), "-o", str(out), "--json")
    n, na = len(SceneBackend.calls), FakeAligner.calls
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--json")
    r = J(o)
    assert code == 0 and r["scenes"]["skipped"] == ["A", "B"]
    assert len(SceneBackend.calls) == n and FakeAligner.calls == na
    # --force bypasses cache
    run(capsys, "synth", str(tsv), "-o", str(out), "--json", "--force", "--scene", "B")
    assert len(SceneBackend.calls) == n + 1


def test_edit_one_line_only_resynths_that_scene(capsys, env):
    tsv, out = env
    run(capsys, "synth", str(tsv), "-o", str(out), "--json")
    tsv.write_text(TSV.replace("结束。", "真的结束了。"), encoding="utf-8")
    code, o, _ = run(capsys, "status", str(out), "--json")
    r = J(o)
    assert code == 7 and r["counts"]["stale"] == 1 and r["incomplete_scenes"] == ["B"]
    assert "--scene B" in r["suggested_command"]
    SceneBackend.calls = []
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--json")
    assert code == 0 and J(o)["scenes"]["skipped"] == ["A"]
    assert [c.text for c in SceneBackend.calls] == ["第二个场景开始了。真的结束了。"]
    assert run(capsys, "status", str(out), "--json")[0] == 0


def test_scene_and_only_selection(capsys, env):
    tsv, out = env
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--scene", "B", "--json")
    r = J(o)
    assert r["scenes"]["done"] == ["B"] and r["full_duration_s"] is not None
    assert not (out / "lines" / "L1.wav").exists()
    code, o, _ = run(capsys, "status", str(out), "--json")
    assert code == 7 and J(o)["counts"]["missing"] == 3
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--only", "L2", "--json")
    assert J(o)["scenes"]["done"] == ["A"]
    assert run(capsys, "synth", str(tsv), "-o", str(out), "--only", "NOPE", "--json")[0] == 2
    assert run(capsys, "synth", str(tsv), "-o", str(out), "--scene", "Z", "--json")[0] == 2


def test_failure_partial_and_fallback(capsys, env):
    tsv, out = env
    SceneBackend.fail_on = {"第二个"}
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--json")
    r = J(o)
    assert code == 7 and r["ok"] is False and r["scenes"]["failed"] == ["B"] and r["scenes"]["done"] == ["A"]
    m = J((out / "manifest.json").read_text())
    assert {x["id"]: x["status"] for x in m["lines"]}["L4"] == "failed"
    # all failed -> backend code 5
    SceneBackend.fail_on = {"。"}
    assert run(capsys, "synth", str(tsv), "-o", str(out / "x"), "--json")[0] == 5
    # with fallback the failing scene is redone on the fallback preset
    SceneBackend.fail_on = {"第二个"}
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--allow-fallback", "--json")
    assert code == 0, o
    m = J((out / "manifest.json").read_text())
    b = next(s for s in m["scenes"] if s["scene"] == "B")
    assert b["fallback"] and b["backend"] == "edgelike" and len(EdgeLike.calls) == 1
    code, o, _ = run(capsys, "status", str(out), "--json")
    assert code == 0 and J(o)["fallback_scenes"] == ["B"]


def test_chunking_long_scene(capsys, env, tmp_path):
    tsv, out = env
    SceneBackend.caps = Capabilities(style=True, prefix=True, max_text_chars=12)
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--json", "--scene", "A")
    assert code == 0, o
    assert all(len(c.text) <= 12 for c in SceneBackend.calls) and len(SceneBackend.calls) >= 3
    m = J((out / "manifest.json").read_text())
    q = [x["cut_after"] for x in m["lines"] if x["scene"] == "A"]
    assert "chunk" in q and len([x for x in m["lines"] if x["status"] == "done"]) == 3


def test_dry_run_and_no_align(capsys, env):
    tsv, out = env
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--dry-run", "--json")
    r = J(o)
    assert code == 0 and r["dry_run"] and len(r["scenes"]) == 2 and not SceneBackend.calls
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--no-align", "--json")
    assert code == 0 and FakeAligner.calls == 0 and (out / "full.wav").is_file()


def test_unsupported_param_fails_before_any_call(capsys, env):
    tsv, out = env
    SceneBackend.caps = Capabilities(style=True, prefix=True, speed=False, max_text_chars=40)
    code, o, _ = run(capsys, "synth", str(tsv), "-o", str(out), "--speed", "1.2", "--json")
    assert code == 4 and not SceneBackend.calls


def test_tsv_errors(capsys, env, tmp_path):
    tsv, out = env
    bad = tmp_path / "bad.tsv"
    bad.write_text("a\tS\tx\na\tS\ty\n", encoding="utf-8")
    code, o, _ = run(capsys, "synth", str(bad), "-o", str(out), "--json")
    assert code == 2 and "重复" in J(o)["error"]["message"]
    bad.write_text("a\tS\tx\nb\tT\ty\nc\tS\tz\n", encoding="utf-8")
    assert "不连续" in J(run(capsys, "synth", str(bad), "-o", str(out), "--json")[1])["error"]["message"]
    bad.write_text("a\tS\n", encoding="utf-8")
    assert run(capsys, "synth", str(bad), "-o", str(out), "--json")[0] == 2


def test_status_without_manifest(capsys, tmp_path):
    assert run(capsys, "status", str(tmp_path / "none"), "--json")[0] == 2


def test_align_command(capsys, env, tmp_path):
    tsv, out = env
    wav = tmp_path / "a.wav"
    SceneBackend({}).synthesize(type("R", (), {"text": "你好世界。今天天气很好，我们出门吧。"})(), wav)
    code, o, _ = run(capsys, "align", str(wav), "-o", str(tmp_path / "al"), "--text", "你好世界。今天天气很好，我们出门吧。",
                     "--max-cue-chars", "6", "--json")
    r = J(o)
    assert code == 0 and r["cues"] == 3 and r["unaligned"] == 0, o
    srt = Path(r["srt"]).read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:00,200" in srt
    code, o, _ = run(capsys, "align", str(wav), "-o", str(tmp_path / "al2"), "--tsv", str(tsv), "--json")
    assert code == 2
    code, o, _ = run(capsys, "align", str(wav), "-o", str(tmp_path / "al2"), "--json")
    assert code == 2


def test_doctor_includes_optional_aligner(capsys, env):
    code, o, _ = run(capsys, "doctor", "--backend", "scene", "--json")
    r = J(o)
    assert code == 0 and any(c["name"] == "aligner.fake" for c in r["checks"])
