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


def doctor(ctx: Context, *, backend: str | None, deep: bool) -> tuple[dict, int]:
    checks: list[dict] = []

    def add(c: CheckResult, required: bool, backend_name: str | None = None):
        d = c.to_dict()
        d["required"] = required
        d["backend"] = backend_name
        d["failing"] = c.status == "fail" or (required and c.status == "skip")
        checks.append(d)

    ff = audio.ffmpeg_version()
    add(CheckResult("ffmpeg", "ok" if ff else "fail", ff or "ffmpeg not found on PATH", EXIT_DEPENDENCY), True)
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
