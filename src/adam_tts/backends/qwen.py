"""Local Qwen3-TTS CustomVoice 1.7B (MPS fp16, 24 kHz). Style -> `instruct`. No numeric speed.

Runs the model in a worker process under the configured python (default: <home>/.venv/bin/python,
the env already used by scripts/batch_scene_voice.py). The model loads once per CLI invocation.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

from adam_tts import audio
from adam_tts.backends.base import Capabilities, CheckResult, SynthRequest, SynthResult, VoiceInfo
from adam_tts.errors import EXIT_BACKEND, EXIT_DEPENDENCY, BackendError, UsageError
from adam_tts.localmodels import language_name, qwen_home, resolve_model, resolve_python
from adam_tts.workers.client import WorkerClient

DEFAULT_MODEL = "models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
# from the model's config.json (spk_id / spk_is_dialect); used when the config cannot be read
STATIC_SPEAKERS = {"Serena": None, "Vivian": None, "Uncle_Fu": None, "Ryan": None, "Aiden": None,
                   "Ono_Anna": None, "Sohee": None, "Eric": "sichuan_dialect", "Dylan": "beijing_dialect"}


class QwenBackend:
    name = "qwen"
    version = "1"
    default_voice = "Serena"
    implemented = True

    def __init__(self, settings: dict, secrets=None):
        self.settings = settings
        self.home = qwen_home(settings, secrets)
        self.python = resolve_python(settings.get("python"), self.home / ".venv" / "bin" / "python")
        self.model = resolve_model(self.home, settings.get("model"), DEFAULT_MODEL)
        self.device = settings.get("device")  # None = mps if available
        self.dtype = settings.get("dtype", "float16")
        self.max_text_chars = int(settings.get("max_text_chars", 400))
        self.model_id = self.model.name
        self.worker: WorkerClient | None = None
        self.load_info: dict | None = None

    def capabilities(self, voice: str | None = None) -> Capabilities:
        return Capabilities(style=True, prefix=True, lang=True, max_text_chars=self.max_text_chars)

    def _speakers(self) -> dict:
        cfg = self.model / "config.json"
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            talker = data.get("talker_config", data)
            ids = talker.get("spk_id") or {}
            dia = talker.get("spk_is_dialect") or {}
            if ids:
                return {k: (dia.get(k) or None) for k in ids}
        except (OSError, ValueError, AttributeError):
            pass
        return {k.lower(): v for k, v in STATIC_SPEAKERS.items()}

    def list_voices(self, lang: str | None = None) -> list[VoiceInfo]:
        names = {k.lower(): k for k in STATIC_SPEAKERS}
        out = []
        for k, dialect in self._speakers().items():
            out.append(VoiceInfo(names.get(k.lower(), k), "zh" if dialect else None, None,
                                 f"dialect: {dialect}" if dialect else None, True))
        return out

    def _check_voice(self, voice: str) -> None:
        known = {k.lower() for k in self._speakers()}
        if voice.lower() not in known:
            raise UsageError(f"qwen 没有音色 {voice}", hint="可选: " + ", ".join(v.name for v in self.list_voices()))

    def _ensure_worker(self) -> WorkerClient:
        if self.worker is None:
            if not self.model.exists():
                from adam_tts.errors import DependencyMissing

                raise DependencyMissing(f"找不到 Qwen3-TTS 模型目录: {self.model}",
                                        hint="设置 ADAM_TTS_QWEN_HOME 或 config.toml [backends.qwen] model")
            self.worker = WorkerClient(self.python, "qwen_tts_worker.py", label="qwen-tts worker",
                                       verbose=bool(self.settings.get("verbose")))
            self.load_info = self.worker.request("load", model=str(self.model), device=self.device,
                                                 dtype=self.dtype)
        return self.worker

    def synthesize(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        self._check_voice(req.voice)
        w = self._ensure_worker()
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / "raw.wav"
            r = w.request("synth", text=(req.prefix or "") + req.text, speaker=req.voice,
                          language=language_name(req.lang), instruct=req.style, out=str(raw))
            audio.to_standard_wav(raw, out_wav)
        return SynthResult(out=out_wav, duration_s=round(audio.wav_duration(out_wav), 3),
                           elapsed_s=round(time.monotonic() - t0, 2), bytes=out_wav.stat().st_size,
                           extra={"model_sr": r.get("sr"), "device": (self.load_info or {}).get("device")})

    def close(self) -> None:
        if self.worker:
            self.worker.close()
            self.worker = None

    def check(self, deep: bool = False) -> list[CheckResult]:
        out = [CheckResult("qwen.home", "ok" if self.home.is_dir() else "fail", str(self.home), EXIT_DEPENDENCY),
               CheckResult("qwen.model", "ok" if self.model.is_dir() else "fail", str(self.model), EXIT_DEPENDENCY)]
        out.append(_python_check("qwen.python", self.python, "import qwen_tts, torch; "
                                 "print(torch.__version__, torch.backends.mps.is_available())"))
        if deep and all(c.status == "ok" for c in out):
            from adam_tts.backends.mimo import _deep_check

            out.append(_deep_check(self, SynthRequest(text="测试。", voice=self.default_voice)))
            self.close()
        return out


def _python_check(name: str, python: str, code: str) -> CheckResult:
    if not Path(python).exists():
        return CheckResult(name, "fail", f"python not found: {python}", EXIT_DEPENDENCY)
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.TimeoutExpired) as e:
        return CheckResult(name, "fail", f"{python}: {type(e).__name__}", EXIT_DEPENDENCY)
    if r.returncode != 0:
        last = (r.stderr.strip().splitlines() or ["?"])[-1]
        return CheckResult(name, "fail", f"{python}: {last[:200]}", EXIT_DEPENDENCY)
    info = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    return CheckResult(name, "ok", f"{python} (torch/mps: {info})")


__all__ = ["QwenBackend", "_python_check", "EXIT_BACKEND", "BackendError"]
