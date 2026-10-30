"""Backends registered in v1 but implemented in later milestones (qwen: M2, doubao: M3)."""

from __future__ import annotations

from pathlib import Path

from adam_tts.backends.base import Capabilities, CheckResult, SynthRequest, SynthResult, VoiceInfo
from adam_tts.errors import EXIT_CONFIG, NotImplementedBackend


class _Stub:
    name = ""
    version = "0"
    default_voice: str | None = None
    milestone = ""
    implemented = False

    def __init__(self, settings: dict, secrets=None):
        self.settings = settings
        self.secrets = secrets

    def _err(self) -> NotImplementedBackend:
        return NotImplementedBackend(f"后端 {self.name} 已注册但尚未实现（计划在 {self.milestone}）",
                                     hint="当前可用后端：mimo、edge")

    def list_voices(self, lang: str | None = None) -> list[VoiceInfo]:
        raise self._err()

    def synthesize(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        raise self._err()

    def check(self, deep: bool = False) -> list[CheckResult]:
        return [CheckResult(f"{self.name}.implemented", "skip", f"not implemented ({self.milestone})",
                            self._err().exit_code)]


class QwenBackend(_Stub):
    name = "qwen"
    milestone = "M2"

    def capabilities(self, voice: str | None = None) -> Capabilities:
        return Capabilities(style=True, prefix=True, lang=True, max_text_chars=600)


class DoubaoBackend(_Stub):
    name = "doubao"
    milestone = "M3"
    KEY_NAME = "DOUBAO_API_KEY"

    def capabilities(self, voice: str | None = None) -> Capabilities:
        return Capabilities(style=True, speed=True, pitch=True, volume=True, max_text_bytes=1024)

    def check(self, deep: bool = False) -> list[CheckResult]:
        out = super().check(deep)
        src = self.secrets.source(self.KEY_NAME) if self.secrets else "missing"
        status = "ok" if src != "missing" else "warn"
        out.insert(0, CheckResult("doubao.key", status, f"{self.KEY_NAME} {'set (' + src + ')' if src != 'missing' else 'missing'}",
                                  EXIT_CONFIG))
        return out
