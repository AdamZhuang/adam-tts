"""Backend interface (design §4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from adam_tts.errors import UnsupportedParam, UsageError

SAMPLE_RATE = 24000
SPEED_RANGE = (0.5, 2.0)
VOLUME_RANGE = (0.5, 2.0)


@dataclass(frozen=True)
class Capabilities:
    style: bool = False
    prefix: bool = False
    speed: bool = False
    pitch: bool = False
    volume: bool = False
    lang: bool = False
    max_text_chars: int | None = None
    max_text_bytes: int | None = None
    sample_rate: int = SAMPLE_RATE

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SynthRequest:
    text: str
    voice: str
    style: str | None = None
    prefix: str | None = None
    speed: float | None = None
    pitch: int | None = None  # Hz offset (edge); doubao mapping decided in M3
    volume: float | None = None
    lang: str | None = None


@dataclass
class SynthResult:
    out: Path
    duration_s: float
    elapsed_s: float
    bytes: int
    extra: dict = field(default_factory=dict)  # backend-specific, must not contain secrets


@dataclass(frozen=True)
class VoiceInfo:
    name: str
    lang: str | None = None
    gender: str | None = None
    description: str | None = None
    supports_style: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckResult:
    name: str
    status: str  # ok | warn | fail | skip
    detail: str = ""
    code: int | None = None  # exit code to use when this check fails

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class TTSBackend(Protocol):
    name: str
    version: str
    default_voice: str | None

    def capabilities(self, voice: str | None = None) -> Capabilities: ...
    def list_voices(self, lang: str | None = None) -> list[VoiceInfo]: ...
    def check(self, deep: bool = False) -> list[CheckResult]: ...
    def synthesize(self, req: SynthRequest, out_wav: Path) -> SynthResult: ...


_HINTS = {
    "speed": "该后端没有数值语速参数；请去掉 --speed，把语速要求写进 --style（如“语速稍快”）",
    "style": "该后端不支持风格指令；请去掉 --style，或换支持风格的后端（mimo / qwen）",
    "prefix": "该后端不支持行首内联标签；请去掉 --prefix",
    "pitch": "该后端不支持 --pitch",
    "volume": "该后端不支持 --volume",
    "lang": "该后端不需要 --lang（只有 qwen 使用）",
}


def validate_request(backend_name: str, caps: Capabilities, req: SynthRequest) -> None:
    """Fail loudly on any param the backend does not support. Never silently ignore."""
    if not req.text or not req.text.strip():
        raise UsageError("要合成的文本为空")
    if not req.voice:
        raise UsageError("没有指定音色", hint=f"用 adam_tts voice list --backend {backend_name} 查看原生音色")
    for f in ("style", "prefix", "speed", "pitch", "volume", "lang"):
        if getattr(req, f) is not None and not getattr(caps, f):
            raise UnsupportedParam(f"后端 {backend_name} 不支持 --{f}", hint=_HINTS[f])
    if req.speed is not None and not (SPEED_RANGE[0] <= req.speed <= SPEED_RANGE[1]):
        raise UsageError(f"--speed 取值 {req.speed} 超出范围 {SPEED_RANGE[0]}–{SPEED_RANGE[1]}（倍率）")
    if req.volume is not None and not (VOLUME_RANGE[0] <= req.volume <= VOLUME_RANGE[1]):
        raise UsageError(f"--volume 取值 {req.volume} 超出范围 {VOLUME_RANGE[0]}–{VOLUME_RANGE[1]}（倍率）")
