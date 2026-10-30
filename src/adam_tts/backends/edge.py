"""Microsoft Edge TTS via the edge-tts package. Fallback backend; numeric rate/pitch/volume, no style."""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

from adam_tts import audio
from adam_tts.backends.base import Capabilities, CheckResult, SynthRequest, SynthResult, VoiceInfo
from adam_tts.backends.mimo import _deep_check
from adam_tts.errors import EXIT_BACKEND, EXIT_DEPENDENCY, BackendError


def _pct(mult: float) -> str:
    n = round((mult - 1.0) * 100)
    return f"{n:+d}%"


def edge_params(req: SynthRequest) -> dict:
    return {
        "rate": _pct(req.speed) if req.speed is not None else "+0%",
        "volume": _pct(req.volume) if req.volume is not None else "+0%",
        "pitch": f"{int(req.pitch):+d}Hz" if req.pitch is not None else "+0Hz",
    }


class EdgeBackend:
    name = "edge"
    version = "1"
    default_voice = "zh-CN-YunxiNeural"

    def __init__(self, settings: dict, secrets=None):
        self.max_text_chars = int(settings.get("max_text_chars", 2000))

    def capabilities(self, voice: str | None = None) -> Capabilities:
        return Capabilities(speed=True, pitch=True, volume=True, max_text_chars=self.max_text_chars)

    def list_voices(self, lang: str | None = None) -> list[VoiceInfo]:
        import edge_tts

        try:
            raw = asyncio.run(edge_tts.list_voices())
        except Exception as e:  # noqa: BLE001
            raise BackendError(f"获取 Edge 音色列表失败: {type(e).__name__}: {e}") from e
        out = []
        for v in raw:
            loc = v.get("Locale")
            if lang and not (loc or "").lower().startswith(lang.lower()):
                continue
            out.append(VoiceInfo(v.get("ShortName"), loc, (v.get("Gender") or "").lower() or None,
                                 v.get("FriendlyName"), False))
        return sorted(out, key=lambda x: x.name)

    def synthesize(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        import edge_tts

        audio.require_ffmpeg()
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as d:
            mp3 = Path(d) / "edge.mp3"
            try:
                comm = edge_tts.Communicate(req.text, req.voice, **edge_params(req))
                asyncio.run(comm.save(str(mp3)))
            except Exception as e:  # noqa: BLE001 - edge-tts raises various types
                raise BackendError(f"Edge TTS 失败: {type(e).__name__}: {e}",
                                   hint="检查音色名（adam_tts voice list --backend edge）和网络") from e
            if not mp3.exists() or mp3.stat().st_size == 0:
                raise BackendError("Edge TTS 没有返回音频", hint="检查音色名是否正确")
            audio.to_standard_wav(mp3, out_wav)
        elapsed = time.monotonic() - t0
        return SynthResult(out=out_wav, duration_s=round(audio.wav_duration(out_wav), 3),
                           elapsed_s=round(elapsed, 2), bytes=out_wav.stat().st_size)

    def check(self, deep: bool = False) -> list[CheckResult]:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            return [CheckResult("edge.package", "fail", "edge-tts not installed", EXIT_DEPENDENCY)]
        out = [CheckResult("edge.package", "ok", "edge-tts importable")]
        try:
            n = len(self.list_voices())
            out.append(CheckResult("edge.reachable", "ok", f"{n} voices"))
        except BackendError as e:
            out.append(CheckResult("edge.reachable", "fail", e.message, EXIT_BACKEND))
        if deep and out[-1].status == "ok":
            out.append(_deep_check(self, SynthRequest(text="测试。", voice=self.default_voice)))
        return out
