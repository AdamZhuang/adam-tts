"""Audio helpers (stdlib wave + ffmpeg). Output standard: 24 kHz mono 16-bit PCM wav."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from adam_tts.errors import DependencyMissing, BackendError

SAMPLE_RATE = 24000


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def require_ffmpeg() -> str:
    p = ffmpeg_path()
    if not p:
        raise DependencyMissing("找不到 ffmpeg", hint="macOS: brew install ffmpeg")
    return p


def ffmpeg_version() -> str | None:
    p = ffmpeg_path()
    if not p:
        return None
    try:
        out = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=10).stdout
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def wav_info(data_or_path) -> tuple[int, int, int, float]:
    """(sample_rate, channels, sampwidth, duration_s)."""
    src = io.BytesIO(data_or_path) if isinstance(data_or_path, (bytes, bytearray)) else str(data_or_path)
    with wave.open(src, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    return sr, ch, sw, (n / sr if sr else 0.0)


def wav_duration(path: Path) -> float:
    return wav_info(path)[3]


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def to_standard_wav(src: Path, dst: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Convert any audio file to 24 kHz mono s16 wav via ffmpeg (atomic)."""
    ff = require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp-", suffix=".wav")
    os.close(fd)
    try:
        r = subprocess.run(
            [ff, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(sample_rate),
             "-sample_fmt", "s16", tmp],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise BackendError(f"ffmpeg 转换失败: {r.stderr.strip()[:500]}")
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_standard_wav_bytes(data: bytes, dst: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Write wav bytes; pass through if already standard, else convert with ffmpeg."""
    try:
        sr, ch, sw, _ = wav_info(data)
        standard = (sr, ch, sw) == (sample_rate, 1, 2)
    except (wave.Error, EOFError):
        standard = False
    if standard:
        atomic_write(dst, data)
        return
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(data)
        raw = Path(f.name)
    try:
        to_standard_wav(raw, dst, sample_rate)
    finally:
        raw.unlink(missing_ok=True)


def concat_wavs(parts: list[Path], dst: Path) -> None:
    """Concatenate standard wavs (same format) into dst (atomic)."""
    frames, params = [], None
    for p in parts:
        with wave.open(str(p), "rb") as w:
            if params is None:
                params = w.getparams()
            elif (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (
                params.framerate, params.nchannels, params.sampwidth):
                raise BackendError(f"拼接的音频格式不一致: {p}")
            frames.append(w.readframes(w.getnframes()))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        w.writeframes(b"".join(frames))
    atomic_write(dst, buf.getvalue())
