"""Audio helpers on numpy + soundfile. Standard output: 24 kHz mono 16-bit PCM wav.

ffmpeg is optional: soundfile (libsndfile >= 1.1) decodes wav/flac/mp3 itself; ffmpeg is only used
as a fallback decoder for formats libsndfile cannot read.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from adam_tts.errors import BackendError, DependencyMissing

SAMPLE_RATE = 24000


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffmpeg_version() -> str | None:
    p = ffmpeg_path()
    if not p:
        return None
    try:
        out = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=10).stdout
        return out.splitlines()[0] if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1).astype(np.float32) if x.ndim == 2 else x.astype(np.float32)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to or len(x) == 0:
        return x.astype(np.float32)
    n = int(round(len(x) * sr_to / sr_from))
    t_old = np.arange(len(x)) / sr_from
    t_new = np.arange(n) / sr_to
    return np.interp(t_new, t_old, x).astype(np.float32)


def _ffmpeg_decode(src: Path, sr: int) -> np.ndarray:
    ff = ffmpeg_path()
    if not ff:
        raise DependencyMissing(f"soundfile 无法解码 {src.suffix}，且找不到 ffmpeg", hint="macOS: brew install ffmpeg")
    r = subprocess.run([ff, "-v", "error", "-i", str(src), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                       capture_output=True)
    if r.returncode != 0:
        raise BackendError(f"ffmpeg 解码失败: {r.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def read_audio(src, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode to mono float32 at `sr`. `src` is a path or bytes."""
    try:
        data, file_sr = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else str(src),
                                dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError, TypeError):
        if isinstance(src, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                f.write(src)
            try:
                return _ffmpeg_decode(Path(f.name), sr)
            finally:
                os.unlink(f.name)
        return _ffmpeg_decode(Path(src), sr)
    return resample(_mono(np.asarray(data)), file_sr, sr)


def write_wav(dst: Path, x: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Atomic 16-bit PCM mono wav write."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp-", suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp, np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0), sr, subtype="PCM_16", format="WAV")
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_bytes(dst: Path, data: bytes) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=".tmp-", suffix=dst.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def is_standard_wav(data: bytes) -> bool:
    try:
        info = sf.info(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        return False
    return info.format == "WAV" and info.samplerate == SAMPLE_RATE and info.channels == 1 and info.subtype == "PCM_16"


def write_standard_wav_bytes(data: bytes, dst: Path) -> None:
    """Pass through if already 24k mono s16 wav; else decode + resample + write."""
    if is_standard_wav(data):
        atomic_write_bytes(dst, data)
    else:
        write_wav(dst, read_audio(data))


def to_standard_wav(src: Path, dst: Path) -> None:
    write_wav(dst, read_audio(src))


def wav_info(path) -> tuple[int, int, float]:
    """(sample_rate, channels, duration_s)."""
    info = sf.info(str(path))
    return info.samplerate, info.channels, info.frames / info.samplerate if info.samplerate else 0.0


def wav_duration(path) -> float:
    return wav_info(path)[2]


def silence(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(round(max(seconds, 0) * sr)), dtype=np.float32)


def concat(parts: list[np.ndarray], gap_s: float = 0.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    out = []
    for i, p in enumerate(parts):
        if i and gap_s > 0:
            out.append(silence(gap_s, sr))
        out.append(np.asarray(p, dtype=np.float32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def concat_wavs(paths: list[Path], dst: Path, gap_s: float = 0.0) -> None:
    write_wav(dst, concat([read_audio(p) for p in paths], gap_s))
