import io
import math
import struct
import wave
from pathlib import Path

import pytest

from adam_tts.backends import registry
from adam_tts.backends.base import Capabilities, CheckResult, SynthResult, VoiceInfo
from adam_tts.errors import BackendError

FAKE_KEY = "sk-fake-UNIT-TEST-0123456789abcdef"


def make_wav(seconds: float, sr: int = 24000) -> bytes:
    n = int(seconds * sr)
    frames = b"".join(struct.pack("<h", int(3000 * math.sin(i / 20))) for i in range(n))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames)
    return buf.getvalue()


class FakeBackend:
    """0.01 s of audio per character; records requests; can be told to fail."""

    name = "fake"
    version = "1"
    default_voice = "fv1"
    calls: list = []
    fail = False
    caps = Capabilities(style=True, prefix=True, max_text_chars=20)

    def __init__(self, settings, secrets=None):
        self.secrets = secrets

    def capabilities(self, voice=None):
        return type(self).caps

    def list_voices(self, lang=None):
        return [VoiceInfo("fv1", "zh"), VoiceInfo("fv2", "en")]

    def check(self, deep=False):
        return [CheckResult("fake.ok", "ok", "fine")]

    def synthesize(self, req, out_wav: Path):
        if type(self).fail:
            raise BackendError("boom")
        type(self).calls.append(req)
        data = make_wav(0.01 * len(req.text))
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(data)
        return SynthResult(out=out_wav, duration_s=0.01 * len(req.text), elapsed_s=0.0, bytes=len(data))


@pytest.fixture(autouse=True)
def isolated(request, tmp_path, monkeypatch):
    """No real config/.env; fake backend registered; known secret env cleared. Not applied to smoke tests."""
    if request.node.get_closest_marker("smoke"):
        yield
        return
    monkeypatch.setenv("ADAM_TTS_CONFIG", str(tmp_path / "nope.toml"))
    monkeypatch.delenv("ADAM_TTS_ENV_FILE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for k in ("MIMO_API_KEY", "DOUBAO_API_KEY", "DOUBAO_RESOURCE_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(registry.BACKENDS, "fake", FakeBackend)
    FakeBackend.calls = []
    FakeBackend.fail = False
    FakeBackend.caps = Capabilities(style=True, prefix=True, max_text_chars=20)
    yield


@pytest.fixture
def fake_config(tmp_path, monkeypatch):
    p = tmp_path / "config.toml"
    p.write_text(
        'default_preset = "fk"\nfallback_preset = "edge"\n'
        '[presets.fk]\nbackend = "fake"\nvoice = "fv2"\nstyle = "平静"\nprefix = "(平静)"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ADAM_TTS_CONFIG", str(p))
    return p
