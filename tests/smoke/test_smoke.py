"""Real backend calls. Run with: ADAM_TTS_SMOKE=1 uv run pytest -m smoke"""

import os
import wave

import pytest

from adam_tts.cli import main

pytestmark = pytest.mark.smoke
SMOKE = os.environ.get("ADAM_TTS_SMOKE") == "1"
TEXT = "这是一次冒烟测试。"


def _ok(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 24000 and w.getnframes() > 0


@pytest.mark.skipif(not (SMOKE and os.environ.get("MIMO_API_KEY")), reason="needs ADAM_TTS_SMOKE=1 + MIMO_API_KEY")
def test_mimo(tmp_path):
    out = tmp_path / "mimo.wav"
    assert main(["say", TEXT, "-o", str(out)]) == 0
    _ok(out)


@pytest.mark.skipif(not SMOKE, reason="needs ADAM_TTS_SMOKE=1")
def test_edge(tmp_path):
    out = tmp_path / "edge.wav"
    assert main(["say", TEXT, "-o", str(out), "--preset", "edge"]) == 0
    _ok(out)


@pytest.mark.skip(reason="qwen backend lands in M2")
def test_qwen():
    pass


@pytest.mark.skip(reason="doubao backend lands in M3 (also needs ADAM_TTS_SMOKE_DOUBAO=1)")
def test_doubao():
    pass
