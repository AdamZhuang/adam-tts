import pytest

from adam_tts.backends.base import Capabilities, SynthRequest, validate_request
from adam_tts.backends.edge import edge_params
from adam_tts.backends.registry import BACKENDS
from adam_tts.errors import UnsupportedParam, UsageError


def caps(name):
    return BACKENDS[name]({}, None).capabilities()


@pytest.mark.parametrize("backend,field,value", [
    ("mimo", "speed", 1.2), ("mimo", "pitch", 5), ("mimo", "volume", 1.1), ("mimo", "lang", "zh"),
    ("qwen", "speed", 1.2), ("qwen", "pitch", 5), ("qwen", "volume", 1.1),
    ("edge", "style", "平静"), ("edge", "prefix", "(x)"), ("edge", "lang", "zh"),
    ("doubao", "prefix", "(x)"), ("doubao", "lang", "zh"),
])
def test_unsupported_params_error(backend, field, value):
    req = SynthRequest(text="你好", voice="v", **{field: value})
    with pytest.raises(UnsupportedParam):
        validate_request(backend, caps(backend), req)


@pytest.mark.parametrize("backend,field,value", [
    ("mimo", "style", "x"), ("mimo", "prefix", "(x)"), ("qwen", "style", "x"), ("qwen", "lang", "zh"),
    ("edge", "speed", 1.2), ("edge", "pitch", -5), ("edge", "volume", 0.8),
    ("doubao", "style", "x"), ("doubao", "speed", 1.2),
])
def test_supported_params_pass(backend, field, value):
    validate_request(backend, caps(backend), SynthRequest(text="你好", voice="v", **{field: value}))


def test_speed_hint_mentions_style():
    with pytest.raises(UnsupportedParam) as e:
        validate_request("mimo", caps("mimo"), SynthRequest(text="x", voice="v", speed=1.1))
    assert "--style" in e.value.hint and e.value.exit_code == 4


@pytest.mark.parametrize("speed", [0.3, 2.5])
def test_speed_range(speed):
    with pytest.raises(UsageError):
        validate_request("edge", caps("edge"), SynthRequest(text="x", voice="v", speed=speed))


def test_empty_text_and_voice():
    c = Capabilities()
    with pytest.raises(UsageError):
        validate_request("x", c, SynthRequest(text="  ", voice="v"))
    with pytest.raises(UsageError):
        validate_request("x", c, SynthRequest(text="hi", voice=""))


def test_edge_param_mapping():
    p = edge_params(SynthRequest(text="x", voice="v", speed=1.15, volume=0.8, pitch=-3))
    assert p == {"rate": "+15%", "volume": "-20%", "pitch": "-3Hz"}
    assert edge_params(SynthRequest(text="x", voice="v")) == {"rate": "+0%", "volume": "+0%", "pitch": "+0Hz"}
