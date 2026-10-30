import base64
import json

import pytest

from adam_tts.backends.base import SynthRequest
from adam_tts.backends.mimo import MimoBackend
from adam_tts.errors import BackendError, ConfigError
from adam_tts.secrets import Secrets
from conftest import FAKE_KEY, make_wav


class Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return self._body if not isinstance(self._body, str) else json.loads(self._body)


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, headers=None, data=None, timeout=None):
        self.posts.append({"url": url, "headers": headers, "payload": json.loads(data)})
        return self.responses.pop(0)

    def get(self, url, headers=None, timeout=None):
        return self.responses.pop(0)


def backend(monkeypatch, responses, key=FAKE_KEY):
    if key:
        monkeypatch.setenv("MIMO_API_KEY", key)
    s = Session(responses)
    return MimoBackend({"retries": 2}, Secrets(), session=s), s


def ok_body(seconds=0.5):
    return {"choices": [{"message": {"audio": {"data": base64.b64encode(make_wav(seconds)).decode()}}}],
            "usage": {"total_tokens": 1}}


def test_payload_style_in_user_text_in_assistant(monkeypatch, tmp_path):
    be, s = backend(monkeypatch, [Resp(200, ok_body())])
    res = be.synthesize(SynthRequest(text="你好。", voice="苏打", style="平静", prefix="(平静 干练)"),
                        tmp_path / "o.wav")
    p = s.posts[0]["payload"]
    assert s.posts[0]["url"].endswith("/chat/completions")
    assert p["model"] == "mimo-v2.5-tts"
    assert p["messages"] == [{"role": "user", "content": "平静"},
                             {"role": "assistant", "content": "(平静 干练)你好。"}]
    assert p["audio"] == {"format": "wav", "voice": "苏打"}
    assert abs(res.duration_s - 0.5) < 0.01


def test_no_style_message_when_none(monkeypatch):
    be, _ = backend(monkeypatch, [])
    assert [m["role"] for m in be.build_payload(SynthRequest(text="x", voice="苏打"))["messages"]] == ["assistant"]


def test_missing_key_is_config_error(monkeypatch, tmp_path):
    be, _ = backend(monkeypatch, [], key=None)
    with pytest.raises(ConfigError):
        be.synthesize(SynthRequest(text="x", voice="苏打"), tmp_path / "o.wav")


def test_http_error_redacts_key(monkeypatch, tmp_path):
    be, _ = backend(monkeypatch, [Resp(401, f"bad key {FAKE_KEY}")])
    with pytest.raises(BackendError) as e:
        be.synthesize(SynthRequest(text="x", voice="苏打"), tmp_path / "o.wav")
    assert FAKE_KEY not in e.value.message and e.value.exit_code == 5


def test_retry_then_success(monkeypatch, tmp_path):
    monkeypatch.setattr("time.sleep", lambda s: None)
    be, s = backend(monkeypatch, [Resp(503, "busy"), Resp(200, ok_body())])
    be.synthesize(SynthRequest(text="x", voice="苏打"), tmp_path / "o.wav")
    assert len(s.posts) == 2


def test_check_reports_source_not_value(monkeypatch):
    be, _ = backend(monkeypatch, [Resp(200, {"data": [{"id": "mimo-v2.5-tts"}]})])
    checks = be.check()
    assert [c.status for c in checks] == ["ok", "ok"]
    assert all(FAKE_KEY not in c.detail for c in checks)


def test_check_rejected_key(monkeypatch):
    be, _ = backend(monkeypatch, [Resp(401, "no")])
    assert be.check()[-1].status == "fail" and be.check.__self__  # noqa
