"""Xiaomi MiMo TTS (mimo-v2.5-tts, preset voices), OpenAI-compatible chat.completions.

Style instruction -> `user` message (not spoken); prefix + text -> `assistant` message.
Ported from the reference script mimo_tts.py.
"""

from __future__ import annotations

import base64
import json
import tempfile
import time
from pathlib import Path

import requests

from adam_tts import audio
from adam_tts.backends.base import Capabilities, CheckResult, SynthRequest, SynthResult, VoiceInfo
from adam_tts.errors import EXIT_BACKEND, EXIT_CONFIG, BackendError, ConfigError

KEY_NAME = "MIMO_API_KEY"

# https://mimo.mi.com/docs/usage-guide/speech-synthesis-v2.5
VOICES = [
    VoiceInfo("苏打", "zh", "male", "中文男声", True),
    VoiceInfo("白桦", "zh", "male", "中文男声", True),
    VoiceInfo("冰糖", "zh", "female", "中文女声", True),
    VoiceInfo("茉莉", "zh", "female", "中文女声", True),
    VoiceInfo("Mia", "en", "female", "English", True),
    VoiceInfo("Chloe", "en", "female", "English", True),
    VoiceInfo("Milo", "en", "male", "English", True),
    VoiceInfo("Dean", "en", "male", "English", True),
    VoiceInfo("mimo_default", None, None, "按集群默认（中国集群=冰糖）", True),
]

RETRY_STATUS = {429, 500, 502, 503, 504}


class MimoBackend:
    name = "mimo"
    version = "1"
    default_voice = "苏打"

    def __init__(self, settings: dict, secrets, session: requests.Session | None = None):
        self.base_url = settings.get("base_url", "https://api.xiaomimimo.com/v1").rstrip("/")
        self.model = settings.get("model", "mimo-v2.5-tts")
        self.max_text_chars = int(settings.get("max_text_chars", 600))
        self.timeout = float(settings.get("timeout_s", 300))
        self.retries = int(settings.get("retries", 3))
        self.secrets = secrets
        self.session = session or requests.Session()

    def capabilities(self, voice: str | None = None) -> Capabilities:
        return Capabilities(style=True, prefix=True, max_text_chars=self.max_text_chars)

    def list_voices(self, lang: str | None = None) -> list[VoiceInfo]:
        if not lang:
            return list(VOICES)
        l = lang.lower().split("-")[0]
        return [v for v in VOICES if v.lang is None or v.lang == l]

    def _key(self) -> str:
        k = self.secrets.get(KEY_NAME)
        if not k:
            raise ConfigError(f"缺少 {KEY_NAME}", hint=f"在环境变量或 .env 中设置 {KEY_NAME}（不要把值贴到聊天里）")
        return k

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}

    def build_payload(self, req: SynthRequest) -> dict:
        messages = []
        if req.style:
            messages.append({"role": "user", "content": req.style})
        messages.append({"role": "assistant", "content": (req.prefix or "") + req.text})
        return {"model": self.model, "messages": messages, "audio": {"format": "wav", "voice": req.voice}}

    def _post(self, payload: dict) -> requests.Response:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_err = None
        for attempt in range(self.retries):
            try:
                r = self.session.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                                      data=body, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = f"{type(e).__name__}: {e}"
            else:
                if r.status_code == 200:
                    return r
                if r.status_code not in RETRY_STATUS:
                    raise self._http_error(r)
                last_err = f"HTTP {r.status_code}"
            if attempt < self.retries - 1:
                time.sleep(2 ** attempt)
        raise BackendError(f"MiMo 请求失败（重试 {self.retries} 次）: {self.secrets.redact(str(last_err))}")

    def _http_error(self, r: requests.Response) -> BackendError:
        text = self.secrets.redact(r.text[:500])
        hint = "运行 adam_tts doctor --backend mimo 检查密钥" if r.status_code in (401, 403) else None
        return BackendError(f"MiMo HTTP {r.status_code}: {text}", hint=hint)

    def synthesize(self, req: SynthRequest, out_wav: Path) -> SynthResult:
        t0 = time.monotonic()
        r = self._post(self.build_payload(req))
        try:
            body = r.json()
            data = base64.b64decode(body["choices"][0]["message"]["audio"]["data"])
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise BackendError(f"MiMo 响应格式异常: {type(e).__name__}") from e
        elapsed = time.monotonic() - t0
        audio.write_standard_wav_bytes(data, out_wav)
        return SynthResult(out=out_wav, duration_s=round(audio.wav_duration(out_wav), 3),
                           elapsed_s=round(elapsed, 2), bytes=out_wav.stat().st_size,
                           extra={"usage": body.get("usage")})

    def check(self, deep: bool = False) -> list[CheckResult]:
        src = self.secrets.source(KEY_NAME)
        if src == "missing":
            return [CheckResult("mimo.key", "fail", f"{KEY_NAME} missing", EXIT_CONFIG)]
        out = [CheckResult("mimo.key", "ok", f"{KEY_NAME} set ({src})")]
        try:
            r = self.session.get(f"{self.base_url}/models", headers=self._headers(), timeout=20)
            if r.status_code == 200:
                ids = [m.get("id") for m in r.json().get("data", [])]
                if self.model in ids:
                    out.append(CheckResult("mimo.reachable", "ok", f"auth ok, {self.model} available"))
                else:
                    out.append(CheckResult("mimo.reachable", "warn", f"auth ok, {self.model} not listed"))
            elif r.status_code in (401, 403):
                out.append(CheckResult("mimo.reachable", "fail", f"HTTP {r.status_code}: key rejected", EXIT_CONFIG))
            else:
                out.append(CheckResult("mimo.reachable", "fail", f"HTTP {r.status_code}", EXIT_BACKEND))
        except requests.RequestException as e:
            out.append(CheckResult("mimo.reachable", "fail", self.secrets.redact(f"{type(e).__name__}: {e}"),
                                   EXIT_BACKEND))
        if deep and out[-1].status != "fail":
            out.append(_deep_check(self, SynthRequest(text="测试。", voice=self.default_voice)))
        return out


def _deep_check(backend, req: SynthRequest) -> CheckResult:
    with tempfile.TemporaryDirectory() as d:
        try:
            res = backend.synthesize(req, Path(d) / "probe.wav")
            return CheckResult(f"{backend.name}.synth", "ok", f"{res.duration_s:.2f}s audio in {res.elapsed_s:.2f}s")
        except Exception as e:  # noqa: BLE001 - report any failure as a check result
            code = getattr(e, "exit_code", EXIT_BACKEND)
            return CheckResult(f"{backend.name}.synth", "fail", str(e), code)
