from __future__ import annotations

from adam_tts.backends.edge import EdgeBackend
from adam_tts.backends.mimo import MimoBackend
from adam_tts.backends.qwen import QwenBackend
from adam_tts.backends.stubs import DoubaoBackend

BACKENDS = {
    "qwen": QwenBackend,
    "mimo": MimoBackend,
    "edge": EdgeBackend,
    "doubao": DoubaoBackend,
}


def names() -> list[str]:
    return list(BACKENDS)


def create(name: str, cfg, secrets):
    from adam_tts.errors import UsageError

    if name not in BACKENDS:
        raise UsageError(f"未知后端: {name}", hint=f"可选: {', '.join(BACKENDS)}")
    return BACKENDS[name](cfg.backend_settings(name), secrets)
