"""Paths for the local Qwen setup (Mac mini). ADAM_TTS_QWEN_HOME (env/.env) > config > ~/qwen3-tts."""

from __future__ import annotations

import sys
from pathlib import Path

LANGS = {"zh": "Chinese", "zh-cn": "Chinese", "cn": "Chinese", "en": "English", "ja": "Japanese",
         "ko": "Korean", "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
         "es": "Spanish", "it": "Italian"}


def language_name(lang: str | None, default: str = "Chinese") -> str:
    if not lang:
        return default
    return LANGS.get(lang.lower(), lang)


def qwen_home(settings: dict, secrets) -> Path:
    v = (secrets.get("ADAM_TTS_QWEN_HOME") if secrets else None) or settings.get("home") or "~/qwen3-tts"
    return Path(v).expanduser()


def resolve_python(configured: str | None, default_venv: Path) -> str:
    """'self' = this interpreter; explicit path; else the existing venv if present; else this interpreter."""
    if configured == "self":
        return sys.executable
    if configured:
        return str(Path(configured).expanduser())
    if default_venv.exists():
        return str(default_venv)
    return sys.executable


def resolve_model(home: Path, configured: str | None, default_rel: str) -> Path:
    p = Path(configured).expanduser() if configured else Path(default_rel)
    return p if p.is_absolute() else home / p
