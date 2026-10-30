"""Secret resolution: process env first, then ONE .env file as a supplement.

Values are never logged or returned in output; only their *source* is reported.
os.environ is not modified.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

KNOWN_SECRETS = ("MIMO_API_KEY", "DOUBAO_API_KEY")
KNOWN_SETTINGS = ("DOUBAO_RESOURCE_ID", "ADAM_TTS_QWEN_HOME")

REDACTED = "***"


def candidate_env_files(explicit: str | None = None, cwd: Path | None = None) -> list[Path]:
    if explicit:
        return [Path(explicit).expanduser()]
    out: list[Path] = []
    if os.environ.get("ADAM_TTS_ENV_FILE"):
        out.append(Path(os.environ["ADAM_TTS_ENV_FILE"]).expanduser())
    out.append((cwd or Path.cwd()) / ".env")
    out.append(Path("~/.config/adam_tts/.env").expanduser())
    return out


class Secrets:
    def __init__(self, env_file: str | None = None, environ: dict | None = None, cwd: Path | None = None):
        self._environ = os.environ if environ is None else environ
        self.env_file: Path | None = None
        self._file_values: dict[str, str] = {}
        explicit = env_file
        for path in candidate_env_files(explicit, cwd):
            if path.is_file():
                self.env_file = path
                self._file_values = {k: v for k, v in dotenv_values(path).items() if v}
                break
        # explicit --env-file that does not exist is a usage problem; surfaced by caller via .missing_explicit
        self.missing_explicit = bool(explicit) and self.env_file is None

    def get(self, name: str) -> str | None:
        v = self._environ.get(name)
        if v:
            return v
        return self._file_values.get(name) or None

    def source(self, name: str) -> str:
        if self._environ.get(name):
            return "env"
        if self._file_values.get(name):
            return ".env"
        return "missing"

    def report(self, names=KNOWN_SECRETS) -> dict[str, str]:
        return {n: self.source(n) for n in names}

    def redact(self, text: str) -> str:
        """Replace any known secret value appearing in text."""
        if not text:
            return text
        for n in KNOWN_SECRETS:
            v = self.get(n)
            if v and len(v) >= 4:
                text = text.replace(v, REDACTED)
        return text
