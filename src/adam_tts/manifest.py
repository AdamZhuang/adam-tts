"""manifest.json (schema_version 1) read/write + hashing helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from adam_tts import __version__
from adam_tts.audio import atomic_write_bytes
from adam_tts.errors import UsageError

SCHEMA_VERSION = 1
NAME = "manifest.json"


def sha(obj, n: int = 16) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def text_hash(text: str) -> str:
    return sha(text.strip())


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load(outdir: Path) -> dict | None:
    p = Path(outdir) / NAME
    if not p.is_file():
        return None
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise UsageError(f"manifest 损坏: {p}: {e}") from e
    if m.get("schema_version") != SCHEMA_VERSION:
        raise UsageError(f"不支持的 manifest 版本: {m.get('schema_version')}")
    return m


def new(tsv: Path) -> dict:
    return {"schema_version": SCHEMA_VERSION, "tool_version": __version__, "created_at": now_iso(),
            "updated_at": now_iso(), "input": {}, "params": {}, "sample_rate": 24000, "full": None,
            "scenes": [], "lines": []}


def save(outdir: Path, m: dict) -> None:
    m["updated_at"] = now_iso()
    m["tool_version"] = __version__
    atomic_write_bytes(Path(outdir) / NAME, (json.dumps(m, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
