"""Two output modes: human text, or exactly one JSON object on stdout (--json). Logs go to stderr."""

from __future__ import annotations

import json
import sys


def emit(result: dict, json_mode: bool, human: str | None = None, stream=None) -> None:
    stream = stream or sys.stdout
    if json_mode:
        stream.write(json.dumps({"ok": True, **result}, ensure_ascii=False, default=str) + "\n")
    elif human is not None:
        stream.write(human.rstrip("\n") + "\n")
    stream.flush()


def emit_error(err: dict, json_mode: bool) -> None:
    if json_mode:
        sys.stdout.write(json.dumps({"ok": False, "error": err}, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    else:
        msg = f"错误[{err['code']} {err['type']}]: {err['message']}"
        if err.get("hint"):
            msg += f"\n提示: {err['hint']}"
        sys.stderr.write(msg + "\n")


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        sys.stderr.write(msg.rstrip("\n") + "\n")
