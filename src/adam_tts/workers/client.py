"""Spawn a worker script under a given python interpreter and talk JSON lines to it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from adam_tts.errors import BackendError, DependencyMissing
from adam_tts.workers import WORKER_DIR


class WorkerClient:
    def __init__(self, python: str, script: str, *, error_cls=BackendError, label: str = "worker",
                 verbose: bool = False):
        self.python = python
        self.script = str(WORKER_DIR / script) if not os.path.isabs(script) else script
        self.error_cls = error_cls
        self.label = label
        self.verbose = verbose
        self.proc: subprocess.Popen | None = None
        self._log = None

    def start(self) -> dict:
        if self.proc is not None:
            return {}
        if not Path(self.python).exists() and self.python != sys.executable:
            raise DependencyMissing(f"{self.label}: 找不到 python 解释器 {self.python}",
                                    hint="检查 ADAM_TTS_QWEN_HOME 或 config.toml 里的 python 路径")
        self._log = None if self.verbose else tempfile.TemporaryFile(mode="w+b")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        self.proc = subprocess.Popen([self.python, self.script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None if self.verbose else self._log, env=env, text=True,
                                     encoding="utf-8", bufsize=1)
        return self._read()

    def _stderr_tail(self, n: int = 1500) -> str:
        if self._log is None:
            return ""
        try:
            self._log.seek(0, os.SEEK_END)
            size = self._log.tell()
            self._log.seek(max(0, size - n))
            return self._log.read().decode("utf-8", errors="ignore")
        except (OSError, ValueError):
            return ""

    def _read(self) -> dict:
        while True:
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.wait(timeout=5) if self.proc.poll() is None else self.proc.poll()
                raise self.error_cls(f"{self.label} 进程意外退出 (code={code})",
                                     hint=self._stderr_tail()[-800:] or "用 -v 查看 worker 日志")
            try:
                resp = json.loads(line)
                if isinstance(resp, dict) and "ok" in resp:
                    break
            except ValueError:
                pass
            sys.stderr.write(line) if self.verbose else None  # stray stdout before serve(): not protocol
        if not resp.get("ok"):
            et = resp.get("type", "")
            cls = DependencyMissing if et in ("ModuleNotFoundError", "ImportError") else self.error_cls
            raise cls(f"{self.label}: {resp.get('error')}", hint=(resp.get("trace") or "")[-600:] or None)
        return resp

    def request(self, op: str, **kw) -> dict:
        self.start()
        try:
            self.proc.stdin.write(json.dumps({"op": op, **kw}, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise self.error_cls(f"{self.label} 进程已退出", hint=self._stderr_tail()[-800:]) from e
        return self._read()

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write('{"op":"quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            self.proc.kill()
        finally:
            self.proc = None
            if self._log:
                self._log.close()
