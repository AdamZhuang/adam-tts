"""Qwen3-ForcedAligner-0.6B via qwen-asr, in a worker under <home>/aligner/.venv/bin/python by default."""

from __future__ import annotations

from pathlib import Path

from adam_tts.align.base import Token
from adam_tts.backends.base import CheckResult
from adam_tts.errors import EXIT_DEPENDENCY, AlignError, DependencyMissing
from adam_tts.localmodels import qwen_home, resolve_model, resolve_python
from adam_tts.workers.client import WorkerClient

DEFAULT_MODEL = "models/Qwen3-ForcedAligner-0.6B"


class QwenAligner:
    name = "qwen"

    def __init__(self, settings: dict, qwen_settings: dict, secrets=None):
        self.home = qwen_home(qwen_settings, secrets)
        self.python = resolve_python(settings.get("python"), self.home / "aligner" / ".venv" / "bin" / "python")
        self.model = resolve_model(self.home, settings.get("model"), DEFAULT_MODEL)
        self.device = settings.get("device")
        self.verbose = bool(settings.get("verbose"))
        self.model_id = self.model.name
        self.worker: WorkerClient | None = None

    def _w(self) -> WorkerClient:
        if self.worker is None:
            if not self.model.is_dir():
                raise DependencyMissing(f"找不到对齐模型目录: {self.model}",
                                        hint="设置 ADAM_TTS_QWEN_HOME 或 config.toml [aligner] model")
            self.worker = WorkerClient(self.python, "aligner_worker.py", error_cls=AlignError,
                                       label="aligner worker", verbose=self.verbose)
            self.worker.request("load", model=str(self.model), device=self.device)
        return self.worker

    def align(self, audio_path: Path, text: str, language: str) -> list[Token]:
        r = self._w().request("align", audio=str(audio_path), text=text, language=language)
        return [Token(t["text"], t["start"], t["end"]) for t in r["tokens"]]

    def unit_counts(self, texts: list[str], language: str) -> list[int]:
        return list(self._w().request("counts", texts=texts, language=language)["counts"])

    def close(self) -> None:
        if self.worker:
            self.worker.close()
            self.worker = None

    def check(self) -> list[CheckResult]:
        from adam_tts.backends.qwen import _python_check

        return [CheckResult("aligner.model", "ok" if self.model.is_dir() else "fail", str(self.model),
                            EXIT_DEPENDENCY),
                _python_check("aligner.python", self.python, "import qwen_asr, torch; "
                              "print(torch.__version__, torch.backends.mps.is_available())")]


ALIGNERS = {"qwen": QwenAligner}


def create(cfg, secrets):
    s = dict(cfg.data.get("aligner", {}))
    kind = s.get("kind", "qwen")
    if kind not in ALIGNERS:
        from adam_tts.errors import ConfigError

        raise ConfigError(f"未知对齐器: {kind}")
    return ALIGNERS[kind](s, cfg.backend_settings("qwen"), secrets)
