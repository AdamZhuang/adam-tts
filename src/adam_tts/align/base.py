from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Token:
    text: str
    start: float
    end: float


class Aligner(Protocol):
    name: str
    model_id: str

    def align(self, audio_path: Path, text: str, language: str) -> list[Token]: ...
    def unit_counts(self, texts: list[str], language: str) -> list[int]: ...
    def check(self) -> list: ...
    def close(self) -> None: ...
