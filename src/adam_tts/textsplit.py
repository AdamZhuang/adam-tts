"""Split long text into chunks at sentence boundaries (never mid-sentence)."""

from __future__ import annotations

import re

_SENT_END = re.compile(r"(?<=[。！？；!?;…\n])")


def split_sentences(text: str) -> list[str]:
    parts = [p for p in _SENT_END.split(text) if p and p.strip()]
    return parts or [text]


def chunk_text(text: str, max_chars: int | None, max_bytes: int | None = None) -> list[str]:
    def fits(s: str) -> bool:
        return (max_chars is None or len(s) <= max_chars) and (
            max_bytes is None or len(s.encode("utf-8")) <= max_bytes)

    if fits(text):
        return [text]
    chunks: list[str] = []
    cur = ""
    for sent in split_sentences(text):
        if not fits(sent):
            # a single sentence over the limit: fall back to commas, then hard split as last resort
            sub = [p for p in re.split(r"(?<=[，,、：:])", sent) if p]
            for piece in sub:
                while not fits(piece):
                    n = max_chars or len(piece) // 2 or 1
                    while n > 1 and not fits(piece[:n]):
                        n -= 1
                    if cur:
                        chunks.append(cur)
                        cur = ""
                    chunks.append(piece[:n])
                    piece = piece[n:]
                if fits(cur + piece):
                    cur += piece
                else:
                    chunks.append(cur)
                    cur = piece
            continue
        if fits(cur + sent):
            cur += sent
        else:
            if cur:
                chunks.append(cur)
            cur = sent
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]
