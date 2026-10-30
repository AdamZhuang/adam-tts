from __future__ import annotations

import re

from adam_tts.textsplit import split_sentences


def _ts(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(cues: list[tuple[float, float, str]]) -> str:
    out = []
    for i, (a, b, text) in enumerate(cues, 1):
        out.append(f"{i}\n{_ts(a)} --> {_ts(b)}\n{text.strip()}\n")
    return "\n".join(out)


def cue_texts(text: str, max_chars: int) -> list[str]:
    """Sentences, further split at commas when longer than max_chars."""
    out = []
    for s in split_sentences(text.replace("\n", "")):
        if len(s) <= max_chars:
            out.append(s)
            continue
        cur = ""
        for p in [x for x in re.split(r"(?<=[，,、：:])", s) if x]:
            if cur and len(cur) + len(p) > max_chars:
                out.append(cur)
                cur = p
            else:
                cur += p
        if cur:
            out.append(cur)
    return [c for c in out if c.strip()]


def strip_display(text: str) -> str:
    """Subtitle text: drop trailing sentence punctuation that looks noisy on screen."""
    return text.strip().rstrip("。；;，,")
