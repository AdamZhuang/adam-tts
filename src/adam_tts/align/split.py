"""Map aligner tokens back to text pieces and choose cut points (ported from split_scene_lines.py).

Cut between two adjacent lines = midpoint of the longest silence inside the aligner gap (±80 ms);
if there is no silence there, the gap midpoint is used and the cut is flagged weak.
"""

from __future__ import annotations

import numpy as np

from adam_tts.align.base import Token

FR = 0.01  # 10 ms frames
SILENCE_DB = -40.0
SEARCH_PAD = 0.08


def frame_db(wav: np.ndarray, sr: int) -> np.ndarray:
    hop = int(sr * FR)
    n = len(wav) // hop
    if n == 0:
        return np.zeros(0)
    x = wav[: n * hop].reshape(n, hop).astype(np.float64)
    rms = np.sqrt((x ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms / (np.abs(wav).max() + 1e-9))


def silent_runs(db: np.ndarray, thr: float, min_len: float) -> list[tuple[float, float]]:
    runs, start = [], None
    for i, v in enumerate(np.append(db, 0.0)):
        if v < thr and start is None:
            start = i
        elif v >= thr and start is not None:
            if (i - start) * FR >= min_len:
                runs.append((start * FR, i * FR))
            start = None
    return runs


def best_silence_mid(db: np.ndarray, lo: float, hi: float, thr: float = SILENCE_DB):
    a, b = max(0, int(lo / FR)), min(len(db), int(np.ceil(hi / FR)))
    best = None
    for s, e in silent_runs(db[a:b], thr, 0.03):
        if best is None or (e - s) > (best[1] - best[0]):
            best = (s, e)
    if best is None:
        return None, 0.0
    return a * FR + (best[0] + best[1]) / 2, best[1] - best[0]


def piece_spans(tokens: list[Token], counts: list[int]) -> tuple[list[tuple[float, float] | None], list[str]]:
    """Assign consecutive tokens to pieces by unit counts. Returns spans and notes."""
    notes = []
    if len(tokens) != sum(counts):
        notes.append(f"unit count mismatch tokens={len(tokens)} expected={sum(counts)}")
    spans, k = [], 0
    for c in counts:
        seg = tokens[k:k + c]
        k += c
        spans.append((seg[0].start, seg[-1].end) if seg else None)
    return spans, notes


def cut_between(db: np.ndarray, end_prev: float, start_next: float) -> tuple[float, str]:
    lo, hi = min(end_prev, start_next), max(end_prev, start_next)
    mid, _ = best_silence_mid(db, lo - SEARCH_PAD, hi + SEARCH_PAD)
    if mid is None:
        return (end_prev + start_next) / 2, "gap_mid"
    return mid, "silence"
