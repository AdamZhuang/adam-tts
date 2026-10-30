import numpy as np

from adam_tts.align import split
from adam_tts.align.base import Token
from adam_tts.align.srt import cue_texts, to_srt


def test_cut_in_silence_and_gap_mid():
    sr = 24000
    x = np.full(sr * 2, 0.3, dtype=np.float32)
    x[int(0.9 * sr):int(1.1 * sr)] = 0
    db = split.frame_db(x, sr)
    t, q = split.cut_between(db, 0.95, 1.05)
    assert q == "silence" and abs(t - 1.0) < 0.02
    t, q = split.cut_between(db, 0.4, 0.5)
    assert q == "gap_mid" and abs(t - 0.45) < 1e-9


def test_piece_spans_mismatch():
    toks = [Token(c, i, i + 1) for i, c in enumerate("abcde")]
    spans, notes = split.piece_spans(toks, [2, 3])
    assert spans == [(0, 2), (2, 5)] and not notes
    spans, notes = split.piece_spans(toks, [2, 4, 1])
    assert spans[2] is None and notes


def test_srt():
    s = to_srt([(0, 1.5, "你好"), (3661.001, 3662, "b")])
    assert "1\n00:00:00,000 --> 00:00:01,500\n你好" in s and "01:01:01,001" in s
    assert cue_texts("短句。这是一个比较长的句子，需要在逗号处切开。", 8) == ["短句。", "这是一个比较长的句子，", "需要在逗号处切开。"]
