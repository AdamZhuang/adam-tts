from adam_tts.textsplit import chunk_text, split_sentences


def test_no_split_when_fits():
    assert chunk_text("你好。世界。", 100) == ["你好。世界。"]


def test_split_at_sentence_boundaries():
    text = "第一句话在这里。第二句话也在这里！第三句？"
    chunks = chunk_text(text, 10)
    assert "".join(chunks) == text
    assert all(len(c) <= 10 for c in chunks)
    assert all(c[-1] in "。！？" for c in chunks)


def test_overlong_sentence_falls_back_to_commas_then_hard():
    text = "甲" * 8 + "，" + "乙" * 25 + "。"
    chunks = chunk_text(text, 10)
    assert "".join(chunks) == text and all(len(c) <= 10 for c in chunks)


def test_byte_limit():
    text = "汉字汉字。" * 10
    chunks = chunk_text(text, None, max_bytes=30)
    assert "".join(chunks) == text and all(len(c.encode()) <= 30 for c in chunks)


def test_split_sentences():
    assert split_sentences("a。b！c") == ["a。", "b！", "c"]
