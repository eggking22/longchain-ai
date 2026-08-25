"""Unit tests for the shared tokenizer."""

from app.services.retrieval.tokenizer import tokenize


def test_chinese_segments_into_words():
    tokens = tokenize("细胞是生命活动的基本单位")
    assert "细胞" in tokens
    assert len(tokens) > 1  # not one giant token


def test_deterministic():
    assert tokenize("细胞的多样性和统一性") == tokenize("细胞的多样性和统一性")


def test_english_lowercased():
    assert tokenize("Cell Structure ATP") == ["cell", "structure", "atp"]


def test_punctuation_dropped():
    tokens = tokenize("细胞学说。、！")
    assert tokens == ["细胞学说"]


def test_mixed_cn_en():
    tokens = tokenize("ATP是细胞的直接能源物质。")
    assert "atp" in tokens
    assert "细胞" in tokens


def test_empty_and_blank():
    assert tokenize("") == []
    assert tokenize("   。、；") == []


def test_digits_kept():
    assert "1" in tokenize("第1章")
