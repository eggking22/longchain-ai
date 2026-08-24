"""Unit tests for numbering patterns and TOC title cleaning."""

from app.services.parser.patterns import (
    clean_toc_title,
    match_numbering,
    normalize_for_match,
)


def test_chapter_patterns():
    assert match_numbering("第1章 走近细胞") == 1
    assert match_numbering("第十二章 遗传与进化") == 1
    assert match_numbering("第3篇 分子与细胞") == 1
    assert match_numbering("Chapter 5 The Cell") == 1


def test_section_patterns():
    assert match_numbering("第1节 细胞是生命活动的基本单位") == 2
    assert match_numbering("1.1 细胞的生活") == 2
    assert match_numbering("1.1.1 更深层的小节") == 3


def test_sub_block_patterns():
    assert match_numbering("一、细胞学说") == 3
    assert match_numbering("（二）实验设计") == 3
    assert match_numbering("3、注意事项") == 3


def test_non_headings():
    assert match_numbering("细胞是生命活动的基本单位。") is None
    assert match_numbering("") is None
    assert match_numbering("观察洋葱鳞片叶内表皮细胞") is None


def test_clean_toc_title():
    assert clean_toc_title("第1章 走近细胞……12") == "第1章 走近细胞"
    assert clean_toc_title("第1节 细胞是生命活动的基本单位 ..... 12") == "第1节 细胞是生命活动的基本单位"
    assert clean_toc_title("第1章 走近细胞") == "第1章 走近细胞"  # digits in title kept


def test_normalize_for_match():
    assert normalize_for_match("第1章  走近细胞") == "第1章走近细胞"
    assert normalize_for_match(" 第1章走近细胞\t") == "第1章走近细胞"
