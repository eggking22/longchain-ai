"""Unit tests for heading detection, hierarchy resolution and paragraphs."""

from app.services.parser import (
    ParserConfig,
    build_document,
    clean_lines,
    detect_headings,
    extract_raw,
)
from app.services.parser.structure import HeadingCandidate, join_line_texts
from tests.conftest import build_english_pdf, build_sample_pdf


def _prep(pdf_path):
    raw = extract_raw(pdf_path)
    clean = clean_lines(raw)
    return raw, clean


def test_toc_headings_take_priority(sample_pdf):
    raw, clean = _prep(sample_pdf)
    result = detect_headings(clean.lines, raw.toc, ParserConfig())

    by_text = {c.text: c for c in result.candidates}
    assert by_text["第1章 走近细胞"].rule == "toc"
    assert by_text["第1章 走近细胞"].level == 1
    assert by_text["第1节 细胞是生命活动的基本单位"].level == 2
    assert by_text["第2节 细胞的多样性和统一性"].level == 2
    assert by_text["第2章 组成细胞的分子"].level == 1
    # sub-heading not in TOC: font rule wins over numbering (priority)
    sub = by_text["一、细胞学说"]
    assert sub.rule == "font"
    assert sub.level == 3
    assert sub.confidence > 0.6  # boosted by the numbering hit

    # every decision is recorded in the debug trace
    assert len(result.debug) == len(result.candidates)
    assert all("hits" in d for d in result.debug)


def test_font_detection_without_toc(sample_pdf_no_toc):
    raw, clean = _prep(sample_pdf_no_toc)
    result = detect_headings(clean.lines, raw.toc, ParserConfig())
    by_text = {c.text: c for c in result.candidates}
    assert by_text["第1章 走近细胞"].rule == "font"
    assert by_text["第1章 走近细胞"].level == 1
    assert by_text["第1节 细胞是生命活动的基本单位"].level == 2


def test_bold_only_heading_detected(tmp_path):
    pdf = tmp_path / "en.pdf"
    build_english_pdf(pdf)
    raw, clean = _prep(pdf)
    result = detect_headings(clean.lines, raw.toc, ParserConfig())
    assert [c.text for c in result.candidates] == ["Cell Structure"]
    assert result.candidates[0].rule == "font"
    assert result.candidates[0].evidence["kind"] == "bold"


def test_build_document_full_tree(sample_pdf):
    raw, clean = _prep(sample_pdf)
    result = detect_headings(clean.lines, raw.toc, ParserConfig())
    tree = build_document("bio", clean.lines, result.candidates, ParserConfig())

    assert tree.node_type == "document"
    chapters = [c for c in tree.children if c.node_type == "chapter"]
    assert [c.title for c in chapters] == ["第1章 走近细胞", "第2章 组成细胞的分子"]
    assert chapters[0].heading_rule == "toc"

    s1 = chapters[0].children[0]
    assert s1.title == "第1节 细胞是生命活动的基本单位"
    paras = [c for c in s1.children if c.node_type == "paragraph"]
    sub = [c for c in s1.children if c.node_type == "section"]
    assert len(paras) == 2
    assert [c.title for c in sub] == ["一、细胞学说"]

    # paragraph 1 = two lines joined without spaces (CJK)
    assert (
        paras[0].text
        == "细胞是生物体结构和功能的基本单位，除病毒外，一切生物体都是由细胞构成的，病毒没有细胞结构，必须寄生在活的宿主细胞内才能生活和增殖。"
    )
    # paragraph 3 spans pages 1-2
    p3 = sub[0].children[0]
    assert p3.node_type == "paragraph"
    assert p3.pages == [1, 2]


def test_level_compaction(tmp_path):
    # candidates jumping 1 -> 3 get compacted to 1 -> 2
    from app.services.parser.parser import LineInfo

    lines = [
        LineInfo(index=0, page_no=1, block_no=0, text="Chapter One", bbox=(0, 0, 100, 10)),
        LineInfo(index=1, page_no=1, block_no=1, text="1.1.1 deep", bbox=(0, 20, 100, 30)),
    ]
    candidates = [
        HeadingCandidate(0, "Chapter One", 1, "toc", 0.9),
        HeadingCandidate(1, "1.1.1 deep", 3, "font", 0.7),
    ]
    tree = build_document("d", lines, candidates, ParserConfig())
    assert tree.children[0].node_type == "chapter"
    assert tree.children[0].children[0].level == 2


def test_synthesized_chapters_from_sections_only(tmp_path):
    from app.services.parser.parser import LineInfo

    lines = [
        LineInfo(index=i, page_no=1, block_no=i, text=t, bbox=(0, i * 20, 100, i * 20 + 10))
        for i, t in enumerate(["第1节 A", "body a", "第2节 B", "body b", "第1节 C", "body c"])
    ]
    candidates = [
        HeadingCandidate(0, "第1节 A", 2, "font", 0.7),
        HeadingCandidate(2, "第2节 B", 2, "font", 0.7),
        HeadingCandidate(4, "第1节 C", 2, "font", 0.7),
    ]
    tree = build_document("d", lines, candidates, ParserConfig())
    chapters = [c for c in tree.children if c.node_type == "chapter"]
    # 1,2 keep running (same chapter); the second 第1节 restarts a new group
    assert len(chapters) == 2
    assert all(c.heading_rule == "synthetic" for c in chapters)
    assert [len(c.children) for c in chapters] == [2, 1]


def test_fallback_chapter_when_no_headings(tmp_path):
    from app.services.parser.parser import LineInfo

    lines = [
        LineInfo(index=i, page_no=1, block_no=i, text=f"正文第{i}行，内容较长一些以便测试段落。", bbox=(0, i * 20, 100, i * 20 + 10))
        for i in range(3)
    ]
    tree = build_document("d", lines, [], ParserConfig())
    assert len(tree.children) == 1
    assert tree.children[0].title == "正文"
    assert tree.children[0].heading_rule == "fallback"
    assert len(tree.children[0].children) == 3


def test_join_line_texts_cjk_aware():
    assert join_line_texts(["细胞是", "基本单位。"]) == "细胞是基本单位。"
    assert join_line_texts(["Cells are", "basic units."]) == "Cells are basic units."
    assert join_line_texts(["  ", "文本", ""]) == "文本"
