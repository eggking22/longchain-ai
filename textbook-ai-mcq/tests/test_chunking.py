"""Unit tests for paragraph-aware chunking."""

from app.schemas.document import DocNode
from app.services.parser import ParserConfig
from app.services.parser.chunking import chunk_document, split_long_paragraph, split_sentences


def _para(node_id: str, text: str, pages=(1,)) -> DocNode:
    return DocNode(node_id=node_id, node_type="paragraph", level=3, text=text, pages=list(pages))


def _tree() -> DocNode:
    ch1 = DocNode(node_id="c1", node_type="chapter", title="第1章 走近细胞", level=1)
    s1 = DocNode(node_id="s1", node_type="section", title="第1节 细胞", level=2)
    s1.children = [_para("p1", "甲" * 300), _para("p2", "乙" * 400)]
    ch1.children = [s1]
    ch2 = DocNode(node_id="c2", node_type="chapter", title="第2章 分子", level=1)
    s2 = DocNode(node_id="s2", node_type="section", title="第1节 元素", level=2)
    s2.children = [_para("p3", "丙" * 200, pages=(4, 5))]
    ch2.children = [s2]
    root = DocNode(node_id="root", node_type="document", title="doc", level=0)
    root.children = [ch1, ch2]
    return root


def test_chunks_never_cross_sections():
    chunks = chunk_document(_tree(), "doc", ParserConfig())
    assert len(chunks) == 3  # [p1 | p2] (300+400 exceeds target), [p3]
    crumbs = [c.breadcrumb for c in chunks]
    assert crumbs[0] == ["第1章 走近细胞", "第1节 细胞"]
    assert crumbs[-1] == ["第2章 分子", "第1节 元素"]
    # p3 chunk records both pages it spans
    assert chunks[-1].pages == [4, 5]


def test_paragraph_ids_are_traceable():
    chunks = chunk_document(_tree(), "doc", ParserConfig())
    assert chunks[0].paragraph_ids == ["p1"]
    assert chunks[1].paragraph_ids == ["p2"]
    assert all(c.chunk_id for c in chunks)
    assert all(c.char_count == len(c.text) for c in chunks)


def test_split_sentences():
    parts = split_sentences("细胞是基本单位。病毒没有细胞结构！它们都必须依赖活细胞；才能增殖。")
    assert parts == ["细胞是基本单位。", "病毒没有细胞结构！", "它们都必须依赖活细胞；", "才能增殖。"]


def test_split_long_paragraph_respects_limits_and_overlap():
    text = "。".join(f"第{i}句话内容固定长度" for i in range(1, 80)) + "。"
    pieces = split_long_paragraph(text, target=100, hard_max=150, overlap=1)
    assert len(pieces) > 3
    assert all(len(p) <= 150 for p in pieces)
    # one-sentence overlap: piece i+1 starts with the last sentence of piece i
    last_sentence = pieces[0].split("。")[-2] + "。"
    assert pieces[1].startswith(last_sentence)


def test_target_accumulation():
    # 300 + 400 > 600 target -> split; but 300 + 250 would fit together
    ch = DocNode(node_id="c", node_type="chapter", title="章", level=1)
    ch.children = [_para("a", "甲" * 300), _para("b", "乙" * 250)]
    root = DocNode(node_id="r", node_type="document", title="d", level=0, children=[ch])
    chunks = chunk_document(root, "d", ParserConfig())
    assert len(chunks) == 1
    assert chunks[0].paragraph_ids == ["a", "b"]
