"""Unit tests for the BM25 leg (Chinese synthetic corpus)."""

from app.schemas.retrieval import IndexedRecord
from app.services.retrieval.bm25 import Bm25Index
from app.services.retrieval.tokenizer import tokenize


def _record(chunk_id: str, text: str) -> IndexedRecord:
    return IndexedRecord(
        document_id="doc",
        chunk_id=chunk_id,
        chunk_index=0,
        text=text,
        breadcrumb=["第1章"],
        pages=[1],
        char_count=len(text),
        paragraph_ids=[],
        tokens=tokenize(text),
    )


CORPUS = [
    _record("c1", "细胞膜是细胞的边界，控制物质进出细胞。"),
    _record("c2", "线粒体是有氧呼吸的主要场所，为细胞提供能量。"),
    _record("c3", "叶绿体是光合作用的场所，将光能转化为化学能。"),
]


def test_chinese_query_hits_expected_chunk():
    index = Bm25Index.build(CORPUS)
    hits = index.search(tokenize("细胞膜的功能"), k=3)
    assert hits and hits[0][0] == "c1"
    assert all(score > 0 for _, score in hits)


def test_scores_descend():
    index = Bm25Index.build(CORPUS)
    hits = index.search(tokenize("细胞的场所"), k=3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_zero_score_hits_omitted():
    index = Bm25Index.build(CORPUS)
    assert index.search(tokenize("核糖体"), k=3) == []


def test_empty_query_tokens():
    assert Bm25Index.build(CORPUS).search([], k=3) == []


def test_empty_index():
    assert Bm25Index.build([]).search(tokenize("细胞"), k=3) == []


def test_k_truncates():
    index = Bm25Index.build(CORPUS)
    assert len(index.search(tokenize("细胞"), k=1)) <= 1
