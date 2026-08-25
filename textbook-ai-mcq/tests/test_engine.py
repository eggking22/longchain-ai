"""End-to-end tests: synthetic PDF -> ingest -> build_index -> retrieve.

The engine derives the offline hash embedder from the manifest, so the full
loop runs without network. Dense-leg semantics are exercised by feeding a
chunk's own text back as the query (identical strings hash to identical
unit vectors -> cosine 1.0).
"""

import json

import pytest

from app.services.parser import ParserConfig, ingest
from app.services.retrieval.config import RetrievalConfig
from app.services.retrieval.embeddings import HashEmbeddingProvider
from app.services.retrieval.engine import RetrievalEngine, RetrievalError
from app.services.retrieval.indexer import build_index

from .conftest import build_sample_pdf

DOC_ID = "bio-e2e"
DIM = 16


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    pdf = root / "sample.pdf"
    build_sample_pdf(pdf, with_toc=True)
    ingest(DOC_ID, pdf, ParserConfig(), root)
    stats = build_index(
        DOC_ID, root, embedder=HashEmbeddingProvider(dim=DIM), config=RetrievalConfig(min_chunk_chars=0)
    )
    assert stats["status"] == "built" and stats["num_chunks"] == 4
    engine = RetrievalEngine.load(DOC_ID, root)  # embedder derived from manifest
    return root, engine


def test_sparse_mode_finds_expected_chunk(indexed):
    _, engine = indexed
    result = engine.retrieve("细胞学说", mode="sparse", top_k=3)
    assert result.hits
    top = result.hits[0]
    assert "细胞学说" in top.text
    assert top.sparse_score is not None and top.sparse_score > 0
    assert top.dense_score is None
    assert top.sources == ["sparse"]


def test_dense_mode_self_match(indexed):
    _, engine = indexed
    record = engine.records[0]
    result = engine.retrieve(record.text, mode="dense", top_k=4)
    top = result.hits[0]
    assert top.chunk_id == record.chunk_id
    assert top.dense_score == pytest.approx(1.0, abs=1e-5)
    assert top.sources == ["dense"]


def test_hybrid_reports_both_legs(indexed):
    _, engine = indexed
    record = engine.records[0]
    result = engine.retrieve(record.text, mode="hybrid", top_k=4)
    top = result.hits[0]
    assert top.chunk_id == record.chunk_id
    # exact-text query also matches lexically -> both legs must be reported
    assert top.sources == ["dense", "sparse"]
    assert top.dense_score is not None and top.sparse_score is not None
    assert top.fused_score is not None


def test_provenance_fields(indexed):
    _, engine = indexed
    result = engine.retrieve("细胞的多样性", mode="hybrid", top_k=4)
    assert result.hits
    for hit in result.hits:
        assert hit.document_id == DOC_ID
        assert hit.breadcrumb  # e.g. ["第1章 走近细胞", "第1节 …"]
        assert hit.pages
        assert hit.char_count > 0
    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))
    assert result.latency_ms >= 0
    assert result.manifest.doc_id == DOC_ID
    assert result.manifest.embedder == "hash"


def test_result_is_deterministic(indexed):
    _, engine = indexed
    first = engine.retrieve("细胞的多样性和统一性", mode="hybrid", top_k=4)
    second = engine.retrieve("细胞的多样性和统一性", mode="hybrid", top_k=4)
    assert [h.chunk_id for h in first.hits] == [h.chunk_id for h in second.hits]


def test_top_k_respected(indexed):
    _, engine = indexed
    result = engine.retrieve("细胞", mode="hybrid", top_k=2)
    assert len(result.hits) <= 2


def test_blank_query_returns_empty(indexed):
    _, engine = indexed
    result = engine.retrieve("   ", mode="hybrid")
    assert result.hits == []


def test_invalid_mode_rejected(indexed):
    _, engine = indexed
    with pytest.raises(ValueError, match="mode"):
        engine.retrieve("细胞", mode="rrf")


def test_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        RetrievalEngine.load("ghost", tmp_path)


def test_dim_mismatch_rejected(indexed):
    root, _ = indexed
    with pytest.raises(RetrievalError, match="dim"):
        RetrievalEngine.load(DOC_ID, root, embedder=HashEmbeddingProvider(dim=32))


def test_records_on_disk_match_engine(indexed):
    root, engine = indexed
    lines = (root / "index" / DOC_ID / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(engine.records)
    first = json.loads(lines[0])
    assert first["document_id"] == DOC_ID
    assert first["tokens"]
