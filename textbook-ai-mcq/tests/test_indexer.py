"""Unit tests for index building (hash embedder, tmp artifacts)."""

import json

import pytest

from app.schemas.document import Chunk
from app.services.retrieval.config import RetrievalConfig
from app.services.retrieval.embeddings import HashEmbeddingProvider
from app.services.retrieval.indexer import build_index, load_chunks


def _write_chunks(root, doc_id: str, chunks: list[Chunk]) -> None:
    d = root / "chunks" / doc_id
    d.mkdir(parents=True)
    with (d / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        breadcrumb=["第1章 走近细胞"],
        pages=[1],
        char_count=len(text),
        paragraph_ids=["p1"],
    )


@pytest.fixture
def corpus(tmp_path):
    _write_chunks(
        tmp_path,
        "bio",
        [
            _chunk("c1", "细胞是生物体结构和功能的基本单位。"),
            _chunk("c2", "线粒体是有氧呼吸的主要场所。"),
            _chunk("c3", "短"),  # filtered by min_chunk_chars
            _chunk("c4", "细胞学说由施莱登和施旺提出。"),
        ],
    )
    return tmp_path


def _embedder():
    return HashEmbeddingProvider(dim=16)


def _config(**kw):
    kw.setdefault("min_chunk_chars", 2)
    return RetrievalConfig(**kw)


def test_build_writes_artifacts(corpus):
    stats = build_index("bio", corpus, embedder=_embedder(), config=_config())
    index_dir = corpus / "index" / "bio"
    assert stats["status"] == "built"
    assert stats["num_chunks"] == 3 and stats["num_dropped"] == 1
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "embeddings.npy").exists()
    records = [
        json.loads(line)
        for line in (index_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3
    assert all(r["document_id"] == "bio" for r in records)
    # original line numbers preserved even though c3 was filtered
    assert [r["chunk_index"] for r in records] == [0, 1, 3]
    assert all(r["tokens"] for r in records)


def test_manifest_fields(corpus):
    build_index("bio", corpus, embedder=_embedder(), config=_config())
    manifest = json.loads((corpus / "index" / "bio" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedder"] == "hash"
    assert manifest["embedding_dim"] == 16
    assert manifest["num_chunks"] == 3
    assert manifest["config_hash"] == _config().config_hash()
    assert len(manifest["chunk_set_hash"]) == 16


def test_idempotent_skip(corpus):
    first = build_index("bio", corpus, embedder=_embedder(), config=_config())
    second = build_index("bio", corpus, embedder=_embedder(), config=_config())
    assert first["status"] == "built"
    assert second["status"] == "skipped"
    assert second["num_chunks"] == 3


def test_force_rebuilds(corpus):
    build_index("bio", corpus, embedder=_embedder(), config=_config())
    again = build_index("bio", corpus, embedder=_embedder(), config=_config(), force=True)
    assert again["status"] == "built"


def test_config_change_rebuilds(corpus):
    build_index("bio", corpus, embedder=_embedder(), config=_config())
    other = build_index("bio", corpus, embedder=_embedder(), config=_config(min_chunk_chars=5))
    assert other["status"] == "built"
    assert other["num_chunks"] == 3  # only "短" (1 char) is below 5


def test_all_filtered_raises(corpus, tmp_path):
    _write_chunks(tmp_path, "tiny", [_chunk("x1", "一")])
    with pytest.raises(ValueError, match="filtered out"):
        build_index("tiny", tmp_path, embedder=_embedder(), config=RetrievalConfig(min_chunk_chars=5))


def test_missing_chunks_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="parse_pdf"):
        build_index("ghost", tmp_path, embedder=_embedder(), config=_config())


def test_load_chunks_roundtrip(corpus):
    chunks = load_chunks("bio", corpus)
    assert [c.chunk_id for c in chunks] == ["c1", "c2", "c3", "c4"]
