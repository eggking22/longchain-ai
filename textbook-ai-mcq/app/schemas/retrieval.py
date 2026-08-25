"""Pydantic schemas for the hybrid retrieval layer (Phase 2).

Index artifacts live under ARTIFACTS_DIR/index/{doc_id}/:

    manifest.json   IndexManifest (embedder identity + reproducibility hashes)
    records.jsonl   IndexedRecord per line (chunk metadata + jieba tokens)
    embeddings.npy  float32 matrix, row order == record line order

Retrieval hydrates hits back from records so every result keeps the full
Phase 1 provenance (breadcrumb / pages / paragraph_ids) plus the scores of
each retrieval leg.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndexManifest(BaseModel):
    """Self-describing snapshot of one built index (the unit of reproducibility)."""

    doc_id: str
    index_version: str = "1"
    embedder: str  # provider name, e.g. "hash" / "http"
    embedding_model: str
    embedding_dim: int
    config_hash: str  # sha256 of canonical retrieval config (see retrieval/config.py)
    chunk_set_hash: str  # sha256 over the sorted chunk ids actually indexed
    num_chunks: int
    created_at: str  # ISO-8601 local time


class IndexedRecord(BaseModel):
    """One indexed chunk: Phase 1 Chunk fields plus indexing-time facts.

    document_id / chunk_index are not stored inside Phase 1 chunks.jsonl;
    the indexer derives them (directory name / line number) so the parser
    schema stays untouched.
    """

    document_id: str
    chunk_id: str
    chunk_index: int  # original line number in chunks.jsonl (0-based)
    text: str
    breadcrumb: list[str]
    pages: list[int]
    char_count: int
    paragraph_ids: list[str]
    tokens: list[str] = Field(default_factory=list)  # jieba tokens, reused by BM25 and future PG text[] column


class RetrievedChunk(BaseModel):
    """A hydrated retrieval hit with per-leg scores kept for debugging/eval.

    Scores are None when the chunk did not appear in that leg's top list.
    """

    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    breadcrumb: list[str]
    pages: list[int]
    char_count: int
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None
    rank: int  # 1-based final rank
    sources: list[str] = Field(default_factory=list)  # subset of ["dense", "sparse"]


class RetrievalResult(BaseModel):
    """Complete answer of RetrievalEngine.retrieve()."""

    query: str
    mode: str  # "hybrid" | "dense" | "sparse"
    top_k: int
    hits: list[RetrievedChunk] = Field(default_factory=list)
    manifest: IndexManifest
    latency_ms: float
