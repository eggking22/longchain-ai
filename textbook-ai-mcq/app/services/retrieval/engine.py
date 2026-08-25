"""RetrievalEngine: dense + sparse legs fused into provenance-rich hits.

Query path::

    query ──┬─ embed_query ──► NumpyVectorStore.search (cosine, top dense_top_k)
            └─ tokenize ─────► Bm25Index.search           (BM25,   top sparse_top_k)
                        ╲────► reciprocal_rank_fusion(k) ──► top_k hydrated hits

Every hit keeps the raw score of each leg it appeared in (None otherwise),
which leg(s) produced it, and the full Phase 1 provenance. The embedder is
derived from the manifest, so an index built with the offline hash provider
never accidentally calls the paid API.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.schemas.retrieval import (
    IndexManifest,
    IndexedRecord,
    RetrievalResult,
    RetrievedChunk,
)

from .bm25 import Bm25Index
from .config import RetrievalConfig
from .embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    build_embedding_provider,
)
from .fusion import reciprocal_rank_fusion
from .tokenizer import tokenize
from .vector_store import NumpyVectorStore

MODES = ("hybrid", "dense", "sparse")


class RetrievalError(RuntimeError):
    """Raised on inconsistent index/provider state (usually: rebuild needed)."""


class RetrievalEngine:
    def __init__(
        self,
        records: list[IndexedRecord],
        store: NumpyVectorStore,
        bm25: Bm25Index,
        manifest: IndexManifest,
        embedder: EmbeddingProvider,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.records = records
        self.store = store
        self.bm25 = bm25
        self.manifest = manifest
        self.embedder = embedder
        self.config = config or RetrievalConfig()

    @classmethod
    def load(
        cls,
        doc_id: str,
        artifacts_root: str | Path = "data",
        embedder: EmbeddingProvider | None = None,
        config: RetrievalConfig | None = None,
    ) -> "RetrievalEngine":
        index_dir = Path(artifacts_root) / "index" / doc_id
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found — build the index first "
                f"(python scripts/build_index.py --doc-id {doc_id})"
            )
        manifest = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        records = [
            IndexedRecord.model_validate_json(line)
            for line in (index_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        store = NumpyVectorStore.load(index_dir)
        bm25 = Bm25Index.build(records)
        if embedder is None:
            embedder = cls._embedder_for(manifest)
        if embedder.dim != manifest.embedding_dim:
            raise RetrievalError(
                f"embedder dim {embedder.dim} != index dim {manifest.embedding_dim} "
                f"({manifest.embedding_model}) — rebuild the index or switch provider"
            )
        if config is None:
            from app.core.config import get_settings

            config = RetrievalConfig.from_settings(get_settings())
        return cls(records, store, bm25, manifest, embedder, config)

    @staticmethod
    def _embedder_for(manifest: IndexManifest) -> EmbeddingProvider:
        if manifest.embedder == "hash":
            return HashEmbeddingProvider(dim=manifest.embedding_dim)
        from app.core.config import get_settings

        provider = build_embedding_provider(get_settings())
        if provider.model != manifest.embedding_model:
            raise RetrievalError(
                f"index was built with '{manifest.embedding_model}' but settings "
                f"provide '{provider.model}' — align EMBEDDING_MODEL or rebuild"
            )
        return provider

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
    ) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got '{mode}'")
        started = time.perf_counter()

        top_k = self.config.default_top_k if top_k is None else top_k
        dense_top_k = self.config.dense_top_k if dense_top_k is None else dense_top_k
        sparse_top_k = self.config.sparse_top_k if sparse_top_k is None else sparse_top_k

        dense_hits: list[tuple[str, float]] = []
        sparse_hits: list[tuple[str, float]] = []
        if query.strip():
            if mode in ("hybrid", "dense"):
                dense_hits = self.store.search(self.embedder.embed_query(query), dense_top_k)
            if mode in ("hybrid", "sparse"):
                sparse_hits = self.bm25.search(tokenize(query), sparse_top_k)

        if mode == "dense":
            fused = dense_hits[:top_k]
        elif mode == "sparse":
            fused = sparse_hits[:top_k]
        else:
            fused = reciprocal_rank_fusion(
                {
                    "dense": [chunk_id for chunk_id, _ in dense_hits],
                    "sparse": [chunk_id for chunk_id, _ in sparse_hits],
                },
                k=self.config.rrf_k,
                top_k=top_k,
            )

        dense_by_id = dict(dense_hits)
        sparse_by_id = dict(sparse_hits)
        by_id = {record.chunk_id: record for record in self.records}

        hits: list[RetrievedChunk] = []
        for rank, (chunk_id, fused_score) in enumerate(fused, start=1):
            record = by_id.get(chunk_id)
            if record is None:
                continue
            sources = [
                leg
                for leg, scored in (("dense", dense_by_id), ("sparse", sparse_by_id))
                if chunk_id in scored
            ]
            hits.append(
                RetrievedChunk(
                    document_id=record.document_id,
                    chunk_id=record.chunk_id,
                    chunk_index=record.chunk_index,
                    text=record.text,
                    breadcrumb=record.breadcrumb,
                    pages=record.pages,
                    char_count=record.char_count,
                    dense_score=dense_by_id.get(chunk_id),
                    sparse_score=sparse_by_id.get(chunk_id),
                    fused_score=fused_score,
                    rank=rank,
                    sources=sources,
                )
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return RetrievalResult(
            query=query,
            mode=mode,
            top_k=top_k,
            hits=hits,
            manifest=self.manifest,
            latency_ms=latency_ms,
        )
