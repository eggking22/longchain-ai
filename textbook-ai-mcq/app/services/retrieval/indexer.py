"""Index orchestration: Phase 1 chunks.jsonl -> searchable index artifacts.

Reads (never rewrites) data/chunks/{doc_id}/chunks.jsonl, stamps each chunk
with document_id (= doc_id) and chunk_index (= line number), filters noise,
tokenizes, embeds and writes::

    data/index/{doc_id}/manifest.json    IndexManifest (reproducibility)
    data/index/{doc_id}/records.jsonl    IndexedRecord per line
    data/index/{doc_id}/embeddings.npy   float32 matrix (row == record line)

Idempotency: when config_hash + chunk_set_hash + embedder identity match the
existing manifest and force is False, the (possibly costly) embedding step
is skipped entirely — re-running on an unchanged corpus costs nothing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from app.schemas.document import Chunk
from app.schemas.retrieval import IndexManifest, IndexedRecord

from .config import RetrievalConfig
from .embeddings import EmbeddingProvider, build_embedding_provider
from .tokenizer import tokenize
from .vector_store import NumpyVectorStore


def load_chunks(doc_id: str, artifacts_root: str | Path = "data") -> list[Chunk]:
    path = Path(artifacts_root) / "chunks" / doc_id / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run the Phase 1 parser first "
            f"(python scripts/parse_pdf.py <pdf> --doc-id {doc_id})"
        )
    chunks = [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not chunks:
        raise ValueError(f"{path} contains no chunks")
    return chunks


def _chunk_set_hash(records: list[IndexedRecord]) -> str:
    ids = ",".join(sorted(r.chunk_id for r in records))
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()[:16]


def _manifest_matches(
    existing: IndexManifest,
    config: RetrievalConfig,
    chunk_set_hash: str,
    embedder: EmbeddingProvider,
) -> bool:
    return (
        existing.config_hash == config.config_hash()
        and existing.chunk_set_hash == chunk_set_hash
        and existing.embedder == embedder.name
        and existing.embedding_model == embedder.model
        and existing.embedding_dim == embedder.dim
    )


def build_index(
    doc_id: str,
    artifacts_root: str | Path = "data",
    embedder: EmbeddingProvider | None = None,
    config: RetrievalConfig | None = None,
    force: bool = False,
) -> dict:
    from app.core.config import get_settings

    if config is None or embedder is None:
        settings = get_settings()
        if config is None:
            config = RetrievalConfig.from_settings(settings)
        if embedder is None:
            embedder = build_embedding_provider(settings)

    index_dir = Path(artifacts_root) / "index" / doc_id
    chunks = load_chunks(doc_id, artifacts_root)

    kept, dropped = [], 0
    for line_no, chunk in enumerate(chunks):
        if len(chunk.text.strip()) < config.min_chunk_chars:
            dropped += 1
            continue
        kept.append(
            IndexedRecord(
                document_id=doc_id,
                chunk_id=chunk.chunk_id,
                chunk_index=line_no,
                text=chunk.text,
                breadcrumb=chunk.breadcrumb,
                pages=chunk.pages,
                char_count=chunk.char_count,
                paragraph_ids=chunk.paragraph_ids,
                tokens=tokenize(chunk.text),
            )
        )
    if not kept:
        raise ValueError(
            f"all {len(chunks)} chunks of '{doc_id}' were filtered out "
            f"(min_chunk_chars={config.min_chunk_chars})"
        )

    chunk_set_hash = _chunk_set_hash(kept)
    manifest_path = index_dir / "manifest.json"
    if not force and manifest_path.exists():
        existing = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if _manifest_matches(existing, config, chunk_set_hash, embedder):
            return {
                "status": "skipped",
                "doc_id": doc_id,
                "num_chunks": existing.num_chunks,
                "num_dropped": dropped,
                "reason": "index up to date (matching config/chunks/embedder)",
            }

    print(f"[index] embedding {len(kept)} chunks with '{embedder.name}' (dim={embedder.dim}) ...")
    vectors = embedder.embed_texts([r.text for r in kept])

    store = NumpyVectorStore()
    store.add([r.chunk_id for r in kept], vectors)

    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(record.model_dump_json() + "\n")
    store.save(index_dir)

    manifest = IndexManifest(
        doc_id=doc_id,
        embedder=embedder.name,
        embedding_model=embedder.model,
        embedding_dim=embedder.dim,
        config_hash=config.config_hash(),
        chunk_set_hash=chunk_set_hash,
        num_chunks=len(kept),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "status": "built",
        "doc_id": doc_id,
        "num_chunks": len(kept),
        "num_dropped": dropped,
        "embedder": embedder.name,
        "embedding_model": embedder.model,
        "embedding_dim": embedder.dim,
        "chunk_set_hash": chunk_set_hash,
    }
