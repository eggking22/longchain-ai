"""Centralised, tunable knobs for the hybrid retrieval engine.

Mirrors ParserConfig: every knob is settable from .env, and the config is
canonicalised into config_hash so identical chunk sets with different
configs never share an index (idempotent re-indexing, see indexer.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass
class RetrievalConfig:
    # --- index-time ---
    min_chunk_chars: int = 10  # chunks shorter than this are dropped at index time

    # --- query-time ---
    dense_top_k: int = 20  # over-fetch per leg, then fuse and cut to top_k
    sparse_top_k: int = 20
    rrf_k: int = 60  # RRF constant (Cormack et al., SIGIR 2009)
    default_top_k: int = 5

    def config_hash(self) -> str:
        """Stable hash of the knobs that shape the index/query behaviour."""
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_settings(cls, settings) -> "RetrievalConfig":
        return cls(
            min_chunk_chars=settings.RETRIEVAL_MIN_CHUNK_CHARS,
            dense_top_k=settings.RETRIEVAL_DENSE_TOP_K,
            sparse_top_k=settings.RETRIEVAL_SPARSE_TOP_K,
            rrf_k=settings.RETRIEVAL_RRF_K,
        )
