"""Hybrid retrieval package (Phase 2).

Pipeline: indexer.py (chunks.jsonl -> manifest + records + embeddings)
-> engine.py (dense + sparse legs -> fusion.py RRF -> hydrated hits).
Shared tokenizer in tokenizer.py; embedders in embeddings.py; stores in
vector_store.py / bm25.py; knobs in config.py.
"""

from .bm25 import Bm25Index
from .config import RetrievalConfig
from .embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    HashEmbeddingProvider,
    HttpEmbeddingProvider,
    build_embedding_provider,
)
from .engine import MODES, RetrievalEngine, RetrievalError
from .fusion import reciprocal_rank_fusion, weighted_relative_fusion
from .indexer import build_index, load_chunks
from .tokenizer import tokenize
from .vector_store import NumpyVectorStore

__all__ = [
    "MODES",
    "Bm25Index",
    "NumpyVectorStore",
    "RetrievalConfig",
    "RetrievalEngine",
    "RetrievalError",
    "EmbeddingError",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "HttpEmbeddingProvider",
    "build_embedding_provider",
    "build_index",
    "load_chunks",
    "tokenize",
    "reciprocal_rank_fusion",
    "weighted_relative_fusion",
]
