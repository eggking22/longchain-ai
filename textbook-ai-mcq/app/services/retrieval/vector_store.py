"""In-memory vector store with exact cosine search.

At textbook scale (10^3–10^4 chunks) brute force over a normalised float32
matrix IS exact search — no ANN structure needed. The interface (add /
search / save / load) is shaped so a future PgVectorStore (vector(1024)
column, HNSW vector_cosine_ops, ORDER BY embedding <=> q) can replace this
class without touching callers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

MATRIX_FILE = "embeddings.npy"
IDS_FILE = "embeddings.ids.json"


class VectorStore(Protocol):
    def add(self, ids: list[str], vectors: list[list[float]]) -> None: ...
    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]: ...
    def save(self, directory: Path) -> None: ...


class NumpyVectorStore:
    """Rows are L2-normalised on add, so search is a plain dot product."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # (n, dim) float32

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[1])

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        if not ids:
            return
        if set(ids) & set(self._ids):
            raise ValueError("duplicate chunk ids cannot be added twice; rebuild the index instead")
        incoming = np.asarray(vectors, dtype=np.float64)
        if incoming.ndim != 2:
            raise ValueError("vectors must be a 2-D list")
        if self._matrix is not None and incoming.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"dimension mismatch: store has {self._matrix.shape[1]}, got {incoming.shape[1]}"
            )
        norms = np.linalg.norm(incoming, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("zero-norm vector cannot be stored")
        incoming = (incoming / norms).astype(np.float32)
        self._matrix = incoming if self._matrix is None else np.vstack([self._matrix, incoming])
        self._ids.extend(ids)

    def search(self, query_vector: list[float], k: int) -> list[tuple[str, float]]:
        if self._matrix is None or k <= 0 or not query_vector:
            return []
        query = np.asarray(query_vector, dtype=np.float64)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        sims = self._matrix @ (query / norm)  # rows are unit vectors -> cosine
        k = min(k, len(self._ids))
        # stable sort keeps insertion (chunk_index) order on score ties
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self._ids[i], float(sims[i])) for i in order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._matrix is None:
            raise ValueError("cannot save an empty store")
        np.save(directory / MATRIX_FILE, self._matrix)
        (directory / IDS_FILE).write_text(
            json.dumps(self._ids, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "NumpyVectorStore":
        store = cls()
        store._matrix = np.load(directory / MATRIX_FILE)
        store._ids = json.loads((directory / IDS_FILE).read_text(encoding="utf-8"))
        if len(store._ids) != store._matrix.shape[0]:
            raise ValueError(
                f"corrupt index: {len(store._ids)} ids vs {store._matrix.shape[0]} vectors"
            )
        return store
