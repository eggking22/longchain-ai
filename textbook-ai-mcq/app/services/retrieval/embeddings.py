"""Embedding providers behind one protocol.

HashEmbeddingProvider: deterministic, semantic-free vectors seeded from
sha256(text) — used by tests, CI and offline smoke runs so the whole
pipeline runs without network or API cost.

HttpEmbeddingProvider: any OpenAI-compatible /embeddings endpoint (default
Zhipu bigmodel, model embedding-3) called through plain httpx with batching
and retry. Returned vectors are L2-normalised client-side so a dot product
equals cosine similarity and maps 1:1 onto pgvector's vector_cosine_ops.
"""

from __future__ import annotations

import hashlib
import time
from typing import Protocol, runtime_checkable

import httpx
import numpy as np


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce vectors."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str
    model: str
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise EmbeddingError("embedding vector has zero norm")
    return vec / norm


class HashEmbeddingProvider:
    """Deterministic offline embedder: identical text -> identical unit vector.

    Semantic-free by design (a random direction per distinct text); it only
    guarantees reproducibility, which is exactly what tests need. Dense-leg
    correctness is exercised by querying a chunk's own text back (hash of
    equal strings collides on purpose).
    """

    def __init__(self, dim: int = 64):
        self.name = "hash"
        self.model = "hash"
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            vectors.append(_normalize(rng.standard_normal(self.dim)).tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class HttpEmbeddingProvider:
    """Batched call of an OpenAI-compatible POST {base_url}/embeddings."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dim: int,
        batch_size: int = 64,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 1.0,
        client: httpx.Client | None = None,
    ):
        self.name = "http"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.backoff = backoff
        self._client = client

    def _post(self, payload: dict) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            client = self._client or httpx.Client(timeout=self.timeout)
            try:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code >= 500 or response.status_code == 429:
                    last_error = EmbeddingError(
                        f"embeddings API returned {response.status_code}: {response.text[:200]}"
                    )
                elif response.status_code != 200:
                    raise EmbeddingError(  # client errors are not retryable
                        f"embeddings API returned {response.status_code}: {response.text[:200]}"
                    )
                else:
                    data = response.json()["data"]
                    # align by the echoed index, not by arrival order
                    data.sort(key=lambda item: item["index"])
                    return [item["embedding"] for item in data]
            except httpx.HTTPError as exc:  # network / timeout errors
                last_error = exc
            finally:
                if self._client is None:
                    client.close()
            if self.backoff > 0:
                time.sleep(self.backoff * (2.0**attempt))
        raise EmbeddingError(f"embeddings API failed after {self.max_retries} attempts: {last_error}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = {"model": self.model, "input": batch, "dimensions": self.dim}
            for raw in self._post(payload):
                vec = np.asarray(raw, dtype=np.float64)
                if vec.shape[0] != self.dim:
                    raise EmbeddingError(
                        f"embedding dim mismatch: expected {self.dim}, got {vec.shape[0]}"
                    )
                vectors.append(_normalize(vec).tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def build_embedding_provider(settings) -> EmbeddingProvider:
    """Factory from Settings; EMBEDDING_MODEL=hash selects the offline provider."""
    model = (settings.EMBEDDING_MODEL or "").strip()
    if model.lower() == "hash":
        return HashEmbeddingProvider(dim=settings.EMBEDDING_DIM)
    api_key = settings.EMBEDDING_API_KEY or settings.LLM_API_KEY
    if not api_key:
        raise EmbeddingError(
            "no embedding API key: set EMBEDDING_API_KEY (or LLM_API_KEY), "
            "or set EMBEDDING_MODEL=hash for the offline deterministic provider"
        )
    return HttpEmbeddingProvider(
        base_url=settings.EMBEDDING_BASE_URL,
        api_key=api_key,
        model=model or "embedding-3",
        dim=settings.EMBEDDING_DIM,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )
