"""Unit tests for embedding providers (all offline; HTTP via MockTransport)."""

import json

import httpx
import numpy as np
import pytest

from app.services.retrieval.embeddings import (
    EmbeddingError,
    HashEmbeddingProvider,
    HttpEmbeddingProvider,
    build_embedding_provider,
)


def _vec(dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


class TestHashEmbeddingProvider:
    def test_deterministic(self):
        a = HashEmbeddingProvider(dim=32)
        assert a.embed_texts(["细胞膜"]) == a.embed_texts(["细胞膜"])

    def test_different_texts_differ(self):
        a = HashEmbeddingProvider(dim=32)
        assert a.embed_texts(["细胞膜"]) != a.embed_texts(["线粒体"])

    def test_dim_and_normalised(self):
        a = HashEmbeddingProvider(dim=16)
        [vec] = a.embed_texts(["x"])
        assert len(vec) == 16
        assert np.linalg.norm(vec) == pytest.approx(1.0)

    def test_empty_input(self):
        assert HashEmbeddingProvider(dim=8).embed_texts([]) == []

    def test_identity(self):
        a = HashEmbeddingProvider(dim=8)
        assert (a.name, a.model, a.dim) == ("hash", "hash", 8)


class TestHttpEmbeddingProvider:
    def _provider(self, handler, **kw):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        defaults = dict(
            base_url="https://api.test/v4",
            api_key="sk-test",
            model="embedding-3",
            dim=4,
            batch_size=2,
            max_retries=2,
            backoff=0.0,
            client=client,
        )
        defaults.update(kw)
        return HttpEmbeddingProvider(**defaults)

    def test_happy_path_sorted_by_index(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "embedding-3"
            assert body["dimensions"] == 4
            assert request.headers["Authorization"] == "Bearer sk-test"
            # return out of order on purpose
            data = [
                {"index": 1, "embedding": _vec(4, 1)},
                {"index": 0, "embedding": _vec(4, 0)},
            ]
            return httpx.Response(200, json={"data": data})

        provider = self._provider(handler)
        vecs = provider.embed_texts(["甲", "乙"])
        assert len(vecs) == 2
        assert vecs[0] == pytest.approx(_vec(4, 0))  # index 0 first
        assert np.linalg.norm(vecs[0]) == pytest.approx(1.0)

    def test_batching(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(len(body["input"]))
            return httpx.Response(
                200,
                json={"data": [{"index": i, "embedding": _vec(4, i)} for i in range(len(body["input"]))]},
            )

        provider = self._provider(handler)
        assert len(provider.embed_texts(["a", "b", "c"])) == 3
        assert calls == [2, 1]

    def test_retry_on_500_then_success(self):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": _vec(4, 0)}]})

        provider = self._provider(handler)
        assert len(provider.embed_texts(["a"])) == 1
        assert len(attempts) == 2

    def test_client_error_not_retried(self):
        attempts = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(401, json={"error": "bad key"})

        provider = self._provider(handler)
        with pytest.raises(EmbeddingError, match="401"):
            provider.embed_texts(["a"])
        assert len(attempts) == 1

    def test_exhausted_retries_raise(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "down"})

        provider = self._provider(handler)
        with pytest.raises(EmbeddingError, match="after 2 attempts"):
            provider.embed_texts(["a"])

    def test_dim_mismatch_detected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": _vec(8, 0)}]})

        provider = self._provider(handler)  # expects dim=4
        with pytest.raises(EmbeddingError, match="dim mismatch"):
            provider.embed_texts(["a"])

    def test_empty_input_no_request(self):
        provider = self._provider(lambda request: httpx.Response(500))
        assert provider.embed_texts([]) == []


class TestFactory:
    def test_hash_selected(self, monkeypatch):
        from app.core.config import Settings

        settings = Settings(EMBEDDING_MODEL="hash", EMBEDDING_DIM=24)
        provider = build_embedding_provider(settings)
        assert provider.name == "hash" and provider.dim == 24

    def test_http_selected_with_key_fallback(self, monkeypatch):
        from app.core.config import Settings

        settings = Settings(
            EMBEDDING_MODEL="embedding-3",
            EMBEDDING_API_KEY="",
            LLM_API_KEY="sk-llm",
            EMBEDDING_BASE_URL="https://api.test/v4",
        )
        provider = build_embedding_provider(settings)
        assert provider.name == "http" and provider.api_key == "sk-llm"

    def test_no_key_raises(self, monkeypatch):
        from app.core.config import Settings

        settings = Settings(EMBEDDING_MODEL="embedding-3", EMBEDDING_API_KEY="", LLM_API_KEY="")
        with pytest.raises(EmbeddingError, match="no embedding API key"):
            build_embedding_provider(settings)
