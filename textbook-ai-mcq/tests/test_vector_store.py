"""Unit tests for the numpy vector store."""

import numpy as np
import pytest

from app.services.retrieval.vector_store import NumpyVectorStore


def _store():
    store = NumpyVectorStore()
    store.add(["a", "b", "c"], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    return store


class TestAddSearch:
    def test_cosine_against_hand_computed(self):
        store = _store()
        hits = store.search([1.0, 0.0], k=3)
        # [1,0] -> a: 1.0 ; c: 1/sqrt(2) ; b: 0.0
        assert hits[0][0] == "a" and hits[0][1] == pytest.approx(1.0)
        assert hits[1][0] == "c" and hits[1][1] == pytest.approx(2**-0.5)
        assert hits[2][0] == "b" and hits[2][1] == pytest.approx(0.0)

    def test_query_normalised_so_scale_invariant(self):
        store = _store()
        assert store.search([5.0, 0.0], k=1)[0][1] == pytest.approx(1.0)

    def test_k_larger_than_n_returns_all(self):
        assert len(_store().search([1.0, 0.0], k=99)) == 3

    def test_k_zero_and_empty_store(self):
        assert _store().search([1.0, 0.0], k=0) == []
        assert NumpyVectorStore().search([1.0, 0.0], k=3) == []

    def test_zero_query_vector(self):
        assert _store().search([0.0, 0.0], k=3) == []

    def test_tie_keeps_insertion_order(self):
        store = NumpyVectorStore()
        store.add(["x", "y"], [[1.0, 0.0], [1.0, 0.0]])
        assert [cid for cid, _ in store.search([1.0, 0.0], k=2)] == ["x", "y"]

    def test_duplicate_ids_rejected(self):
        store = _store()
        with pytest.raises(ValueError, match="duplicate"):
            store.add(["a"], [[1.0, 0.0]])

    def test_dim_mismatch_rejected(self):
        store = _store()
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add(["d"], [[1.0, 0.0, 0.0]])

    def test_zero_norm_vector_rejected(self):
        with pytest.raises(ValueError, match="zero-norm"):
            NumpyVectorStore().add(["z"], [[0.0, 0.0]])


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        store = _store()
        store.save(tmp_path)
        loaded = NumpyVectorStore.load(tmp_path)
        assert len(loaded) == 3
        assert loaded.search([1.0, 0.0], k=3) == store.search([1.0, 0.0], k=3)
        assert loaded.dim == 2

    def test_save_empty_store_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            NumpyVectorStore().save(tmp_path)
