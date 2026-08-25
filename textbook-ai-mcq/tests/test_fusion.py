"""Unit tests for RRF and weighted relative fusion (exact math)."""

import pytest

from app.services.retrieval.fusion import reciprocal_rank_fusion, weighted_relative_fusion


class TestRRF:
    def test_single_leg_exact_formula(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b"]}, k=60, top_k=5)
        assert fused[0] == ("a", pytest.approx(1 / 61))
        assert fused[1] == ("b", pytest.approx(1 / 62))

    def test_overlap_accumulates(self):
        fused = reciprocal_rank_fusion(
            {"dense": ["a"], "sparse": ["a"]}, k=60, top_k=5
        )
        assert fused[0][0] == "a"
        assert fused[0][1] == pytest.approx(2 / 61)

    def test_disjoint_union(self):
        fused = reciprocal_rank_fusion(
            {"dense": ["a", "b"], "sparse": ["c"]}, k=60, top_k=5
        )
        ids = [cid for cid, _ in fused]
        assert set(ids) == {"a", "b", "c"}
        assert ids[0] == "a"  # 1/61 beats 1/62 and 1/61? c also gets 1/61...
        # a: 1/61, c: 1/61, b: 1/62 -> tie between a and c broken by id asc
        assert ids == ["a", "c", "b"]

    def test_overlap_beats_single_leg_leader(self):
        # x ranks 2nd in both legs; y ranks 1st only in one leg.
        fused = reciprocal_rank_fusion(
            {"dense": ["y", "x"], "sparse": ["z", "x"]}, k=60, top_k=5
        )
        assert fused[0][0] == "x"  # 2/62 > 1/61
        assert fused[0][1] == pytest.approx(2 / 62)

    def test_tie_break_chunk_id_asc(self):
        fused = reciprocal_rank_fusion(
            {"dense": ["b", "a"], "sparse": ["a", "b"]}, k=60, top_k=5
        )
        # both have 1/61 + 1/62 -> deterministic tie-break by id
        assert [cid for cid, _ in fused] == ["a", "b"]

    def test_top_k_truncates(self):
        fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"]}, k=60, top_k=2)
        assert len(fused) == 2

    def test_empty_legs(self):
        assert reciprocal_rank_fusion({}, k=60, top_k=5) == []
        assert reciprocal_rank_fusion({"dense": [], "sparse": ["a"]})[0][0] == "a"

    def test_larger_k_flattens(self):
        near = reciprocal_rank_fusion({"dense": ["a", "b"]}, k=1, top_k=2)[0][1]
        far = reciprocal_rank_fusion({"dense": ["a", "b"]}, k=1000, top_k=2)[0][1]
        assert near > far


class TestWeightedRelative:
    def test_minmax_normalisation(self):
        # dense scores 0.9 / 0.5 -> 1.0 / 0.0 with equal weights
        fused = weighted_relative_fusion(
            {"dense": [("a", 0.9), ("b", 0.5)]},
            weights={"dense": 0.5},
            top_k=5,
        )
        assert fused[0] == ("a", pytest.approx(0.5))
        assert fused[1] == ("b", pytest.approx(0.0))

    def test_weights_favour_a_leg(self):
        fused = weighted_relative_fusion(
            {
                "dense": [("a", 1.0), ("b", 0.5)],
                "sparse": [("b", 10.0), ("a", 0.0)],
            },
            weights={"dense": 0.9, "sparse": 0.1},
            top_k=5,
        )
        # a: 0.9*1.0 + 0.1*0.0 = 0.9 ; b: 0.9*0.5 + 0.1*1.0 = 0.55
        assert fused[0][0] == "a"
        assert fused[0][1] == pytest.approx(0.9)

    def test_degenerate_all_equal_positive(self):
        fused = weighted_relative_fusion(
            {"dense": [("a", 3.0), ("b", 3.0)]},
            weights={"dense": 1.0},
            top_k=5,
        )
        assert fused[0][1] == pytest.approx(1.0)
        assert fused[1][1] == pytest.approx(1.0)

    def test_missing_leg_contributes_zero(self):
        fused = weighted_relative_fusion(
            {"dense": [("a", 2.0)]},
            weights={"dense": 0.5, "sparse": 0.5},
            top_k=5,
        )
        assert fused == [("a", pytest.approx(0.5))]

    def test_empty_input(self):
        assert weighted_relative_fusion({}, weights={"dense": 1.0}) == []
