"""Hybrid score fusion — pure functions, no I/O, fully unit-testable.

RRF is the default: it consumes only ranks, so it is immune to the scale
mismatch between cosine scores (in [0, 1]) and BM25 scores (unbounded).
The weighted mode exists for later alpha-tuning experiments.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked: dict[str, list[str]],
    k: int = 60,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists with RRF: score(id) = Σ 1 / (k + rank).

    Args:
        ranked: leg name -> chunk ids in rank order (best first, 1-based).
        k: RRF constant, 60 per Cormack et al. (SIGIR 2009).
        top_k: keep only this many results.

    Returns:
        (chunk_id, fused_score) sorted by score desc, ties broken by
        chunk_id asc so output is deterministic.
    """
    scores: dict[str, float] = {}
    for ids in ranked.values():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:top_k]


def weighted_relative_fusion(
    scored: dict[str, list[tuple[str, float]]],
    weights: dict[str, float],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Fuse scored hits with min-max normalisation + per-leg weights.

    Each leg's scores are squeezed into [0, 1]; a chunk absent from a leg
    contributes 0 from that leg. Degenerate legs (all scores equal) map to
    1.0 when the shared score is positive, else 0.0.
    """
    totals: dict[str, float] = {}
    for leg, hits in scored.items():
        weight = weights.get(leg, 0.0)
        if not hits:
            continue
        values = [score for _, score in hits]
        lo, hi = min(values), max(values)
        for chunk_id, score in hits:
            if hi > lo:
                norm = (score - lo) / (hi - lo)
            else:
                norm = 1.0 if score > 0 else 0.0
            totals[chunk_id] = totals.get(chunk_id, 0.0) + weight * norm
    fused = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:top_k]
