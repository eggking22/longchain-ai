"""Centralised, tunable knobs for the evidence gate.

Mirrors ParserConfig / RetrievalConfig: every knob is settable from .env.
Weights must sum to 1 so coverage_score stays a proper [0, 1] blend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class EvidenceConfig:
    coverage_threshold: float = 0.8  # sufficient iff coverage_score >= threshold
    term_weight: float = 0.75  # query-term coverage signal (union+focus blend)
    sparse_weight: float = 0.15  # saturating BM25 strength of the best hit
    hit_weight: float = 0.1  # how many hits clear the score floor
    sparse_saturate: float = 12.0  # BM25 top score mapped to a full 1.0 signal
    hit_floor: float = 2.0  # BM25 score a hit needs to count toward depth
    evidence_window: int = 5  # hits cited in the report
    coverage_pool: int = 8  # deeper pool inspected for term coverage (>= window)
    llm_max_chars_per_chunk: int = 300  # Level-2 prompt truncation

    def __post_init__(self) -> None:
        total = self.term_weight + self.sparse_weight + self.hit_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"evidence weights must sum to 1.0, got {total}")
        if self.coverage_pool < self.evidence_window:
            raise ValueError("coverage_pool must be >= evidence_window")

    @classmethod
    def from_settings(cls, settings) -> "EvidenceConfig":
        return cls(
            coverage_threshold=settings.EVIDENCE_COVERAGE_THRESHOLD,
            term_weight=settings.EVIDENCE_TERM_WEIGHT,
            sparse_weight=settings.EVIDENCE_SPARSE_WEIGHT,
            hit_weight=settings.EVIDENCE_HIT_WEIGHT,
            sparse_saturate=settings.EVIDENCE_SPARSE_SATURATE,
            hit_floor=settings.EVIDENCE_HIT_FLOOR,
            evidence_window=settings.EVIDENCE_WINDOW,
            coverage_pool=settings.EVIDENCE_COVERAGE_POOL,
        )

    def as_dict(self) -> dict:
        return asdict(self)
