"""Evidence gate package (Phase 3).

Query → hybrid retrieval (Phase 2) → evidence evaluation (Level 1 heuristic,
Level 2 optional LLM judge) → CoverageReport / InsufficientEvidenceError.
"""

from .config import EvidenceConfig
from .evaluator import (
    EvidenceError,
    EvidenceEvaluator,
    HeuristicEvidenceEvaluator,
    LlmEvidenceEvaluator,
    informative_tokens,
)
from .gate import EvidenceGate, InsufficientEvidenceError, build_evidence_gate

__all__ = [
    "EvidenceConfig",
    "EvidenceError",
    "EvidenceEvaluator",
    "EvidenceGate",
    "HeuristicEvidenceEvaluator",
    "InsufficientEvidenceError",
    "LlmEvidenceEvaluator",
    "build_evidence_gate",
    "informative_tokens",
]
