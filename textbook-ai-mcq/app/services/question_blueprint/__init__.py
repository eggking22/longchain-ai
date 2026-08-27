"""Question Blueprint layer (Phase 5): deterministic question plans over Experiment Models.

Pipeline (read-only reuse of the paper semantics layer):

    reconstruct_figures(persist=False)
        ↓ per figure / panel, strict gates
    RESULT_INTERPRETATION / EXPERIMENTAL_DESIGN / SIMPLE_PREDICTION / DATA_STATEMENT
        ↓ evidence-bound templates (association never upgraded to causal,
          numbers only when literally reported)
    data/paper_semantics/{doc_id}/question_blueprints.json

This phase deliberately produces blueprints only — no MCQ stem, no options,
no answer key, no LLM (a future L2 refinement layer can extend this baseline).
"""

from __future__ import annotations

from .config import BlueprintConfig
from .generators import (
    CAUSAL_LICENSED,
    SkipTracker,
    generate_data_statements,
    generate_experimental_design,
    generate_result_interpretation,
    generate_simple_prediction,
)
from .numeric import NumericFinding, extract_numeric_findings
from .pipeline import generate_blueprints, persist_blueprints

__all__ = [
    "BlueprintConfig",
    "CAUSAL_LICENSED",
    "NumericFinding",
    "SkipTracker",
    "extract_numeric_findings",
    "generate_blueprints",
    "generate_data_statements",
    "generate_experimental_design",
    "generate_result_interpretation",
    "generate_simple_prediction",
    "persist_blueprints",
]
