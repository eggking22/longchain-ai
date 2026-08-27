"""Pydantic schemas for the Question Blueprint layer (Phase 5).

A Question Blueprint is a deterministic, evidence-bound *plan* for a future
MCQ — it is NOT a question: no stem, no options, no answer key. Everything a
blueprint claims must trace back to the Evidence Store via evidence_ids, and
the semantic layer's relationship strength (association ≠ causation) is
preserved verbatim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal[
    "RESULT_INTERPRETATION",
    "EXPERIMENTAL_DESIGN",
    "SIMPLE_PREDICTION",
    "DATA_STATEMENT",
]
ReasoningOperation = Literal[
    "comparison",
    "result_interpretation",
    "experimental_design_reasoning",
    "local_prediction",
    "quantitative_reading",
]
BlueprintStatus = Literal["READY", "INSUFFICIENT"]  # this phase emits READY only; gaps mean no blueprint


class QuestionBlueprint(BaseModel):
    """One deterministic question plan derived from an Experiment Model."""

    blueprint_id: str  # qb_{experiment key}_{type abbreviation}_{seq}, e.g. qb_f02_ri_001
    question_type: QuestionType
    experiment_id: str  # exp_f02 (figure level) / exp_f02a (panel level)
    figure_id: str  # "Figure 2" / "Extended Data Figure 1" / "Table 1"
    panel_ids: list[str] = Field(default_factory=list)  # ["2a"] for panel-level blueprints
    question_focus: str  # what the future question examines (comparison, purpose, prediction, data)
    required_evidence: list[str] = Field(default_factory=list)  # evidence types this question needs
    reasoning_operation: ReasoningOperation
    expected_answer: str  # deterministic template answer derived from recorded slots only
    evidence_ids: list[str] = Field(default_factory=list)  # must resolve into the Evidence Store
    confidence: float = 0.0  # inherited from the source figure/panel reconstruction
    status: BlueprintStatus = "READY"
    detail: dict = Field(default_factory=dict)  # structured slots for future MCQ rendering


class QuestionBlueprintReport(BaseModel):
    """Top-level artifact for one document."""

    doc_id: str
    summary: dict = Field(default_factory=dict)  # total, by_type counts, skipped-reason counts
    blueprints: list[QuestionBlueprint] = Field(default_factory=list)
