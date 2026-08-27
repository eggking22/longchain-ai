"""Pydantic schemas for the Statement Draft layer (MCQ Generator step 1).

A Statement Draft is one structured statement for a future "which of the
following statements are correct/incorrect?" question: exactly one TRUE
statement per set (derived from a Question Blueprint's evidence-bound answer)
plus controlled false statements produced by minimal, evidence-based
perturbations. No Chinese rendering, no A/B/C/D layout, no LLM, no reviewer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

PerturbationType = Literal[
    "NONE",  # the true statement
    "SIGNIFICANCE_FLIP",
    "DIRECTION_FLIP",
    "GROUP_SWAP",
    "VARIABLE_SWAP",
    "DV_SWAP",
    "CAUSALITY_UPGRADE",
    "CONCLUSION_FLIP",
    "CONDITION_MUTATION",
    "NUMERIC_MUTATION",
    "PANEL_MISATTRIBUTION",
]

DraftStatus = Literal["READY", "INSUFFICIENT"]  # this phase emits READY only


class StatementDraft(BaseModel):
    """One statement (true or perturbed-false) inside a draft set."""

    draft_id: str  # {draft_set_id}_{nn}
    blueprint_id: str
    figure_id: str
    panel_ids: list[str] = Field(default_factory=list)
    statement: str
    is_correct: bool
    perturbation_type: PerturbationType
    evidence_ids: list[str] = Field(default_factory=list)  # traceable to the Evidence Store
    confidence: float
    status: DraftStatus = "READY"
    detail: dict = Field(default_factory=dict)  # perturbation-specific slots


class StatementDraftSet(BaseModel):
    """One true statement plus its controlled perturbations, per blueprint."""

    draft_set_id: str  # qd_{experiment key}_{seq}
    blueprint_id: str
    figure_id: str
    question_type: str  # inherited from the blueprint
    panel_ids: list[str] = Field(default_factory=list)
    statements: list[StatementDraft] = Field(default_factory=list)  # exactly one is_correct=True
    detail: dict = Field(default_factory=dict)


class QuestionDraftReport(BaseModel):
    """Top-level artifact for one document."""

    doc_id: str
    summary: dict = Field(default_factory=dict)  # sets/statements counts, by_perturbation, skipped
    draft_sets: list[StatementDraftSet] = Field(default_factory=list)
