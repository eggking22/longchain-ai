"""Pydantic schemas for the Chinese MCQ statement layer (translation step).

The Chinese layer is a *display-side projection* of the Statement Drafts: the
English statements are kept verbatim (never overwritten), a ``statement_zh``
field is added, and all semantic invariants (numbers/units, gene names,
direction, significance, relationship strength, group roles, TRUE/FALSE) are
preserved — translation only changes the language, never the science.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TranslationMethod = Literal["template", "term_fallback"]


class MCQStatementZh(BaseModel):
    """One statement with its Chinese rendering (original English preserved)."""

    draft_id: str
    blueprint_id: str
    figure_id: str
    panel_ids: list[str] = Field(default_factory=list)
    statement: str  # original English, verbatim
    statement_zh: str = ""
    is_correct: bool  # unchanged by translation
    perturbation_type: str  # unchanged by translation
    evidence_ids: list[str] = Field(default_factory=list)  # unchanged by translation
    confidence: float = 0.0
    detail: dict = Field(default_factory=dict)  # gains translation_method
    translation_method: TranslationMethod = "template"


class MCQDraftSetZh(BaseModel):
    """One question set (1 true + N false statements) with Chinese renderings."""

    draft_set_id: str
    blueprint_id: str
    figure_id: str
    question_type: str
    panel_ids: list[str] = Field(default_factory=list)
    statements: list[MCQStatementZh] = Field(default_factory=list)


class MCQDraftReportZh(BaseModel):
    """Top-level Chinese artifact for one document."""

    doc_id: str
    summary: dict = Field(default_factory=dict)
    draft_sets: list[MCQDraftSetZh] = Field(default_factory=list)
