"""Pydantic schemas for the evidence gate (Phase 3).

A CoverageReport answers one question: "can the retrieved evidence support
answering this query from the textbook?" It carries the gate decision, a
0-1 coverage score, the supporting evidence with provenance, and what is
missing when the verdict is negative.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """One supporting chunk as cited by the gate."""

    chunk_id: str
    score: float  # retrieval score the gate saw (fused score by default)
    rank: int = 0
    document_id: str = ""
    breadcrumb: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    text_preview: str = ""


class CoverageReport(BaseModel):
    """Gate decision for one (query, retrieval) pair — the spec output."""

    query: str
    sufficient: bool
    coverage_score: float  # 0.0 - 1.0, compared against the threshold
    threshold: float
    evidence: list[EvidenceItem] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    level: str = "heuristic"  # "heuristic" | "heuristic+llm"
    detail: dict = Field(default_factory=dict)  # per-level diagnostics


class LlmCoverageVerdict(BaseModel):
    """Level-2 LLM judge output (strict JSON contract)."""

    sufficient: bool
    coverage_score: float = 0.0
    missing_information: list[str] = Field(default_factory=list)
    reasoning: str = ""
