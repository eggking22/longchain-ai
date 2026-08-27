"""Pydantic schemas for Scientific Paper Figure Semantic Reconstruction.

A "figure" here is a Scientific Evidence Unit defined by text evidence
(caption + Results/Methods/Discussion paragraphs), not an image. Every
reconstructed statement must cite the evidence (evidence_id) it was derived
from, so downstream MCQ generation can always trace claims back to the paper.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# --- Vocabulary types -----------------------------------------------------------

PaperSectionType = Literal[
    "introduction", "methods", "results", "discussion", "abstract", "other"
]
EvidenceRole = Literal["caption", "direct", "supporting", "interpretation"]
EvidenceAssignment = Literal["anchor", "continuation", ""]  # how a paragraph was bound to its figure
FigureKind = Literal["figure", "table"]
ReconstructionStatus = Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"]
Direction = Literal["increase", "decrease", "no_change", "unspecified"]
Significance = Literal["significant", "not_significant", "unspecified"]
RelationshipType = Literal[
    "causal",
    "correlation",
    "association",
    "inhibition",
    "activation",
    "knockout",
    "overexpression",
    "dose_response",
    "time_dependent",
    "unspecified",
]
ReconstructionMethod = Literal["deterministic", "deterministic+llm"]
EvidenceType = Literal[
    "direct_observation",  # what was measured/observed (Results statements)
    "experimental_design",  # how the experiment was set up (groups, treatments, Methods)
    "statistical_result",  # statistics notes with literal p-values
    "author_interpretation",  # Discussion claims — attributed to the authors, not asserted
    "mixed",  # retrieval-supplemented text of unknown provenance
]


# --- Evidence -------------------------------------------------------------------


class PaperEvidence(BaseModel):
    """One piece of text evidence bound to a figure, with full provenance.

    role is the *source* dimension (where the text came from); evidence_type is
    the *semantic* dimension (what kind of information it carries), so future
    question blueprints can impose rules like "conclusion questions require
    direct_observation evidence".
    """

    evidence_id: str = ""  # ev_001, assigned when the per-figure bundle is finalized
    figure_id: str  # canonical id, e.g. "Figure 2" / "Table 1"
    text: str
    role: EvidenceRole  # caption=figure caption, direct=Results statement, supporting=Methods, interpretation=Discussion
    evidence_type: EvidenceType = "mixed"
    section_type: PaperSectionType = "other"
    section_title: str = ""
    breadcrumb: list[str] = Field(default_factory=list)  # chapter/section titles from the parsed tree
    paragraph_id: str = ""  # DocNode node_id of the source paragraph
    page_no: int = 0
    chunk_id: Optional[str] = None  # set when sourced via figure-aware retrieval
    panel_id: str = ""  # "2a" when bound to a specific panel, "" for figure-level evidence
    assignment: EvidenceAssignment = ""  # anchor=cited explicitly, continuation=inherited from the text block


class FigureReference(BaseModel):
    """A canonical figure/table identity extracted from the paper text."""

    figure_id: str  # canonical: "Figure 2" / "Table 1"
    kind: FigureKind
    number: int
    raw_forms: list[str] = Field(default_factory=list)  # distinct surface forms seen, e.g. "Fig. 2B"
    subfigures: list[str] = Field(default_factory=list)  # e.g. "2A", "2B" (panel labels seen in text)
    mention_paragraph_ids: list[str] = Field(default_factory=list)
    panel_mention_paragraph_ids: dict[str, list[str]] = Field(default_factory=dict)  # "a" -> [paragraph ids citing Fig. 2A]
    caption_paragraph_id: Optional[str] = None
    caption_text: str = ""
    caption_offset: int = 0  # >0 when the caption is embedded mid-paragraph (sliced from this offset)
    panel_texts: dict[str, str] = Field(default_factory=dict)  # "a" -> caption chunk describing panel a


# --- Experiment model ------------------------------------------------------------


class Observation(BaseModel):
    """A single result statement extracted from direct evidence.

    direction/significance come only from explicit text markers;
    "significantly increased" yields significance="significant" but never a
    fabricated p-value (p_value is set only when literally present in text).
    """

    statement: str
    direction: Direction = "unspecified"
    significance: Significance = "unspecified"
    relationship_type: RelationshipType = "unspecified"
    p_value: Optional[str] = None  # literal text like "p < 0.05", never inferred
    evidence_ids: list[str] = Field(default_factory=list)


class Interpretation(BaseModel):
    """An author's mechanistic/explicative claim, recorded verbatim.

    Interpretations come from Discussion (author_interpretation) evidence and
    are *attributed claims*, not system assertions: the statement is the
    author's sentence, so no causal strength is ever added or removed.
    """

    interpretation_id: str = ""  # int_001, assigned after the evidence bundle is numbered
    statement: str
    direction: Direction = "unspecified"
    relationship_type: RelationshipType = "unspecified"
    evidence_ids: list[str] = Field(default_factory=list)


class Conclusion(BaseModel):
    """A conclusion statement; must cite the evidence supporting it.

    Layering: conclusions are synthesized from Observations only; when an
    Interpretation of the same direction backs the claim, its id is linked in
    interpretation_ids without changing the conclusion text.
    """

    statement: str
    relationship_type: RelationshipType = "unspecified"
    evidence_ids: list[str] = Field(default_factory=list)
    interpretation_ids: list[str] = Field(default_factory=list)


class ExperimentModel(BaseModel):
    """The experiment logic behind a figure, reconstructed from text evidence."""

    experiment_id: str  # exp_002 (derived from the figure number)
    research_question: str = ""
    hypothesis: str = ""
    subjects: list[str] = Field(default_factory=list)
    independent_variables: list[str] = Field(default_factory=list)
    dependent_variables: list[str] = Field(default_factory=list)
    experimental_groups: list[str] = Field(default_factory=list)
    control_groups: list[str] = Field(default_factory=list)
    intervention: str = ""
    measurements: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    interpretations: list[Interpretation] = Field(default_factory=list)
    statistical_results: list[str] = Field(default_factory=list)  # literal text only, e.g. "p < 0.05"
    conclusions: list[Conclusion] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


# --- Figure-level and report models ------------------------------------------------


class FigureTextBlock(BaseModel):
    """L1 deterministic partition of the Results text flow around one figure.

    Anchors are paragraphs that cite the figure explicitly (hard facts from
    the text); continuation paragraphs are uncited narrative inherited from the
    last anchor — they widen the evidence scope only and never add semantics
    by themselves.
    """

    figure_id: str
    anchor_paragraph_ids: list[str] = Field(default_factory=list)
    continuation_paragraph_ids: list[str] = Field(default_factory=list)
    shared_with: list[str] = Field(default_factory=list)  # secondary figures cited in the same paragraphs

    @property
    def paragraph_ids(self) -> list[str]:
        return [*self.anchor_paragraph_ids, *self.continuation_paragraph_ids]


class PaperBackground(BaseModel):
    """Extractive paper-level background: the authors' own words, never rewritten."""

    abstract: dict = Field(default_factory=dict)  # {text, source, paragraph_ids, pages}
    introduction: dict = Field(default_factory=dict)  # {text, paragraph_ids, pages}


class PanelSemantic(BaseModel):
    """An independent semantic unit for one panel of a figure (2A/2B/...).

    Panels get their own evidence bundles (panel caption chunk + paragraphs
    citing that panel), their own experiment reconstruction and their own gate
    verdict. The figure-level status is never auto-upgraded from panels.
    """

    panel_id: str  # "2a" (lowercase-normalized)
    label: str  # "a"
    title: str = ""  # leading fragment of the panel's caption chunk
    experiment: Optional[ExperimentModel] = None
    evidence: list[PaperEvidence] = Field(default_factory=list)
    reconstruction_status: ReconstructionStatus = "INSUFFICIENT"
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    method: ReconstructionMethod = "deterministic"


class FigureSemantic(BaseModel):
    """The reconstructed semantics of one figure/table."""

    figure_id: str
    kind: FigureKind
    title: str = ""  # leading fragment of the caption
    caption: str = ""
    references: list[str] = Field(default_factory=list)  # raw mention forms found in the text
    experiment: Optional[ExperimentModel] = None
    evidence: list[PaperEvidence] = Field(default_factory=list)
    panels: list[PanelSemantic] = Field(default_factory=list)
    text_block: Optional[FigureTextBlock] = None  # L1 partition of the Results flow around this figure
    reconstruction_status: ReconstructionStatus = "INSUFFICIENT"
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = 0.0  # weighted fraction of core slots recovered
    method: ReconstructionMethod = "deterministic"
    detail: dict = Field(default_factory=dict)  # diagnostics (e.g. llm rejection reason)

    def evidence_of_type(self, evidence_type: str) -> list[PaperEvidence]:
        """Query the bundle by semantic type (e.g. 'direct_observation').

        Supports downstream question rules like "conclusion questions require
        Results evidence" without any MCQ-specific logic living here.
        """
        return [e for e in self.evidence if e.evidence_type == evidence_type]


class PaperSemanticsReport(BaseModel):
    """Top-level artifact for one document."""

    doc_id: str
    num_figures: int = 0
    background: PaperBackground = Field(default_factory=PaperBackground)
    figures: list[FigureSemantic] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)  # counts by status/kind, config snapshot
    created_at: str = ""


# --- LLM normalization contract ----------------------------------------------------


class LlmConclusion(BaseModel):
    """One normalized conclusion proposed by the optional LLM normalizer."""

    statement: str
    relationship_type: RelationshipType = "unspecified"
    evidence_ids: list[str] = Field(default_factory=list)


class LlmNormalizationVerdict(BaseModel):
    """Strict-JSON parse target for the optional LLM normalizer.

    Every field is a *patch* over the deterministic draft; conclusions must
    cite evidence_ids that exist in the figure's evidence bundle or the whole
    verdict is rejected (see services/paper_semantics/llm_normalizer.py).
    """

    research_question: str = ""
    hypothesis: str = ""
    subjects: list[str] = Field(default_factory=list)
    independent_variables: list[str] = Field(default_factory=list)
    dependent_variables: list[str] = Field(default_factory=list)
    experimental_groups: list[str] = Field(default_factory=list)
    control_groups: list[str] = Field(default_factory=list)
    intervention: str = ""
    measurements: list[str] = Field(default_factory=list)
    conclusions: list[LlmConclusion] = Field(default_factory=list)
