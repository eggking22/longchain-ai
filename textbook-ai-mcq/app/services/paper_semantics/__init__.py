"""Scientific Paper Figure Semantic Reconstruction.

Recovers the experiment semantics and conclusions behind each Figure/Table of
a biology paper from *text evidence only* (caption + Results/Methods/
Discussion paragraphs) — no image parsing, no OCR, no vision model. Reuses the
Phase 1 structure tree read-only; optionally supplements evidence through a
read-only wrapper of the Phase 2 RetrievalEngine.

Pipeline:

    document.json → flatten + IMRaD classify → figure references →
    evidence collection → deterministic experiment model → conclusions →
    semantic evidence gate (SUFFICIENT/PARTIAL/INSUFFICIENT) →
    optional LLM normalization (evidence-bound, rejected on violation) →
    data/paper_semantics/{doc_id}/

Core safety constraints:
- no text evidence → no recovered semantics (INSUFFICIENT, never a guess);
- association/correlation are never upgraded to causation;
- p-values/numbers are recorded only when literally present in the text.
"""

from __future__ import annotations

from .config import PaperSemanticsConfig
from .conclusion import build_conclusions
from .evidence_collector import collect_evidence, collect_panel_evidence, infer_evidence_type, number_evidence
from .experiment_model import (
    build_experiment,
    classify_relationship,
    detect_direction,
    detect_significance,
    extract_intervention,
    extract_interpretations,
)
from .figure_reference import extract_figure_references, find_inline_caption, find_mentions, parse_caption, split_caption_panels
from .gate import GateVerdict, SemanticEvidenceGate
from .llm_normalizer import DEFAULT_LLM_BASE_URL, LlmNormalizationError, LlmSemanticNormalizer
from .panel import panel_labels, reconstruct_panel
from .patterns import split_sentences_en, word_tokens
from .persistence import build_figures_document, build_report_markdown, load_document_tree, persist_report
from .pipeline import reconstruct_figures
from .retrieval_adapter import FigureAwareRetriever
from .sections import ParagraphRecord, classify_section, flatten_document
from .text_partition import extract_background, partition_flow, partition_results_by_figures

__all__ = [
    "PaperSemanticsConfig",
    "ParagraphRecord",
    "GateVerdict",
    "SemanticEvidenceGate",
    "LlmSemanticNormalizer",
    "LlmNormalizationError",
    "DEFAULT_LLM_BASE_URL",
    "FigureAwareRetriever",
    "build_conclusions",
    "build_experiment",
    "build_figures_document",
    "build_report_markdown",
    "classify_relationship",
    "classify_section",
    "collect_evidence",
    "collect_panel_evidence",
    "detect_direction",
    "detect_significance",
    "extract_background",
    "extract_figure_references",
    "extract_intervention",
    "extract_interpretations",
    "find_inline_caption",
    "find_mentions",
    "flatten_document",
    "infer_evidence_type",
    "load_document_tree",
    "number_evidence",
    "panel_labels",
    "parse_caption",
    "partition_flow",
    "partition_results_by_figures",
    "persist_report",
    "reconstruct_figures",
    "reconstruct_panel",
    "split_caption_panels",
    "split_sentences_en",
    "word_tokens",
]
