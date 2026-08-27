"""Pipeline orchestration: paper structure → per-figure semantics → artifacts.

Data flow (all Phase 1/2/3 inputs are read-only):

    data/structure/{doc_id}/document.json
        ↓ flatten + IMRaD classify                     (sections.py)
        ↓ caption/mention extraction                   (figure_reference.py)
        ↓ evidence collection + numbering              (evidence_collector.py)
        ↓ deterministic experiment model + conclusions (experiment_model.py, conclusion.py)
        ↓ semantic evidence gate                       (gate.py)
        ↓ optional LLM normalization + validation      (llm_normalizer.py)
    FigureSemantic[] → data/paper_semantics/{doc_id}/  (persistence.py)

The optional figure-aware retriever (retrieval_adapter.py) supplements bundles
with extra supporting evidence when a Phase 2 index exists.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.schemas.paper_semantics import FigureSemantic, PaperBackground, PaperSemanticsReport

from .config import PaperSemanticsConfig
from .conclusion import build_conclusions
from .evidence_collector import collect_evidence, number_evidence
from .experiment_model import build_experiment
from .figure_reference import extract_figure_references, figure_key
from .gate import SemanticEvidenceGate
from .llm_normalizer import LlmSemanticNormalizer
from .panel import panel_labels, reconstruct_panel
from .patterns import CAPTION_START_RE
from .persistence import load_document_tree, persist_report
from .retrieval_adapter import FigureAwareRetriever
from .sections import flatten_document
from .text_partition import extract_background, partition_flow, partition_results_by_figures


def _title_from_caption(caption: str, max_chars: int = 120) -> str:
    match = CAPTION_START_RE.match(caption)
    body = caption[match.end() :] if match else caption
    first = body.split(". ", 1)[0].strip()
    return first[:max_chars]


def _reading_index(paragraphs, references, figures, text_blocks) -> dict:
    """Paper-level reading view: IMRaD distribution, figure inventory, partition coverage."""
    from .text_partition import partition_flow as _flow

    section_counts: dict[str, int] = {}
    for paragraph in paragraphs:
        section_counts[paragraph.section_type] = section_counts.get(paragraph.section_type, 0) + 1

    status_by_figure = {f.figure_id: f.reconstruction_status for f in figures}
    inventory = [
        {
            "figure_id": ref.figure_id,
            "kind": ref.kind,
            "status": status_by_figure.get(ref.figure_id),
            "caption": _title_from_caption(ref.caption_text, 80),
        }
        for ref in references
    ]

    flow_size = len(_flow(paragraphs, references))
    anchors = sum(len(block.anchor_paragraph_ids) for block in text_blocks.values())
    continuations = sum(len(block.continuation_paragraph_ids) for block in text_blocks.values())
    partition = {
        "method": "L1-deterministic",
        "flow_paragraphs": flow_size,
        "assigned": anchors + continuations,
        "anchors": anchors,
        "continuations": continuations,
        "unassigned": max(flow_size - anchors - continuations, 0),
        "coverage": round((anchors + continuations) / flow_size, 3) if flow_size else 0.0,
    }
    return {
        "section_paragraph_counts": section_counts,
        "figure_inventory": inventory,
        "partition": partition,
    }


def reconstruct_figures(
    doc_id: str,
    artifacts_root: str | Path = "data",
    config: Optional[PaperSemanticsConfig] = None,
    normalizer: Optional[LlmSemanticNormalizer] = None,
    retriever: Optional[FigureAwareRetriever] = None,
    persist: bool = True,
) -> PaperSemanticsReport:
    """Reconstruct experiment semantics for every figure/table in a document."""
    if config is None:
        from app.core.config import get_settings

        config = PaperSemanticsConfig.from_settings(get_settings())

    document_path, tree = load_document_tree(doc_id, artifacts_root)
    paragraphs = flatten_document(tree)
    references = extract_figure_references(paragraphs, config)
    background = PaperBackground(**extract_background(paragraphs, references))
    text_blocks = partition_results_by_figures(paragraphs, references)
    gate = SemanticEvidenceGate()

    figures: list[FigureSemantic] = []
    for ref in references:
        text_block = text_blocks.get(ref.figure_id)
        evidences = collect_evidence(ref, paragraphs, config, text_block=text_block)
        if retriever is not None:
            covered = {e.paragraph_id for e in evidences if e.paragraph_id}
            evidences = evidences + retriever.collect(ref, covered)
        number_evidence(evidences, id_prefix=f"ev_{figure_key(ref)}_")

        experiment = build_experiment(ref, evidences)
        experiment.conclusions = build_conclusions(experiment)
        verdict = gate.evaluate(ref, experiment, evidences)

        method: str = "deterministic"
        detail: dict = {}
        if normalizer is not None and verdict.status != "INSUFFICIENT":
            try:
                llm_verdict = normalizer.normalize(ref, evidences, experiment)
                merged, rejection = normalizer.apply(
                    experiment,
                    llm_verdict,
                    valid_evidence_ids={e.evidence_id for e in evidences},
                    evidence_text="\n".join(e.text for e in evidences),
                )
                if rejection is None:
                    experiment = merged
                    method = "deterministic+llm"
                else:
                    detail["llm_rejected"] = rejection
            except Exception as exc:  # network/JSON failure must never break reconstruction
                detail["llm_error"] = str(exc)

        figures.append(
            FigureSemantic(
                figure_id=ref.figure_id,
                kind=ref.kind,
                title=_title_from_caption(ref.caption_text),
                caption=ref.caption_text,
                references=list(ref.raw_forms),
                experiment=experiment,
                evidence=evidences,
                panels=[
                    reconstruct_panel(ref, label, paragraphs, config, gate)
                    for label in panel_labels(ref)
                ],
                text_block=text_block,
                reconstruction_status=verdict.status,
                missing_information=verdict.missing_information,
                confidence=verdict.confidence,
                method=method,
                detail=detail,
            )
        )

    status_counts = {"SUFFICIENT": 0, "PARTIAL": 0, "INSUFFICIENT": 0}
    for figure in figures:
        status_counts[figure.reconstruction_status] += 1
    panel_status_counts = {"SUFFICIENT": 0, "PARTIAL": 0, "INSUFFICIENT": 0}
    num_panels = 0
    for figure in figures:
        for panel in figure.panels:
            num_panels += 1
            panel_status_counts[panel.reconstruction_status] += 1
    report = PaperSemanticsReport(
        doc_id=doc_id,
        num_figures=len(figures),
        background=background,
        figures=figures,
        stats={
            "status_counts": status_counts,
            "num_panels": num_panels,
            "panel_status_counts": panel_status_counts,
            "num_paragraphs": len(paragraphs),
            "reading_index": _reading_index(paragraphs, references, figures, text_blocks),
            "config": config.as_dict(),
        },
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    if persist:
        persist_report(report, document_path, artifacts_root)
    return report
