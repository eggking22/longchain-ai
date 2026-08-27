"""Artifact persistence for paper semantics.

Storage layout (``data/paper_semantics/{doc_id}/``), following patterns from
the open-source literature-reading stacks researched for this project:

- **S2ORC** (Allen AI) stores figure/table reference entries once in a side
  table and lets text refer to them by id — we do the same: full evidence
  texts live in ``evidence.jsonl`` (one unit per line, the repo's chunks.jsonl
  convention) and every other file references evidence by id only.
- **Docling** (IBM) always ships a human-readable Markdown companion next to
  the machine JSON — ``report.md`` is that companion, fully deterministic.
- **PDFFigures2** (Allen AI) keeps per-figure records compact (name/type/
  caption/page) with bulk data separated — ``figures.json`` is a scannable
  semantic index headed by the paper background (Abstract + Introduction).

Files:

- figures.json     — background → summary → compact per-figure semantic blocks
                     (evidence referenced by id only; includes the L1 text
                     block partition of the Results narrative);
- evidence.jsonl   — the single evidence store (figure-level and panel-level
                     units, deduplicated by evidence_id, with assignment);
- experiments.json — full ExperimentModels (figures + panels);
- report.md        — human-readable Markdown report: background first, then
                     one section per figure;
- manifest.json    — doc_id, status counts, reading index (IMRaD map, figure
                     inventory, partition coverage), input sha256, file
                     inventory, created_at (the only timestamp anywhere).

All content files are deterministic: same input tree → byte-identical files.
Existing artifact directories (raw/structure/chunks/index) are never touched.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.paper_semantics import FigureSemantic, PaperBackground, PaperSemanticsReport

_TYPE_ORDER = ("direct_observation", "experimental_design", "statistical_result", "author_interpretation", "mixed")


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _input_hash(document_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(document_path.read_bytes())
    return digest.hexdigest()[:16]


def _evidence_store(report: PaperSemanticsReport) -> list[dict]:
    """All evidence units (figure-level + panel-level), deduplicated by id, in order."""
    store: dict[str, dict] = {}
    for figure in report.figures:
        for evidence in figure.evidence + [e for panel in figure.panels for e in panel.evidence]:
            if evidence.evidence_id and evidence.evidence_id not in store:
                store[evidence.evidence_id] = evidence.model_dump()
    return list(store.values())


def _counts_by_status(items) -> dict:
    counts = {"SUFFICIENT": 0, "PARTIAL": 0, "INSUFFICIENT": 0}
    for item in items:
        counts[item.reconstruction_status] += 1
    return counts


def _type_histogram(evidences) -> dict:
    histogram: dict[str, int] = {}
    for evidence in evidences:
        key = evidence.get("evidence_type") if isinstance(evidence, dict) else evidence.evidence_type
        histogram[key] = histogram.get(key, 0) + 1
    return {key: histogram[key] for key in _TYPE_ORDER if key in histogram}


def _text_block_index(block) -> dict | None:
    if block is None:
        return None
    return {
        "anchors": block.anchor_paragraph_ids,
        "continuations": block.continuation_paragraph_ids,
        "shared_with": block.shared_with,
        "paragraph_count": len(block.paragraph_ids),
    }


def _figure_index(figure: FigureSemantic) -> dict:
    """Compact, human-scannable semantic block: evidence referenced by id only."""
    experiment = figure.experiment
    index: dict = {
        "figure_id": figure.figure_id,
        "kind": figure.kind,
        "status": figure.reconstruction_status,
        "confidence": figure.confidence,
        "method": figure.method,
        "title": figure.title,
        "caption": figure.caption,
    }
    if experiment is not None:
        index.update(
            {
                "research_question": experiment.research_question,
                "groups": {
                    "experimental": experiment.experimental_groups,
                    "control": experiment.control_groups,
                },
                "variables": {
                    "independent": experiment.independent_variables,
                    "dependent": experiment.dependent_variables,
                },
                "observations": [
                    {
                        "statement": o.statement,
                        "direction": o.direction,
                        "significance": o.significance,
                        "relationship": o.relationship_type,
                        "p_value": o.p_value,
                        "evidence": o.evidence_ids,
                    }
                    for o in experiment.observations
                ],
                "interpretations": [
                    {
                        "id": i.interpretation_id,
                        "statement": i.statement,
                        "direction": i.direction,
                        "relationship": i.relationship_type,
                        "evidence": i.evidence_ids,
                    }
                    for i in experiment.interpretations
                ],
                "conclusions": [
                    {
                        "statement": c.statement,
                        "relationship": c.relationship_type,
                        "evidence": c.evidence_ids,
                        "interpretations": c.interpretation_ids,
                    }
                    for c in experiment.conclusions
                ],
                "statistical_results": experiment.statistical_results,
            }
        )
    if figure.missing_information:
        index["missing_information"] = figure.missing_information
    if figure.panels:
        index["panels"] = [
            {
                "panel_id": panel.panel_id,
                "status": panel.reconstruction_status,
                "confidence": panel.confidence,
                "title": panel.title,
                "missing_information": panel.missing_information,
                "experiment_id": panel.experiment.experiment_id if panel.experiment else None,
                "evidence": [e.evidence_id for e in panel.evidence],
            }
            for panel in figure.panels
        ]
    if figure.text_block is not None:
        index["text_block"] = _text_block_index(figure.text_block)
    index["evidence"] = [e.evidence_id for e in figure.evidence]
    index["evidence_by_type"] = _type_histogram(figure.evidence)
    return index


def build_figures_document(report: PaperSemanticsReport) -> dict:
    """The readable figures.json document: background → summary → figure blocks."""
    panels = [panel for figure in report.figures for panel in figure.panels]
    evidence_units = _evidence_store(report)
    return {
        "doc_id": report.doc_id,
        "background": report.background.model_dump(),
        "summary": {
            "figures": {"total": len(report.figures), **_counts_by_status(report.figures)},
            "panels": {"total": len(panels), **_counts_by_status(panels)},
            "evidence_units": {"total": len(evidence_units), **_type_histogram(evidence_units)},
        },
        "figures": [_figure_index(figure) for figure in report.figures],
    }


def _preview(text: str, limit: int = 96) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _status_line(name: str, items) -> str:
    counts = _counts_by_status(items)
    return (
        f"**{len(items)} {name}** · SUFFICIENT {counts['SUFFICIENT']}"
        f" · PARTIAL {counts['PARTIAL']} · INSUFFICIENT {counts['INSUFFICIENT']}"
    )


def build_report_markdown(report: PaperSemanticsReport) -> str:
    """Deterministic human-readable Markdown companion (Docling-style)."""
    figures = report.figures
    panels = [panel for figure in figures for panel in figure.panels]
    lines: list[str] = [f"# {report.doc_id} — Figure Semantic Reconstruction", ""]

    background: PaperBackground = report.background
    if background.abstract:
        lines.append(f"**Abstract** {_preview(background.abstract.get('text', ''), 600)}")
        lines.append("")
    if background.introduction:
        lines.append(f"**Introduction** {_preview(background.introduction.get('text', ''), 600)}")
        lines.append("")
    lines.append(_status_line("figures", figures))
    lines.append(_status_line("panels", panels))
    lines.append("")

    for figure in figures:
        experiment = figure.experiment
        lines.append(
            f"## {figure.figure_id} — {figure.reconstruction_status} "
            f"(confidence {figure.confidence:.2f}, {figure.method})"
        )
        if figure.title:
            lines.append(f"**Caption:** {_preview(figure.caption or figure.title, 160)}")
        if experiment is not None and figure.reconstruction_status != "INSUFFICIENT":
            if experiment.research_question:
                lines.append(f"**Q:** {experiment.research_question}")
            if experiment.experimental_groups or experiment.control_groups:
                lines.append(
                    f"**Groups:** {', '.join(experiment.experimental_groups) or '?'} "
                    f"vs {', '.join(experiment.control_groups) or '?'}"
                )
            if experiment.dependent_variables:
                lines.append(f"**DV:** {', '.join(experiment.dependent_variables[:3])}")
            lines.append("")
            for observation in experiment.observations:
                flags = f"direction={observation.direction}"
                if observation.significance != "unspecified":
                    flags += f", {observation.significance}"
                if observation.relationship_type != "unspecified":
                    flags += f", {observation.relationship_type}"
                lines.append(
                    f"- **OBS** {_preview(observation.statement)} — {flags} "
                    f"[{', '.join(observation.evidence_ids) or '-'}]"
                )
            for interpretation in experiment.interpretations:
                lines.append(
                    f"- **INT** {_preview(interpretation.statement)} "
                    f"[{', '.join(interpretation.evidence_ids) or '-'}]"
                )
            for conclusion in experiment.conclusions:
                linked = f" ← {', '.join(conclusion.interpretation_ids)}" if conclusion.interpretation_ids else ""
                lines.append(
                    f"- **CON** {conclusion.statement} [{', '.join(conclusion.evidence_ids) or '-'}]{linked}"
                )
        if figure.reconstruction_status != "SUFFICIENT" and figure.missing_information:
            lines.append(f"**Missing:** {'; '.join(figure.missing_information)}")
        if figure.text_block is not None:
            block = figure.text_block
            shared = f", shared with {', '.join(block.shared_with)}" if block.shared_with else ""
            lines.append(
                f"**Text block:** {len(block.paragraph_ids)} paragraphs "
                f"({len(block.anchor_paragraph_ids)} anchored, "
                f"{len(block.continuation_paragraph_ids)} continuation{shared})"
            )
        if figure.panels:
            summary = " · ".join(
                f"{panel.panel_id} {panel.reconstruction_status}({panel.confidence:.2f})" for panel in figure.panels
            )
            lines.append(f"**Panels:** {summary}")
        types = _type_histogram(figure.evidence)
        type_note = ", ".join(f"{key} {value}" for key, value in types.items())
        lines.append(f"**Evidence:** {len(figure.evidence)} units ({type_note})")
        lines.append("")
    return "\n".join(lines)


def persist_report(
    report: PaperSemanticsReport,
    document_path: Path,
    artifacts_root: str | Path = "data",
) -> Path:
    """Persist report artifacts; returns the figure directory."""
    out_dir = Path(artifacts_root) / "paper_semantics" / report.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    figures_document = build_figures_document(report)
    (out_dir / "figures.json").write_text(
        json.dumps(figures_document, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    evidence_units = _evidence_store(report)
    with (out_dir / "evidence.jsonl").open("w", encoding="utf-8") as handle:
        for unit in evidence_units:
            handle.write(json.dumps(unit, ensure_ascii=False) + "\n")

    experiments = []
    for figure in report.figures:
        if figure.experiment is not None:
            experiments.append(figure.experiment.model_dump())
        experiments.extend(panel.experiment.model_dump() for panel in figure.panels if panel.experiment is not None)
    (out_dir / "experiments.json").write_text(
        json.dumps(experiments, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (out_dir / "report.md").write_text(build_report_markdown(report), encoding="utf-8")

    manifest = {
        "doc_id": report.doc_id,
        "num_figures": report.num_figures,
        "status_counts": report.stats.get("status_counts", {}),
        "panel_status_counts": report.stats.get("panel_status_counts", {}),
        "reading_index": report.stats.get("reading_index", {}),
        "config": report.stats.get("config", {}),
        "files": {
            "figures": "figures.json",
            "evidence": "evidence.jsonl",
            "experiments": "experiments.json",
            "report": "report.md",
        },
        "input_document_sha256": _input_hash(document_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


def load_document_tree(doc_id: str, artifacts_root: str | Path = "data"):
    """Read-only load of the Phase 1 structure artifact."""
    from app.schemas.document import DocNode

    path = Path(artifacts_root) / "structure" / doc_id / "document.json"
    if not path.exists():
        raise FileNotFoundError(
            f"document.json not found for doc_id={doc_id!r} under {path.parent} — run scripts/parse_pdf.py first"
        )
    return path, DocNode.model_validate_json(path.read_text(encoding="utf-8"))
