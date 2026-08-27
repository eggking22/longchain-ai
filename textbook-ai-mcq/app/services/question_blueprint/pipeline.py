"""Pipeline: Experiment Model → Question Blueprints → question_blueprints.json.

Deterministic end to end: the semantic layer is rebuilt in memory
(``reconstruct_figures(persist=False)`` — no existing artifact is rewritten),
blueprints are generated under strict gates, and the result is persisted as a
single new file next to the other paper_semantics artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.question_blueprint import QuestionBlueprintReport
from app.services.paper_semantics import PaperSemanticsConfig, reconstruct_figures

from .config import BlueprintConfig
from .generators import (
    _IdAssigner,
    SkipTracker,
    generate_data_statements,
    generate_experimental_design,
    generate_result_interpretation,
    generate_simple_prediction,
)
from .numeric import extract_numeric_findings


def generate_blueprints(
    doc_id: str,
    artifacts_root: str | Path = "data",
    config: BlueprintConfig | None = None,
    semantic_config: PaperSemanticsConfig | None = None,
    persist: bool = True,
    semantic_report=None,
) -> QuestionBlueprintReport:
    """Generate deterministic question blueprints for every figure/panel of a document.

    semantic_report (optional): a precomputed PaperSemanticsReport to reuse
    instead of reconstructing — additive parameter, default behavior unchanged.
    """
    config = config or BlueprintConfig()
    if semantic_report is None:
        semantic_report = reconstruct_figures(doc_id, artifacts_root, config=semantic_config, persist=False)
    report = semantic_report

    ids = _IdAssigner()
    skips = SkipTracker()
    blueprints = []

    for figure in report.figures:
        experiment = figure.experiment
        if experiment is None or figure.reconstruction_status == "INSUFFICIENT":
            for question_type in ("RESULT_INTERPRETATION", "EXPERIMENTAL_DESIGN", "SIMPLE_PREDICTION", "DATA_STATEMENT"):
                skips.skip(question_type, "figure_insufficient")
            continue

        blueprints.extend(generate_result_interpretation(figure, experiment, config=config, ids=ids, skips=skips))
        blueprints.extend(generate_experimental_design(figure, experiment, config=config, ids=ids, skips=skips))
        blueprints.extend(generate_simple_prediction(figure, experiment, config=config, ids=ids, skips=skips))
        blueprints.extend(
            generate_data_statements(
                figure,
                findings=extract_numeric_findings(figure.evidence),
                experiment=experiment,
                config=config,
                ids=ids,
                skips=skips,
            )
        )

        for panel in figure.panels:
            if panel.experiment is None or panel.reconstruction_status == "INSUFFICIENT":
                for question_type in ("RESULT_INTERPRETATION", "DATA_STATEMENT"):
                    skips.skip(question_type, "panel_insufficient")
                continue
            blueprints.extend(
                generate_result_interpretation(figure, panel.experiment, panel=panel, config=config, ids=ids, skips=skips)
            )
            blueprints.extend(
                generate_data_statements(
                    figure,
                    panel=panel,
                    findings=extract_numeric_findings(panel.evidence),
                    experiment=panel.experiment,
                    config=config,
                    ids=ids,
                    skips=skips,
                )
            )

    by_type: dict[str, int] = {question_type: 0 for question_type in (
        "RESULT_INTERPRETATION", "EXPERIMENTAL_DESIGN", "SIMPLE_PREDICTION", "DATA_STATEMENT"
    )}
    for blueprint in blueprints:
        by_type[blueprint.question_type] += 1

    blueprint_report = QuestionBlueprintReport(
        doc_id=doc_id,
        summary={
            "total": len(blueprints),
            "by_type": by_type,
            "skipped": skips.as_dict(),
            "method": "deterministic",
        },
        blueprints=blueprints,
    )
    if persist:
        persist_blueprints(blueprint_report, artifacts_root)
    return blueprint_report


def persist_blueprints(report: QuestionBlueprintReport, artifacts_root: str | Path = "data") -> Path:
    """Write question_blueprints.json (deterministic content, no timestamps)."""
    out_dir = Path(artifacts_root) / "paper_semantics" / report.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": report.doc_id,
        "summary": report.summary,
        "blueprints": [blueprint.model_dump() for blueprint in report.blueprints],
    }
    path = out_dir / "question_blueprints.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
