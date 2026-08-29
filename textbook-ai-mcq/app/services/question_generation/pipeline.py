"""Pipeline: Question Blueprints → Statement Draft sets → question_drafts.json.

Per figure, one blueprint per question type is selected (first of each type,
fixed priority order); its TRUE statement (built from blueprint-bound content
only) is perturbed with controlled minimal edits whose replacement material
comes from evidence-derived pools. A set with no applicable perturbation is
skipped entirely — a "which statements are correct" item needs at least one
false statement, and we never force one.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.paper_semantics import FigureSemantic, PaperSemanticsReport
from app.schemas.question_blueprint import QuestionBlueprint, QuestionBlueprintReport
from app.schemas.question_draft import QuestionDraftReport, StatementDraft, StatementDraftSet
from app.services.paper_semantics import PaperSemanticsConfig, reconstruct_figures
from app.services.paper_semantics.patterns import TREATMENT_SPAN_RE
from app.services.question_blueprint import BlueprintConfig, generate_blueprints
from app.services.question_blueprint.numeric import extract_numeric_findings

from .config import DraftConfig
from .llm_object_extractor import LlmObjectExtractionError, LlmObjectExtractor
from .perturbations import (
    PERTURBERS,
    PERTURBATION_ORDER,
    PerturbationContext,
    build_true_statement,
    needs_object_extraction,
)

# fixed priority order for blueprint selection within a figure
_TYPE_PRIORITY = ("RESULT_INTERPRETATION", "SIMPLE_PREDICTION", "DATA_STATEMENT", "EXPERIMENTAL_DESIGN")


def generate_question_drafts(
    doc_id: str,
    artifacts_root: str | Path = "data",
    config: DraftConfig | None = None,
    semantic_config: PaperSemanticsConfig | None = None,
    persist: bool = True,
    object_extractor: LlmObjectExtractor | None = None,
    semantic_report: PaperSemanticsReport | None = None,
) -> QuestionDraftReport:
    """Generate deterministic statement-draft sets for a document.

    object_extractor (optional LLM patch, verbatim-span-gated) only upgrades DATA
    statements that already fell back to a kind label; without it — or whenever it
    errors or is rejected — the deterministic result stands unchanged.

    semantic_report (optional): a precomputed PaperSemanticsReport to reuse instead
    of reconstructing — additive parameter, default behavior unchanged (same seam
    as generate_blueprints; lets an LLM-normalized report flow into the drafts).
    """
    config = config or DraftConfig()
    if semantic_report is None:
        semantic_report = reconstruct_figures(doc_id, artifacts_root, config=semantic_config, persist=False)
    semantics = semantic_report
    blueprints = generate_blueprints(
        doc_id, artifacts_root, config=BlueprintConfig(), semantic_report=semantics, persist=False
    )

    numeric_pool = _numeric_pool(blueprints)
    set_counter: dict[str, int] = {}
    skipped: dict[str, int] = {}
    draft_sets: list[StatementDraftSet] = []
    extraction_stats = {"extracted": 0, "rejected": 0, "errors": 0}

    figures_by_id = {figure.figure_id: figure for figure in semantics.figures}
    for figure in semantics.figures:
        selected = _select_blueprints(figure.figure_id, blueprints, config)
        for blueprint in selected:
            context = _context_for(blueprint, figure, figures_by_id, numeric_pool)
            object_phrase, extraction_note = _object_for(
                blueprint, figure, object_extractor, extraction_stats
            )
            draft_set = _build_set(
                blueprint,
                context,
                set_counter,
                config.max_perturbations_per_set,
                object_phrase=object_phrase,
                extraction_note=extraction_note,
            )
            if draft_set is None:
                skipped["no_perturbation_applied"] = skipped.get("no_perturbation_applied", 0) + 1
                continue
            draft_sets.append(draft_set)

    statement_count = sum(len(draft_set.statements) for draft_set in draft_sets)
    by_perturbation: dict[str, int] = {}
    for draft_set in draft_sets:
        for statement in draft_set.statements:
            if not statement.is_correct:
                by_perturbation[statement.perturbation_type] = (
                    by_perturbation.get(statement.perturbation_type, 0) + 1
                )

    report = QuestionDraftReport(
        doc_id=doc_id,
        summary={
            "sets": len(draft_sets),
            "statements": statement_count,
            "true_statements": len(draft_sets),
            "false_statements": statement_count - len(draft_sets),
            "by_perturbation": dict(sorted(by_perturbation.items())),
            "skipped": dict(sorted(skipped.items())),
            "method": "deterministic+llm"
            if object_extractor is not None and extraction_stats["extracted"]
            else "deterministic",
            **({"object_extraction": dict(extraction_stats)} if object_extractor is not None else {}),
        },
        draft_sets=draft_sets,
    )
    if persist:
        persist_drafts(report, artifacts_root)
    return report


# Pool hygiene: upstream extraction occasionally yields noisy surface forms
# ("treated with the X", "than in X-treated", "not upregulate CCR7 expression").
# Perturbation material must be clean names — skip noisy candidates rather than
# generate ungrammatical drafts from them.
_NOISY_TREATMENT_PREFIXES = ("treated with", "than in")
_NOISY_TREATMENT_MARKERS = ("versus", " vs")


def _is_clean_treatment(name: str) -> bool:
    lowered = name.lower()
    return not lowered.startswith(_NOISY_TREATMENT_PREFIXES) and not any(
        marker in lowered for marker in _NOISY_TREATMENT_MARKERS
    )


def _is_clean_endpoint(name: str) -> bool:
    return " not " not in f" {name.lower()} " and not name.lower().startswith(("not ", "showing "))


def _select_blueprints(
    figure_id: str, blueprints: QuestionBlueprintReport, config: DraftConfig
) -> list[QuestionBlueprint]:
    """First blueprint per question type (fixed priority), capped per figure."""
    by_type: dict[str, QuestionBlueprint] = {}
    for blueprint in blueprints.blueprints:
        if blueprint.figure_id != figure_id or blueprint.panel_ids:
            continue  # figure-level blueprints only; panel results surface via their figure's RI/DATA sets
        if blueprint.question_type not in by_type:
            by_type[blueprint.question_type] = blueprint
    selected = [by_type[t] for t in _TYPE_PRIORITY if t in by_type]
    return selected[: config.max_sets_per_figure]


def _numeric_pool(blueprints: QuestionBlueprintReport) -> dict[str, list[tuple[str, str]]]:
    """Paper-wide literal value pool by kind, deduplicated (first occurrence)."""
    pool: dict[str, list[tuple[str, str]]] = {}
    for blueprint in blueprints.blueprints:
        if blueprint.question_type != "DATA_STATEMENT":
            continue
        kind = blueprint.detail.get("kind")
        value = blueprint.detail.get("data_value")
        evidence_id = blueprint.evidence_ids[0] if blueprint.evidence_ids else ""
        if not kind or not value:
            continue
        bucket = pool.setdefault(kind, [])
        if all(existing != value for existing, _eid in bucket):
            bucket.append((value, evidence_id))
    return pool


def _context_for(
    blueprint: QuestionBlueprint,
    figure: FigureSemantic,
    figures_by_id: dict[str, FigureSemantic],
    numeric_pool: dict[str, list[tuple[str, str]]],
) -> PerturbationContext:
    detail = blueprint.detail
    comparison = detail.get("comparison", {}) or {}
    experiment = figure.experiment

    treatments: list[str] = []
    if experiment and experiment.intervention:
        for evidence in figure.evidence:
            for match in TREATMENT_SPAN_RE.finditer(evidence.text):
                name = match.group(0).strip()
                if not _is_clean_treatment(name):
                    continue
                if name.lower() not in [t.lower() for t in treatments]:
                    treatments.append(name)

    intervention = comparison.get("experimental") or (experiment.intervention if experiment else "")
    other_treatments = [
        t for t in treatments if t.lower() != (intervention or "").lower() and _is_clean_treatment(t)
    ]

    other_endpoints: list[str] = []
    if experiment and blueprint.question_type != "DATA_STATEMENT":
        other_endpoints = [
            dv
            for dv in experiment.dependent_variables[1:]
            if dv.lower() != (comparison.get("endpoint") or "").lower() and _is_clean_endpoint(dv)
        ]

    if blueprint.panel_ids:
        siblings = [p.panel_id for p in figure.panels if p.panel_id != blueprint.panel_ids[0]]
        if not siblings:
            siblings = [other.figure_id for other in figures_by_id.values() if other.figure_id != figure.figure_id]
    else:
        siblings = [other.figure_id for other in figures_by_id.values() if other.figure_id != figure.figure_id]

    condition_findings = [
        (finding.sentence, finding.value, finding.evidence_id)
        for finding in extract_numeric_findings(figure.evidence)
        if finding.kind in ("concentration", "time")
    ]

    return PerturbationContext(
        intervention=intervention or "",
        dv=comparison.get("endpoint", "") or (experiment.dependent_variables[0] if experiment and experiment.dependent_variables else ""),
        experimental_group=comparison.get("experimental", "") or (experiment.experimental_groups[0] if experiment and experiment.experimental_groups else ""),
        control_group=comparison.get("control", "") or (experiment.control_groups[0] if experiment and experiment.control_groups else ""),
        relationship_type=detail.get("relationship_type", "unspecified"),
        direction=detail.get("direction", "unspecified"),
        evidence_ids=list(blueprint.evidence_ids),
        other_treatments=other_treatments[:3],
        other_endpoints=other_endpoints[:2],
        sibling_labels=siblings[:3],
        numeric_pool=numeric_pool,
        condition_findings=condition_findings[:4],
    )


def _object_for(
    blueprint: QuestionBlueprint,
    figure: FigureSemantic,
    extractor: LlmObjectExtractor | None,
    stats: dict[str, int],
) -> tuple[str | None, str | None]:
    """Optional LLM object extraction — only for DATA fallbacks, never for quotable ones."""
    if extractor is None or not needs_object_extraction(blueprint):
        return None, None
    texts = [blueprint.detail.get("sentence", "")] + [
        evidence.text for evidence in figure.evidence if evidence.evidence_id in blueprint.evidence_ids
    ]
    try:
        phrase = extractor.extract(
            value=blueprint.detail.get("data_value", ""),
            kind=blueprint.detail.get("kind", ""),
            texts=texts,
        )
    except LlmObjectExtractionError:
        stats["errors"] += 1
        return None, "llm_error"
    if phrase is None:
        stats["rejected"] += 1
        return None, "llm_rejected"
    stats["extracted"] += 1
    return phrase, None


def _build_set(
    blueprint: QuestionBlueprint,
    context: PerturbationContext,
    set_counter: dict[str, int],
    max_perturbations: int,
    object_phrase: str | None = None,
    extraction_note: str | None = None,
) -> StatementDraftSet | None:
    true_statement = build_true_statement(blueprint, object_phrase=object_phrase)
    if not true_statement or not blueprint.evidence_ids:
        return None

    unit = blueprint.experiment_id.removeprefix("exp_") or blueprint.figure_id.replace(" ", "")
    set_counter[unit] = set_counter.get(unit, 0) + 1
    set_id = f"qd_{unit}_{set_counter[unit]:03d}"

    true_detail = {"source": "blueprint_expected_answer"}
    if object_phrase:
        true_detail.update(object=object_phrase, object_extraction="llm")
    elif extraction_note:
        true_detail["object_extraction"] = extraction_note

    statements = [
        StatementDraft(
            draft_id=f"{set_id}_00",
            blueprint_id=blueprint.blueprint_id,
            figure_id=blueprint.figure_id,
            panel_ids=list(blueprint.panel_ids),
            statement=true_statement,
            is_correct=True,
            perturbation_type="NONE",
            evidence_ids=list(blueprint.evidence_ids),
            confidence=blueprint.confidence,
            detail=true_detail,
        )
    ]
    seen = {true_statement}
    for name in PERTURBATION_ORDER:
        if len(statements) >= 1 + max_perturbations:
            break
        perturber = PERTURBERS[name]
        result, extra_evidence = perturber(true_statement, context, blueprint)
        if result is None or result in seen:
            continue
        seen.add(result)
        statements.append(
            StatementDraft(
                draft_id=f"{set_id}_{len(statements):02d}",
                blueprint_id=blueprint.blueprint_id,
                figure_id=blueprint.figure_id,
                panel_ids=list(blueprint.panel_ids),
                statement=result,
                is_correct=False,
                perturbation_type=name,  # type: ignore[arg-type]
                evidence_ids=_merged(blueprint.evidence_ids, extra_evidence),
                confidence=blueprint.confidence,
                detail={"base": true_statement},
            )
        )
    if len(statements) < 2:
        return None  # a correct/incorrect question needs at least one false statement
    return StatementDraftSet(
        draft_set_id=set_id,
        blueprint_id=blueprint.blueprint_id,
        figure_id=blueprint.figure_id,
        question_type=blueprint.question_type,
        panel_ids=list(blueprint.panel_ids),
        statements=statements,
    )


def _merged(base: list[str], extra: list[str]) -> list[str]:
    merged = list(base)
    for evidence_id in extra:
        if evidence_id and evidence_id not in merged:
            merged.append(evidence_id)
    return merged


def persist_drafts(report: QuestionDraftReport, artifacts_root: str | Path = "data") -> Path:
    """Write question_drafts.json (deterministic content, no timestamps)."""
    out_dir = Path(artifacts_root) / "paper_semantics" / report.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": report.doc_id,
        "summary": report.summary,
        "draft_sets": [draft_set.model_dump() for draft_set in report.draft_sets],
    }
    path = out_dir / "question_drafts.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
