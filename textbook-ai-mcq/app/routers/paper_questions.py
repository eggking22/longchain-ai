"""API for the Paper Question Review page: question sets grouped by figure.

Read-only projection over the existing artifacts (figures.json /
question_blueprints.json / question_drafts.json / evidence.jsonl); the Chinese
layer is translated on the fly (deterministic) unless mcq_drafts_zh.json
already exists. No existing endpoint or artifact is modified.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from app.core.config import get_settings
from app.services.question_translation import translate_document

router = APIRouter(tags=["paper-questions"])


def _artifacts_root() -> Path:
    return Path(get_settings().ARTIFACTS_DIR)


def _read_json(doc_id: str, filename: str) -> dict:
    path = _artifacts_root() / "paper_semantics" / doc_id / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found for doc_id={doc_id!r} — run the generation pipeline first",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_store(doc_id: str) -> dict[str, dict]:
    path = _artifacts_root() / "paper_semantics" / doc_id / "evidence.jsonl"
    if not path.exists():
        return {}
    store: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                unit = json.loads(line)
                store[unit.get("evidence_id", "")] = unit
    return store


def _figure_summaries(doc_id: str) -> dict[str, dict]:
    document = _read_json(doc_id, "figures.json")
    summaries: dict[str, dict] = {}
    for figure in document.get("figures", []):
        experiment = {
            key: figure.get(key)
            for key in ("research_question", "groups", "variables", "conclusions")
            if figure.get(key)
        }
        summaries[figure["figure_id"]] = {
            "figure_id": figure["figure_id"],
            "kind": figure.get("kind"),
            "status": figure.get("status"),
            "confidence": figure.get("confidence"),
            "title": figure.get("title"),
            "caption": figure.get("caption"),
            "experiment": experiment,
            "panels": [
                {"panel_id": panel.get("panel_id"), "status": panel.get("status"), "title": panel.get("title")}
                for panel in figure.get("panels", [])
            ],
        }
    return summaries


def _blueprint_map(doc_id: str) -> dict[str, dict]:
    """Blueprint context for each set; empty map when the artifact is absent
    (sets still render, just without focus/expected-answer enrichment)."""
    path = _artifacts_root() / "paper_semantics" / doc_id / "question_blueprints.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        blueprint["blueprint_id"]: {
            "blueprint_id": blueprint["blueprint_id"],
            "question_type": blueprint["question_type"],
            "question_focus": blueprint["question_focus"],
            "expected_answer": blueprint["expected_answer"],
            "reasoning_operation": blueprint["reasoning_operation"],
            "required_evidence": blueprint["required_evidence"],
        }
        for blueprint in payload.get("blueprints", [])
    }


def _translate_or_404(doc_id: str):
    try:
        return translate_document(doc_id, _artifacts_root(), persist=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/paper-questions/{doc_id}")
def list_question_sets(doc_id: str, figure_id: str | None = Query(default=None)):
    """Question sets of one paper, grouped by figure (optionally filtered)."""
    zh = _translate_or_404(doc_id)
    figures = _figure_summaries(doc_id)
    blueprints = _blueprint_map(doc_id)
    evidence = _evidence_store(doc_id)

    wanted = None
    if figure_id:
        wanted = figure_id.strip()
        normalized = wanted.lower().replace("fig.", "figure").replace("fig ", "figure ")
        wanted = next(
            (name for name in figures if normalized in name.lower() or name.lower() in normalized),
            wanted,
        )

    grouped: dict[str, dict] = {}
    for draft_set in zh.draft_sets:
        if wanted and draft_set.figure_id != wanted:
            continue
        entry = grouped.setdefault(
            draft_set.figure_id,
            {**figures.get(draft_set.figure_id, {"figure_id": draft_set.figure_id}), "question_sets": []},
        )
        statements = []
        for index, statement in enumerate(draft_set.statements):
            statements.append(
                {
                    "draft_id": statement.draft_id,
                    "label": chr(ord("A") + index),
                    "statement": statement.statement,
                    "statement_zh": statement.statement_zh,
                    "is_correct": statement.is_correct,
                    "perturbation_type": statement.perturbation_type,
                    "evidence_ids": statement.evidence_ids,
                    "confidence": statement.confidence,
                    "evidence_previews": [
                        {
                            "evidence_id": evidence_id,
                            "text": (evidence.get(evidence_id, {}).get("text", "") or "")[:120],
                        }
                        for evidence_id in statement.evidence_ids
                    ],
                }
            )
        entry["question_sets"].append(
            {
                "draft_set_id": draft_set.draft_set_id,
                "question_type": draft_set.question_type,
                "blueprint": blueprints.get(draft_set.blueprint_id, {"blueprint_id": draft_set.blueprint_id}),
                "statements": statements,
            }
        )

    figure_list = [grouped[name] for name in figures if name in grouped] + [
        grouped[name] for name in grouped if name not in figures
    ]
    if wanted and not figure_list:
        raise HTTPException(status_code=404, detail=f"no question sets for figure_id={figure_id!r}")
    return {
        "doc_id": doc_id,
        "summary": zh.summary,
        "figures": figure_list,
    }


@router.get("/paper-questions/{doc_id}/{set_id}")
def get_question_set(doc_id: str, set_id: str):
    """One question set with full evidence texts for review."""
    zh = _translate_or_404(doc_id)
    blueprints = _blueprint_map(doc_id)
    evidence = _evidence_store(doc_id)

    for draft_set in zh.draft_sets:
        if draft_set.draft_set_id != set_id:
            continue
        statements = []
        for index, statement in enumerate(draft_set.statements):
            statements.append(
                {
                    "draft_id": statement.draft_id,
                    "label": chr(ord("A") + index),
                    "statement": statement.statement,
                    "statement_zh": statement.statement_zh,
                    "is_correct": statement.is_correct,
                    "perturbation_type": statement.perturbation_type,
                    "evidence_ids": statement.evidence_ids,
                    "confidence": statement.confidence,
                    "translation_method": statement.translation_method,
                    "evidence": [
                        {
                            "evidence_id": evidence_id,
                            **{
                                key: value
                                for key, value in evidence.get(evidence_id, {}).items()
                                if key in ("text", "role", "evidence_type", "section_type", "page_no", "paragraph_id")
                            },
                        }
                        for evidence_id in statement.evidence_ids
                    ],
                }
            )
        return {
            "doc_id": doc_id,
            "draft_set_id": draft_set.draft_set_id,
            "figure_id": draft_set.figure_id,
            "question_type": draft_set.question_type,
            "panel_ids": draft_set.panel_ids,
            "blueprint": blueprints.get(draft_set.blueprint_id, {"blueprint_id": draft_set.blueprint_id}),
            "statements": statements,
        }
    raise HTTPException(status_code=404, detail=f"question set {set_id!r} not found")
