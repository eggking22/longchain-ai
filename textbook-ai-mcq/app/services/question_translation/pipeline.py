"""Pipeline: Statement Drafts → Chinese MCQ statements → mcq_drafts_zh.json.

Reads question_drafts.json (read-only — the English artifact is never
rewritten), translates every statement with the deterministic registry /
template translator, and writes mcq_drafts_zh.json next to it. Numbers,
units, gene names, TRUE/FALSE flags and evidence_ids are carried over
untouched; only the language changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.schemas.mcq_zh import MCQDraftReportZh, MCQDraftSetZh, MCQStatementZh
from app.schemas.question_draft import QuestionDraftReport

from .llm_translator import LlmStatementTranslator, LlmTranslationError
from .translator import translate_statement


def _translate_one(
    text: str, is_data_set: bool, translator: LlmStatementTranslator | None
) -> tuple[str, str]:
    """LLM first (invariant-gated, whole sentence); deterministic result as fallback."""
    if translator is not None:
        try:
            chinese = translator.translate(text)
        except LlmTranslationError:
            chinese = None
        if chinese is not None:
            return chinese, "llm"
    return translate_statement(text, data_statement=is_data_set)


def translate_drafts(
    report: QuestionDraftReport, translator: LlmStatementTranslator | None = None
) -> MCQDraftReportZh:
    """Translate a draft report into its Chinese projection.

    translator (optional LLM layer) retranslates every statement as a whole
    sentence behind the deterministic invariant gate; per-statement failure or
    rejection falls back to the deterministic registry/template result.
    """
    draft_sets: list[MCQDraftSetZh] = []
    method_counts: dict[str, int] = {"template": 0, "term_fallback": 0}
    if translator is not None:
        method_counts["llm"] = 0
    for draft_set in report.draft_sets:
        is_data_set = draft_set.question_type == "DATA_STATEMENT"
        statements: list[MCQStatementZh] = []
        for statement in draft_set.statements:
            chinese, method = _translate_one(statement.statement, is_data_set, translator)
            method_counts[method] += 1
            detail = dict(statement.detail)
            detail["translation_method"] = method
            statements.append(
                MCQStatementZh(
                    draft_id=statement.draft_id,
                    blueprint_id=statement.blueprint_id,
                    figure_id=statement.figure_id,
                    panel_ids=list(statement.panel_ids),
                    statement=statement.statement,
                    statement_zh=chinese,
                    is_correct=statement.is_correct,
                    perturbation_type=statement.perturbation_type,
                    evidence_ids=list(statement.evidence_ids),
                    confidence=statement.confidence,
                    detail=detail,
                    translation_method=method,
                )
            )
        draft_sets.append(
            MCQDraftSetZh(
                draft_set_id=draft_set.draft_set_id,
                blueprint_id=draft_set.blueprint_id,
                figure_id=draft_set.figure_id,
                question_type=draft_set.question_type,
                panel_ids=list(draft_set.panel_ids),
                statements=statements,
            )
        )
    return MCQDraftReportZh(
        doc_id=report.doc_id,
        summary={
            **report.summary,
            "translation": {
                "method": "deterministic-registry+llm" if translator is not None else "deterministic-registry",
                "counts": method_counts,
            },
        },
        draft_sets=draft_sets,
    )


def load_drafts(doc_id: str, artifacts_root: str | Path = "data") -> QuestionDraftReport:
    """Read question_drafts.json (read-only)."""
    path = Path(artifacts_root) / "paper_semantics" / doc_id / "question_drafts.json"
    if not path.exists():
        raise FileNotFoundError(
            f"question_drafts.json not found for doc_id={doc_id!r} under {path.parent} — "
            "run scripts/generate_question_drafts.py first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return QuestionDraftReport.model_validate(
        {
            "doc_id": payload["doc_id"],
            "summary": payload["summary"],
            "draft_sets": payload["draft_sets"],
        }
    )


def translate_document(
    doc_id: str,
    artifacts_root: str | Path = "data",
    persist: bool = True,
    translator: LlmStatementTranslator | None = None,
) -> MCQDraftReportZh:
    """Load the English drafts artifact and produce (and optionally persist) the Chinese one."""
    report = translate_drafts(load_drafts(doc_id, artifacts_root), translator=translator)
    if persist:
        persist_mcq_zh(report, artifacts_root)
    return report


def persist_mcq_zh(report: MCQDraftReportZh, artifacts_root: str | Path = "data") -> Path:
    """Write mcq_drafts_zh.json (deterministic content, no timestamps)."""
    out_dir = Path(artifacts_root) / "paper_semantics" / report.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "doc_id": report.doc_id,
        "summary": report.summary,
        "draft_sets": [draft_set.model_dump() for draft_set in report.draft_sets],
    }
    path = out_dir / "mcq_drafts_zh.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
