"""CLI entry for Chinese MCQ statement translation.

Usage (from the project root, with the bio-ai env active or via its python):

    python scripts/generate_mcq_zh.py --doc-id paper_1 [--figure 2] [--json]

Deterministic registry + template translation: English statements are kept
verbatim, statement_zh is added, and numbers/genes/direction/TRUE-FALSE/
evidence_ids are carried over untouched. Optional LLM whole-sentence
translation activates automatically when LLM_API_KEY and LLM_MODEL are
configured in .env (and --no-llm is not passed); every LLM result passes the
deterministic invariant gate (numbers/anchors/direction/gene names) or the
statement falls back to the deterministic translation. The artifact lands in
data/paper_semantics/{doc_id}/mcq_drafts_zh.json; question_drafts.json is
never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.question_translation import translate_document  # noqa: E402


def _print_human(report) -> None:
    print(f"doc: {report.doc_id}  sets: {len(report.draft_sets)}  "
          f"translation: {json.dumps(report.summary.get('translation', {}))}")
    print("=" * 78)
    for draft_set in report.draft_sets:
        print(f"{draft_set.draft_set_id}  {draft_set.question_type:22s} {draft_set.figure_id}")
        for statement in draft_set.statements:
            marker = "✔ TRUE " if statement.is_correct else f"✘ {statement.perturbation_type}"
            print(f"  [{statement.draft_id}] {marker}")
            print(f"      zh: {statement.statement_zh}")
            print(f"      en: {statement.statement}")
        print("-" * 78)


def _figure_matches(figure_id: str, query: str) -> bool:
    lowered = figure_id.lower()
    query = query.strip().lower().replace("fig.", "figure").replace("fig ", "figure ")
    if not query:
        return True
    return query in lowered or lowered in query


def main() -> int:
    parser = argparse.ArgumentParser(description="Translate statement drafts into Chinese MCQ statements")
    parser.add_argument("--doc-id", required=True, help="document id with existing question_drafts.json")
    parser.add_argument("--figure", default="", help="restrict output to one figure/table")
    parser.add_argument("--artifacts-root", default=None, help="artifacts root (default: settings.ARTIFACTS_DIR)")
    parser.add_argument("--no-llm", action="store_true",
                        help="skip optional LLM whole-sentence translation even when LLM_API_KEY/LLM_MODEL are configured")
    parser.add_argument("--json", action="store_true", help="print the full MCQDraftReportZh as JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from app.core.config import get_settings

    settings = get_settings()
    artifacts = args.artifacts_root or settings.ARTIFACTS_DIR

    translator = None
    if not args.no_llm and settings.LLM_API_KEY and settings.LLM_MODEL:
        from app.services.question_translation.llm_translator import (
            DEFAULT_LLM_BASE_URL,
            LlmStatementTranslator,
        )

        translator = LlmStatementTranslator(
            base_url=settings.LLM_BASE_URL or DEFAULT_LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
        )

    try:
        report = translate_document(args.doc_id, artifacts, translator=translator)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.figure:
        report.draft_sets = [s for s in report.draft_sets if _figure_matches(s.figure_id, args.figure)]
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_human(report)
    artifact = Path(artifacts) / "paper_semantics" / args.doc_id / "mcq_drafts_zh.json"
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
