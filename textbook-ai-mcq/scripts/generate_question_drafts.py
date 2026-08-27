"""CLI entry for Statement Draft generation (MCQ step 1).

Usage (from the project root, with the bio-ai env active or via its python):

    python scripts/generate_question_drafts.py --doc-id paper_1
    python scripts/generate_question_drafts.py --doc-id paper_1 --figure 2 --json

Deterministic only: one TRUE statement per set plus controlled minimal-edit
false statements (material from the paper's own evidence). No Chinese text,
no A/B/C/D layout, no LLM. The artifact lands in
data/paper_semantics/{doc_id}/question_drafts.json without touching any
existing artifact file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.question_generation import DraftConfig, generate_question_drafts  # noqa: E402


def _print_human(report) -> None:
    print(f"doc: {report.doc_id}  draft sets: {report.summary['sets']}  "
          f"statements: {report.summary['statements']} "
          f"(true {report.summary['true_statements']} / false {report.summary['false_statements']})")
    print(f"by_perturbation: {json.dumps(report.summary['by_perturbation'])}")
    print("=" * 78)
    for draft_set in report.draft_sets:
        panels = f"[{', '.join(draft_set.panel_ids)}]" if draft_set.panel_ids else ""
        print(f"{draft_set.draft_set_id}  {draft_set.question_type:22s} {draft_set.figure_id}{panels}")
        for statement in draft_set.statements:
            marker = "✔ TRUE " if statement.is_correct else f"✘ {statement.perturbation_type}"
            print(f"  [{statement.draft_id}] {marker}")
            print(f"      {statement.statement}")
            print(f"      evidence: {', '.join(statement.evidence_ids)}")
        print("-" * 78)


def _figure_matches(figure_id: str, query: str) -> bool:
    lowered = figure_id.lower()
    query = query.strip().lower().replace("fig.", "figure").replace("fig ", "figure ")
    if not query:
        return True
    return query in lowered or lowered in query


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic statement drafts from question blueprints")
    parser.add_argument("--doc-id", required=True, help="document id parsed by Phase 1 (see data/structure)")
    parser.add_argument("--figure", default="", help="restrict to one figure/table, e.g. 2 or 'Figure 2'")
    parser.add_argument("--artifacts-root", default=None, help="artifacts root (default: settings.ARTIFACTS_DIR)")
    parser.add_argument("--json", action="store_true", help="print the full QuestionDraftReport as JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from app.core.config import get_settings

    artifacts = args.artifacts_root or get_settings().ARTIFACTS_DIR

    try:
        report = generate_question_drafts(args.doc_id, artifacts, config=DraftConfig())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.figure:
        report.draft_sets = [s for s in report.draft_sets if _figure_matches(s.figure_id, args.figure)]
    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_human(report)
    artifact = Path(artifacts) / "paper_semantics" / args.doc_id / "question_drafts.json"
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
