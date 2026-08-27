"""CLI entry for Question Blueprint generation (Phase 5).

Usage (from the project root, with the bio-ai env active or via its python):

    python scripts/generate_blueprints.py --doc-id paper_1
    python scripts/generate_blueprints.py --doc-id paper_1 --type DATA_STATEMENT --json
    python scripts/generate_blueprints.py --doc-id paper_1 --figure 2

Deterministic only — no LLM, no MCQ. Blueprints are question *plans* bound to
the Evidence Store; the artifact lands in data/paper_semantics/{doc_id}/
question_blueprints.json without touching any existing artifact file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.question_blueprint import BlueprintConfig, generate_blueprints  # noqa: E402

QUESTION_TYPES = ("RESULT_INTERPRETATION", "EXPERIMENTAL_DESIGN", "SIMPLE_PREDICTION", "DATA_STATEMENT")


def _print_human(report) -> None:
    print(f"doc: {report.doc_id}  blueprints: {report.summary['total']}  "
          f"by_type: {json.dumps(report.summary['by_type'])}")
    print("=" * 78)
    for blueprint in report.blueprints:
        panels = f"[{', '.join(blueprint.panel_ids)}]" if blueprint.panel_ids else ""
        print(f"{blueprint.question_type:22s} {blueprint.figure_id}{panels}  ({blueprint.blueprint_id})")
        print(f"  focus:    {blueprint.question_focus}")
        print(f"  answer:   {blueprint.expected_answer}")
        print(f"  evidence: {', '.join(blueprint.evidence_ids)}")
        print("-" * 78)
    if report.summary.get("skipped"):
        print("skipped (why no blueprint was generated):")
        for question_type, reasons in report.summary["skipped"].items():
            print(f"  {question_type}: {json.dumps(reasons)}")


def _figure_matches(figure_id: str, query: str) -> bool:
    lowered = figure_id.lower()
    query = query.strip().lower().replace("fig.", "figure").replace("fig ", "figure ")
    if not query:
        return True
    return query in lowered or lowered in query


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic question blueprints from paper semantics")
    parser.add_argument("--doc-id", required=True, help="document id parsed by Phase 1 (see data/structure)")
    parser.add_argument("--figure", default="", help="restrict to one figure/table, e.g. 2 or 'Figure 2'")
    parser.add_argument("--type", choices=QUESTION_TYPES, default="", help="restrict to one question type")
    parser.add_argument("--artifacts-root", default=None, help="artifacts root (default: settings.ARTIFACTS_DIR)")
    parser.add_argument("--json", action="store_true", help="print the full QuestionBlueprintReport as JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from app.core.config import get_settings

    artifacts = args.artifacts_root or get_settings().ARTIFACTS_DIR

    try:
        report = generate_blueprints(args.doc_id, artifacts, config=BlueprintConfig())
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.figure:
        report.blueprints = [b for b in report.blueprints if _figure_matches(b.figure_id, args.figure)]
    if args.type:
        report.blueprints = [b for b in report.blueprints if b.question_type == args.type]
    report.summary = {
        **report.summary,
        "total": len(report.blueprints),
        "by_type": {t: sum(1 for b in report.blueprints if b.question_type == t) for t in QUESTION_TYPES},
    }

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_human(report)
    artifact = Path(artifacts) / "paper_semantics" / args.doc_id / "question_blueprints.json"
    print(f"artifact: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
