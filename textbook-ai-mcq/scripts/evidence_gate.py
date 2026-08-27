"""CLI entry for the evidence gate: query → retrieval → sufficiency verdict.

Usage (from the project root, with the bio-ai env active or via its python):

    python scripts/evidence_gate.py "糖酵解的发生部位是哪里？" --doc-id physiology
    python scripts/evidence_gate.py "诺贝尔奖的影响" --doc-id physiology --json

Level 2 (LLM judge) activates automatically when LLM_API_KEY and LLM_MODEL
are configured in .env; otherwise the heuristic Level 1 decides alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.evidence import build_evidence_gate  # noqa: E402
from app.services.retrieval import RetrievalConfig, RetrievalEngine  # noqa: E402


def _print_human(report) -> None:
    verdict = "SUFFICIENT ✔" if report.sufficient else "INSUFFICIENT_EVIDENCE ✘"
    print(f"query: {report.query}")
    print(f"verdict: {verdict}  coverage={report.coverage_score} (threshold={report.threshold}, "
          f"level={report.level})")
    if report.missing_information:
        print(f"missing_information: {report.missing_information}")
    print("-" * 78)
    for item in report.evidence:
        crumb = " / ".join(item.breadcrumb[:2])
        pages = f"p{item.pages[0]}" + (f"-{item.pages[-1]}" if len(item.pages) > 1 else "")
        print(f"  [{item.rank}] {item.chunk_id}  score={item.score:.5f}  {crumb} ({pages})")
        print(f"      {item.text_preview}")
    detail = {k: v for k, v in report.detail.items() if k != "config"}
    print(f"detail: {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate retrieved evidence for sufficiency")
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("--doc-id", required=True, help="document id whose index to search")
    parser.add_argument("--top-k", type=int, default=10,
                        help="hits retrieved for the gate pool (default 10; gate cites 5)")
    parser.add_argument("--artifacts-root", default=None, help="artifacts root (default: settings.ARTIFACTS_DIR)")
    parser.add_argument("--json", action="store_true", help="print the full CoverageReport as JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    artifacts = args.artifacts_root or settings.ARTIFACTS_DIR
    try:
        engine = RetrievalEngine.load(args.doc_id, artifacts, config=RetrievalConfig.from_settings(settings))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    gate = build_evidence_gate(settings)
    result = engine.retrieve(args.query, mode="hybrid", top_k=args.top_k)
    report = gate.evaluate(args.query, result)

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_human(report)
    return 0 if report.sufficient else 2


if __name__ == "__main__":
    raise SystemExit(main())
