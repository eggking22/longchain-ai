"""CLI entry for Scientific Paper Figure Semantic Reconstruction.

Usage (from the project root, with the bio-ai env active or via its python):

    python scripts/reconstruct_figures.py --doc-id example-paper
    python scripts/reconstruct_figures.py --doc-id example-paper --figure 2
    python scripts/reconstruct_figures.py --doc-id example-paper --json

LLM normalization activates automatically when LLM_API_KEY and LLM_MODEL are
configured in .env (and --no-llm is not passed); otherwise the deterministic
result stands. Exit codes: 0 = reconstruction completed (regardless of
per-figure SUFFICIENT/PARTIAL/INSUFFICIENT statuses), 1 = error (e.g. the
document has not been parsed yet).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.paper_semantics import (  # noqa: E402
    DEFAULT_LLM_BASE_URL,
    FigureAwareRetriever,
    LlmSemanticNormalizer,
    PaperSemanticsConfig,
    reconstruct_figures,
)


def _print_human(report) -> None:
    print(f"doc: {report.doc_id}  figures: {report.num_figures}  "
          f"status: {json.dumps(report.stats.get('status_counts', {}))}  "
          f"panels: {report.stats.get('num_panels', 0)} {json.dumps(report.stats.get('panel_status_counts', {}))}")
    print("=" * 78)
    for figure in report.figures:
        print(figure.figure_id)
        print(f"  status: {figure.reconstruction_status}  confidence={figure.confidence}  method={figure.method}")
        experiment = figure.experiment
        if experiment is not None and figure.reconstruction_status != "INSUFFICIENT":
            if experiment.research_question:
                print(f"  experiment: {experiment.research_question}")
            if experiment.experimental_groups or experiment.control_groups:
                groups = " vs ".join(
                    [", ".join(experiment.experimental_groups) or "?", ", ".join(experiment.control_groups) or "?"]
                )
                print(f"  groups: {groups}")
            for observation in experiment.observations:
                print(f"  observation: {observation.statement}")
                print(f"      direction={observation.direction}  significance={observation.significance}  "
                      f"relationship={observation.relationship_type}"
                      + (f"  p={observation.p_value}" if observation.p_value else ""))
            for interpretation in experiment.interpretations:
                cites = ", ".join(interpretation.evidence_ids) or "-"
                print(f"  interpretation: {interpretation.statement}  [{cites}]")
            for conclusion in experiment.conclusions:
                cites = ", ".join(conclusion.evidence_ids) or "-"
                linked = f"  ← {', '.join(conclusion.interpretation_ids)}" if conclusion.interpretation_ids else ""
                print(f"  conclusion: {conclusion.statement}  [{cites}]{linked}")
        if figure.reconstruction_status != "SUFFICIENT" and figure.missing_information:
            print("  missing:")
            for item in figure.missing_information:
                print(f"    - {item}")
        types = {}
        for evidence in figure.evidence:
            types[evidence.evidence_type] = types.get(evidence.evidence_type, 0) + 1
        print(f"  evidence: {len(figure.evidence)} item(s)  types: {json.dumps(types)}")
        for panel in figure.panels:
            print(f"  panel {panel.panel_id}: {panel.reconstruction_status}  confidence={panel.confidence}  "
                  f"evidence={len(panel.evidence)}")
            if panel.reconstruction_status != "SUFFICIENT" and panel.missing_information:
                print(f"      missing: {panel.missing_information}")
        print("-" * 78)


def _figure_matches(figure_id: str, query: str) -> bool:
    lowered = figure_id.lower()  # "figure 2" / "table 1"
    query = query.strip().lower().replace("fig.", "figure").replace("fig ", "figure ")
    if not query:
        return True
    return query in lowered or lowered in query


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct figure/table experiment semantics from paper text")
    parser.add_argument("--doc-id", required=True, help="document id parsed by Phase 1 (see data/structure)")
    parser.add_argument("--figure", default="", help="restrict output to one figure/table, e.g. 2 or 'Figure 2'")
    parser.add_argument("--no-llm", action="store_true", help="skip LLM normalization even if configured")
    parser.add_argument("--no-retrieval", action="store_true", help="skip the figure-aware retrieval supplement")
    parser.add_argument("--artifacts-root", default=None, help="artifacts root (default: settings.ARTIFACTS_DIR)")
    parser.add_argument("--json", action="store_true", help="print the full PaperSemanticsReport as JSON")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    artifacts = args.artifacts_root or settings.ARTIFACTS_DIR
    config = PaperSemanticsConfig.from_settings(settings)

    normalizer = None
    if not args.no_llm and settings.LLM_API_KEY and settings.LLM_MODEL:
        normalizer = LlmSemanticNormalizer(
            base_url=settings.LLM_BASE_URL or DEFAULT_LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            config=config,
        )

    retriever = None
    if not args.no_retrieval:
        retriever = FigureAwareRetriever.try_load(args.doc_id, artifacts, config)

    try:
        report = reconstruct_figures(
            args.doc_id, artifacts, config=config, normalizer=normalizer, retriever=retriever
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.figure:
        report.figures = [f for f in report.figures if _figure_matches(f.figure_id, args.figure)]
        report.num_figures = len(report.figures)

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_human(report)
    out_dir = Path(artifacts) / "paper_semantics" / args.doc_id
    for name in ("figures.json", "evidence.jsonl", "experiments.json", "report.md", "manifest.json"):
        if (out_dir / name).exists():
            print(f"artifact: {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
