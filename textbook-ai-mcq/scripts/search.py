"""CLI entry for hybrid retrieval over a built index.

Usage (from the project root, with the bio-ai env active or via its python):

    python scripts/search.py "心动周期的概念" --doc-id physiology
    python scripts/search.py "细胞膜" --doc-id physiology --mode sparse --top-k 3 --json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.services.retrieval import MODES, RetrievalConfig, RetrievalEngine  # noqa: E402


def _print_human(result) -> None:
    print(f"query={result.query!r}  mode={result.mode}  top_k={result.top_k}  "
          f"latency={result.latency_ms}ms  doc={result.manifest.doc_id} "
          f"({result.manifest.embedding_model}, dim={result.manifest.embedding_dim}, "
          f"{result.manifest.num_chunks} chunks)")
    print("-" * 78)
    for hit in result.hits:
        crumb = " / ".join(hit.breadcrumb[:2])
        pages = f"p{hit.pages[0]}" + (f"-{hit.pages[-1]}" if len(hit.pages) > 1 else "")
        scores = (
            f"dense={hit.dense_score:.4f} " if hit.dense_score is not None else "dense=—     "
        ) + (f"sparse={hit.sparse_score:.3f} " if hit.sparse_score is not None else "sparse=—    ") \
            + f"fused={hit.fused_score:.5f}"
        preview = hit.text[:60] + ("…" if len(hit.text) > 60 else "")
        print(f"#{hit.rank} [{'+'.join(hit.sources)}] {scores}")
        print(f"    {crumb} ({pages}, chunk_index={hit.chunk_index})")
        print(f"    {preview}")
    if not result.hits:
        print("(no hits)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid search over a built index")
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("--doc-id", required=True, help="document id whose index to search")
    parser.add_argument("--mode", choices=MODES, default="hybrid", help="retrieval mode (default: hybrid)")
    parser.add_argument("--top-k", type=int, default=None, help="final hits to keep (default: 5)")
    parser.add_argument("--artifacts-root", default=None, help="artifacts root (default: settings.ARTIFACTS_DIR)")
    parser.add_argument("--json", action="store_true", help="print the full RetrievalResult as JSON")
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

    result = engine.retrieve(args.query, mode=args.mode, top_k=args.top_k)
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_human(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
