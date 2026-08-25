"""CLI entry for building the hybrid retrieval index.

Usage (from the project root, with the bio-ai env active or via its python):

    python scripts/build_index.py --doc-id physiology
    python scripts/build_index.py --doc-id physiology --force --embedder hash
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
from app.services.retrieval import (  # noqa: E402
    HashEmbeddingProvider,
    RetrievalConfig,
    build_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the hybrid retrieval index from Phase 1 chunks")
    parser.add_argument("--doc-id", required=True, help="document id (must have data/chunks/{doc-id}/chunks.jsonl)")
    parser.add_argument("--artifacts-root", default=None, help="artifacts root (default: settings.ARTIFACTS_DIR)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the manifest matches")
    parser.add_argument("--embedder", choices=["hash", "api"], default=None,
                        help="hash = offline deterministic provider (default: from settings)")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = get_settings()
    artifacts = args.artifacts_root or settings.ARTIFACTS_DIR
    config = RetrievalConfig.from_settings(settings)
    embedder = HashEmbeddingProvider(dim=settings.EMBEDDING_DIM) if args.embedder == "hash" else None

    try:
        stats = build_index(args.doc_id, artifacts, embedder=embedder, config=config, force=args.force)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("=" * 60)
    print(f"Index: {args.doc_id}  (artifacts: {Path(artifacts) / 'index' / args.doc_id})")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
