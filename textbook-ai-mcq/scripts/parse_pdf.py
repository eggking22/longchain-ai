"""CLI entry for the hierarchical PDF parser.

Usage (from the project root, with the bio-ai env active or via its python):

    python scripts/parse_pdf.py uploads/example.pdf --doc-id bio-grade1
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
from app.services.parser import ParserConfig, ingest  # noqa: E402


def print_tree(node: dict, depth: int = 0) -> None:
    label = node["title"] or (node["text"][:40] + ("…" if len(node["text"]) > 40 else ""))
    meta = f"  (rule={node['heading_rule']}, conf={node['heading_confidence']})" if node["heading_rule"] else ""
    pages = f"  p{node['pages']}" if node.get("pages") else ""
    print("  " * depth + f"[{node['node_type']}] {label}{meta}{pages}")
    for child in node["children"]:
        print_tree(child, depth + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse a textbook PDF into structure + chunks")
    parser.add_argument("pdf", nargs="?", default="uploads/example.pdf", help="path to the PDF")
    parser.add_argument("--doc-id", default=None, help="document id (default: PDF file stem)")
    parser.add_argument("--artifacts-root", default=None, help="artifacts root (default: settings.ARTIFACTS_DIR)")
    parser.add_argument("--reuse-raw", action="store_true", help="reload raw artifacts if present")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"ERROR: {pdf} not found", file=sys.stderr)
        return 1
    doc_id = args.doc_id or pdf.stem
    settings = get_settings()
    artifacts = args.artifacts_root or settings.ARTIFACTS_DIR

    stats = ingest(doc_id, pdf, ParserConfig.from_settings(settings), artifacts, reuse_raw=args.reuse_raw)

    print("=" * 60)
    print(f"Document: {doc_id}  ({pdf})")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("=" * 60)
    document_path = Path(artifacts) / "structure" / doc_id / "document.json"
    print_tree(json.loads(document_path.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
