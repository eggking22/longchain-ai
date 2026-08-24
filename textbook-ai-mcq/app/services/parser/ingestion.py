"""Pipeline orchestration: parse -> clean -> structure -> chunk.

Every stage persists its artifacts under ARTIFACTS_DIR so runs can be
inspected and re-executed without re-reading the PDF::

    data/raw/{doc_id}/lines.json | toc.json | fonts.json | lines.clean.json
    data/structure/{doc_id}/document.json | headings.debug.json | stats.json
    data/chunks/{doc_id}/chunks.jsonl
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .chunking import chunk_document
from .cleaner import clean_lines
from .config import ParserConfig
from .parser import extract_raw, load_raw, save_raw
from .structure import build_document, detect_headings


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _count_tree(root) -> Counter:
    counts: Counter = Counter()
    stack = [root]
    while stack:
        node = stack.pop()
        counts[node.node_type] += 1
        if node.heading_rule:
            counts[f"rule:{node.heading_rule}"] += 1
        stack.extend(node.children)
    return counts


def _stats(doc_id: str, tree, detection, chunks, raw, clean_result) -> dict:
    counts = _count_tree(tree)
    char_counts = [c.char_count for c in chunks] or [0]
    return {
        "doc_id": doc_id,
        "num_pages": raw.num_pages,
        "raw_lines": len(raw.lines),
        "cleaned_lines": clean_result.report["kept"],
        "chapters": counts["chapter"],
        "sections": counts["section"],
        "paragraphs": counts["paragraph"],
        "heading_rules": {k.split(":", 1)[1]: v for k, v in counts.items() if k.startswith("rule:")},
        "toc_entries": len(raw.toc),
        "chunks": {
            "count": len(chunks),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "avg_chars": round(sum(char_counts) / len(char_counts), 1),
        },
    }


def ingest(
    doc_id: str,
    pdf_path: str | Path,
    config: ParserConfig | None = None,
    artifacts_root: str | Path = "data",
    reuse_raw: bool = False,
) -> dict:
    from app.core.config import get_settings

    if config is None:
        config = ParserConfig.from_settings(get_settings())

    root_dir = Path(artifacts_root)
    raw_dir = root_dir / "raw" / doc_id
    struct_dir = root_dir / "structure" / doc_id
    chunk_dir = root_dir / "chunks" / doc_id

    # --- stage 1: raw extraction ---
    if reuse_raw and (raw_dir / "lines.json").exists():
        raw = load_raw(raw_dir, pdf_path=str(pdf_path))
    else:
        raw = extract_raw(pdf_path)
        save_raw(raw, raw_dir)

    # --- stage 2: cleaning ---
    clean_result = clean_lines(raw, config.header_footer_band, config.repeat_ratio)
    _write_json(raw_dir / "lines.clean.json", [l.to_dict() for l in clean_result.lines])
    _write_json(raw_dir / "cleaner.debug.json", clean_result.report)

    # --- stage 3: heading detection + hierarchy ---
    detection = detect_headings(clean_result.lines, raw.toc, config)
    tree = build_document(doc_id, clean_result.lines, detection.candidates, config)
    _write_json(struct_dir / "document.json", tree.model_dump())
    _write_json(
        struct_dir / "headings.debug.json",
        {
            "accepted": [c.__dict__ for c in detection.candidates],
            "trace": detection.debug,
        },
    )

    # --- stage 4: chunking ---
    chunks = chunk_document(tree, doc_id, config)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    with (chunk_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    stats = _stats(doc_id, tree, detection, chunks, raw, clean_result)
    _write_json(struct_dir / "stats.json", stats)
    return stats
