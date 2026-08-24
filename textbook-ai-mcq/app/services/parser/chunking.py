"""Stage 4 — paragraph-aware chunking with heading breadcrumbs.

Chunks accumulate whole paragraphs under one leaf heading and never cross
a chapter/section boundary. Oversized paragraphs are split at sentence
boundaries with a one-sentence overlap. Every chunk carries the heading
breadcrumb and back-references to its paragraphs (Docling-style metadata).
"""

from __future__ import annotations

import hashlib
import re

from app.schemas.document import Chunk, DocNode

from .config import ParserConfig

SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])")


def split_sentences(text: str) -> list[str]:
    return [p for p in (s.strip() for s in SENT_SPLIT_RE.split(text)) if p]


def split_long_paragraph(text: str, target: int, hard_max: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    cur = ""
    for sent in split_sentences(text):
        if len(sent) > hard_max:  # pathological sentence: hard character split
            if cur:
                pieces.append(cur)
                cur = ""
            for i in range(0, len(sent), hard_max):
                pieces.append(sent[i : i + hard_max])
            continue
        if cur and len(cur) + len(sent) > target:
            pieces.append(cur)
            cur = sent
        else:
            cur = cur + sent if cur else sent
    if cur:
        pieces.append(cur)
    if overlap > 0 and len(pieces) > 1:
        overlapped: list[str] = []
        for i, piece in enumerate(pieces):
            if i == 0:
                overlapped.append(piece)
                continue
            prev_last = split_sentences(pieces[i - 1])[-1] if pieces[i - 1] else ""
            if prev_last and len(piece) + len(prev_last) <= hard_max:
                overlapped.append(prev_last + piece)
            else:
                overlapped.append(piece)
        pieces = overlapped
    return pieces


def _paragraph_pages(node: DocNode) -> list[int]:
    if node.pages:
        return list(node.pages)
    if node.provenance:
        return [node.provenance.page_no]
    return []


def _chunk_paragraph_group(
    paragraphs: list[DocNode], crumb: list[str], doc_id: str, config: ParserConfig, seq: list[int]
) -> list[Chunk]:
    groups: list[tuple[str, list[str], list[int]]] = []
    cur_text = ""
    cur_ids: list[str] = []
    cur_pages: set[int] = set()

    def flush() -> None:
        nonlocal cur_text, cur_ids, cur_pages
        if cur_text.strip():
            groups.append((cur_text.strip(), list(dict.fromkeys(cur_ids)), sorted(cur_pages)))
        cur_text, cur_ids, cur_pages = "", [], set()

    for para in paragraphs:
        text = para.text.strip()
        if not text:
            continue
        pages = _paragraph_pages(para)
        if len(text) > config.chunk_max_chars:
            flush()
            for piece in split_long_paragraph(
                text, config.chunk_target_chars, config.chunk_max_chars, config.chunk_overlap_sentences
            ):
                groups.append((piece, [para.node_id], pages))
            continue
        if cur_text and len(cur_text) + len(text) > config.chunk_target_chars:
            flush()
        cur_text = cur_text + text if cur_text else text
        cur_ids.append(para.node_id)
        cur_pages.update(pages)
    flush()

    chunks = []
    for text, ids, pages in groups:
        seq[0] += 1
        chunk_id = hashlib.sha1(
            f"{doc_id}|{'/'.join(crumb)}|{seq[0]}".encode("utf-8")
        ).hexdigest()[:16]
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                breadcrumb=list(crumb),
                pages=pages,
                char_count=len(text),
                paragraph_ids=ids,
            )
        )
    return chunks


def chunk_document(root: DocNode, doc_id: str, config: ParserConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    seq = [0]

    def walk(node: DocNode, crumb: list[str]) -> None:
        my_crumb = crumb + [node.title] if node.node_type in ("chapter", "section") else crumb
        paragraphs = [c for c in node.children if c.node_type == "paragraph"]
        if paragraphs:
            chunks.extend(_chunk_paragraph_group(paragraphs, my_crumb, doc_id, config, seq))
        for child in node.children:
            if child.node_type in ("chapter", "section"):
                walk(child, my_crumb)

    walk(root, [])
    return chunks
