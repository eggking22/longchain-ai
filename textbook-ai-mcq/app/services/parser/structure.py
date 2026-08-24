"""Stage 3 — heading detection (priority waterfall) and hierarchy resolution.

Heading candidates are produced by four detectors applied per line, in the
project's priority order:

    1. toc       PDF bookmarks matched against page lines (conf ~0.9)
    2. font      font size above body size -> level map, or bold-only (conf ~0.7/0.55)
    3. numbering 第X章 / 第X节 / 1.1 / 一、 patterns (conf ~0.6)
    4. spatial   short isolated line at margin with vertical gaps (conf ~0.4)

When several detectors hit the same line, the highest-priority rule decides
the level and the others only add confidence. Every hit is recorded in the
debug report so parsing decisions stay inspectable.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import statistics
from dataclasses import dataclass, field

from app.schemas.document import DocNode, Provenance

from .config import ParserConfig
from .parser import LineInfo, TocEntry
from .patterns import clean_toc_title, match_numbering, normalize_for_match

RULE_PRIORITY = ("toc", "font", "numbering", "spatial")

# line-final punctuation that closes a sentence (and thus can't end a heading)
SENT_END_RE = re.compile(r"[。！？；…!?;”』」）)]$")
# CJK characters or CJK/fullwidth punctuation at a join boundary => no space
_CJK_JOIN_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef…—]")

SYNTH_PREFIX_DOTTED_RE = re.compile(r"^(\d{1,2})\.\d")
_CN_NUM_CLASS = "0-9０-９一二三四五六七八九十百零〇两"
SYNTH_PREFIX_JIE_RE = re.compile(rf"^第\s*([{_CN_NUM_CLASS}]+)\s*节")


@dataclass
class HeadingCandidate:
    line_index: int
    text: str
    level: int
    rule: str
    confidence: float
    evidence: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    candidates: list[HeadingCandidate]
    debug: list[dict]


# --------------------------------------------------------------------------
# detectors
# --------------------------------------------------------------------------

def build_font_profile(lines: list[LineInfo], max_levels: int) -> tuple[float, dict[float, int]]:
    """Char-weighted size histogram -> (body_size, {heading_size: level}).

    Same idea as pymupdf4llm's IdentifyHeaders: the most popular size is
    body text; larger sizes map to heading levels, largest first.
    """
    weights: dict[float, int] = {}
    for l in lines:
        n = len(l.text.strip())
        if n:
            weights[l.size] = weights.get(l.size, 0) + n
    if not weights:
        return 0.0, {}
    body = max(weights, key=weights.get)
    bigger = sorted((s for s in weights if s > body + 0.3), reverse=True)[:max_levels]
    return body, {s: i + 1 for i, s in enumerate(bigger)}


def match_toc(lines: list[LineInfo], toc: list[TocEntry]) -> dict[int, tuple[TocEntry, float]]:
    """Map TOC entries onto actual page lines.

    Search window is the entry's target page +/- 1. Match quality:
    exact (normalized) > containment > difflib ratio >= 0.8. A line can be
    claimed by at most one entry (best score wins).
    """
    used: dict[int, tuple[TocEntry, float]] = {}
    for entry in toc:
        title = clean_toc_title(entry.title)
        if not title:
            continue
        target = normalize_for_match(title)
        if not target:
            continue
        best: tuple[float, LineInfo] | None = None
        for line in lines:
            if not (entry.page_no - 1 <= line.page_no <= entry.page_no + 1):
                continue
            lt = normalize_for_match(line.text)
            if not lt:
                continue
            if lt == target:
                score = 3.0
            elif target in lt or lt in target:
                score = 2.0
            else:
                ratio = difflib.SequenceMatcher(None, lt, target).ratio()
                if ratio < 0.8:
                    continue
                score = 1.0 + ratio * 0.1
            score += 0.5 if line.page_no == entry.page_no else 0.0
            if best is None or score > best[0]:
                best = (score, line)
        if best is None:
            continue
        score, line = best
        if line.index in used and used[line.index][1] >= score:
            continue
        used[line.index] = (entry, score)
    return used


def _gap_contexts(lines: list[LineInfo]) -> tuple[dict[int, float], dict[int, float], set[int]]:
    """Per-line vertical gap to the previous/next line on the same page."""
    by_page: dict[int, list[LineInfo]] = {}
    for l in lines:
        by_page.setdefault(l.page_no, []).append(l)
    gap_before: dict[int, float] = {}
    gap_after: dict[int, float] = {}
    page_first: set[int] = set()
    for ls in by_page.values():
        ls = sorted(ls, key=lambda l: l.bbox[1])
        page_first.add(ls[0].index)
        for a, b in zip(ls, ls[1:]):
            gap = b.bbox[1] - a.bbox[1]
            gap_after[a.index] = gap
            gap_before[b.index] = gap
    return gap_before, gap_after, page_first


def _median_gap(lines: list[LineInfo]) -> float:
    _, gap_after, _ = _gap_contexts(lines)
    gaps = list(gap_after.values())
    return statistics.median(gaps) if gaps else 0.0


def detect_headings(
    lines: list[LineInfo],
    toc: list[TocEntry],
    config: ParserConfig,
) -> DetectionResult:
    body_size, size_levels = build_font_profile(lines, config.max_heading_levels)
    toc_map = match_toc(lines, toc)
    gap_before, gap_after, page_first = _gap_contexts(lines)
    median_gap = _median_gap(lines)

    candidates: list[HeadingCandidate] = []
    debug: list[dict] = []
    for line in lines:
        hits: list[dict] = []

        if line.index in toc_map:
            entry, score = toc_map[line.index]
            hits.append(
                {"rule": "toc", "level": entry.level, "confidence": 0.9,
                 "evidence": {"toc_title": entry.title, "toc_page": entry.page_no, "match_score": round(score, 2)}}
            )

        text = line.text.strip()
        guarded = 0 < len(text) <= config.heading_max_chars and not SENT_END_RE.search(text)
        if guarded:
            if line.size in size_levels:
                hits.append(
                    {"rule": "font", "level": size_levels[line.size], "confidence": 0.7,
                     "evidence": {"kind": "size", "size": line.size, "body_size": body_size}}
                )
            elif line.bold and line.size >= body_size - 0.2:
                hits.append(
                    {"rule": "font", "level": min(2, config.max_heading_levels), "confidence": 0.55,
                     "evidence": {"kind": "bold", "font": line.font}}
                )
            level = match_numbering(text)
            if level is not None:
                hits.append({"rule": "numbering", "level": level, "confidence": 0.6, "evidence": {}})
            if (
                len(text) < 40
                and median_gap > 0
                and gap_after.get(line.index, 0.0) >= 1.5 * median_gap
                and (line.index in page_first or gap_before.get(line.index, 0.0) >= 1.2 * median_gap)
            ):
                hits.append(
                    {"rule": "spatial", "level": 2, "confidence": 0.4,
                     "evidence": {"gap_after": round(gap_after[line.index], 1), "median_gap": round(median_gap, 1)}}
                )

        if not hits:
            continue
        hits.sort(key=lambda h: RULE_PRIORITY.index(h["rule"]))
        chosen = hits[0]
        confidence = min(0.95, chosen["confidence"] + 0.08 * (len(hits) - 1))
        if confidence < config.min_heading_confidence:
            debug.append({"line_index": line.index, "page": line.page_no, "text": text,
                          "accepted": False, "hits": hits})
            continue
        candidates.append(
            HeadingCandidate(
                line_index=line.index,
                text=text,
                level=chosen["level"],
                rule=chosen["rule"],
                confidence=round(confidence, 2),
                evidence=chosen.get("evidence", {}),
            )
        )
        debug.append(
            {"line_index": line.index, "page": line.page_no, "text": text, "accepted": True,
             "hits": hits, "chosen_rule": chosen["rule"], "final_level": chosen["level"],
             "final_confidence": round(confidence, 2)}
        )
    candidates.sort(key=lambda c: c.line_index)
    return DetectionResult(candidates=candidates, debug=debug)


# --------------------------------------------------------------------------
# hierarchy resolution + paragraph assembly
# --------------------------------------------------------------------------

def join_line_texts(parts: list[str]) -> str:
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not out:
            out = p
        elif _CJK_JOIN_RE.search(out[-1]) or _CJK_JOIN_RE.match(p[0]):
            out += p
        else:
            out += " " + p
    return out


def _body_x0(lines: list[LineInfo]) -> float:
    xs = [l.bbox[0] for l in lines if len(l.text.strip()) >= 15]
    return statistics.median(xs) if xs else 0.0


def _node_id(doc_id: str, path: str) -> str:
    return hashlib.sha1(f"{doc_id}|{path}".encode("utf-8")).hexdigest()[:16]


def build_document(
    doc_id: str,
    lines: list[LineInfo],
    candidates: list[HeadingCandidate],
    config: ParserConfig,
) -> DocNode:
    heading_by_index = {c.line_index: c for c in candidates}

    # compact level jumps (1 -> 3 becomes 1 -> 2); the first heading keeps
    # its own level so chapter-less documents still reach section synthesis
    prev: int | None = None
    for c in sorted(candidates, key=lambda c: c.line_index):
        if prev is not None and c.level > prev + 1:
            c.level = prev + 1
        prev = c.level

    body_x0 = _body_x0(lines)
    root = DocNode(node_id=_node_id(doc_id, ""), node_type="document", title=doc_id, level=0)
    stack: list[DocNode] = [root]

    para_lines: list[LineInfo] = []

    def flush_paragraph() -> None:
        if not para_lines:
            return
        parent = stack[-1]
        text = join_line_texts([l.text for l in para_lines])
        first = para_lines[0]
        node = DocNode(
            node_id=_node_id(doc_id, "/".join(n.title for n in stack[1:]) + f"/para-{len(parent.children)}"),
            node_type="paragraph",
            level=parent.level + 1,
            text=text,
            provenance=Provenance(page_no=first.page_no, bbox=tuple(first.bbox)),
            pages=sorted({l.page_no for l in para_lines}),
        )
        parent.children.append(node)
        para_lines.clear()

    for line in lines:
        cand = heading_by_index.get(line.index)
        if cand is not None:
            flush_paragraph()
            while stack[-1].level >= cand.level:
                stack.pop()
            node_type = "chapter" if cand.level == 1 else "section"
            node = DocNode(
                node_id=_node_id(doc_id, "/".join([*[n.title for n in stack[1:]], cand.text])),
                node_type=node_type,
                title=cand.text,
                level=cand.level,
                provenance=Provenance(page_no=line.page_no, bbox=tuple(line.bbox)),
                heading_rule=cand.rule,
                heading_confidence=cand.confidence,
            )
            stack[-1].children.append(node)
            stack.append(node)
            continue

        if para_lines:
            prev_line = para_lines[-1]
            same_block = prev_line.page_no == line.page_no and prev_line.block_no == line.block_no
            prev_ends = bool(SENT_END_RE.search(prev_line.text.strip()))
            indented = (line.bbox[0] - body_x0) > config.indent_min_pt
            if not same_block and (prev_ends or indented):
                flush_paragraph()
        para_lines.append(line)
    flush_paragraph()

    _synthesize_chapters(root, doc_id)
    _fallback_chapter(root, doc_id)
    return root


def _chapter_group_key(title: str, state: dict) -> str | None:
    """Key for grouping sibling sections into a synthetic chapter.

    Dotted numbering ("1.1") groups by the leading integer; "第N节" runs in
    a group until N restarts at 1 or is not previous+1.
    """
    m = SYNTH_PREFIX_DOTTED_RE.match(title)
    if m:
        return f"n{m.group(1)}"
    m = SYNTH_PREFIX_JIE_RE.match(title)
    if m:
        token = m.group(1)
        if token.isdigit():
            n = int(token)
            if n == 1 or state.get("jie_prev") is None or n != state["jie_prev"] + 1:
                state["jie_group"] = f"jie-{state.get('jie_count', 0)}"
                state["jie_count"] = state.get("jie_count", 0) + 1
            state["jie_prev"] = n
            return state["jie_group"]
    return None


def _synthesize_chapters(root: DocNode, doc_id: str) -> None:
    if any(c.node_type == "chapter" for c in root.children):
        return
    if not any(c.node_type == "section" for c in root.children):
        return
    state: dict = {}
    groups: list[tuple[str | None, list[DocNode]]] = []
    for node in root.children:
        key = _chapter_group_key(node.title, state)
        if key is not None and groups and groups[-1][0] == key:
            groups[-1][1].append(node)
        else:
            groups.append((key, [node]))
    new_children: list[DocNode] = []
    for key, nodes in groups:
        if key is None:
            new_children.extend(nodes)
            continue
        first_title = nodes[0].title
        m = SYNTH_PREFIX_DOTTED_RE.match(first_title) or SYNTH_PREFIX_JIE_RE.match(first_title)
        num = m.group(1) if m else ""
        chapter = DocNode(
            node_id=_node_id(doc_id, f"synthetic/{key}/{num}"),
            node_type="chapter",
            title=f"第{num}章" if num else "未命名章节",
            level=1,
            heading_rule="synthetic",
            heading_confidence=0.5,
            provenance=nodes[0].provenance,
            children=nodes,
        )
        new_children.append(chapter)
    root.children = new_children


def _fallback_chapter(root: DocNode, doc_id: str) -> None:
    """No headings at all: wrap everything in one fallback chapter."""
    if any(c.node_type in ("chapter", "section") for c in root.children):
        return
    if not root.children:
        return
    chapter = DocNode(
        node_id=_node_id(doc_id, "fallback"),
        node_type="chapter",
        title="正文",
        level=1,
        heading_rule="fallback",
        heading_confidence=0.3,
        children=root.children,
    )
    root.children = [chapter]
