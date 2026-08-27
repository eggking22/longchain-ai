"""Paper-level background extraction and L1 deterministic text partitioning.

Two independent, fully deterministic passes over the flattened paragraphs:

1. :func:`extract_background` — the paper's own Abstract (heading path, or the
   Nature-style leading paragraphs when no Abstract heading exists) and an
   extractive Introduction summary, both verbatim author text with
   paragraph/pages provenance (Docling-style coverage).
2. :func:`partition_results_by_figures` — L1 partitioning of the Results
   narrative into per-figure text blocks: paragraphs citing a figure are
   anchors; uncited paragraphs that follow an anchor are inherited as
   continuation text (authors typically cite once and narrate on). Anchors
   are hard facts from the text; continuation only widens the evidence scope
   and never adds semantics by itself.
"""

from __future__ import annotations

import re

from app.schemas.paper_semantics import FigureReference, FigureTextBlock

from .patterns import CAPTION_START_RE, INLINE_CAPTION_RE, split_sentences_en
from .sections import ParagraphRecord

_ABSTRACT_WORD_BUDGET = 400
_INTRODUCTION_WORD_BUDGET = 400
_INTRODUCTION_CHAR_BUDGET = 1500
_MIN_PARAGRAPH_WORDS = 10  # title fragments / author lists / running heads


# --- Background -------------------------------------------------------------------


def _word_budget(text: str, budget: int) -> str:
    """Truncate at a sentence boundary within a word budget (never mid-sentence)."""
    sentences = split_sentences_en(text)
    kept: list[str] = []
    count = 0
    for sentence in sentences:
        words = len(sentence.split())
        if count + words > budget and kept:
            break
        kept.append(sentence)
        count += words
        if count >= budget:
            break
    return " ".join(kept) if kept else " ".join(text.split()[:budget])


def _summary_record(paragraphs: list[ParagraphRecord], text: str, source: str = "") -> dict:
    return {
        "text": text,
        "source": source,
        "paragraph_ids": [p.paragraph_id for p in paragraphs],
        "pages": sorted({p.page_no for p in paragraphs if p.page_no}),
    }


def _is_caption_like(text: str) -> bool:
    return CAPTION_START_RE.match(text) is not None or INLINE_CAPTION_RE.search(text[:120]) is not None


def _looks_like_author_list(text: str) -> bool:
    """Author/affiliation lines: most tokens carry superscript markers ("Sanséau1,19")."""
    tokens = text.split()
    if not tokens:
        return False
    marked = sum(1 for token in tokens if re.search(r"[a-zA-Z]\d", token) or re.match(r"^\d", token))
    return marked / len(tokens) >= 0.4


def _usable(paragraph: ParagraphRecord) -> bool:
    if len(paragraph.text.split()) < _MIN_PARAGRAPH_WORDS:
        return False  # title fragments / running heads
    if _is_caption_like(paragraph.text):
        return False
    if _looks_like_author_list(paragraph.text):
        return False  # author lists and affiliations on the first page
    return True


def _cites_any_figure(paragraph: ParagraphRecord, known: dict) -> bool:
    return bool(_figure_mentions(paragraph, known))


def _first_anchor_order(paragraphs: list[ParagraphRecord], known: dict) -> int | None:
    for paragraph in paragraphs:
        if _cites_any_figure(paragraph, known):
            return paragraph.order
    return None


def extract_background(paragraphs: list[ParagraphRecord], refs: list[FigureReference] | None = None) -> dict:
    """Extractive background: Abstract (dual-path) + Introduction summary.

    Abstract path 1: paragraphs under an Abstract/Overview heading.
    Abstract path 2 (Nature style, no Abstract heading): among the leading
    body paragraphs (before the first figure anchor / first IMRaD heading),
    the longest one — author/title lines are short, the summary paragraph is
    the longest block on the first page.
    Introduction path 1: introduction-section paragraphs, sentence-boundary
    truncated. Path 2 (Nature style): leading paragraphs after the abstract
    and before the first figure anchor, word-budgeted.
    All text is the authors' own wording, never rewritten.
    """
    known = {ref.figure_id: ref for ref in (refs or [])}
    abstract_paras = [p for p in paragraphs if p.section_type == "abstract"]

    abstract: dict = {}
    if abstract_paras:
        abstract = _summary_record(abstract_paras, " ".join(p.text for p in abstract_paras), "abstract_heading")
    else:
        leading = _leading_paragraphs(paragraphs, known)
        if leading:
            longest = max(leading, key=lambda p: len(p.text.split()))
            text = longest.text
            if len(text.split()) > _ABSTRACT_WORD_BUDGET:
                text = _word_budget(text, _ABSTRACT_WORD_BUDGET)
            abstract = _summary_record([longest], text, "leading_paragraphs")

    introduction: dict = {}
    introduction_paras = [p for p in paragraphs if p.section_type == "introduction"]
    if introduction_paras:
        full = " ".join(p.text for p in introduction_paras)
        text = full if len(full) <= _INTRODUCTION_CHAR_BUDGET else _char_budget(full, _INTRODUCTION_CHAR_BUDGET)
        introduction = _summary_record(introduction_paras, text)
    else:
        # Nature style: the narrative between the abstract and the first figure
        # anchor (or the first Methods/Results heading) is the introduction.
        boundary = _first_anchor_order(paragraphs, known)
        if boundary is None:
            for paragraph in paragraphs:
                if paragraph.section_type in ("methods", "results"):
                    boundary = paragraph.order
                    break
        abstract_orders = {p.order for p in abstract_paras} | {
            p.order for p in _paragraphs_by_ids(paragraphs, abstract)
        }
        candidates = [
            p
            for p in paragraphs
            if _usable(p)
            and p.order not in abstract_orders
            and p.section_type in ("introduction", "other")
            and (boundary is None or p.order < boundary)
        ]
        if candidates:
            full = " ".join(p.text for p in candidates)
            introduction = _summary_record(candidates, _word_budget(full, _INTRODUCTION_WORD_BUDGET))

    return {"abstract": abstract, "introduction": introduction}


def _paragraphs_by_ids(paragraphs: list[ParagraphRecord], record: dict) -> list[ParagraphRecord]:
    ids = set(record.get("paragraph_ids", [])) if record else set()
    return [p for p in paragraphs if p.paragraph_id in ids]


def _leading_paragraphs(paragraphs: list[ParagraphRecord], known: dict) -> list[ParagraphRecord]:
    """Leading body paragraphs before the first figure anchor / IMRaD heading."""
    boundary = _first_anchor_order(paragraphs, known)
    leading: list[ParagraphRecord] = []
    for paragraph in paragraphs:
        if paragraph.section_type in ("introduction", "methods", "results", "discussion"):
            break
        if boundary is not None and paragraph.order >= boundary:
            break
        if _usable(paragraph):
            leading.append(paragraph)
    return leading


def _char_budget(text: str, budget: int) -> str:
    sentences = split_sentences_en(text)
    kept: list[str] = []
    count = 0
    for sentence in sentences:
        if count + len(sentence) > budget and kept:
            break
        kept.append(sentence)
        count += len(sentence) + 1
        if count >= budget:
            break
    return " ".join(kept)


def _is_caption_like(text: str) -> bool:
    return CAPTION_START_RE.match(text) is not None or INLINE_CAPTION_RE.search(text[:120]) is not None


# --- L1 partitioning -----------------------------------------------------------------


def _figure_mentions(paragraph: ParagraphRecord, known: dict[str, FigureReference]) -> list[str]:
    """Canonical figure ids cited by a paragraph (panels roll up to parents), in order."""
    from .figure_reference import find_mentions

    cited: list[str] = []
    for mention in find_mentions(paragraph.text):
        label = "Figure" if mention.kind == "figure" else "Table"
        figure_id = f"{mention.prefix} {label} {mention.number}".strip()
        if figure_id in known and figure_id not in cited:
            cited.append(figure_id)
    return cited


def partition_flow(paragraphs: list[ParagraphRecord], refs: list[FigureReference]) -> list[ParagraphRecord]:
    """The Results paragraph flow that L1 partitions.

    Classified Results sections when available; otherwise (Nature-style papers
    whose section headings the Phase 1 detector misses) the other/results
    paragraphs between the first and last figure anchor — Methods sits outside
    that span in both IMRaD orders.
    """
    flow = [p for p in paragraphs if p.section_type == "results"]
    if flow:
        return flow
    known = {ref.figure_id: ref for ref in refs}
    anchor_orders = [p.order for p in paragraphs if _cites_any_figure(p, known)]
    if not anchor_orders:
        return []
    first, last = min(anchor_orders), max(anchor_orders)
    return [p for p in paragraphs if p.section_type in ("results", "other") and first <= p.order <= last]


def partition_results_by_figures(
    paragraphs: list[ParagraphRecord], refs: list[FigureReference]
) -> dict[str, FigureTextBlock]:
    """L1: partition the Results narrative into per-figure text blocks.

    Rules (document order over the flow from :func:`partition_flow`):
    - a paragraph citing figure F is an anchor: it closes the current block and
      opens/extends F's block;
    - an uncited paragraph is inherited by the current block (continuation);
    - paragraphs citing several figures belong to the first-cited figure, and
      the others are recorded in shared_with;
    - paragraphs before the first anchor are left unassigned (safety valve —
      never guess an owner);
    - Methods/Discussion paragraphs never take part.
    """
    known = {ref.figure_id: ref for ref in refs}
    blocks: dict[str, FigureTextBlock] = {}

    def block_for(figure_id: str) -> FigureTextBlock:
        if figure_id not in blocks:
            blocks[figure_id] = FigureTextBlock(figure_id=figure_id)
        return blocks[figure_id]

    current: str = ""  # figure_id of the open block
    for paragraph in partition_flow(paragraphs, refs):
        cited = _figure_mentions(paragraph, known)
        if not cited:
            if current:
                block_for(current).continuation_paragraph_ids.append(paragraph.paragraph_id)
            continue  # before the first anchor → unassigned (safety valve)

        primary = cited[0]
        block = block_for(primary)
        block.anchor_paragraph_ids.append(paragraph.paragraph_id)
        for other in cited[1:]:
            if other not in block.shared_with:
                block.shared_with.append(other)
        current = primary

    return blocks
