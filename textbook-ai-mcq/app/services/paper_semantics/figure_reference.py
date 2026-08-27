"""Figure/Table reference extraction over flattened paragraphs.

Distinguishes caption paragraphs (label at paragraph start followed by a
separator: "Figure 2. ...", "Fig. 1: ...", "Fig. 1 | ..." (Nature), "Table 3.")
from in-text mentions ("Figure 2 shows ..."), canonicalizes ids ("Fig. 2B" →
Figure 2 with panel B), and keeps prefixed namespaces ("Extended Data Fig. 1",
"Supplementary Fig. 2") separate from the main figure numbering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.paper_semantics import FigureKind, FigureReference

from .config import PaperSemanticsConfig
from .patterns import CAPTION_START_RE, FIGURE_MENTION_RE, INLINE_CAPTION_RE, PANEL_BOUNDARY_RE, TABLE_MENTION_RE
from .sections import ParagraphRecord


@dataclass
class MentionMatch:
    """One surface occurrence of a figure/table reference."""

    kind: FigureKind
    number: int
    panel: str  # "" or single letter
    raw: str  # surface form up to the panel, e.g. "Fig. 2B"
    prefix: str = ""  # normalized namespace: "", "Extended Data", "Supplementary"


def _normalize_prefix(prefix: str | None) -> str:
    prefix = (prefix or "").strip()
    lowered = prefix.lower().replace(".", "")
    collapsed = re.sub(r"\s+", " ", lowered)
    if collapsed.startswith("extended data"):
        return "Extended Data"
    if collapsed.startswith("suppl"):
        return "Supplementary"
    return ""


def _raw_form(text: str, match: re.Match, panel_group: int) -> str:
    end = match.end(panel_group) if match.group(panel_group) else match.end(panel_group - 1)
    return text[match.start() : end].strip()


def find_mentions(text: str) -> list[MentionMatch]:
    """All figure/table mentions in a text, in order of appearance."""
    found: list[tuple[int, MentionMatch]] = []
    for regex, kind in ((FIGURE_MENTION_RE, "figure"), (TABLE_MENTION_RE, "table")):
        for match in regex.finditer(text):
            number = int(match.group(2))
            panel = (match.group(3) or "").upper()
            found.append(
                (
                    match.start(),
                    MentionMatch(
                        kind=kind,
                        number=number,
                        panel=panel,
                        raw=_raw_form(text, match, 3),
                        prefix=_normalize_prefix(match.group(1)),
                    ),
                )
            )
            extra = match.group(4)  # "Figures 2 and 3" / "Figure 2-4" second number
            if extra and 0 < int(extra) - number < 10:
                found.append(
                    (
                        match.end(),
                        MentionMatch(
                            kind=kind,
                            number=int(extra),
                            panel="",
                            raw=f"{kind.title()} {extra}",
                            prefix=_normalize_prefix(match.group(1)),
                        ),
                    )
                )
    return [mention for _, mention in sorted(found, key=lambda item: item[0])]


def parse_caption(text: str) -> MentionMatch | None:
    """Return the caption label if the paragraph *starts* with a caption form."""
    match = CAPTION_START_RE.match(text)
    if match is None:
        return None
    label = text[: match.start(2)].strip()
    kind: FigureKind = "table" if re.match(r"(?i)^((Extended\s+Data|Supplementary|Suppl\.?)\s+)?Tables?\b", label) else "figure"
    panel = (match.group(3) or "").upper()
    raw = f"{label} {match.group(2)}{panel}".strip()
    return MentionMatch(kind=kind, number=int(match.group(2)), panel=panel, raw=raw, prefix=_normalize_prefix(match.group(1)))


def _canonical(kind: FigureKind, number: int, prefix: str = "") -> str:
    label = "Figure" if kind == "figure" else "Table"
    return f"{prefix} {label} {number}".strip()


def _namespace_rank(prefix: str) -> int:
    return {"": 0, "Extended Data": 1, "Supplementary": 2}.get(prefix, 3)


def _register(refs: dict[str, FigureReference], mention: MentionMatch) -> FigureReference:
    figure_id = _canonical(mention.kind, mention.number, mention.prefix)
    ref = refs.get(figure_id)
    if ref is None:
        ref = FigureReference(figure_id=figure_id, kind=mention.kind, number=mention.number)
        refs[figure_id] = ref
    return ref


def find_inline_caption(text: str) -> tuple[MentionMatch, int] | None:
    """Locate a Nature-style pipe caption ("Fig. 1 | Title") anywhere in a paragraph.

    Two-column PDFs routinely glue body text onto the caption line, so the label
    is not always at the paragraph start. The pipe separator is unambiguous —
    body prose never references figures as "Fig. 1 |".
    """
    match = INLINE_CAPTION_RE.search(text)
    if match is None:
        return None
    label = text[match.start() : match.start(2)].strip()
    kind: FigureKind = "table" if re.match(r"(?i)^((Extended\s+Data|Supplementary|Suppl\.?)\s+)?Tables?\b", label) else "figure"
    panel = (match.group(3) or "").upper()
    raw = f"{label} {match.group(2)}{panel}".strip()
    mention = MentionMatch(kind=kind, number=int(match.group(2)), panel=panel, raw=raw, prefix=_normalize_prefix(match.group(1)))
    return mention, match.start()


def split_caption_panels(caption_text: str) -> dict[str, str]:
    """Split a caption into per-panel chunks: {"a": "a, GFP intensity ...", ...}.

    The text before the first panel boundary is the figure-level title and is
    not returned. Captions without panel boundaries return {} (single-panel
    figure).
    """
    boundaries = [match for match in PANEL_BOUNDARY_RE.finditer(caption_text)]
    if not boundaries:
        return {}
    panels: dict[str, str] = {}
    for index, match in enumerate(boundaries):
        end = boundaries[index + 1].start() if index + 1 < len(boundaries) else len(caption_text)
        label = match.group(1).lower()
        if label in panels:  # duplicate label — keep the first chunk
            continue
        panels[label] = caption_text[match.start() : end].strip()
    return panels


def _truncate_caption(text: str, max_chars: int) -> str:
    """Keep leading whole sentences up to ~max_chars (multi-panel captions)."""
    kept: list[str] = []
    total = 0
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if total + len(sentence) > max_chars and kept:
            break
        kept.append(sentence)
        total += len(sentence) + 1
        if total >= max_chars:
            break
    return " ".join(kept) if kept else text[:max_chars]


def extract_figure_references(
    paragraphs: list[ParagraphRecord], config: PaperSemanticsConfig
) -> list[FigureReference]:
    """Build canonical FigureReference records from caption + in-text mentions."""
    refs: dict[str, FigureReference] = {}

    def register_mentions(text: str, paragraph_id: str, skip_figure_id: str = "") -> None:
        for mention in find_mentions(text):
            ref = _register(refs, mention)
            if ref.figure_id == skip_figure_id:
                continue  # the caption's own label is not a self-mention
            if mention.raw not in ref.raw_forms:
                ref.raw_forms.append(mention.raw)
            if mention.panel and f"{mention.number}{mention.panel}" not in ref.subfigures:
                ref.subfigures.append(f"{mention.number}{mention.panel}")
            if paragraph_id not in ref.mention_paragraph_ids:
                ref.mention_paragraph_ids.append(paragraph_id)
            if mention.panel:
                label = mention.panel.lower()
                if paragraph_id not in ref.panel_mention_paragraph_ids.setdefault(label, []):
                    ref.panel_mention_paragraph_ids[label].append(paragraph_id)

    def attach_caption(caption: MentionMatch, paragraph_id: str, caption_text: str, offset: int = 0) -> FigureReference:
        ref = _register(refs, caption)
        ref.caption_paragraph_id = paragraph_id
        ref.caption_text = caption_text
        ref.caption_offset = offset
        if caption.raw not in ref.raw_forms:
            ref.raw_forms.append(caption.raw)
        if caption.panel and caption.panel not in ref.subfigures:
            ref.subfigures.append(caption.panel)
        ref.panel_texts = split_caption_panels(caption_text)
        return ref

    for paragraph in paragraphs:
        caption = parse_caption(paragraph.text)
        if caption is not None and len(paragraph.text) <= config.caption_max_chars:
            attach_caption(caption, paragraph.paragraph_id, paragraph.text)
            continue  # a caption paragraph is not scanned for further mentions

        inline = find_inline_caption(paragraph.text)
        if inline is not None:
            caption, offset = inline
            caption_text = paragraph.text[offset:]
            if len(caption_text) > config.caption_max_chars:
                # the pipe form unambiguously marks a caption; long multi-panel
                # captions are truncated at a sentence boundary instead of rejected
                caption_text = _truncate_caption(caption_text, config.caption_max_chars)
            ref = attach_caption(caption, paragraph.paragraph_id, caption_text, offset=offset)
            # the body prefix and the caption text may still reference other figures
            register_mentions(paragraph.text[:offset], paragraph.paragraph_id, skip_figure_id=ref.figure_id)
            register_mentions(caption_text, paragraph.paragraph_id, skip_figure_id=ref.figure_id)
            continue

        register_mentions(paragraph.text, paragraph.paragraph_id)

    return [
        refs[key]
        for key in sorted(
            refs,
            key=lambda k: (0 if refs[k].kind == "figure" else 1, _namespace_rank_of(refs[k]), refs[k].number),
        )
    ]


def _namespace_rank_of(ref: FigureReference) -> int:
    if ref.figure_id.startswith("Extended Data "):
        return 1
    if ref.figure_id.startswith("Supplementary "):
        return 2
    return 0


def figure_key(ref: FigureReference) -> str:
    """Globally unique short key for a figure: used in experiment ids and as
    the evidence-id prefix so evidence ids never collide across figures.

    "Figure 2" → f02, "Table 1" → t01,
    "Extended Data Figure 1" → edf01, "Supplementary Figure 2" → sf02.
    """
    namespace = ""
    lowered = ref.figure_id.lower()
    if lowered.startswith("extended data "):
        namespace = "ed"
    elif lowered.startswith("supplementary "):
        namespace = "s"
    kind_letter = "f" if ref.kind == "figure" else "t"
    return f"{namespace}{kind_letter}{ref.number:02d}"
