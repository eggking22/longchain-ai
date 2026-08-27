"""Flatten the parsed DocNode tree and classify paragraphs into IMRaD sections.

The Phase 1 parser is structural only (chapter/section/paragraph); it knows
nothing about paper sections. This module is a read-only second pass that maps
breadcrumb titles onto Introduction/Methods/Results/Discussion via English
keywords. Sections that classify as none of these keep type "other" — the
downstream evidence collector still works, it just cannot infer roles from
section type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.document import DocNode
from app.schemas.paper_semantics import PaperSectionType

# Keyword -> section type; a title matches on substring (case-insensitive).
# Most specific keywords first where a title could match several. Variants follow
# GROBID / PubReader section-heading inventories (journals differ in punctuation
# and wording: "Materials & methods", "Experimental section", ...).
SECTION_KEYWORDS: tuple[tuple[str, PaperSectionType], ...] = (
    # NB: bare "summary" is deliberately NOT mapped — Nature papers end with a
    # "Reporting summary" section that must not be mistaken for the Abstract.
    ("abstract", "abstract"),
    ("overview", "abstract"),
    ("introduction", "introduction"),
    ("background", "introduction"),
    ("materials and methods", "methods"),
    ("materials & methods", "methods"),
    ("methods and materials", "methods"),
    ("material and methods", "methods"),
    ("experimental procedures", "methods"),
    ("experimental section", "methods"),
    ("patients and methods", "methods"),
    ("study design", "methods"),
    ("method", "methods"),
    ("material", "methods"),
    ("results and discussion", "results"),
    ("result", "results"),
    ("discussion", "discussion"),
    ("conclusion", "discussion"),
)


def classify_section(title: str) -> PaperSectionType:
    """Map a heading title onto an IMRaD section type via keywords."""
    lowered = title.lower()
    for keyword, section_type in SECTION_KEYWORDS:
        if keyword in lowered:
            return section_type
    return "other"


@dataclass
class ParagraphRecord:
    """A flattened paragraph with its structural context."""

    paragraph_id: str
    text: str
    breadcrumb: list[str] = field(default_factory=list)
    section_type: PaperSectionType = "other"
    section_title: str = ""
    page_no: int = 0
    order: int = 0  # position in document flow, used for stable ordering


def flatten_document(root: DocNode) -> list[ParagraphRecord]:
    """Depth-first flatten of the DocNode tree into paragraph records.

    The section type is inherited from the nearest ancestor heading whose
    title classifies to a known IMRaD section; unknown headings inherit the
    outer section (e.g. a "2.1 Cell culture" sub-heading inside Methods).
    """

    records: list[ParagraphRecord] = []

    def walk(node: DocNode, breadcrumb: list[str], section_type: PaperSectionType, section_title: str) -> None:
        current_type, current_title = section_type, section_title
        if node.node_type in ("chapter", "section") and node.title.strip():
            classified = classify_section(node.title)
            if classified != "other":
                current_type, current_title = classified, node.title
            breadcrumb = [*breadcrumb, node.title]
        elif node.node_type == "document" and node.title.strip():
            breadcrumb = [*breadcrumb, node.title]

        if node.node_type == "paragraph" and node.text.strip():
            if node.pages:
                page_no = node.pages[0]
            elif node.provenance is not None:
                page_no = node.provenance.page_no
            else:
                page_no = 0
            records.append(
                ParagraphRecord(
                    paragraph_id=node.node_id,
                    text=node.text.strip(),
                    breadcrumb=list(breadcrumb),
                    section_type=current_type,
                    section_title=current_title,
                    page_no=page_no,
                )
            )

        for child in node.children:
            walk(child, breadcrumb, current_type, current_title)

    walk(root, [], "other", "")
    for index, record in enumerate(records):
        record.order = index
    return records
