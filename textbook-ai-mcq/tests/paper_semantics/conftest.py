"""Synthetic scientific-paper fixtures for the paper_semantics suite.

Two construction levels, mirroring the repo's established test strategy:

- ``build_paper_tree`` builds a DocNode tree directly (unit tests; precise
  control over sections/captions/mentions);
- ``build_paper_pdf`` builds a real (synthetic) English paper PDF that goes
  through the actual Phase 1 ``ingest()`` for the e2e test.

The synthetic paper deliberately contains:

  Figure 2  — full evidence (caption + Results + Methods + Discussion)  → SUFFICIENT
  Figure 3  — bare mention ("Figure 3 shows the experimental results.") → INSUFFICIENT
  Figure 4  — caption with DV but no result statement                   → PARTIAL
  Figure 5  — association study wording                                  → association, not causation
  Table 1   — table with decrease direction                             → SUFFICIENT
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from app.schemas.document import DocNode

A4_W, A4_H = 595, 842


def _node(node_id: str, node_type: str, title: str = "", level: int = 0, text: str = "", children=None) -> DocNode:
    return DocNode(
        node_id=node_id,
        node_type=node_type,  # type: ignore[arg-type]
        title=title,
        level=level,
        text=text,
        children=children or [],
    )


def _para(node_id: str, text: str, page: int = 1) -> DocNode:
    return DocNode(node_id=node_id, node_type="paragraph", level=3, text=text, pages=[page])


def build_paper_tree() -> DocNode:
    intro = _node("ch-intro", "chapter", title="Introduction", level=1, children=[
        _para("p-intro-1", "Gene X is a key regulator of cellular metabolism.", page=1),
    ])
    methods = _node("ch-methods", "chapter", title="Materials and Methods", level=1, children=[
        _para(
            "p-methods-1",
            "Cells were divided into control and treatment groups. "
            "Cells in the treatment group received Treatment A for 24 hours.",
            page=2,
        ),
    ])
    results = _node("ch-results", "chapter", title="Results", level=1, children=[
        _para(
            "p-results-1",
            "Treatment A significantly increased expression of gene X compared with control (Figure 2).",
            page=2,
        ),
        _para("p-cap-fig2", "Figure 2. Relative expression of gene X in control and treatment groups.", page=3),
        _para(
            "p-results-3",
            "The samples were analyzed as described previously. Figure 3 shows the experimental results.",
            page=3,
        ),
        _para("p-results-4", "Figure 4 shows the apparatus used in the assay.", page=3),
        _para("p-cap-fig4", "Figure 4. Experimental setup of the assay for gene Y expression.", page=4),
        _para(
            "p-results-6",
            "As shown in Figure 5, Treatment C was associated with increased expression of gene Z.",
            page=4,
        ),
        _para(
            "p-cap-fig5",
            "Figure 5. Association between Treatment C exposure and gene Z expression in patient samples.",
            page=5,
        ),
        _para(
            "p-results-8",
            "Treatment B significantly decreased body weight of mice compared with control (Table 1).",
            page=5,
        ),
        _para("p-cap-tab1", "Table 1. Relative body weight of mice in control and treatment groups.", page=6),
    ])
    discussion = _node("ch-discussion", "chapter", title="Discussion", level=1, children=[
        _para(
            "p-disc-1",
            "Our findings for Figure 2 suggest that Treatment A promotes gene X expression in cultured cells.",
            page=7,
        ),
    ])
    return _node("root", "document", title="Synthetic Paper", level=0, children=[intro, methods, results, discussion])


def write_document_artifact(root: DocNode, artifacts_root: Path, doc_id: str) -> Path:
    """Write the Phase 1 structure artifact the pipeline reads (read-only reuse)."""
    directory = Path(artifacts_root) / "structure" / doc_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "document.json"
    path.write_text(root.model_dump_json(indent=2), encoding="utf-8")
    return path


def build_paper_pdf(path) -> None:
    """A one-page synthetic English research paper with bold IMRaD headings.

    Layout note: the Phase 1 paragraph splitter only breaks between blocks when
    the previous line ends with CJK sentence punctuation or the new line is
    indented (ASCII "." is not in SENT_END_RE — the parser targets Chinese
    textbooks). Each synthetic paragraph therefore starts with an indented
    first line (x=92 vs body x=72) and continues at x=72, which the parser
    reliably splits on. This mirrors how the module behaves on real English
    papers whose captions sit in their own paragraphs.
    """
    doc = fitz.open()
    page = doc.new_page(width=A4_W, height=A4_H)
    y = 90

    def heading(text: str) -> None:
        nonlocal y
        page.insert_text((72, y), text, fontsize=14, fontname="hebo")
        y += 32

    def para(first: str, second: str) -> None:
        nonlocal y
        page.insert_text((92, y), first, fontsize=10.5, fontname="helv")  # indented paragraph start
        page.insert_text((72, y + 24), second, fontsize=10.5, fontname="helv")  # continuation line
        y += 60

    heading("Introduction")
    para("Gene X is a key regulator of cellular metabolism", "in eukaryotic cells and other model systems")
    heading("Materials and Methods")
    para("Cells were divided into control and treatment groups", "for this study of metabolic regulation")
    para("Cells in the treatment group received Treatment A", "for 24 hours before harvest")
    heading("Results")
    para(
        "Treatment A significantly increased expression of gene X",
        "compared with control (Figure 2)",
    )
    para("Figure 2. Relative expression of gene X", "in control and treatment groups")
    heading("Discussion")
    para("Our findings for Figure 2 suggest that Treatment A promotes", "gene X expression in cells")
    doc.save(str(path))
    doc.close()
