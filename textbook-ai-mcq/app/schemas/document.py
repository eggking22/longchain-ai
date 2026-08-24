"""Pydantic schemas for the parsed document tree and retrieval chunks.

Design borrows from Docling's DoclingDocument: a hierarchical tree where
every node carries provenance (page + bbox) so downstream phases can cite
the original textbook location.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Where a node came from in the source PDF (1-based page number)."""

    page_no: int
    bbox: tuple[float, float, float, float]


class DocNode(BaseModel):
    """A node of the Document -> Chapter -> Section -> Paragraph tree."""

    node_id: str
    node_type: Literal["document", "chapter", "section", "paragraph"]
    title: str = ""
    level: int  # document=0, chapter=1, section>=2
    text: str = ""  # paragraph content only
    children: list[DocNode] = Field(default_factory=list)
    provenance: Optional[Provenance] = None
    heading_rule: Optional[str] = None  # toc / font / numbering / spatial / synthetic / fallback
    heading_confidence: Optional[float] = None
    pages: list[int] = Field(default_factory=list)  # pages a paragraph spans


class Chunk(BaseModel):
    """A retrieval-oriented chunk produced by paragraph-aware chunking."""

    chunk_id: str
    text: str
    breadcrumb: list[str]  # chapter/section titles leading to this chunk
    pages: list[int]
    char_count: int
    paragraph_ids: list[str]  # back-reference to DocNode paragraphs


DocNode.model_rebuild()
