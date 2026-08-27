"""Knobs for Scientific Paper Figure Semantic Reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PaperSemanticsConfig:
    """Configuration for the paper_semantics pipeline.

    All knobs are advisory caps that bound evidence bundles; they never
    change reconstruction semantics, only how much text is considered.
    """

    max_evidence_per_figure: int = 12  # hard cap on the per-figure evidence bundle
    max_methods_paragraphs: int = 4  # top-N Methods paragraphs kept as supporting evidence
    max_continuation_evidence: int = 3  # continuation paragraphs stored per figure (context, not semantics)
    caption_max_chars: int = 400  # a caption-start paragraph longer than this is not treated as a caption
    retrieval_top_k: int = 5  # hits pulled per figure by the optional figure-aware retriever

    def __post_init__(self) -> None:
        if self.max_evidence_per_figure < 1:
            raise ValueError("max_evidence_per_figure must be >= 1")
        if self.max_methods_paragraphs < 0:
            raise ValueError("max_methods_paragraphs must be >= 0")
        if self.max_continuation_evidence < 0:
            raise ValueError("max_continuation_evidence must be >= 0")
        if self.caption_max_chars < 50:
            raise ValueError("caption_max_chars must be >= 50")
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be >= 1")

    @classmethod
    def from_settings(cls, settings) -> "PaperSemanticsConfig":
        return cls(
            max_evidence_per_figure=settings.PAPER_SEMANTICS_MAX_EVIDENCE_PER_FIGURE,
            max_methods_paragraphs=settings.PAPER_SEMANTICS_MAX_METHODS_PARAGRAPHS,
            caption_max_chars=settings.PAPER_SEMANTICS_CAPTION_MAX_CHARS,
        )

    def as_dict(self) -> dict:
        return asdict(self)
