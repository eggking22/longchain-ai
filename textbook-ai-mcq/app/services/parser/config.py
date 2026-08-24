"""Centralised, tunable knobs for the hierarchical PDF parser.

Every threshold that shapes parsing behaviour lives here so the pipeline
stays controllable: behaviour can be adjusted from .env without code
changes, and every decision is logged with the rule that made it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParserConfig:
    # --- cleaner ---
    header_footer_band: float = 0.09  # top/bottom page fraction treated as band
    repeat_ratio: float = 0.30  # fraction of pages a band text must repeat on

    # --- heading detection ---
    max_heading_levels: int = 4  # cap hierarchy depth (like pymupdf4llm max_levels)
    heading_max_chars: int = 60  # a heading line is never longer than this
    min_heading_confidence: float = 0.0  # drop candidates below this score

    # --- paragraph assembly ---
    indent_min_pt: float = 10.0  # x0 offset (pt) vs body margin => new paragraph

    # --- chunking ---
    chunk_target_chars: int = 600
    chunk_max_chars: int = 1200
    chunk_overlap_sentences: int = 1

    @classmethod
    def from_settings(cls, settings) -> "ParserConfig":
        return cls(
            header_footer_band=settings.PARSER_HEADER_FOOTER_BAND,
            repeat_ratio=settings.PARSER_REPEAT_RATIO,
            max_heading_levels=settings.PARSER_MAX_HEADING_LEVELS,
            chunk_target_chars=settings.PARSER_CHUNK_TARGET_CHARS,
            chunk_max_chars=settings.PARSER_CHUNK_MAX_CHARS,
            chunk_overlap_sentences=settings.PARSER_CHUNK_OVERLAP_SENTENCES,
        )
