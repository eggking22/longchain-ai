"""Chinese MCQ statement translation (deterministic registry + templates).

    Statement Draft (English, untouched artifact)
        ↓ terminology registry + statement templates
    mcq_drafts_zh.json（statement_zh 增量字段；数值/基因/方向/真假/证据全部原样）

本阶段不做：L2 LLM、A/B/C/D 排版、自动 Reviewer、任何科学性再判断。
"""

from __future__ import annotations

from .pipeline import load_drafts, persist_mcq_zh, translate_document, translate_drafts
from .terminology import TERMINOLOGY, translate_entity
from .translator import translate_statement

__all__ = [
    "TERMINOLOGY",
    "load_drafts",
    "persist_mcq_zh",
    "translate_document",
    "translate_drafts",
    "translate_entity",
    "translate_statement",
]
