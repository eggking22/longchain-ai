"""Chinese MCQ statement translation (deterministic registry + templates).

    Statement Draft (English, untouched artifact)
        ↓ terminology registry + statement templates
    mcq_drafts_zh.json（statement_zh 增量字段；数值/基因/方向/真假/证据全部原样）

可选 LLM 整句翻译层（LlmStatementTranslator）：仅当注入时生效，每句译文必须通过
确定性不变量门（数值/图锚点/方向词/基因试剂名逐字保留），任一失败回退确定性翻译。
不做：A/B/C/D 排版、自动 Reviewer、任何科学性再判断。
"""

from __future__ import annotations

from .llm_translator import LlmStatementTranslator, LlmTranslationError
from .pipeline import load_drafts, persist_mcq_zh, translate_document, translate_drafts
from .terminology import TERMINOLOGY, translate_entity
from .translator import translate_statement

__all__ = [
    "LlmStatementTranslator",
    "LlmTranslationError",
    "TERMINOLOGY",
    "load_drafts",
    "persist_mcq_zh",
    "translate_document",
    "translate_drafts",
    "translate_entity",
    "translate_statement",
]
