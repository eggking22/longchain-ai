"""Question Generation — Statement Draft layer (MCQ step 1).

Pipeline (read-only reuse of Phase 5 and the semantic layer):

    Question Blueprint（每图每题型取首条）
        ↓ 真陈述（仅由蓝图绑定内容构造，evidence-bound）
        ↓ 十类受控扰动（最小编辑；替换素材全部来自论文证据池）
    StatementDraftSet（恰 1 真 + ≤4 假）
        ↓
    data/paper_semantics/{doc_id}/question_drafts.json

确定性核心；可选 LLM 对象抽取（LlmObjectExtractor，verbatim-span 门控）只在
DATA 类型标签回退时作为补丁生效。本阶段不做：中文翻译、A/B/C/D 排版、
MCQ Reviewer。
"""

from __future__ import annotations

from .config import DraftConfig
from .llm_object_extractor import LlmObjectExtractionError, LlmObjectExtractor
from .perturbations import (
    PERTURBATION_ORDER,
    PerturbationContext,
    build_true_statement,
    needs_object_extraction,
)
from .pipeline import generate_question_drafts, persist_drafts

__all__ = [
    "DraftConfig",
    "LlmObjectExtractionError",
    "LlmObjectExtractor",
    "PERTURBATION_ORDER",
    "PerturbationContext",
    "build_true_statement",
    "generate_question_drafts",
    "needs_object_extraction",
    "persist_drafts",
]
