"""Shared Chinese/mixed tokenizer for indexing AND retrieval.

The same function must tokenize corpus and query, otherwise BM25 term
statistics are computed over a different vocabulary. Tokens are jieba
segments, lowercased, with punctuation/whitespace-only segments dropped.
"""

from __future__ import annotations

import logging

import jieba

# Silence "Building prefix dict..." on first use (it dumps to stderr).
jieba.setLogLevel(logging.ERROR)


def tokenize(text: str) -> list[str]:
    """Segment text into lowercase tokens usable by BM25.

    Keeps any segment containing at least one alphanumeric character
    (Chinese characters are alnum), so pure punctuation like "。" or
    "、" disappears while "细胞" or "atp" survives.
    """
    return [t.lower() for t in jieba.lcut(text) if any(ch.isalnum() for ch in t)]
