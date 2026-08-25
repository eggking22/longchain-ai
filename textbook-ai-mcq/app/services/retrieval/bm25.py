"""Lexical retrieval leg: rank-bm25 BM25Okapi over pre-tokenized records.

Tokens were produced by the shared tokenizer at index time and stored in
records.jsonl, so query and corpus always share one vocabulary. Rebuilding
BM25Okapi from tokens for ~2k chunks takes well under a second, which is
why nothing is pickled.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas.retrieval import IndexedRecord


class Bm25Index:
    def __init__(self, ids: list[str], corpus_tokens: list[list[str]]):
        self._ids = ids
        self._bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    def __len__(self) -> int:
        return len(self._ids)

    @classmethod
    def build(cls, records: list[IndexedRecord]) -> "Bm25Index":
        return cls([r.chunk_id for r in records], [r.tokens for r in records])

    def search(self, query_tokens: list[str], k: int) -> list[tuple[str, float]]:
        """Top-k (chunk_id, bm25_score); zero-score hits are omitted."""
        if self._bm25 is None or not query_tokens or k <= 0:
            return []
        scores = self._bm25.get_scores(query_tokens)
        k = min(k, len(self._ids))
        order = np.argsort(-scores, kind="stable")[:k]
        hits = [(self._ids[i], float(scores[i])) for i in order if scores[i] > 0]
        return hits[:k]
