"""Evidence coverage evaluators (Level 1 heuristic, Level 2 LLM judge).

Level 1 (HeuristicEvidenceEvaluator) is deterministic, provider-independent
and free: it scores how much of the query's informative vocabulary the
retrieved evidence actually covers, plus two saturating retrieval-strength
signals. Research consensus behind the design: term-coverage ratios are the
most stable LLM-free sufficiency signal, while raw BM25/cosine cutoffs
drift with IDF and query length (see docs/evidence.md).

Level 2 (LlmEvidenceEvaluator) asks an OpenAI-compatible chat model to
decompose the query into required facts and verdict each against the
evidence (RAGAS context-recall pattern), which counteracts the documented
over-generosity of LLM judges.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

import httpx

from app.schemas.evidence import CoverageReport, LlmCoverageVerdict
from app.schemas.retrieval import RetrievedChunk

from .config import EvidenceConfig
from .stopwords import STOPWORDS_ZH, SYNONYMS

# a term is covered when the evidence contains it OR any listed equivalent;
# keys and values are lowercase tokens (the shared tokenizer lowercases)
TOKEN_EQUIVALENTS: dict[str, frozenset[str]] = {
    token: frozenset({token, *alts}) for token, alts in SYNONYMS.items()
}


def _equivalents(token: str) -> frozenset[str]:
    return TOKEN_EQUIVALENTS.get(token, frozenset({token}))


def informative_tokens(query: str) -> list[str]:
    """Query tokens that carry topical signal (lowercased, dedup-keeping-order).

    Drops stopwords, single characters and pure punctuation; empty when the
    query contains nothing substantive.
    """
    from app.services.retrieval.tokenizer import tokenize

    seen: set[str] = set()
    out: list[str] = []
    for token in tokenize(query):
        if len(token) < 2 or token in STOPWORDS_ZH or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


@runtime_checkable
class EvidenceEvaluator(Protocol):
    name: str

    def evaluate(self, query: str, hits: list[RetrievedChunk]) -> CoverageReport: ...


class HeuristicEvidenceEvaluator:
    """Level 1: term coverage (union + focus) + saturating retrieval signals.

    Union coverage alone is too lenient: scattered one-term matches across
    unrelated chunks can "cover" a query no single chunk addresses. The
    focus factor requires the best single chunk to cover a good share of
    the query vocabulary, which separates on-topic evidence from scatter.
    """

    name = "heuristic"

    def __init__(self, config: EvidenceConfig | None = None):
        self.config = config or EvidenceConfig()

    def evaluate(self, query: str, hits: list[RetrievedChunk]) -> CoverageReport:
        from app.services.retrieval.tokenizer import tokenize

        cfg = self.config
        window = hits[: cfg.evidence_window]  # cited evidence
        pool = hits[: cfg.coverage_pool]  # deeper pool inspected for coverage
        terms = informative_tokens(query)

        pool_token_sets = [set(tokenize(hit.text)) for hit in pool]
        union_tokens: set[str] = set().union(*pool_token_sets) if pool_token_sets else set()

        covered_flags = [
            bool(_equivalents(t) & union_tokens) for t in terms
        ] if terms else []
        union_coverage = sum(covered_flags) / len(terms) if terms else 0.0

        focus = 0.0
        if terms and pool_token_sets:
            focus = max(
                sum(1 for t in terms if _equivalents(t) & chunk_tokens) / len(terms)
                for chunk_tokens in pool_token_sets
            )
        term_signal = 0.5 * union_coverage + 0.5 * focus

        missing = [t for t, covered in zip(terms, covered_flags) if not covered]
        sparse_scores = [h.sparse_score for h in pool if h.sparse_score is not None]
        top_sparse = max(sparse_scores) if sparse_scores else 0.0
        sparse_strength = min(1.0, top_sparse / cfg.sparse_saturate) if cfg.sparse_saturate > 0 else 0.0
        hits_above_floor = sum(1 for s in sparse_scores if s >= cfg.hit_floor)
        hit_depth = min(1.0, hits_above_floor / 3)

        coverage_score = round(
            cfg.term_weight * term_signal
            + cfg.sparse_weight * sparse_strength
            + cfg.hit_weight * hit_depth,
            4,
        )
        sufficient = coverage_score >= cfg.coverage_threshold
        if not terms:
            sufficient = False
            missing_info = ["（问题中不含可检索的实质内容词）"]
        else:
            missing_info = missing

        return CoverageReport(
            query=query,
            sufficient=sufficient,
            coverage_score=coverage_score,
            threshold=cfg.coverage_threshold,
            evidence=_evidence_items(window),
            missing_information=missing_info,
            level=self.name,
            detail={
                "union_coverage": round(union_coverage, 4),
                "focus_coverage": round(focus, 4),
                "term_signal": round(term_signal, 4),
                "query_terms": terms,
                "sparse_strength": round(sparse_strength, 4),
                "top_sparse_score": round(top_sparse, 4),
                "hit_depth": round(hit_depth, 4),
                "hits_above_floor": hits_above_floor,
                "window_size": len(window),
                "pool_size": len(pool),
                "config": cfg.as_dict(),
            },
        )


def _evidence_items(window: list[RetrievedChunk]) -> list:
    from app.schemas.evidence import EvidenceItem

    return [
        EvidenceItem(
            chunk_id=hit.chunk_id,
            score=hit.fused_score if hit.fused_score is not None else 0.0,
            rank=hit.rank,
            document_id=hit.document_id,
            breadcrumb=hit.breadcrumb,
            pages=hit.pages,
            text_preview=hit.text[:80],
        )
        for hit in window
    ]


SYSTEM_PROMPT = (
    "你是教材检索证据的充分性评审员。给定一个问题和若干从教材中检索到的证据片段，"
    "你必须判断：仅依据这些证据，是否足以回答该问题。\n"
    "评审步骤（必须执行）：\n"
    "1. 先把问题拆解为回答它所必需的关键事实点；\n"
    "2. 逐条核对每个事实点是否在证据片段中出现；\n"
    "3. 统计被证据覆盖的事实点比例，作为 coverage_score；\n"
    "4. 缺失的事实点用简短中文短语列入 missing_information。\n"
    "严格标准：只有当证据包含回答问题所需的全部关键信息时才 sufficient=true；"
    "证据只覆盖部分、含糊或与之无关时必须 sufficient=false；不确定时判 false。"
)

_USER_TEMPLATE = (
    "【问题】\n{query}\n\n【证据片段】\n{chunks}\n\n"
    "请按系统指令评审，并只输出一个 JSON 对象（不要输出其他文字）：\n"
    '{{"sufficient": true/false, "coverage_score": 0.0-1.0, '
    '"missing_information": ["缺失事实点", ...], "reasoning": "简要理由"}}'
)


class EvidenceError(RuntimeError):
    """Raised when the Level-2 judge cannot produce a usable verdict."""


class LlmEvidenceEvaluator:
    """Level 2: OpenAI-compatible chat model as a strict coverage judge."""

    name = "llm"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        config: EvidenceConfig | None = None,
        timeout: float = 60.0,
        max_attempts: int = 2,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.config = config or EvidenceConfig()
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self._client = client

    def _format_chunks(self, hits: list[RetrievedChunk]) -> str:
        limit = self.config.llm_max_chars_per_chunk
        lines = []
        for hit in hits[: self.config.evidence_window]:
            source = " / ".join(hit.breadcrumb[:2]) or "（无出处）"
            pages = f"p{hit.pages[0]}" if hit.pages else "p?"
            text = hit.text[:limit] + ("…" if len(hit.text) > limit else "")
            lines.append(f"[{hit.rank}] （{source}，{pages}）{text}")
        return "\n".join(lines) if lines else "（无检索结果）"

    def _parse_verdict(self, content: str) -> LlmCoverageVerdict:
        stripped = re.sub(r"```(?:json)?", "", content).strip()
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise EvidenceError(f"no JSON object in judge reply: {content[:120]!r}")
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"malformed judge JSON: {exc}") from exc
        verdict = LlmCoverageVerdict.model_validate(payload)
        verdict.coverage_score = min(1.0, max(0.0, verdict.coverage_score))
        return verdict

    def _chat(self, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            client = self._client or httpx.Client(timeout=self.timeout)
            try:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code in (429,) or response.status_code >= 500:
                    last_error = EvidenceError(
                        f"judge API returned {response.status_code}: {response.text[:200]}"
                    )
                elif response.status_code != 200:
                    raise EvidenceError(  # client errors are deterministic, not retryable
                        f"judge API returned {response.status_code}: {response.text[:200]}"
                    )
                else:
                    return response.json()["choices"][0]["message"]["content"]
            except httpx.HTTPError as exc:  # network / timeout errors
                last_error = exc
            finally:
                if self._client is None:
                    client.close()
        raise EvidenceError(f"judge API failed after {self.max_attempts} attempts: {last_error}")

    def evaluate(self, query: str, hits: list[RetrievedChunk]) -> CoverageReport:
        user_prompt = _USER_TEMPLATE.format(query=query, chunks=self._format_chunks(hits))
        content = self._chat(user_prompt)
        try:
            verdict = self._parse_verdict(content)
        except EvidenceError:
            # one re-ask for malformed JSON (model sometimes adds prose around it)
            verdict = self._parse_verdict(self._chat(user_prompt))
        window = hits[: self.config.evidence_window]
        return CoverageReport(
            query=query,
            sufficient=verdict.sufficient,
            coverage_score=round(verdict.coverage_score, 4),
            threshold=self.config.coverage_threshold,
            evidence=_evidence_items(window),
            missing_information=verdict.missing_information,
            level=self.name,
            detail={"reasoning": verdict.reasoning, "model": self.model},
        )
