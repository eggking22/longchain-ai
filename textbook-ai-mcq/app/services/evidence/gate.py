"""EvidenceGate: the pass/refuse decision point before any generation.

Semantics (CRAG-style cheap-gate-first):
- Level 1 (heuristic) always runs. If it passes, the gate returns
  immediately — the LLM judge is never billed for clear passes.
- If Level 1 fails and an LLM judge is configured, Level 2 re-examines the
  same evidence semantically (it can rescue paraphrased queries whose
  vocabulary does not overlap the textbook wording) and its verdict is
  authoritative.
- If Level 2 is absent or errors, the Level 1 decision stands (degraded
  gracefully, error recorded in detail.llm_error).

evaluate() never raises; require() raises InsufficientEvidenceError on a
negative verdict so downstream generators physically cannot proceed on
insufficient evidence.
"""

from __future__ import annotations

from app.schemas.evidence import CoverageReport
from app.schemas.retrieval import RetrievalResult, RetrievedChunk

from .config import EvidenceConfig
from .evaluator import (
    EvidenceError,
    EvidenceEvaluator,
    HeuristicEvidenceEvaluator,
    LlmEvidenceEvaluator,
)

DEFAULT_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class InsufficientEvidenceError(RuntimeError):
    """Raised by EvidenceGate.require() when evidence cannot support a query."""

    def __init__(self, report: CoverageReport):
        self.report = report
        missing = "、".join(report.missing_information[:5]) or "—"
        super().__init__(
            f"INSUFFICIENT_EVIDENCE: query={report.query!r} "
            f"coverage={report.coverage_score} < threshold={report.threshold} "
            f"(level={report.level}, missing: {missing})"
        )


class EvidenceGate:
    def __init__(
        self,
        heuristic: HeuristicEvidenceEvaluator,
        llm: LlmEvidenceEvaluator | None = None,
        config: EvidenceConfig | None = None,
    ):
        self.heuristic = heuristic
        self.llm = llm
        self.config = config or heuristic.config

    @property
    def levels(self) -> str:
        return "heuristic+llm" if self.llm is not None else "heuristic"

    def evaluate(
        self, query: str, retrieval: RetrievalResult | list[RetrievedChunk]
    ) -> CoverageReport:
        hits = retrieval.hits if isinstance(retrieval, RetrievalResult) else list(retrieval)
        level1 = self.heuristic.evaluate(query, hits)

        if level1.sufficient:
            level1.detail["llm"] = "skipped: level-1 already above threshold"
            return level1

        if self.llm is None:
            return level1

        try:
            final = self.llm.evaluate(query, hits)
        except EvidenceError as exc:
            level1.detail["llm_error"] = str(exc)
            return level1

        final.level = "heuristic+llm"
        final.detail = {"heuristic": level1.detail, **final.detail}
        return final

    def require(
        self, query: str, retrieval: RetrievalResult | list[RetrievedChunk]
    ) -> CoverageReport:
        report = self.evaluate(query, retrieval)
        if not report.sufficient:
            raise InsufficientEvidenceError(report)
        return report


def build_evidence_gate(settings) -> EvidenceGate:
    """Factory: heuristic always; LLM judge only when LLM_* is configured."""
    config = EvidenceConfig.from_settings(settings)
    heuristic = HeuristicEvidenceEvaluator(config)
    llm = None
    if settings.LLM_API_KEY and settings.LLM_MODEL:
        llm = LlmEvidenceEvaluator(
            base_url=settings.LLM_BASE_URL or DEFAULT_LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            config=config,
        )
    return EvidenceGate(heuristic, llm, config)
