"""Unit tests for EvidenceGate combination / degradation / refusal semantics."""

import pytest

from app.schemas.evidence import CoverageReport
from app.schemas.retrieval import RetrievedChunk
from app.services.evidence.config import EvidenceConfig
from app.services.evidence.evaluator import EvidenceError, HeuristicEvidenceEvaluator
from app.services.evidence.gate import (
    EvidenceGate,
    InsufficientEvidenceError,
    build_evidence_gate,
)


def _hit(text, sparse_score):
    return RetrievedChunk(
        document_id="doc",
        chunk_id="c1",
        chunk_index=0,
        text=text,
        breadcrumb=["第1章"],
        pages=[1],
        char_count=len(text),
        sparse_score=sparse_score,
        fused_score=0.0164,
        rank=1,
        sources=["sparse"],
    )


PASS_HITS = [
    _hit("糖酵解发生在细胞质基质中，不需要氧参与。", 13.0),
    _hit("糖酵解将葡萄糖分解为丙酮酸。", 8.0),
    _hit("糖酵解是细胞呼吸的第一阶段。", 5.0),
]
PASS_QUERY = "糖酵解在哪里发生"
FAIL_QUERY = "诺贝尔生理学奖的影响"
FAIL_HITS = [_hit("糖酵解在细胞质基质中进行。", 10.0)]


class _FakeLlm:
    """Duck-typed stand-in for LlmEvidenceEvaluator in gate-level tests."""

    name = "llm"

    def __init__(self, report=None, error=None):
        self.report = report
        self.error = error
        self.calls = 0

    def evaluate(self, query, hits):
        self.calls += 1
        if self.error:
            raise self.error
        report = self.report
        return CoverageReport(
            query=query,
            sufficient=report["sufficient"],
            coverage_score=report["coverage_score"],
            threshold=0.8,
            evidence=[],
            missing_information=report["missing_information"],
            level="llm",
            detail={"reasoning": "fake"},
        )


def _gate(llm=None):
    return EvidenceGate(HeuristicEvidenceEvaluator(EvidenceConfig()), llm)


def test_level1_pass_skips_llm():
    llm = _FakeLlm(report={"sufficient": False, "coverage_score": 0.1, "missing_information": ["x"]})
    report = _gate(llm).evaluate(PASS_QUERY, PASS_HITS)
    assert report.sufficient is True
    assert report.level == "heuristic"
    assert llm.calls == 0
    assert "skipped" in report.detail["llm"]


def test_level1_fail_llm_rescues():
    llm = _FakeLlm(report={"sufficient": True, "coverage_score": 0.9, "missing_information": []})
    report = _gate(llm).evaluate(FAIL_QUERY, FAIL_HITS)
    assert report.sufficient is True
    assert report.level == "heuristic+llm"
    assert llm.calls == 1
    assert report.detail["heuristic"]["union_coverage"] < 1.0  # L1 diagnostics kept


def test_level1_fail_llm_confirms_rejection():
    llm = _FakeLlm(report={"sufficient": False, "coverage_score": 0.3, "missing_information": ["奖项信息"]})
    report = _gate(llm).evaluate(FAIL_QUERY, FAIL_HITS)
    assert report.sufficient is False
    assert report.level == "heuristic+llm"
    assert report.missing_information == ["奖项信息"]


def test_level1_fail_llm_error_degrades():
    llm = _FakeLlm(error=EvidenceError("judge API failed"))
    report = _gate(llm).evaluate(FAIL_QUERY, FAIL_HITS)
    assert report.sufficient is False
    assert report.level == "heuristic"
    assert "judge API failed" in report.detail["llm_error"]


def test_no_llm_configured_pure_level1():
    report = _gate(None).evaluate(FAIL_QUERY, FAIL_HITS)
    assert report.sufficient is False
    assert report.level == "heuristic"
    assert "诺贝尔" in " ".join(report.missing_information)


def test_require_raises_on_insufficient():
    gate = _gate(None)
    with pytest.raises(InsufficientEvidenceError) as excinfo:
        gate.require(FAIL_QUERY, FAIL_HITS)
    assert excinfo.value.report.sufficient is False
    assert "INSUFFICIENT_EVIDENCE" in str(excinfo.value)


def test_require_returns_report_on_pass():
    report = _gate(None).require(PASS_QUERY, PASS_HITS)
    assert report.sufficient is True


def test_accepts_retrieval_result(tmp_path):
    from app.schemas.retrieval import IndexManifest, RetrievalResult

    result = RetrievalResult(
        query=PASS_QUERY,
        mode="hybrid",
        top_k=5,
        hits=PASS_HITS,
        manifest=IndexManifest(
            doc_id="doc",
            embedder="hash",
            embedding_model="hash",
            embedding_dim=16,
            config_hash="x",
            chunk_set_hash="y",
            num_chunks=3,
            created_at="2026-01-01T00:00:00",
        ),
        latency_ms=1.0,
    )
    assert _gate(None).evaluate(PASS_QUERY, result).sufficient is True


class TestFactory:
    def test_no_llm_settings(self, monkeypatch):
        from app.core.config import Settings

        gate = build_evidence_gate(Settings(LLM_API_KEY="", LLM_MODEL=""))
        assert gate.llm is None
        assert gate.levels == "heuristic"

    def test_llm_settings(self, monkeypatch):
        from app.core.config import Settings

        gate = build_evidence_gate(
            Settings(LLM_API_KEY="sk", LLM_MODEL="glm-4", LLM_BASE_URL="")
        )
        assert gate.llm is not None
        assert gate.llm.base_url == "https://open.bigmodel.cn/api/paas/v4"  # default
        assert gate.levels == "heuristic+llm"
