"""Unit tests for evidence evaluators: L1 exact math, L2 judge via MockTransport."""

import json

import httpx
import pytest

from app.schemas.evidence import CoverageReport
from app.schemas.retrieval import RetrievedChunk
from app.services.evidence.config import EvidenceConfig
from app.services.evidence.evaluator import (
    EvidenceError,
    HeuristicEvidenceEvaluator,
    LlmEvidenceEvaluator,
    informative_tokens,
)


def _hit(chunk_id, text, sparse_score=None, rank=1):
    return RetrievedChunk(
        document_id="doc",
        chunk_id=chunk_id,
        chunk_index=rank - 1,
        text=text,
        breadcrumb=["第1章", "第1节"],
        pages=[rank],
        char_count=len(text),
        sparse_score=sparse_score,
        fused_score=0.0164,
        rank=rank,
        sources=["sparse"] if sparse_score else ["dense"],
    )


class TestInformativeTokens:
    def test_interrogatives_and_function_words_dropped(self):
        assert informative_tokens("糖酵解在哪里发生？") == ["糖酵解", "发生"]

    def test_single_char_tokens_dropped(self):
        assert informative_tokens("血 压 的 形成") == ["形成"]

    def test_dedup(self):
        assert informative_tokens("糖酵解 糖酵解 过程") == ["糖酵解", "过程"]

    def test_empty(self):
        assert informative_tokens("是什么？为什么？") == []


class TestHeuristic:
    def test_exact_formula(self):
        cfg = EvidenceConfig()
        evaluator = HeuristicEvidenceEvaluator(cfg)
        hits = [
            _hit("a", "线粒体是有氧呼吸的主要场所。", sparse_score=12.0, rank=1),
            _hit("b", "细胞呼吸释放能量。", sparse_score=6.0, rank=2),
            _hit("c", "呼吸作用产生 ATP。", sparse_score=3.0, rank=3),
        ]
        # nothing in the evidence window matches the query vocabulary
        report = evaluator.evaluate("百慕大三角区", hits)
        assert report.coverage_score == pytest.approx(0.15 * 1.0 + 0.1 * 1.0)  # 0.25
        assert report.missing_information == informative_tokens("百慕大三角区")

    def test_full_coverage_passes(self):
        evaluator = HeuristicEvidenceEvaluator(EvidenceConfig())
        hits = [
            _hit("a", "糖酵解发生在细胞质基质中，不需要氧参与。", sparse_score=13.0),
            _hit("b", "糖酵解将葡萄糖分解为丙酮酸。", sparse_score=8.0, rank=2),
            _hit("c", "糖酵解是呼吸作用的第一阶段。", sparse_score=5.0, rank=3),
        ]
        report = evaluator.evaluate("糖酵解在哪里发生", hits)
        assert report.coverage_score == pytest.approx(0.75 * 1.0 + 0.15 * 1.0 + 0.1 * 1.0)
        assert report.sufficient is True
        assert report.missing_information == []
        assert report.evidence[0].chunk_id == "a"
        assert report.evidence[0].breadcrumb == ["第1章", "第1节"]

    def test_partial_coverage_fails_with_missing_terms(self):
        evaluator = HeuristicEvidenceEvaluator(EvidenceConfig())
        hits = [
            _hit("a", "糖酵解在细胞质基质中进行。", sparse_score=12.0),
            _hit("b", "糖酵解产生少量 ATP。", sparse_score=6.0, rank=2),
            _hit("c", "糖酵解不需要氧气参与。", sparse_score=3.0, rank=3),
        ]
        report = evaluator.evaluate("2025 年诺贝尔生理学奖对糖酵解研究的影响", hits)
        assert report.sufficient is False
        assert "诺贝尔" in report.missing_information
        assert "2025" in report.missing_information
        assert report.coverage_score < report.threshold

    def test_no_hits(self):
        report = HeuristicEvidenceEvaluator().evaluate("糖酵解过程", [])
        assert report.sufficient is False
        assert report.coverage_score == 0.0
        assert report.missing_information == ["糖酵解", "过程"]

    def test_query_without_substance(self):
        report = HeuristicEvidenceEvaluator().evaluate("是什么", [])
        assert report.sufficient is False
        assert report.missing_information == ["（问题中不含可检索的实质内容词）"]

    def test_evidence_window_truncates(self):
        cfg = EvidenceConfig(evidence_window=1, coverage_pool=1)
        evaluator = HeuristicEvidenceEvaluator(cfg)
        hits = [
            _hit("a", "完全无关的内容。", sparse_score=12.0),
            _hit("b", "红细胞运输氧气。", sparse_score=1.0, rank=2),  # outside pool
        ]
        report = evaluator.evaluate("红细胞运输氧气", hits)
        # the only chunk in the pool misses both terms; it alone clears the floor
        assert report.coverage_score == pytest.approx(0.15 * 1.0 + 0.1 * (1 / 3), abs=1e-4)
        assert report.missing_information == ["红细胞", "运输", "氧气"]

    def test_synonym_equivalents_cover_symbol_notation(self):
        evaluator = HeuristicEvidenceEvaluator(EvidenceConfig())
        hits = [
            _hit("a", "血液中绝大部分 O2 与血红蛋白结合，运输到组织。", sparse_score=12.0),
            _hit("b", "CO2 以碳酸氢盐形式运输。", sparse_score=6.0, rank=2),
            _hit("c", "气体运输依赖血液中的红细胞。", sparse_score=3.0, rank=3),
        ]
        report = evaluator.evaluate("氧气在血液中如何运输", hits)
        # 氧气 covered via the o2 equivalent even though the word never appears
        assert "氧气" not in report.missing_information
        assert report.detail["union_coverage"] == pytest.approx(1.0, abs=1e-4)

    def test_scattered_terms_focus_penalty(self):
        # each chunk matches ONE query term; union covers all but focus is low
        evaluator = HeuristicEvidenceEvaluator(EvidenceConfig())
        hits = [
            _hit("a", "人工智能在教育中的应用。", sparse_score=12.0),
            _hit("b", "心电图导联体系的创立。", sparse_score=9.0, rank=2),
            _hit("c", "临床诊断依赖生理学参考值。", sparse_score=6.0, rank=3),
        ]
        report = evaluator.evaluate("人工智能在心电图诊断中的应用现状", hits)
        assert report.sufficient is False
        assert report.detail["union_coverage"] >= 0.8  # scattered hits cover terms
        assert report.detail["focus_coverage"] <= 0.4  # but no chunk addresses the query

    def test_pool_must_not_be_smaller_than_window(self):
        with pytest.raises(ValueError, match="coverage_pool"):
            EvidenceConfig(evidence_window=5, coverage_pool=3)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            EvidenceConfig(term_weight=0.5, sparse_weight=0.2, hit_weight=0.1)


def _judge_handler(body_fn=None, status=200):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if body_fn is not None:
            content = body_fn(len(calls))
        else:
            content = json.dumps(
                {
                    "sufficient": False,
                    "coverage_score": 0.3,
                    "missing_information": ["X 的作用机制"],
                    "reasoning": "证据未覆盖 X",
                },
                ensure_ascii=False,
            )
        return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})

    return handler, calls


def _llm(handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    defaults = dict(
        base_url="https://api.test/v4",
        api_key="sk-test",
        model="glm-4",
        config=EvidenceConfig(),
        max_attempts=2,
        client=client,
    )
    defaults.update(kw)
    return LlmEvidenceEvaluator(**defaults)


class TestLlmEvaluator:
    def test_happy_path_and_prompt_content(self):
        handler, calls = _judge_handler()
        evaluator = _llm(handler)
        hits = [_hit("a", "糖酵解在细胞质基质中进行。", 12.0)]
        report = evaluator.evaluate("糖酵解过程", hits)
        assert isinstance(report, CoverageReport)
        assert report.sufficient is False
        assert report.missing_information == ["X 的作用机制"]
        assert report.level == "llm"
        (payload,) = calls
        assert payload["model"] == "glm-4"
        assert payload["temperature"] == 0.0
        assert "糖酵解过程" in payload["messages"][1]["content"]
        assert "第1章" in payload["messages"][1]["content"]  # provenance included

    def test_fenced_json_parsed(self):
        def body(_):
            return (
                '```json\n{"sufficient": true, "coverage_score": 1.0, '
                '"missing_information": [], "reasoning": "ok"}\n```'
            )

        handler, _ = _judge_handler(body)
        report = _llm(handler).evaluate("q", [_hit("a", "文本", 5.0)])
        assert report.sufficient is True

    def test_coverage_clamped(self):
        def body(_):
            return '{"sufficient": false, "coverage_score": 7.3, "missing_information": [], "reasoning": ""}'

        handler, _ = _judge_handler(body)
        report = _llm(handler).evaluate("q", [])
        assert report.coverage_score == 1.0

    def test_malformed_then_valid_reasks(self):
        def body(n):
            if n == 1:
                return "抱歉，我无法输出 JSON。"
            return '{"sufficient": false, "coverage_score": 0.2, "missing_information": ["a"], "reasoning": ""}'

        handler, calls = _judge_handler(body)
        report = _llm(handler).evaluate("q", [_hit("a", "文本", 5.0)])
        assert report.coverage_score == 0.2
        assert len(calls) == 2

    def test_malformed_twice_raises(self):
        def body(_):
            return "不是 JSON"

        handler, calls = _judge_handler(body)
        with pytest.raises(EvidenceError, match="no JSON object"):
            _llm(handler, max_attempts=1).evaluate("q", [])
        assert len(calls) == 2  # one re-ask

    def test_client_error_fatal(self):
        handler, calls = _judge_handler(status=401)
        with pytest.raises(EvidenceError, match="401"):
            _llm(handler).evaluate("q", [])
        assert len(calls) == 1

    def test_server_error_retried_then_raises(self):
        handler, calls = _judge_handler(status=500)
        with pytest.raises(EvidenceError, match="after 2 attempts"):
            _llm(handler).evaluate("q", [])
        assert len(calls) == 2
