"""LLM object extractor: offline unit tests + pipeline integration.

All HTTP is mocked with httpx.MockTransport (same seam as Phase 3/4 LLM tests)
— the suite stays fully offline. The extractor is a patcher over the DATA
kind-label fallback only; every result must be a verbatim span of the evidence
texts or the deterministic fallback stands.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.services.question_generation import (
    DraftConfig,
    LlmObjectExtractionError,
    LlmObjectExtractor,
    generate_question_drafts,
)

from .conftest import _para, build_paper_tree, write_document_artifact

BASE = "https://llm.test/v1"
SENTENCE = "a, GFP intensity in DCs treated with the cPLA2 inhibitor AACOF3 (25 µM) or control."
SPAN = "the cPLA2 inhibitor AACOF3"


def _content(payload) -> bytes:
    return json.dumps({"choices": [{"message": {"content": json.dumps(payload)}}]}).encode()


def _raw_content(message: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": message}}]}).encode()


def _extractor(handler):
    return LlmObjectExtractor(
        base_url=BASE,
        api_key="test-key",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class TestExtractorUnits:
    def test_returns_verbatim_span(self):
        extractor = _extractor(lambda request: httpx.Response(200, content=_content({"object": SPAN})))
        assert extractor.extract("25 µM", "concentration", [SENTENCE]) == SPAN

    def test_thin_space_text_normalized_before_span_check(self):
        thin = "treated with the cPLA2 inhibitor AACOF3 (25\u2009µM) or control."
        extractor = _extractor(lambda request: httpx.Response(200, content=_content({"object": SPAN})))
        assert extractor.extract("25 µM", "concentration", [thin]) == SPAN

    def test_rejects_non_verbatim_phrase(self):
        extractor = _extractor(lambda request: httpx.Response(200, content=_content({"object": "the AACOF3 treatment"})))
        assert extractor.extract("25 µM", "concentration", [SENTENCE]) is None

    def test_rejects_empty_and_overlong(self):
        empty = _extractor(lambda request: httpx.Response(200, content=_content({"object": ""})))
        assert empty.extract("25 µM", "concentration", [SENTENCE]) is None
        overlong = _extractor(lambda request: httpx.Response(200, content=_content({"object": "word " * 40})))
        assert overlong.extract("25 µM", "concentration", [SENTENCE]) is None

    def test_strips_padding_punctuation(self):
        extractor = _extractor(lambda request: httpx.Response(200, content=_content({"object": f"{SPAN}."})))
        assert extractor.extract("25 µM", "concentration", [SENTENCE]) == SPAN

    def test_markdown_fenced_json_accepted(self):
        fenced = f"```json\n{json.dumps({'object': SPAN})}\n```"
        raw = json.dumps({"choices": [{"message": {"content": fenced}}]}).encode()
        extractor = _extractor(lambda request: httpx.Response(200, content=raw))
        assert extractor.extract("25 µM", "concentration", [SENTENCE]) == SPAN

    def test_invalid_json_reasked_then_ok(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(200, content=_raw_content("not json at all"))
            return httpx.Response(200, content=_content({"object": SPAN}))

        extractor = _extractor(handler)
        assert extractor.extract("25 µM", "concentration", [SENTENCE]) == SPAN
        assert len(calls) == 2

    def test_persistent_invalid_json_raises(self):
        extractor = _extractor(lambda request: httpx.Response(200, content=_raw_content("still not json")))
        with pytest.raises(LlmObjectExtractionError):
            extractor.extract("25 µM", "concentration", [SENTENCE])

    def test_http_error_raises(self):
        extractor = _extractor(lambda request: httpx.Response(500))
        with pytest.raises(LlmObjectExtractionError):
            extractor.extract("25 µM", "concentration", [SENTENCE])

    def test_prompt_shape_and_temperature_zero(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, content=_content({"object": SPAN}))

        extractor = _extractor(handler)
        extractor.extract("25 µM", "concentration", [SENTENCE])
        assert seen["body"]["temperature"] == 0.0
        user = seen["body"]["messages"][1]["content"]
        assert "25 µM" in user and "concentration" in user and SENTENCE in user


class _FakeExtractor:
    """Duck-typed stand-in for pipeline-level tests (validation already unit-tested)."""

    def __init__(self, phrase=None, error=False):
        self.phrase = phrase
        self.error = error
        self.calls = []

    def extract(self, value, kind, texts):
        self.calls.append({"value": value, "kind": kind, "texts": list(texts)})
        if self.error:
            raise LlmObjectExtractionError("boom")
        return self.phrase


@pytest.fixture(scope="module")
def fallback_corpus(tmp_path_factory):
    """Figure 6's first percentage lives in a <5-word fragment sentence → fallback path."""
    root = tmp_path_factory.mktemp("obj-extract")
    tree = build_paper_tree()
    results = next(c for c in tree.children if c.title == "Results")
    results.children.append(
        _para("p-results-9", "Treatment D reduced migration distance. The reduction reached 30% (Figure 6).", page=6)
    )
    results.children.append(
        _para("p-results-10", "Treatment E reduced migration distance by 50% (Figure 6).", page=6)
    )
    results.children.append(_para("p-cap-fig6", "Figure 6. Migration distance of cells after treatment.", page=6))
    write_document_artifact(tree, root, "obj-paper")
    return root


def _true_data_statement(report):
    data_sets = [s for s in report.draft_sets if s.question_type == "DATA_STATEMENT"]
    assert data_sets, "corpus must produce a DATA set"
    return next(st for s in data_sets for st in s.statements if st.is_correct)


class TestPipelineIntegration:
    def test_extractor_upgrades_fallback(self, fallback_corpus):
        fake = _FakeExtractor(phrase="migration distance")
        report = generate_question_drafts("obj-paper", fallback_corpus, config=DraftConfig(), object_extractor=fake)
        true = _true_data_statement(report)
        assert true.statement == (
            "According to Figure 6, the reported percentage for migration distance is 30%."
        )
        assert true.detail["object"] == "migration distance"
        assert true.detail["object_extraction"] == "llm"
        assert report.summary["method"] == "deterministic+llm"
        assert report.summary["object_extraction"] == {"extracted": 1, "rejected": 0, "errors": 0}

    def test_without_extractor_unchanged(self, fallback_corpus):
        report = generate_question_drafts("obj-paper", fallback_corpus, config=DraftConfig())
        true = _true_data_statement(report)
        assert true.statement == "According to Figure 6, the reported percentage is 30%."
        assert report.summary["method"] == "deterministic"
        assert "object_extraction" not in report.summary
        assert "object" not in true.detail

    def test_rejection_keeps_fallback(self, fallback_corpus):
        fake = _FakeExtractor(phrase=None)
        report = generate_question_drafts("obj-paper", fallback_corpus, config=DraftConfig(), object_extractor=fake)
        true = _true_data_statement(report)
        assert true.statement == "According to Figure 6, the reported percentage is 30%."
        assert true.detail["object_extraction"] == "llm_rejected"
        assert report.summary["method"] == "deterministic"
        assert report.summary["object_extraction"] == {"extracted": 0, "rejected": 1, "errors": 0}

    def test_error_keeps_fallback(self, fallback_corpus):
        fake = _FakeExtractor(error=True)
        report = generate_question_drafts("obj-paper", fallback_corpus, config=DraftConfig(), object_extractor=fake)
        true = _true_data_statement(report)
        assert true.statement == "According to Figure 6, the reported percentage is 30%."
        assert true.detail["object_extraction"] == "llm_error"
        assert report.summary["object_extraction"]["errors"] == 1

    def test_extraction_texts_carry_the_evidence(self, fallback_corpus):
        fake = _FakeExtractor(phrase="migration distance")
        generate_question_drafts("obj-paper", fallback_corpus, config=DraftConfig(), object_extractor=fake)
        assert len(fake.calls) == 1  # one fallback blueprint only — quotable ones are never asked
        assert fake.calls[0]["kind"] == "percentage"
        assert any("reduction reached 30%" in text for text in fake.calls[0]["texts"])
