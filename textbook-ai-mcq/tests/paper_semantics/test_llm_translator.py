"""LLM statement translator: offline unit tests + pipeline integration.

All HTTP is mocked with httpx.MockTransport (same seam as the other LLM layers)
— the suite stays fully offline. Every translation must pass the deterministic
invariant gate (numbers / figure anchors / direction / gene names verbatim) or
the deterministic registry/template result stands.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.services.question_generation import DraftConfig, generate_question_drafts
from app.services.question_translation import (
    LlmStatementTranslator,
    LlmTranslationError,
    translate_drafts,
)

from .conftest import _para, build_paper_tree, write_document_artifact

BASE = "https://llm.test/v1"

EN = "CCR7 expression (GFP intensity) would increase, as already observed in the DCs treated with the cPLA2 inhibitor AACOF3 (25 µM) group."
GOOD_ZH = "CCR7 表达（GFP 强度）将会提高，这在经 cPLA2 抑制剂 AACOF3（25 µM）处理的 DC 组中已被观察到。"


def _content(zh: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": json.dumps({"zh": zh}, ensure_ascii=False)}}]}).encode()


def _raw_content(message: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": message}}]}).encode()


def _translator(handler):
    return LlmStatementTranslator(
        base_url=BASE,
        api_key="test-key",
        model="test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class TestValidationGate:
    def test_valid_translation_passes(self):
        assert LlmStatementTranslator.validate(EN, GOOD_ZH) is None

    def test_missing_number_rejected(self):
        reason = LlmStatementTranslator.validate(EN, GOOD_ZH.replace("25", "30"))
        assert reason is not None and "25" in reason

    def test_flipped_direction_rejected(self):
        reason = LlmStatementTranslator.validate(EN, GOOD_ZH.replace("提高", "降低"))
        assert reason is not None and "direction" in reason

    def test_direction_synonyms_accepted(self):
        # natural Chinese synonyms pass as long as the polarity is not flipped
        assert LlmStatementTranslator.validate(EN, GOOD_ZH.replace("提高", "增加")) is None
        down_en = "Genes encoding MHC class I are expressed at lower levels in confined DCs."
        down_zh = "编码 MHC I 类的基因在受限 DCs 中的表达水平较低。"
        assert LlmStatementTranslator.validate(down_en, down_zh) is None

    def test_up_statement_with_down_word_rejected(self):
        reason = LlmStatementTranslator.validate(EN, GOOD_ZH.replace("提高", "下降"))
        assert reason is not None and "direction" in reason

    def test_compound_direction_allowed(self):
        # "decreases the upregulation" legitimately carries both words in Chinese
        en = "CK666 decreases LPS-induced CCR7 upregulation."
        zh = "CK666 降低 LPS 诱导的 CCR7 上调。"
        assert LlmStatementTranslator.validate(en, zh) is None

    def test_missing_gene_token_rejected(self):
        reason = LlmStatementTranslator.validate(EN, GOOD_ZH.replace("AACOF3", "某抑制剂"))
        assert reason is not None and "AACOF3" in reason

    def test_missing_figure_anchor_rejected(self):
        en = "According to Figure 2, DCs spent 35% of their time at diameters of >4 µm."
        zh = "根据该图，DC 有 35% 的时间直径大于 4 µm。"
        reason = LlmStatementTranslator.validate(en, zh)
        assert reason is not None and "Figure 2" in reason

    def test_localized_anchor_with_same_number_accepted(self):
        en = "According to Figure 2, DCs spent 35% of their time at diameters of >4 µm."
        zh = "根据图2，DC 有 35% 的时间直径大于 4 µm。"
        assert LlmStatementTranslator.validate(en, zh) is None

    def test_wrong_anchor_number_rejected(self):
        en = "According to Figure 2, DCs spent 35% of their time at diameters of >4 µm."
        zh = "根据图3，DC 有 35% 的时间直径大于 4 µm。"
        reason = LlmStatementTranslator.validate(en, zh)
        assert reason is not None and "Figure 2" in reason


class TestTranslatorUnits:
    def test_translate_returns_validated_zh(self):
        translator = _translator(lambda request: httpx.Response(200, content=_content(GOOD_ZH)))
        assert translator.translate(EN) == GOOD_ZH

    def test_rejected_translation_returns_none(self):
        translator = _translator(lambda request: httpx.Response(200, content=_content("CCR7 表达将会降低。")))
        assert translator.translate(EN) is None

    def test_invalid_json_reasked_then_ok(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(200, content=_raw_content("nope"))
            return httpx.Response(200, content=_content(GOOD_ZH))

        translator = _translator(handler)
        assert translator.translate(EN) == GOOD_ZH
        assert len(calls) == 2

    def test_persistent_invalid_json_raises(self):
        translator = _translator(lambda request: httpx.Response(200, content=b"still nope"))
        with pytest.raises(LlmTranslationError):
            translator.translate(EN)

    def test_http_error_raises(self):
        translator = _translator(lambda request: httpx.Response(500))
        with pytest.raises(LlmTranslationError):
            translator.translate(EN)

    def test_prompt_shape(self):
        seen = {}

        def handler(request):
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, content=_content(GOOD_ZH))

        _translator(handler).translate(EN)
        assert seen["body"]["temperature"] == 0.0
        assert EN in seen["body"]["messages"][1]["content"]


class TestPipelineIntegration:
    @pytest.fixture(scope="class")
    def drafts(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("zh-llm")
        tree = build_paper_tree()
        results = next(c for c in tree.children if c.title == "Results")
        results.children.append(
            _para("p-results-9", "Treatment D reduced migration distance by 30% (p < 0.01) (Figure 6).", page=6)
        )
        results.children.append(_para("p-cap-fig6", "Figure 6. Migration distance of cells after treatment.", page=6))
        write_document_artifact(tree, root, "zh-llm-paper")
        return generate_question_drafts("zh-llm-paper", root, config=DraftConfig(), persist=False)

    def test_llm_translations_used_and_counted(self, drafts):
        class _Fake:
            def translate(self, statement):
                return f"【译】{statement}"  # passes: numbers/anchors/genes copied verbatim

        zh = translate_drafts(drafts, translator=_Fake())
        counts = zh.summary["translation"]["counts"]
        assert counts["llm"] == sum(len(s.statements) for s in drafts.draft_sets)
        assert zh.summary["translation"]["method"] == "deterministic-registry+llm"
        assert all(st.statement_zh.startswith("【译】") for s in zh.draft_sets for st in s.statements)
        # English preserved verbatim
        assert all(
            st_zh.statement == st.statement
            for s_en, s_zh in zip(drafts.draft_sets, zh.draft_sets)
            for st, st_zh in zip(s_en.statements, s_zh.statements)
        )

    def test_rejection_falls_back_per_statement(self, drafts):
        class _Flaky:
            def translate(self, statement):
                return None if "30%" in statement else "【译】ok"

        zh = translate_drafts(drafts, translator=_Flaky())
        counts = zh.summary["translation"]["counts"]
        assert counts["llm"] > 0 and counts["template"] + counts["term_fallback"] > 0

    def test_error_falls_back_deterministic(self, drafts):
        class _Broken:
            def translate(self, statement):
                raise LlmTranslationError("boom")

        zh = translate_drafts(drafts, translator=_Broken())
        counts = zh.summary["translation"]["counts"]
        assert counts["llm"] == 0
        baseline = translate_drafts(drafts)
        assert [st.statement_zh for s in zh.draft_sets for st in s.statements] == [
            st.statement_zh for s in baseline.draft_sets for st in s.statements
        ]

    def test_without_translator_summary_unchanged(self, drafts):
        zh = translate_drafts(drafts)
        assert zh.summary["translation"]["method"] == "deterministic-registry"
        assert set(zh.summary["translation"]["counts"]) == {"template", "term_fallback"}
