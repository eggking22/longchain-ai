"""Chinese MCQ statement translation: invariants on language change only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.question_generation import DraftConfig, generate_question_drafts
from app.services.question_translation import (
    TERMINOLOGY,
    load_drafts,
    persist_mcq_zh,
    translate_document,
    translate_drafts,
    translate_statement,
)

from .conftest import build_paper_tree, write_document_artifact


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Golden drafts + Chinese projection built once."""
    from .conftest import _para

    root = tmp_path_factory.mktemp("mcq-zh")
    tree = build_paper_tree()
    results = next(c for c in tree.children if c.title == "Results")
    results.children.append(
        _para("p-results-9", "Treatment D reduced migration distance by 30% (p < 0.01) (Figure 6).", page=6)
    )
    results.children.append(_para("p-cap-fig6", "Figure 6. Migration distance of cells after treatment.", page=6))
    write_document_artifact(tree, root, "zh-paper")
    drafts = generate_question_drafts("zh-paper", root, config=DraftConfig())
    return root, drafts, translate_drafts(drafts)


class TestStatementLevelTranslation:
    def test_basic_templates(self):
        zh, method = translate_statement("cPLA2 inhibitor increases GFP expression.")
        assert method == "template"
        assert zh == "cPLA2 抑制剂可提高 GFP 表达。"

    def test_direction_pair(self):
        up, _ = translate_statement("Treatment A increases gene X expression.")
        down, _ = translate_statement("Treatment A decreases gene X expression.")
        assert "提高" in up and "降低" in down
        assert "降低" not in up and "提高" not in down  # direction never blurred

    def test_significance_flip(self):
        zh, _ = translate_statement("Treatment A does not significantly increase gene X expression.")
        assert zh == "处理 A 不会显著提高 gene X 表达。"

    def test_association_wording_not_causal(self):
        zh, _ = translate_statement("Treatment C is associated with increased gene Z expression.")
        assert "相关" in zh and "导致" not in zh

    def test_causality_upgrade_false_statement(self):
        zh, _ = translate_statement("Treatment C causes increased gene Z expression.")
        assert "导致" in zh

    def test_reported_value_keeps_number(self):
        zh, _ = translate_statement("The reported value is 30%.")
        assert zh == "论文报告的数值为 30%。"

    def test_prediction_template(self):
        zh, _ = translate_statement(
            "GFP expression would increase, as already observed in the cPLA2 inhibitor group."
        )
        assert "将会提高" in zh and "cPLA2 抑制剂组" in zh

    def test_design_template_control(self):
        zh, _ = translate_statement(
            "The untreated group provides the baseline level of body weight against which Treatment B is compared."
        )
        assert zh.startswith("未处理组提供体重")
        assert "基线水平" in zh and "处理 B" in zh

    def test_fallback_preserves_numbers_and_genes(self):
        zh, method = translate_statement("Cells were treated with AACOF3 (25 uM) for 6 h and GFP increased 3-fold.")
        assert method == "term_fallback"
        assert "AACOF3" in zh and "25 uM" in zh and "6 h" in zh and "3-fold" in zh

    def test_terminology_registry_direction_is_fixed(self):
        # direction verbs have exactly one Chinese counterpart each — a
        # translation can never flip or blur a direction
        assert TERMINOLOGY["increases"] == "提高"
        assert TERMINOLOGY["decreases"] == "降低"
        assert TERMINOLOGY["expression"] == "表达"


class TestReportLevelInvariants:
    def test_english_preserved_and_zh_present(self, corpus):
        _, drafts, zh = corpus
        for draft_set, zh_set in zip(drafts.draft_sets, zh.draft_sets):
            for statement, zh_statement in zip(draft_set.statements, zh_set.statements):
                assert zh_statement.statement == statement.statement  # English verbatim
                assert zh_statement.statement_zh  # Chinese exists

    def test_true_false_and_evidence_unchanged(self, corpus):
        _, drafts, zh = corpus
        for draft_set, zh_set in zip(drafts.draft_sets, zh.draft_sets):
            for statement, zh_statement in zip(draft_set.statements, zh_set.statements):
                assert zh_statement.is_correct == statement.is_correct
                assert zh_statement.evidence_ids == statement.evidence_ids
                assert zh_statement.perturbation_type == statement.perturbation_type

    def test_numbers_units_untouched_in_zh(self, corpus):
        import re

        for draft_set in zh_report_sets(corpus):
            for statement in draft_set.statements:
                for number in re.findall(r"\d[\d.]*%?", statement.statement):
                    assert number in statement.statement_zh, (number, statement.statement_zh)

    def test_translation_summary_counts(self, corpus):
        _, _, zh = corpus
        counts = zh.summary["translation"]["counts"]
        assert counts["template"] + counts["term_fallback"] == zh.summary["statements"]


def zh_report_sets(corpus):
    return corpus[2].draft_sets


class TestArtifactIsolation:
    def test_english_artifact_untouched_and_zh_written(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "zh-iso")
        generate_question_drafts("zh-iso", tmp_path, config=DraftConfig())
        drafts_path = Path(tmp_path) / "paper_semantics" / "zh-iso" / "question_drafts.json"
        before = drafts_path.read_bytes()

        report = translate_drafts(load_drafts("zh-iso", tmp_path))
        persist_mcq_zh(report, tmp_path)

        assert drafts_path.read_bytes() == before  # English artifact untouched
        payload = json.loads(
            (Path(tmp_path) / "paper_semantics" / "zh-iso" / "mcq_drafts_zh.json").read_text(encoding="utf-8")
        )
        assert list(payload) == ["doc_id", "summary", "draft_sets"]
        assert payload["draft_sets"][0]["statements"][0]["statement_zh"]

    def test_zh_artifact_byte_reproducible(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "zh-repr")
        generate_question_drafts("zh-repr", tmp_path, config=DraftConfig())
        translate_document("zh-repr", tmp_path)
        artifact = Path(tmp_path) / "paper_semantics" / "zh-repr" / "mcq_drafts_zh.json"
        first = artifact.read_bytes()
        translate_document("zh-repr", tmp_path)
        assert artifact.read_bytes() == first
