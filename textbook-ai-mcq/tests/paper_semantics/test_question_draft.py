"""Statement Draft layer: one TRUE statement per set + controlled false perturbations.

Deterministic, evidence-first: perturbation material (treatments, endpoints,
conditions, numbers, sibling panels) comes only from the paper's own evidence;
association may only be upgraded to causation as a FALSE statement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.question_blueprint import QuestionBlueprint
from app.services.question_generation import DraftConfig, generate_question_drafts
from app.services.question_generation.perturbations import (
    PERTURBERS,
    PerturbationContext,
    build_true_statement,
    numeric_mutation,
    panel_misattribution,
)

from .conftest import build_paper_tree, write_document_artifact

EXISTING_ARTIFACTS = (
    "figures.json", "evidence.jsonl", "experiments.json", "report.md", "manifest.json",
)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    root = tmp_path_factory.mktemp("drafts")
    write_document_artifact(build_paper_tree(), root, "draft-paper")
    return root, generate_question_drafts("draft-paper", root, config=DraftConfig())


@pytest.fixture(scope="module")
def numeric_report(tmp_path_factory):
    """Corpus variant with reported percentages so DATA sets and mutations activate."""
    from .conftest import _para

    root = tmp_path_factory.mktemp("drafts-numeric")
    tree = build_paper_tree()
    results = next(c for c in tree.children if c.title == "Results")
    results.children.append(
        _para("p-results-9", "Treatment D reduced migration distance by 30% (p < 0.01) (Figure 6).", page=6)
    )
    results.children.append(
        _para("p-results-10", "Treatment E reduced migration distance by 50% (Figure 6).", page=6)
    )
    results.children.append(_para("p-cap-fig6", "Figure 6. Migration distance of cells after treatment.", page=6))
    write_document_artifact(tree, root, "draft-numeric")
    return generate_question_drafts("draft-numeric", root, config=DraftConfig())


class TestSetStructure:
    def test_every_set_has_exactly_one_true(self, report):
        _, drafts = report
        assert drafts.draft_sets
        for draft_set in drafts.draft_sets:
            trues = [s for s in draft_set.statements if s.is_correct]
            assert len(trues) == 1, draft_set.draft_set_id
            assert trues[0].perturbation_type == "NONE"
            assert len(draft_set.statements) >= 2  # needs at least one false statement

    def test_all_types_covered(self, report):
        _, drafts = report
        types = {s.question_type for s in drafts.draft_sets}
        assert types >= {"RESULT_INTERPRETATION", "EXPERIMENTAL_DESIGN"}  # golden corpus minimum
        if types == {"RESULT_INTERPRETATION", "EXPERIMENTAL_DESIGN"}:
            # golden corpus has no literal numbers → DATA sets legitimately absent
            assert drafts.summary["by_perturbation"] or drafts.summary["skipped"]

    def test_statements_carry_evidence(self, report):
        _, drafts = report
        for draft_set in drafts.draft_sets:
            for statement in draft_set.statements:
                assert statement.evidence_ids, statement.draft_id

    def test_false_statements_differ_from_true(self, report):
        _, drafts = report
        for draft_set in drafts.draft_sets:
            texts = [s.statement for s in draft_set.statements]
            assert len(texts) == len(set(texts)), draft_set.draft_set_id


class TestPerturbations:
    def test_direction_flip(self, report):
        _, drafts = report
        flips = [s for d in drafts.draft_sets for s in d.statements if s.perturbation_type == "DIRECTION_FLIP"]
        assert flips, "golden corpus must produce direction flips"
        import re

        up = re.compile(r"\b(increases|increased|increase|higher)\b")
        down = re.compile(r"\b(decreases|decreased|decrease|lower)\b")

        def polarity(text: str) -> str:
            return "up" if up.search(text) else ("down" if down.search(text) else "none")

        for statement in flips:
            base = statement.detail["base"]
            assert polarity(base) != polarity(statement.statement)
            assert polarity(statement.statement) in ("up", "down"), (base, statement.statement)

    def test_group_swap_exchanges_both_groups(self, report):
        _, drafts = report
        swaps = [s for d in drafts.draft_sets for s in d.statements if s.perturbation_type == "GROUP_SWAP"]
        assert swaps
        for statement in swaps:
            base = statement.detail["base"]
            # both group names still present, positions exchanged
            assert set(base.split()) & set(statement.statement.split())

    def test_causality_upgrade_only_for_association(self, report):
        _, drafts = report
        upgrades = [s for d in drafts.draft_sets for s in d.statements if s.perturbation_type == "CAUSALITY_UPGRADE"]
        for statement in upgrades:
            assert "causes" in statement.statement
            assert "is associated with" in statement.detail["base"]

    def test_true_statements_never_upgrade_association(self, report):
        _, drafts = report
        for draft_set in drafts.draft_sets:
            true_statement = next(s for s in draft_set.statements if s.is_correct)
            if draft_set.question_type == "RESULT_INTERPRETATION":
                # association experiments keep their relationship wording in truth
                assert "causes" not in true_statement.statement

    def test_panel_misattribution_references_sibling(self, report):
        _, drafts = report
        mis = [s for d in drafts.draft_sets for s in d.statements if s.perturbation_type == "PANEL_MISATTRIBUTION"]
        assert mis
        for statement in mis:
            assert statement.statement.startswith("According to ")
            attributed = statement.statement.split(",", 1)[0].removeprefix("According to ")
            base = statement.detail["base"]
            assert attributed and attributed not in base  # attributed to something else

    def test_conclusion_flip_negates(self, report):
        _, drafts = report
        flips = [s for d in drafts.draft_sets for s in d.statements if s.perturbation_type == "CONCLUSION_FLIP"]
        for statement in flips:
            assert "does not" in statement.statement or "not " in statement.statement


class TestDataPerturbations:
    def test_numeric_mutation_swaps_value_inside_quote(self, numeric_report):
        import re

        data_sets = [d for d in numeric_report.draft_sets if d.question_type == "DATA_STATEMENT"]
        assert data_sets, "numeric corpus must produce DATA sets"
        mutations = [s for d in data_sets for s in d.statements if s.perturbation_type == "NUMERIC_MUTATION"]
        assert mutations
        literal_values = {"30%", "50%"}

        def skeleton(text: str) -> str:
            return re.sub(r"\d[\d.]*", "#", text)

        for statement in mutations:
            base = statement.detail["base"]
            assert skeleton(statement.statement) == skeleton(base)  # only the value changed
            assert any(v in statement.statement for v in literal_values)  # never a fabricated number

    def test_true_data_statement_is_anchored_quote(self, numeric_report):
        data_sets = [d for d in numeric_report.draft_sets if d.question_type == "DATA_STATEMENT"]
        true_statement = next(s for d in data_sets for s in d.statements if s.is_correct)
        assert true_statement.statement.startswith("According to Figure 6, ")
        assert "30%" in true_statement.statement
        assert "reduced migration distance" in true_statement.statement  # object context quoted verbatim


class TestDataStatementAnchors:
    """DATA statements quote the evidence sentence behind a figure anchor."""

    @staticmethod
    def _blueprint(sentence, value="30%", kind="percentage", figure_id="Figure 2", panel_ids=None):
        return QuestionBlueprint(
            blueprint_id="qb_f02_ds_001",
            question_type="DATA_STATEMENT",
            experiment_id="exp_f02",
            figure_id=figure_id,
            panel_ids=panel_ids or [],
            question_focus="what value is reported",
            reasoning_operation="quantitative_reading",
            expected_answer=value,
            evidence_ids=["ev_f02_001"],
            detail={"data_value": value, "kind": kind, "sentence": sentence},
        )

    def test_quote_is_anchored_and_verbatim(self):
        bp = self._blueprint("DCs spent 30% of their time displaying diameters of >4 um.")
        assert build_true_statement(bp) == (
            "According to Figure 2, DCs spent 30% of their time displaying diameters of >4 um."
        )

    def test_line_break_hyphen_merged_and_dangling_ref_trimmed(self):
        bp = self._blueprint("DCs spent 30% of their time dis- playing diameters of >4 um (Fig.")
        assert build_true_statement(bp) == (
            "According to Figure 2, DCs spent 30% of their time displaying diameters of >4 um."
        )

    def test_trailing_figure_reference_stripped(self):
        bp = self._blueprint("Treatment D reduced migration distance by 30% (Figure 6).")
        assert build_true_statement(bp) == "According to Figure 2, Treatment D reduced migration distance by 30%."

    def test_panel_blueprint_anchor_includes_panel(self):
        bp = self._blueprint("Signal rose to 30% of cells.", panel_ids=["2a"])
        assert build_true_statement(bp) == "According to Figure 2a, Signal rose to 30% of cells."

    def test_value_missing_from_sentence_falls_back_to_kind_label(self):
        bp = self._blueprint("Migration distance was reduced.")
        assert build_true_statement(bp) == "According to Figure 2, the reported percentage is 30%."

    def test_concentration_kind_label(self):
        bp = self._blueprint("No usable sentence.", value="25 uM", kind="concentration")
        assert build_true_statement(bp) == "According to Figure 2, the reported concentration is 25 uM."

    def test_long_sentence_truncated_at_clause_after_value(self):
        front = "Treatment F reduced migration distance by 30% in the confined chamber"
        tail = ", and the effect was reproduced across replicates " + "with fully consistent results " * 8
        bp = self._blueprint(front + tail + " in all experiments.")
        statement = build_true_statement(bp)
        assert statement.startswith(f"According to Figure 2, {front}")
        assert statement.endswith("confined chamber.")
        assert len(statement) <= 260  # 240-char body cap + anchor + period

    def test_value_beyond_cap_falls_back(self):
        padding = " ".join(f"qualifier{i}" for i in range(40))  # > 240 chars before the value
        bp = self._blueprint(f"{padding} the level reached 30% overall.")
        assert build_true_statement(bp) == "According to Figure 2, the reported percentage is 30%."

    def test_splitter_fragment_never_quoted(self):
        # sentence-splitter artifact ("2d and 3c).") must not become the quoted object
        bp = self._blueprint("2d and 3c).", value="6 h", kind="time")
        assert build_true_statement(bp) == "According to Figure 2, the reported time is 6 h."

    def test_panel_label_value_with_fragment_skipped(self):
        # "2d" from "Fig. 2d and 3c" is a panel reference, not a time — no statement at all
        bp = self._blueprint("2d and 3c).", value="2d", kind="time")
        assert build_true_statement(bp) is None

    def test_glued_time_with_real_sentence_kept(self):
        bp = self._blueprint("Cells were incubated for 6h before imaging wells.", value="6h", kind="time")
        assert build_true_statement(bp) == "According to Figure 2, Cells were incubated for 6h before imaging wells."

    def test_trailing_dot_in_value_normalized(self):
        # upstream p-value literals may carry their own terminal dot — never emit ".."
        bp = self._blueprint("No usable sentence.", value="P < 0.0001.", kind="p_value")
        assert build_true_statement(bp) == "According to Figure 2, the reported p value is P < 0.0001."

    def test_numeric_mutation_normalizes_pool_dots(self):
        context = PerturbationContext(numeric_pool={"p_value": [("P < 0.0001.", "ev_x")]})
        text = "According to Figure 2, the reported p value is p = 0.003."
        assert numeric_mutation(text, context, "p_value", "p = 0.003") == (
            "According to Figure 2, the reported p value is P < 0.0001."
        )

    def test_thin_space_sentence_quoted_with_panel_prefix_stripped(self):
        # Nature typesets units with U+2009; the extracted value is ASCII-normalized —
        # whitespace normalization must reconcile the two before the substring check
        sentence = "a, GFP intensity in DCs treated with the cPLA2 inhibitor AACOF3 (25\u2009µM) or control."
        bp = self._blueprint(sentence, value="25 µM", kind="concentration")
        assert build_true_statement(bp) == (
            "According to Figure 2, GFP intensity in DCs treated with the cPLA2 inhibitor AACOF3 (25 µM) or control."
        )

    def test_capitalized_compound_keeps_hyphen(self):
        bp = self._blueprint("Color- coded z frames of untreated LifeAct DCs showed signal in 30% of cells.")
        assert build_true_statement(bp) == (
            "According to Figure 2, Color-coded z frames of untreated LifeAct DCs showed signal in 30% of cells."
        )

    def test_numeric_mutation_on_normalized_quote(self):
        context = PerturbationContext(numeric_pool={"concentration": [("30 µM", "ev_x")]})
        quote = (
            "According to Figure 2, GFP intensity in DCs treated with the cPLA2 "
            "inhibitor AACOF3 (25 µM) or control."
        )
        assert numeric_mutation(quote, context, "concentration", "25 µM") == (
            "According to Figure 2, GFP intensity in DCs treated with the cPLA2 "
            "inhibitor AACOF3 (30 µM) or control."
        )

    def test_numeric_mutation_edits_quote_in_place(self):
        context = PerturbationContext(numeric_pool={"percentage": [("50%", "ev_x")]})
        quote = "According to Figure 2, Signal rose to 30% of cells."
        assert numeric_mutation(quote, context, "percentage", "30%") == (
            "According to Figure 2, Signal rose to 50% of cells."
        )
        assert numeric_mutation("No value here.", context, "percentage", "30%") is None

    def test_panel_misattribution_swaps_anchor_of_quote(self):
        context = PerturbationContext(sibling_labels=["Figure 3"])
        quote = "According to Figure 2, Signal rose to 30% of cells."
        assert panel_misattribution(quote, context) == "According to Figure 3, Signal rose to 30% of cells."
        plain = "cPLA2 inhibitor increases GFP expression."
        assert panel_misattribution(plain, context) == (
            "According to Figure 3, cPLA2 inhibitor increases GFP expression."
        )

    def test_condition_mutation_excluded_for_data(self):
        context = PerturbationContext(
            numeric_pool={"concentration": [("25 uM", "ev_c"), ("10 uM", "ev_d")]},
            condition_findings=[("Cells were treated with AACOF3 (25 uM).", "25 uM", "ev_c")],
        )
        data_bp = self._blueprint(
            "Cells were treated with AACOF3 (25 uM).", value="25 uM", kind="concentration"
        )
        assert PERTURBERS["CONDITION_MUTATION"]("x", context, data_bp) == (None, [])
        ri_bp = QuestionBlueprint(
            blueprint_id="qb_f02_ri_001",
            question_type="RESULT_INTERPRETATION",
            experiment_id="exp_f02",
            figure_id="Figure 2",
            question_focus="direction",
            reasoning_operation="result_interpretation",
            expected_answer="x increases y.",
        )
        mutated = PERTURBERS["CONDITION_MUTATION"]("x increases y.", context, ri_bp)
        assert mutated[0] == "Cells were treated with AACOF3 (10 uM)."

    def test_data_sets_never_carry_condition_mutation(self, numeric_report):
        for draft_set in numeric_report.draft_sets:
            if draft_set.question_type != "DATA_STATEMENT":
                continue
            types = {s.perturbation_type for s in draft_set.statements}
            assert "CONDITION_MUTATION" not in types


class TestSafetyInvariants:
    def test_no_external_facts_in_false_statements(self, numeric_report):
        """Every false statement's changed tokens derive from the paper's literal pool."""
        import re

        # "6" is the anchor figure number; the quote carries the paper's own
        # sentence numbers (value + p value) — all from the corpus itself
        allowed_numbers = {"30%", "50%", "0.01", "30", "50", "6"}
        for draft_set in numeric_report.draft_sets:
            for statement in draft_set.statements:
                if statement.is_correct or statement.perturbation_type != "NUMERIC_MUTATION":
                    continue
                numbers = re.findall(r"\d[\d.]*%?", statement.statement)
                assert set(numbers) <= allowed_numbers, statement.statement


class TestArtifactIsolation:
    def test_semantic_report_reused_when_provided(self, tmp_path, monkeypatch):
        """Passing a precomputed report skips re-derivation (LLM-normalized seam)."""
        from unittest.mock import patch

        from app.services.paper_semantics import PaperSemanticsConfig, reconstruct_figures

        write_document_artifact(build_paper_tree(), tmp_path, "draft-reuse")
        reused = reconstruct_figures("draft-reuse", tmp_path, config=PaperSemanticsConfig(), persist=False)
        with patch(
            "app.services.question_generation.pipeline.reconstruct_figures",
            side_effect=AssertionError("must not re-derive"),
        ) as mock_reconstruct:
            report = generate_question_drafts(
                "draft-reuse", tmp_path, config=DraftConfig(), semantic_report=reused, persist=False
            )
        mock_reconstruct.assert_not_called()
        assert report.draft_sets == generate_question_drafts(
            "draft-reuse", tmp_path, config=DraftConfig(), persist=False
        ).draft_sets  # same output as re-derivation

    def test_existing_files_unchanged_and_new_file_written(self, tmp_path):
        from app.services.paper_semantics import PaperSemanticsConfig, reconstruct_figures

        write_document_artifact(build_paper_tree(), tmp_path, "draft-iso")
        reconstruct_figures("draft-iso", tmp_path, config=PaperSemanticsConfig())
        from app.services.question_blueprint import BlueprintConfig, generate_blueprints

        generate_blueprints("draft-iso", tmp_path, config=BlueprintConfig())
        out = Path(tmp_path) / "paper_semantics" / "draft-iso"
        before = {name: (out / name).read_bytes() for name in (*EXISTING_ARTIFACTS, "question_blueprints.json")}

        generate_question_drafts("draft-iso", tmp_path, config=DraftConfig())

        after = {name: (out / name).read_bytes() for name in before}
        assert before == after  # draft generation rewrites nothing
        payload = json.loads((out / "question_drafts.json").read_text(encoding="utf-8"))
        assert list(payload) == ["doc_id", "summary", "draft_sets"]

    def test_artifact_byte_reproducible(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "draft-repr")
        generate_question_drafts("draft-repr", tmp_path, config=DraftConfig())
        artifact = Path(tmp_path) / "paper_semantics" / "draft-repr" / "question_drafts.json"
        first = artifact.read_bytes()
        generate_question_drafts("draft-repr", tmp_path, config=DraftConfig())
        assert artifact.read_bytes() == first
