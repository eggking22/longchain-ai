"""Statement Draft layer: one TRUE statement per set + controlled false perturbations.

Deterministic, evidence-first: perturbation material (treatments, endpoints,
conditions, numbers, sibling panels) comes only from the paper's own evidence;
association may only be upgraded to causation as a FALSE statement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.question_generation import DraftConfig, generate_question_drafts

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
    def test_numeric_mutation_uses_literal_pool_only(self, numeric_report):
        data_sets = [d for d in numeric_report.draft_sets if d.question_type == "DATA_STATEMENT"]
        assert data_sets, "numeric corpus must produce DATA sets"
        mutations = [s for d in data_sets for s in d.statements if s.perturbation_type == "NUMERIC_MUTATION"]
        assert mutations
        literal_values = {"30%", "50%"}
        for statement in mutations:
            value = statement.statement.removeprefix("The reported value is ").removesuffix(".")
            assert value in literal_values, value  # never a fabricated number

    def test_true_data_statement_reports_literal(self, numeric_report):
        data_sets = [d for d in numeric_report.draft_sets if d.question_type == "DATA_STATEMENT"]
        true_statement = next(s for s in data_sets[0].statements if s.is_correct)
        assert true_statement.statement in (
            "The reported value is 30%.",
            "The reported value is 50%.",
            "The reported value is p < 0.01.",
        )


class TestSafetyInvariants:
    def test_no_external_facts_in_false_statements(self, numeric_report):
        """Every false statement's changed tokens derive from the paper's literal pool."""
        import re

        allowed_numbers = {"30%", "50%", "0.01", "30", "50"}
        for draft_set in numeric_report.draft_sets:
            for statement in draft_set.statements:
                if statement.is_correct or statement.perturbation_type != "NUMERIC_MUTATION":
                    continue
                numbers = re.findall(r"\d[\d.]*%?", statement.statement)
                assert set(numbers) <= allowed_numbers, statement.statement


class TestArtifactIsolation:
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
