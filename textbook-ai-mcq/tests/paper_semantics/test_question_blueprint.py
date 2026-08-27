"""Question Blueprint layer: four deterministic question types over Experiment Models.

Evidence-first is enforced throughout: blueprint evidence_ids must resolve into
the Evidence Store, numbers are only literal text, association never becomes
causal, and gaps mean no blueprint (counted in summary.skipped).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.question_blueprint import BlueprintConfig, generate_blueprints
from app.services.paper_semantics import PaperSemanticsConfig

from .conftest import build_paper_tree, write_document_artifact

EXISTING_ARTIFACTS = ("figures.json", "evidence.jsonl", "experiments.json", "report.md", "manifest.json")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """(artifacts_root, blueprint_report) built once for the golden corpus."""
    from app.services.question_blueprint import generate_blueprints as _generate

    root = tmp_path_factory.mktemp("blueprints")
    write_document_artifact(build_paper_tree(), root, "bp-paper")
    return root, _generate("bp-paper", root, config=BlueprintConfig())


@pytest.fixture(scope="module")
def report(corpus):
    return corpus[1]


def _by_type(report, question_type, figure_id=None):
    return [
        b
        for b in report.blueprints
        if b.question_type == question_type and (figure_id is None or b.figure_id == figure_id)
    ]


class TestFourTypesGenerated:
    def test_all_types_present(self, report):
        assert set(report.summary["by_type"]) == {
            "RESULT_INTERPRETATION", "EXPERIMENTAL_DESIGN", "SIMPLE_PREDICTION", "DATA_STATEMENT",
        }
        for question_type, count in report.summary["by_type"].items():
            if question_type != "DATA_STATEMENT":  # golden corpus has no literal numbers
                assert count > 0, question_type

    def test_result_interpretation_for_canonical_figure(self, report):
        blueprints = _by_type(report, "RESULT_INTERPRETATION", "Figure 2")
        assert blueprints
        first = blueprints[0]
        assert "Treatment A" in first.question_focus and "control" in first.question_focus
        assert first.expected_answer == "Treatment A increases gene X expression."
        assert first.reasoning_operation == "comparison"

    def test_experimental_design_three_subquestions(self, report):
        blueprints = _by_type(report, "EXPERIMENTAL_DESIGN", "Figure 2")
        elements = {b.detail["design_element"] for b in blueprints}
        assert elements == {"control_group", "experimental_group", "measured_endpoint"}
        # answers restate recorded slots only — no invented purposes
        assert all("gene X expression" in b.expected_answer for b in blueprints)

    def test_simple_prediction_grounded_in_control(self, report):
        blueprints = _by_type(report, "SIMPLE_PREDICTION", "Figure 2")
        assert len(blueprints) == 1
        prediction = blueprints[0]
        assert prediction.question_focus.startswith("If Treatment A were applied to the control group")
        assert "as already observed in the" in prediction.expected_answer
        assert prediction.reasoning_operation == "local_prediction"


class TestBindingAndProvenance:
    def test_bindings_match_semantics(self, report, corpus):
        from app.services.paper_semantics import reconstruct_figures

        root = corpus[0]
        semantics = reconstruct_figures("bp-paper", root, config=PaperSemanticsConfig(), persist=False)
        figure_ids = {f.figure_id for f in semantics.figures}
        panel_ids = {(f.figure_id, p.panel_id) for f in semantics.figures for p in f.panels}
        experiment_ids = {f.experiment.experiment_id for f in semantics.figures if f.experiment}
        experiment_ids |= {
            p.experiment.experiment_id for f in semantics.figures for p in f.panels if p.experiment
        }
        for blueprint in report.blueprints:
            assert blueprint.figure_id in figure_ids
            assert blueprint.experiment_id in experiment_ids
            for panel_id in blueprint.panel_ids:
                assert (blueprint.figure_id, panel_id) in panel_ids

    def test_evidence_ids_resolve_to_store(self, report, corpus):
        from app.services.paper_semantics import reconstruct_figures

        root = corpus[0]
        semantics = reconstruct_figures("bp-paper", root, config=PaperSemanticsConfig(), persist=False)
        store = set()
        for figure in semantics.figures:
            store |= {e.evidence_id for e in figure.evidence}
            store |= {e.evidence_id for p in figure.panels for e in p.evidence}
        for blueprint in report.blueprints:
            assert blueprint.evidence_ids
            assert set(blueprint.evidence_ids) <= store, blueprint.blueprint_id

    def test_panel_level_blueprint_uses_panel_scoped_ids(self, report):
        panel_blueprints = [b for b in report.blueprints if b.panel_ids]
        if not panel_blueprints:  # golden corpus figures are single-panel
            pytest.skip("no panel-level blueprints in this corpus")
        for blueprint in panel_blueprints:
            assert blueprint.experiment_id.endswith(tuple("abcdefgh"))
            assert any(eid.startswith(f"ev_{blueprint.panel_ids[0]}_") for eid in blueprint.evidence_ids)

    def test_blueprint_ids_deterministic_and_unique(self, report):
        ids = [b.blueprint_id for b in report.blueprints]
        assert len(ids) == len(set(ids))
        assert all(bid.startswith("qb_") for bid in ids)


class TestAssociationNotUpgraded:
    def test_association_focus_and_answer_wording(self, report):
        blueprints = _by_type(report, "RESULT_INTERPRETATION", "Figure 5")
        assert blueprints
        for blueprint in blueprints:
            assert blueprint.detail["relationship_type"] == "association"
            assert "associated with" in blueprint.expected_answer
            assert "increases" not in blueprint.expected_answer.replace("is associated with increased", "")

    def test_no_prediction_for_association_experiment(self, report):
        assert _by_type(report, "SIMPLE_PREDICTION", "Figure 5") == []
        skipped = report.summary["skipped"].get("SIMPLE_PREDICTION", {})
        assert sum(skipped.values()) >= 1  # gated out (this corpus: no control comparison)

    def test_non_causal_relationship_skips_prediction_directly(self):
        """An association experiment WITH both groups must still refuse prediction."""
        from app.schemas.paper_semantics import (
            ExperimentModel,
            FigureSemantic,
            Observation,
        )
        from app.services.question_blueprint.generators import SkipTracker, generate_simple_prediction

        experiment = ExperimentModel(
            experiment_id="exp_f07",
            intervention="Treatment C",
            independent_variables=["Treatment C"],
            dependent_variables=["gene Z expression"],
            experimental_groups=["Treatment C"],
            control_groups=["control"],
            observations=[
                Observation(
                    statement="Treatment C was associated with increased expression of gene Z.",
                    direction="increase",
                    relationship_type="association",
                    evidence_ids=["ev_001"],
                )
            ],
        )
        figure = FigureSemantic(
            figure_id="Figure 7",
            kind="figure",
            reconstruction_status="SUFFICIENT",
            confidence=1.0,
            experiment=experiment,
        )
        skips = SkipTracker()
        assert generate_simple_prediction(figure, experiment, skips=skips) == []
        assert skips.reasons["SIMPLE_PREDICTION"].get("non_causal_relationship") == 1


class TestGating:
    def test_partial_figure_without_direction_blocked(self, report):
        # Figure 4 (PARTIAL: has endpoint, no groups/direction) → no RI/ED/prediction
        assert _by_type(report, "RESULT_INTERPRETATION", "Figure 4") == []
        assert _by_type(report, "EXPERIMENTAL_DESIGN", "Figure 4") == []
        assert _by_type(report, "SIMPLE_PREDICTION", "Figure 4") == []

    def test_insufficient_figure_fully_blocked(self, report):
        for question_type in ("RESULT_INTERPRETATION", "EXPERIMENTAL_DESIGN", "SIMPLE_PREDICTION", "DATA_STATEMENT"):
            assert _by_type(report, question_type, "Figure 3") == []


class TestDataStatement:
    def _numeric_report(self, tmp_path):
        """Corpus variant with an explicitly reported percentage and p-value."""
        tree = build_paper_tree()
        results = next(c for c in tree.children if c.title == "Results")
        from .conftest import _para

        results.children.append(_para("p-results-9", "Treatment D reduced migration distance by 30% (p < 0.01) (Figure 6).", page=6))
        results.children.append(_para("p-cap-fig6", "Figure 6. Migration distance of cells after Treatment D and vehicle.", page=6))
        write_document_artifact(tree, tmp_path, "bp-numeric")
        return generate_blueprints("bp-numeric", tmp_path, config=BlueprintConfig())

    def test_percentage_and_p_value_extracted(self, tmp_path):
        report = self._numeric_report(tmp_path)
        statements = _by_type(report, "DATA_STATEMENT", "Figure 6")
        values = {b.expected_answer for b in statements}
        assert any("%" in v for v in values)
        p_values = [v for v in values if "p" in v.lower() and "<" in v]
        assert p_values, values
        for blueprint in statements:
            assert blueprint.reasoning_operation == "quantitative_reading"
            assert blueprint.detail["sentence"]  # value bound to its literal sentence

    def test_no_fabricated_numbers_without_literals(self, report):
        """Golden corpus says only 'significantly increased' → zero DATA_STATEMENT anywhere."""
        assert report.summary["by_type"]["DATA_STATEMENT"] == 0
        assert report.summary["skipped"]["DATA_STATEMENT"]["no_literal_numeric_value"] >= 1

    def test_expected_answers_contain_no_unbound_numbers(self, report):
        import re

        for blueprint in report.blueprints:
            if blueprint.question_type == "DATA_STATEMENT":
                continue
            assert not re.search(r"\d", blueprint.expected_answer), blueprint.expected_answer


class TestPanelLabelRegression:
    """STEP 0: figure-panel citations must never be read as quantities."""

    @staticmethod
    def _findings(text: str):
        from app.schemas.paper_semantics import PaperEvidence
        from app.services.question_blueprint.numeric import extract_numeric_findings

        evidence = PaperEvidence(
            evidence_id="ev_001", figure_id="Figure 1", text=text, role="direct",
            evidence_type="direct_observation",
        )
        return extract_numeric_findings([evidence])

    @pytest.mark.parametrize(
        "sentence",
        [
            "Expression increased in confined cells (Fig. 1g).",   # paren after panel letter
            "As shown in Extended Data Fig. 2b, CCR7 was upregulated.",  # Fig context
            "Treatment reduced motility (1g) across all donors.",
            "The response is quantified in 3a, together with controls.",
        ],
    )
    def test_panel_citations_not_extracted(self, sentence):
        values = [f.value for f in self._findings(sentence)]
        assert not any(re.fullmatch(r"\d{1,3}[a-hA-H]", v) for v in values), values

    @pytest.mark.parametrize(
        "sentence, expected",
        [
            ("Cells were pelleted at 1 g for 5 min.", "1 g"),          # spaced quantity
            ("DCs were treated with the inhibitor AACOF3 (25 µM).", "25 µM"),
            ("The culture received 10 mg of supplement.", "10 mg"),
            ("Cells were incubated for 6h before staining.", "6h"),    # glued, followed by word
            ("Confinement was applied for 4-6 h at 3 um.", "6 h"),
        ],
    )
    def test_real_quantities_still_extracted(self, sentence, expected):
        values = [f.value for f in self._findings(sentence)]
        assert expected in values, values


class TestArtifactIsolation:
    def test_existing_files_unchanged_and_new_file_written(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "bp-iso")
        from app.services.paper_semantics import reconstruct_figures

        reconstruct_figures("bp-iso", tmp_path, config=PaperSemanticsConfig())  # write baseline artifacts
        out = Path(tmp_path) / "paper_semantics" / "bp-iso"
        before = {name: (out / name).read_bytes() for name in EXISTING_ARTIFACTS}

        generate_blueprints("bp-iso", tmp_path, config=BlueprintConfig())

        after = {name: (out / name).read_bytes() for name in EXISTING_ARTIFACTS}
        assert before == after  # blueprint generation rewrites nothing
        artifact = out / "question_blueprints.json"
        assert artifact.exists()
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert list(payload) == ["doc_id", "summary", "blueprints"]

    def test_artifact_byte_reproducible(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "bp-repr")
        generate_blueprints("bp-repr", tmp_path, config=BlueprintConfig())
        artifact = Path(tmp_path) / "paper_semantics" / "bp-repr" / "question_blueprints.json"
        first = artifact.read_bytes()
        generate_blueprints("bp-repr", tmp_path, config=BlueprintConfig())
        assert artifact.read_bytes() == first
