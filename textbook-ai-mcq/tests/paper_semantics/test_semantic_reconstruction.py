"""Golden reconstruction suite + reproducibility + provenance acceptance.

Mirrors the Phase 3 evidence-suite style: a synthetic corpus with known
expected verdicts and a confusion-style acceptance criterion.

  Figure 2  → SUFFICIENT   (caption + Results + Methods + Discussion)
  Figure 3  → INSUFFICIENT (bare mention)
  Figure 4  → PARTIAL      (caption with endpoint, no result statement)
  Figure 5  → SUFFICIENT   (association study, IV+DV+direction)
  Table 1   → SUFFICIENT   (table, decrease direction)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.paper_semantics import PaperSemanticsConfig, reconstruct_figures

from .conftest import build_paper_tree, write_document_artifact

EXPECTED = {
    "Figure 2": "SUFFICIENT",
    "Figure 3": "INSUFFICIENT",
    "Figure 4": "PARTIAL",
    "Figure 5": "SUFFICIENT",
    "Table 1": "SUFFICIENT",
}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    artifacts = tmp_path_factory.mktemp("paper-golden")
    write_document_artifact(build_paper_tree(), artifacts, "golden-paper")
    return reconstruct_figures("golden-paper", artifacts, config=PaperSemanticsConfig())


class TestGoldenSuite:
    def test_all_verdicts_correct(self, report):
        actual = {f.figure_id: f.reconstruction_status for f in report.figures}
        assert actual == EXPECTED

    def test_no_false_positives_on_insufficient(self, report):
        figure3 = next(f for f in report.figures if f.figure_id == "Figure 3")
        assert figure3.experiment.conclusions == []
        assert figure3.experiment.observations == []

    def test_canonical_semantics_recovered(self, report):
        figure2 = next(f for f in report.figures if f.figure_id == "Figure 2")
        experiment = figure2.experiment
        assert experiment.intervention == "Treatment A"
        assert "control" in experiment.control_groups
        assert experiment.dependent_variables[0] == "gene X expression"
        observation = experiment.observations[0]
        assert observation.direction == "increase"
        assert observation.significance == "significant"
        assert [c.statement for c in experiment.conclusions] == ["Treatment A increases gene X expression."]

    def test_association_is_not_causation(self, report):
        figure5 = next(f for f in report.figures if f.figure_id == "Figure 5")
        experiment = figure5.experiment
        assert experiment.observations[0].relationship_type == "association"
        conclusion = experiment.conclusions[0]
        assert conclusion.statement == "Treatment C is associated with increased gene Z expression."
        assert conclusion.relationship_type == "association"

    def test_table_reconstruction(self, report):
        table1 = next(f for f in report.figures if f.figure_id == "Table 1")
        assert table1.kind == "table"
        assert table1.experiment.observations[0].direction == "decrease"
        assert table1.experiment.conclusions[0].statement == "Treatment B decreases body weight."


class TestProvenance:
    def test_conclusion_traces_to_paragraph(self, report):
        figure2 = next(f for f in report.figures if f.figure_id == "Figure 2")
        conclusion = figure2.experiment.conclusions[0]
        cited = {e.evidence_id: e for e in figure2.evidence}
        for evidence_id in conclusion.evidence_ids:
            evidence = cited[evidence_id]
            assert evidence.paragraph_id  # → DocNode node_id
            assert evidence.page_no >= 1
            assert evidence.breadcrumb  # → chapter/section path
            assert evidence.figure_id == "Figure 2"


class TestArtifacts:
    def test_files_written(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "art-paper")
        reconstruct_figures("art-paper", tmp_path, config=PaperSemanticsConfig())
        out = Path(tmp_path) / "paper_semantics" / "art-paper"
        assert (out / "figures.json").exists()
        assert (out / "experiments.json").exists()
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["doc_id"] == "art-paper"
        assert manifest["num_figures"] == 5
        assert manifest["status_counts"] == {"SUFFICIENT": 3, "PARTIAL": 1, "INSUFFICIENT": 1}
        assert manifest["input_document_sha256"]

    def test_figures_json_is_byte_reproducible(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "repr-paper")
        reconstruct_figures("repr-paper", tmp_path, config=PaperSemanticsConfig())
        first = (Path(tmp_path) / "paper_semantics" / "repr-paper" / "figures.json").read_bytes()
        reconstruct_figures("repr-paper", tmp_path, config=PaperSemanticsConfig())
        second = (Path(tmp_path) / "paper_semantics" / "repr-paper" / "figures.json").read_bytes()
        assert first == second  # same input → same structured output

    def test_missing_document_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            reconstruct_figures("no-such-doc", tmp_path, config=PaperSemanticsConfig())
