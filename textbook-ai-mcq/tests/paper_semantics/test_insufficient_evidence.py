"""Evidence-safety: insufficient evidence must yield INSUFFICIENT, never a guess."""

from __future__ import annotations

import pytest

from app.schemas.paper_semantics import ExperimentModel
from app.services.paper_semantics import (
    PaperSemanticsConfig,
    SemanticEvidenceGate,
    build_conclusions,
    build_experiment,
    collect_evidence,
    extract_figure_references,
    flatten_document,
    number_evidence,
)
from app.services.paper_semantics.figure_reference import FigureReference
from app.services.paper_semantics.sections import ParagraphRecord

from .conftest import build_paper_tree


@pytest.fixture
def paragraphs():
    return flatten_document(build_paper_tree())


def _reconstruct(paragraphs, figure_id):
    refs = {ref.figure_id: ref for ref in extract_figure_references(paragraphs, PaperSemanticsConfig())}
    ref = refs[figure_id]
    evidences = number_evidence(collect_evidence(ref, paragraphs, PaperSemanticsConfig()))
    experiment = build_experiment(ref, evidences)
    experiment.conclusions = build_conclusions(experiment)
    verdict = SemanticEvidenceGate().evaluate(ref, experiment, evidences)
    return ref, experiment, verdict


class TestInsufficient:
    def test_bare_mention_is_insufficient(self, paragraphs):
        """Only 'Figure 3 shows the experimental results.' → INSUFFICIENT, no guessing."""
        ref, experiment, verdict = _reconstruct(paragraphs, "Figure 3")
        assert verdict.status == "INSUFFICIENT"
        assert verdict.slots["result_direction"] is False
        assert experiment.observations == []
        assert experiment.conclusions == []
        assert "direction of change" in verdict.missing_information
        assert "experimental vs control group comparison" in verdict.missing_information

    def test_insufficient_gate_unit(self):
        gate = SemanticEvidenceGate()
        ref = FigureReference(figure_id="Figure 3", kind="figure", number=3)
        experiment = ExperimentModel(experiment_id="exp_f03")
        verdict = gate.evaluate(ref, experiment, [])
        assert verdict.status == "INSUFFICIENT"
        assert verdict.confidence == 0.0
        assert len(verdict.missing_information) == 6


class TestPartial:
    def test_known_experiment_but_no_direction(self, paragraphs):
        """Caption names the endpoint but no result statement exists → PARTIAL."""
        _, _, verdict = _reconstruct(paragraphs, "Figure 4")
        assert verdict.status == "PARTIAL"
        assert "direction of change" in verdict.missing_information
        assert 0.0 < verdict.confidence < 1.0

    def test_partial_gate_unit(self):
        gate = SemanticEvidenceGate()
        ref = FigureReference(figure_id="Figure 4", kind="figure", number=4, caption_text="Assay setup.")
        experiment = ExperimentModel(experiment_id="exp_f04", dependent_variables=["gene Y expression"])
        verdict = gate.evaluate(ref, experiment, [])
        assert verdict.status == "PARTIAL"


class TestSufficient:
    def test_full_evidence_is_sufficient(self, paragraphs):
        _, experiment, verdict = _reconstruct(paragraphs, "Figure 2")
        assert verdict.status == "SUFFICIENT"
        assert verdict.missing_information == []
        assert verdict.confidence >= 0.8
        assert experiment.conclusions  # a conclusion could be safely drawn


class TestNoFabrication:
    def test_significantly_is_not_a_p_value(self, paragraphs):
        _, experiment, _ = _reconstruct(paragraphs, "Figure 2")
        observation = experiment.observations[0]
        assert observation.significance == "significant"
        assert observation.p_value is None  # "significantly" must not become "p < 0.05"
        assert experiment.statistical_results == []

    def test_no_numbers_invented(self, paragraphs):
        _, experiment, _ = _reconstruct(paragraphs, "Figure 2")
        for conclusion in experiment.conclusions:
            assert not any(char.isdigit() for char in conclusion.statement)

    def test_unclassified_sections_still_direct(self):
        """Section classification failing (fallback heading) must not lose direct evidence."""
        records = [
            ParagraphRecord(paragraph_id="p1", text="Figure 2. Relative expression of gene X.", section_type="other", order=0),
            ParagraphRecord(
                paragraph_id="p2",
                text="Treatment A significantly increased expression of gene X compared with control (Figure 2).",
                section_type="other",
                order=1,
            ),
        ]
        refs = extract_figure_references(records, PaperSemanticsConfig())
        evidences = number_evidence(collect_evidence(refs[0], records, PaperSemanticsConfig()))
        assert [e.role for e in evidences] == ["caption", "direct"]
        experiment = build_experiment(refs[0], evidences)
        verdict = SemanticEvidenceGate().evaluate(refs[0], experiment, evidences)
        assert verdict.status == "SUFFICIENT"
