"""Deterministic experiment-model extraction: groups, variables, observations."""

from __future__ import annotations

import pytest

from app.schemas.paper_semantics import PaperEvidence
from app.services.paper_semantics.figure_reference import FigureReference
from app.services.paper_semantics import (
    PaperSemanticsConfig,
    build_experiment,
    classify_relationship,
    collect_evidence,
    detect_direction,
    detect_significance,
    extract_figure_references,
    extract_intervention,
    flatten_document,
    number_evidence,
)

from .conftest import build_paper_tree


def _ev(text: str, role: str = "direct", evidence_id: str = "ev_001") -> PaperEvidence:
    return PaperEvidence(evidence_id=evidence_id, figure_id="Figure 2", text=text, role=role)  # type: ignore[arg-type]


@pytest.fixture
def paragraphs():
    return flatten_document(build_paper_tree())


def _bundle(paragraphs, figure_id):
    refs = {ref.figure_id: ref for ref in extract_figure_references(paragraphs, PaperSemanticsConfig())}
    return refs[figure_id], number_evidence(collect_evidence(refs[figure_id], paragraphs, PaperSemanticsConfig()))


class TestDetectDirection:
    @pytest.mark.parametrize(
        "sentence,expected",
        [
            ("Treatment A increased expression of gene X.", "increase"),
            ("Expression of gene X was reduced by the inhibitor.", "decrease"),
            ("There was no significant difference between groups.", "no_change"),
            ("The apparatus was cleaned before use.", "unspecified"),
            ("Levels decreased then increased over time.", "decrease"),  # first marker wins
        ],
    )
    def test_direction(self, sentence, expected):
        assert detect_direction(sentence) == expected


class TestDetectSignificance:
    @pytest.mark.parametrize(
        "sentence,expected",
        [
            ("Expression increased significantly.", "significant"),
            ("The difference was not significant.", "not_significant"),
            ("Expression increased.", "unspecified"),
        ],
    )
    def test_significance(self, sentence, expected):
        assert detect_significance(sentence) == expected

    def test_significantly_does_not_fabricate_p_value(self):
        observation_texts = "Treatment A significantly increased expression of gene X."
        assert detect_significance(observation_texts) == "significant"
        ref, evidences = _bundle(flatten_document(build_paper_tree()), "Figure 2")
        experiment = build_experiment(ref, evidences)
        assert all(o.p_value is None for o in experiment.observations)
        assert experiment.statistical_results == []

    def test_literal_p_value_is_recorded(self):
        ref, evidences = _bundle(flatten_document(build_paper_tree()), "Figure 2")
        evidences.append(
            _ev("Expression differed between the groups (p < 0.05).", role="direct", evidence_id="ev_099")
        )
        experiment = build_experiment(ref, evidences)
        assert experiment.statistical_results == ["p<0.05"]
        matching = [o for o in experiment.observations if o.p_value]
        assert [o.p_value for o in matching] == ["p < 0.05"]


class TestClassifyRelationship:
    @pytest.mark.parametrize(
        "sentence,expected",
        [
            ("gene A expression correlated with gene B expression.", "correlation"),
            ("Treatment C was associated with increased expression of gene Z.", "association"),
            ("Loss of the transporter caused iron accumulation.", "causal"),
            ("Treatment A increased expression of gene X compared with control.", "causal"),
            ("The inhibitor suppressed phosphorylation of AKT.", "inhibition"),
            ("The ligand activated downstream signaling.", "activation"),
            ("CRISPR knockout of gene Y impaired growth.", "knockout"),
            ("Overexpression of gene Y rescued the phenotype.", "overexpression"),
            ("Growth increased in a dose-dependent manner.", "dose_response"),
            ("Expression changed over time after stimulus.", "time_dependent"),
            ("The samples were stored at 4 degrees.", "unspecified"),
        ],
    )
    def test_relationship(self, sentence, expected):
        assert classify_relationship(sentence) == expected

    def test_correlation_is_never_causal(self):
        sentence = "gene A expression correlated with gene B expression."
        assert classify_relationship(sentence) == "correlation"


class TestGroupExtraction:
    def test_canonical_groups(self, paragraphs):
        ref, evidences = _bundle(paragraphs, "Figure 2")
        experiment = build_experiment(ref, evidences)
        assert "Treatment A" in experiment.experimental_groups
        assert "control" in experiment.control_groups
        assert experiment.intervention == "Treatment A"

    def test_generic_treatment_group_is_not_intervention(self):
        assert extract_intervention(["treatment", "Treatment A"]) == "Treatment A"
        assert extract_intervention(["treatment"]) == "treatment"  # generic fallback
        assert extract_intervention([]) == ""

    def test_treated_with_prefix_is_stripped(self):
        evidences = [
            _ev("GFP intensity in DCs treated with the cPLA2 inhibitor AACOF3 (25 uM) or control.", role="caption")
        ]
        experiment = build_experiment(
            FigureReference(figure_id="Figure 2", kind="figure", number=2), evidences
        )
        assert "cPLA2 inhibitor" in experiment.experimental_groups  # not "treated with the ..."


class TestStatsNoteGuard:
    def test_caption_stats_notes_are_not_observations(self):
        evidences = [
            _ev(
                "Fig. 1 | Shape sensing of DCs. "
                "Data were analyzed by Kruskal-Wallis test; ****P < 0.0001; NS, not significant (P > 0.999). "
                "Only immature DCs confined at 3 um exhibited a significant increase in CCR7 expression (Fig. 1).",
                role="caption",
            )
        ]
        experiment = build_experiment(FigureReference(figure_id="Figure 1", kind="figure", number=1), evidences)
        statements = [o.statement for o in experiment.observations]
        assert len(statements) == 1
        assert statements[0].startswith("Only immature DCs confined")
        # both literal p-values of the stats note are recorded verbatim
        assert set(experiment.statistical_results) == {"P<0.0001", "P>0.999"}


class TestVariables:
    def test_canonical_variables(self, paragraphs):
        ref, evidences = _bundle(paragraphs, "Figure 2")
        experiment = build_experiment(ref, evidences)
        assert experiment.dependent_variables[0] == "gene X expression"
        assert experiment.independent_variables == ["Treatment A"]
        assert experiment.research_question == "Does Treatment A affect gene X expression?"
        assert "cells" in experiment.subjects

    def test_body_weight_suffix_preferred(self, paragraphs):
        ref, evidences = _bundle(paragraphs, "Table 1")
        experiment = build_experiment(ref, evidences)
        assert experiment.dependent_variables[0] == "body weight"

    def test_observation_bindings(self, paragraphs):
        ref, evidences = _bundle(paragraphs, "Figure 2")
        experiment = build_experiment(ref, evidences)
        assert len(experiment.observations) >= 1
        observation = experiment.observations[0]
        assert observation.statement.startswith("Treatment A significantly increased")
        assert observation.direction == "increase"
        assert observation.significance == "significant"
        assert observation.relationship_type == "causal"
        assert observation.evidence_ids  # bound to the Results paragraph's evidence id
