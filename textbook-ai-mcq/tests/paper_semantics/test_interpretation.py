"""Three-layer reasoning chain: Observation → Interpretation → Conclusion."""

from __future__ import annotations

from app.schemas.paper_semantics import PaperEvidence
from app.services.paper_semantics import (
    PaperSemanticsConfig,
    build_conclusions,
    build_experiment,
    collect_evidence,
    extract_figure_references,
    extract_interpretations,
    flatten_document,
    number_evidence,
    reconstruct_figures,
)
from app.services.paper_semantics.figure_reference import FigureReference

from .conftest import build_paper_tree, write_document_artifact


def _figure(paragraphs, figure_id):
    refs = {ref.figure_id: ref for ref in extract_figure_references(paragraphs, PaperSemanticsConfig())}
    ref = refs[figure_id]
    evidences = number_evidence(collect_evidence(ref, paragraphs, PaperSemanticsConfig()))
    experiment = build_experiment(ref, evidences)
    experiment.conclusions = build_conclusions(experiment)
    return ref, evidences, experiment


class TestInterpretationExtraction:
    def test_discussion_sentence_becomes_interpretation(self):
        paragraphs = flatten_document(build_paper_tree())
        _, evidences, experiment = _figure(paragraphs, "Figure 2")
        interpretation_evidence = [e for e in evidences if e.role == "interpretation"]
        assert len(interpretation_evidence) == 1  # the Discussion paragraph

        assert len(experiment.interpretations) == 1
        interpretation = experiment.interpretations[0]
        # verbatim author claim — recorded, not paraphrased
        assert interpretation.statement.startswith("Our findings for Figure 2 suggest")
        assert interpretation.direction == "increase"
        assert interpretation.interpretation_id == "int_001"
        assert interpretation.evidence_ids == [interpretation_evidence[0].evidence_id]

    def test_stats_notes_never_become_interpretations(self):
        evidences = [
            PaperEvidence(
                evidence_id="ev_001",
                figure_id="Figure 1",
                text="We conclude that this pathway promotes gene X expression. Data were analyzed by t-test; ****P < 0.0001.",
                role="interpretation",
            )
        ]
        interpretations = extract_interpretations(evidences)
        assert [i.statement for i in interpretations] == ["We conclude that this pathway promotes gene X expression."]

    def test_no_discussion_no_interpretations(self):
        paragraphs = flatten_document(build_paper_tree())
        _, _, experiment = _figure(paragraphs, "Figure 4")
        assert experiment.interpretations == []

    def test_association_kept_verbatim(self):
        evidences = [
            PaperEvidence(
                evidence_id="ev_005",
                figure_id="Figure 5",
                text="Treatment C was associated with increased expression of gene Z.",
                role="interpretation",
            )
        ]
        interpretations = extract_interpretations(evidences)
        assert interpretations[0].relationship_type == "association"
        assert interpretations[0].direction == "increase"


class TestLayerSeparation:
    def test_three_layers_are_distinct(self):
        paragraphs = flatten_document(build_paper_tree())
        _, evidences, experiment = _figure(paragraphs, "Figure 2")
        assert experiment.observations and experiment.interpretations and experiment.conclusions

        observation_evidence = {eid for o in experiment.observations for eid in o.evidence_ids}
        interpretation_evidence = {eid for i in experiment.interpretations for eid in i.evidence_ids}
        assert observation_evidence & interpretation_evidence == set()  # Results vs Discussion ids

        conclusion = experiment.conclusions[0]
        assert conclusion.statement == "Treatment A increases gene X expression."
        assert conclusion.evidence_ids  # conclusions cite observation evidence...
        assert set(conclusion.evidence_ids) <= observation_evidence
        assert conclusion.interpretation_ids == ["int_001"]  # ...and *link* interpretations

    def test_interpretation_linking_requires_matching_direction(self):
        evidences = [
            PaperEvidence(evidence_id="ev_001", figure_id="Figure 1", text="Setup.", role="caption"),
            PaperEvidence(
                evidence_id="ev_002",
                figure_id="Figure 1",
                text="Expression of gene X increased in treated cells.",
                role="direct",
            ),
            PaperEvidence(
                evidence_id="ev_003",
                figure_id="Figure 1",
                text="We conclude that gene Y decreased independently.",
                role="interpretation",
            ),
        ]
        ref = FigureReference(figure_id="Figure 1", kind="figure", number=1, caption_text="Setup.")
        experiment = build_experiment(ref, evidences)
        experiment.intervention = "Treatment A"
        experiment.independent_variables = ["Treatment A"]
        experiment.dependent_variables = ["gene X expression"]
        experiment.conclusions = build_conclusions(experiment)
        conclusion = experiment.conclusions[0]
        assert conclusion.interpretation_ids == []  # direction mismatch → not linked


class TestPipelineOutput:
    def test_report_contains_all_three_layers(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "layers-paper")
        report = reconstruct_figures("layers-paper", tmp_path, config=PaperSemanticsConfig())
        figure2 = next(f for f in report.figures if f.figure_id == "Figure 2")
        experiment = figure2.experiment
        assert experiment.observations[0].statement.startswith("Treatment A significantly increased")
        assert experiment.interpretations[0].statement.startswith("Our findings")
        assert experiment.conclusions[0].interpretation_ids == ["int_001"]
