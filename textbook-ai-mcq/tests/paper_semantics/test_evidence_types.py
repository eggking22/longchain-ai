"""Evidence semantic typing: source role vs evidence_type, stats retention, queryability."""

from __future__ import annotations

from app.services.paper_semantics import (
    PaperSemanticsConfig,
    collect_evidence,
    extract_figure_references,
    flatten_document,
    infer_evidence_type,
    number_evidence,
)

from .conftest import build_paper_tree


def _bundle(paragraphs, figure_id):
    refs = {ref.figure_id: ref for ref in extract_figure_references(paragraphs, PaperSemanticsConfig())}
    return refs[figure_id], number_evidence(collect_evidence(refs[figure_id], paragraphs, PaperSemanticsConfig()))


class TestInferEvidenceType:
    def test_source_role_mapping(self):
        assert infer_evidence_type("direct", "Anything from Results.") == "direct_observation"
        assert infer_evidence_type("supporting", "Methods text.") == "experimental_design"
        assert infer_evidence_type("interpretation", "Discussion text.") == "author_interpretation"
        assert infer_evidence_type("unknown-role", "text") == "mixed"

    def test_caption_content_mapping(self):
        design = infer_evidence_type("caption", "a, GFP intensity in DCs treated with the inhibitor or control.")
        assert design == "experimental_design"
        observation = infer_evidence_type("caption", "Expression of gene X increased in treated cells.")
        assert observation == "direct_observation"
        stats = infer_evidence_type("caption", "Data were analyzed by Mann-Whitney test.")
        assert stats == "statistical_result"


class TestGoldenBundleTypes:
    def test_figure2_roles_and_types(self):
        paragraphs = flatten_document(build_paper_tree())
        _, evidences = _bundle(paragraphs, "Figure 2")
        by_role = {e.role: e.evidence_type for e in evidences}
        assert by_role["caption"] in ("experimental_design", "direct_observation")
        assert by_role["direct"] == "direct_observation"
        assert by_role["supporting"] == "experimental_design"
        assert by_role["interpretation"] == "author_interpretation"


class TestStatsNoteRetention:
    def test_stats_note_becomes_statistical_result_evidence(self):
        paragraphs = flatten_document(build_paper_tree())
        # inject a stats note into the Figure 2 caption paragraph
        for paragraph in paragraphs:
            if paragraph.paragraph_id == "p-cap-fig2":
                paragraph.text = (
                    "Figure 2. Relative expression of gene X in control and treatment groups. "
                    "Data were analyzed by Kruskal-Wallis test; ****P < 0.0001."
                )
        _, evidences = _bundle(paragraphs, "Figure 2")
        stats = [e for e in evidences if e.evidence_type == "statistical_result"]
        assert len(stats) == 1
        assert stats[0].text.startswith("Data were analyzed")
        assert stats[0].paragraph_id == "p-cap-fig2"
        assert stats[0].role == "caption"  # source dimension preserved
        # observations are unaffected by the stats note
        from app.services.paper_semantics import build_experiment

        ref = extract_figure_references(paragraphs, PaperSemanticsConfig())[0]
        experiment = build_experiment(ref, evidences)
        assert all(not o.statement.startswith("Data were") for o in experiment.observations)


class TestQuestionRuleSupport:
    def test_evidence_of_type_filter(self):
        """Example MCQ rule: 'conclusion questions require direct_observation evidence'."""
        from app.services.paper_semantics import reconstruct_figures

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            write = Path(tmp)
            from .conftest import write_document_artifact

            write_document_artifact(build_paper_tree(), write, "types-paper")
            report = reconstruct_figures("types-paper", write, config=PaperSemanticsConfig())
        figure2 = next(f for f in report.figures if f.figure_id == "Figure 2")
        assert figure2.evidence_of_type("direct_observation"), "conclusion-grade evidence available"
        figure3 = next(f for f in report.figures if f.figure_id == "Figure 3")
        direct = figure3.evidence_of_type("direct_observation")
        # Figure 3 does have Results-sourced evidence (typed direct_observation), but it is a
        # bare mention — typing alone is not sufficient; the gate + conclusions enforce substance.
        assert direct and all("shows the experimental results" in e.text for e in direct)
        assert figure3.experiment.conclusions == []
