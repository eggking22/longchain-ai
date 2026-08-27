"""Panel-level semantics: caption splitting, mention attribution, panel reconstruction."""

from __future__ import annotations

import pytest

from app.services.paper_semantics import (
    PaperSemanticsConfig,
    SemanticEvidenceGate,
    build_experiment,
    collect_panel_evidence,
    extract_figure_references,
    flatten_document,
    number_evidence,
    panel_labels,
    reconstruct_figures,
    reconstruct_panel,
    split_caption_panels,
)
from app.services.paper_semantics.sections import ParagraphRecord

from .conftest import build_paper_tree, write_document_artifact

CAPTION = (
    "Fig. 2 | CCR7 upregulation in DCs under confinement. "
    "a, GFP intensity in DCs treated with the cPLA2 inhibitor or control. "
    "b, Expression of CCR7 in confined cells. "
    "c, Migration speed of DCs at different heights."
)


class TestSplitCaptionPanels:
    def test_nature_style_split(self):
        panels = split_caption_panels(CAPTION)
        assert sorted(panels) == ["a", "b", "c"]
        assert panels["a"].startswith("a, GFP intensity")
        assert panels["c"].endswith("heights.")

    def test_no_panels_in_single_panel_caption(self):
        assert split_caption_panels("Figure 2. Relative expression of gene X.") == {}

    def test_enumeration_is_not_split(self):
        text = "Cells a, b, c and d were pooled before analysis of GFP expression."
        assert split_caption_panels(text) == {}

    def test_uppercase_panel_labels(self):
        panels = split_caption_panels("Overview. A, Left, migrating cells. B, Right, quantification.")
        assert sorted(panels) == ["a", "b"]  # normalized to lowercase


def _panel_paragraphs():
    return [
        ParagraphRecord(paragraph_id="p-cap", text=CAPTION, section_type="results", order=0),
        ParagraphRecord(
            paragraph_id="p-res-a",
            text="As shown in Fig. 2a, GFP expression decreased in inhibitor-treated cells.",
            section_type="results",
            order=1,
        ),
        ParagraphRecord(
            paragraph_id="p-res-figure",
            text="Figure 2 shows that CCR7 upregulation is shape sensitive overall.",
            section_type="results",
            order=2,
        ),
        ParagraphRecord(
            paragraph_id="p-methods",
            text="Cells were divided into control and treatment groups and cultured for 24 hours.",
            section_type="methods",
            order=3,
        ),
    ]


@pytest.fixture
def panel_setup():
    paragraphs = _panel_paragraphs()
    refs = extract_figure_references(paragraphs, PaperSemanticsConfig())
    return refs[0], paragraphs


class TestPanelAttribution:
    def test_panel_mentions_archived_by_label(self, panel_setup):
        ref, _ = panel_setup
        assert ref.panel_mention_paragraph_ids == {"a": ["p-res-a"]}
        assert "p-res-figure" in ref.mention_paragraph_ids  # figure-level mention stays figure-level

    def test_caption_split_recorded_on_reference(self, panel_setup):
        ref, _ = panel_setup
        assert sorted(ref.panel_texts) == ["a", "b", "c"]

    def test_panel_labels_union_of_caption_and_mentions(self, panel_setup):
        ref, _ = panel_setup
        assert panel_labels(ref) == ["a", "b", "c"]


class TestPanelEvidence:
    def test_panel_bundle_scopes_to_panel(self, panel_setup):
        ref, paragraphs = panel_setup
        evidences = number_evidence(
            collect_panel_evidence(ref, "a", paragraphs, PaperSemanticsConfig()), id_prefix="ev_f02a_"
        )
        assert [e.paragraph_id for e in evidences] == ["p-cap", "p-res-a"]
        assert all(e.panel_id == "2a" for e in evidences)
        assert all(e.paragraph_id != "p-methods" for e in evidences)  # Methods stay at figure level
        assert evidences[0].evidence_id == "ev_f02a_001"

    def test_panel_without_text_mention_is_caption_only(self, panel_setup):
        ref, paragraphs = panel_setup
        evidences = collect_panel_evidence(ref, "b", paragraphs, PaperSemanticsConfig())
        assert [e.role for e in evidences] == ["caption"]


class TestPanelReconstruction:
    def test_panel_a_reaches_sufficient(self, panel_setup):
        ref, paragraphs = panel_setup
        panel = reconstruct_panel(ref, "a", paragraphs, PaperSemanticsConfig())
        assert panel.panel_id == "2a"
        assert panel.reconstruction_status == "SUFFICIENT", panel.missing_information
        assert panel.experiment.experiment_id == "exp_f02a"  # panel-suffixed id
        assert panel.experiment.dependent_variables[0] == "GFP expression"
        assert panel.experiment.observations[0].direction == "decrease"

    def test_panel_without_result_direction_is_partial(self, panel_setup):
        ref, paragraphs = panel_setup
        panel = reconstruct_panel(ref, "b", paragraphs, PaperSemanticsConfig())
        assert panel.reconstruction_status == "PARTIAL"
        assert "direction of change" in panel.missing_information

    def test_panel_never_upgrades_figure_status(self, tmp_path):
        """Baseline lock: panels are additive; the figure-level verdict is frozen."""
        paragraphs = _panel_paragraphs()
        # Figure-level: caption + figure-level mention, no group-bearing Results sentence
        # with treatments → figure PARTIAL while panel a is SUFFICIENT.
        tree = build_paper_tree()
        write_document_artifact(tree, tmp_path, "panel-lock")
        report = reconstruct_figures("panel-lock", tmp_path, config=PaperSemanticsConfig())
        for figure in report.figures:
            assert figure.panels == [] or figure.reconstruction_status in ("SUFFICIENT", "PARTIAL", "INSUFFICIENT")
        status_counts = report.stats["status_counts"]
        assert status_counts == {"SUFFICIENT": 3, "PARTIAL": 1, "INSUFFICIENT": 1}  # golden baseline unchanged

    def test_gate_uses_panel_caption_view(self, panel_setup):
        """The gate is reused unchanged via a panel view of FigureReference."""
        ref, paragraphs = panel_setup
        evidences = number_evidence(
            collect_panel_evidence(ref, "a", paragraphs, PaperSemanticsConfig()), id_prefix="ev_f02a_"
        )
        from app.services.paper_semantics.figure_reference import FigureReference

        panel_view = FigureReference(
            figure_id="Figure 2A", kind="figure", number=2, caption_text=ref.panel_texts["a"]
        )
        experiment = build_experiment(ref, evidences, id_suffix="a")
        verdict = SemanticEvidenceGate().evaluate(panel_view, experiment, evidences)
        assert verdict.status == "SUFFICIENT"
