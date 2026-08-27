"""Figure/Table reference extraction: mentions, captions, canonicalization."""

from __future__ import annotations

import pytest

from app.services.paper_semantics import PaperSemanticsConfig, extract_figure_references, find_mentions, parse_caption
from app.services.paper_semantics.sections import ParagraphRecord

from .conftest import build_paper_tree
from app.services.paper_semantics import flatten_document


@pytest.fixture
def paragraphs():
    return flatten_document(build_paper_tree())


class TestFindMentions:
    def test_basic_forms(self):
        mentions = find_mentions("Figure 1 shows data. (Fig. 2B) figure 3a and Table 4 were omitted.")
        assert [(m.kind, m.number, m.panel) for m in mentions] == [
            ("figure", 1, ""),
            ("figure", 2, "B"),
            ("figure", 3, "A"),
            ("table", 4, ""),
        ]

    def test_multi_number_forms(self):
        mentions = find_mentions("Figures 2 and 3 were combined. Figure 5-6 show dose response.")
        assert [m.number for m in mentions] == [2, 3, 5, 6]

    def test_no_false_positives(self):
        assert find_mentions("Configuration 7 was applied to the figuratively large dataset.") == []

    def test_raw_surface_form(self):
        mentions = find_mentions("As shown in Fig. 2B, expression increased.")
        assert mentions[0].raw == "Fig. 2B"


class TestParseCaption:
    @pytest.mark.parametrize(
        "text,kind,number",
        [
            ("Figure 2. Relative expression of gene X.", "figure", 2),
            ("Fig. 1: Growth curves of mutants.", "figure", 1),
            ("Table 3. Summary of patient characteristics.", "table", 3),
            ("Fig. 1 | CCR7 upregulation in immature DCs is shape sensitive.", "figure", 1),  # Nature style
            ("Extended Data Fig. 2 | Gating strategy.", "figure", 2),
        ],
    )
    def test_caption_forms(self, text, kind, number):
        caption = parse_caption(text)
        assert caption is not None
        assert caption.kind == kind
        assert caption.number == number

    def test_mention_is_not_caption(self):
        assert parse_caption("Figure 2 shows the experimental results.") is None
        assert parse_caption("We refer to Table 1 in the next section.") is None


class TestNamespacedFigures:
    def test_extended_data_is_isolated_from_main_figure(self):
        records = [
            ParagraphRecord(paragraph_id="p1", text="Figure 1 shows the main result.", section_type="results", order=0),
            ParagraphRecord(
                paragraph_id="p2", text="Extended Data Fig. 1 shows the gating strategy.", section_type="results", order=1
            ),
        ]
        refs = extract_figure_references(records, PaperSemanticsConfig())
        assert sorted(ref.figure_id for ref in refs) == ["Extended Data Figure 1", "Figure 1"]
        main = next(r for r in refs if r.figure_id == "Figure 1")
        assert main.mention_paragraph_ids == ["p1"]  # Extended Data mention did not leak in

    def test_supplementary_prefix_normalized(self):
        mentions = find_mentions("See Supplementary Fig. 3 and Supplementary Table 2 for details.")
        assert [m.prefix for m in mentions] == ["Supplementary", "Supplementary"]
        assert [(m.number, m.kind if hasattr(m, "kind") else None) for m in mentions] == [(3, "figure"), (2, "table")]

    def test_namespaced_sorting(self):
        records = [
            ParagraphRecord(paragraph_id="p1", text="Supplementary Figure 1 shows A.", section_type="results", order=0),
            ParagraphRecord(paragraph_id="p2", text="Extended Data Fig. 1 shows B.", section_type="results", order=1),
            ParagraphRecord(paragraph_id="p3", text="Figure 1 shows C.", section_type="results", order=2),
        ]
        refs = extract_figure_references(records, PaperSemanticsConfig())
        assert [ref.figure_id for ref in refs] == ["Figure 1", "Extended Data Figure 1", "Supplementary Figure 1"]


class TestExtractFigureReferences:
    def test_canonicalization_and_captions(self, paragraphs):
        refs = {ref.figure_id: ref for ref in extract_figure_references(paragraphs, PaperSemanticsConfig())}
        assert set(refs) == {"Figure 2", "Figure 3", "Figure 4", "Figure 5", "Table 1"}

        fig2 = refs["Figure 2"]
        assert fig2.caption_paragraph_id == "p-cap-fig2"
        assert fig2.caption_text.startswith("Figure 2. Relative expression")
        assert "p-results-1" in fig2.mention_paragraph_ids
        assert "p-disc-1" in fig2.mention_paragraph_ids

        fig3 = refs["Figure 3"]
        assert fig3.caption_paragraph_id is None  # bare mention, no caption anywhere
        assert fig3.mention_paragraph_ids == ["p-results-3"]

    def test_panel_mentions_roll_up(self):
        records = [
            ParagraphRecord(
                paragraph_id="p1",
                text="Figure 2A shows mRNA levels. Figure 2B shows protein levels.",
                section_type="results",
                order=0,
            ),
            ParagraphRecord(paragraph_id="p2", text="Figure 2. Overview of results.", section_type="results", order=1),
        ]
        refs = extract_figure_references(records, PaperSemanticsConfig())
        assert len(refs) == 1
        assert refs[0].figure_id == "Figure 2"
        assert sorted(refs[0].subfigures) == ["2A", "2B"]

    def test_figure_and_table_numbers_are_distinct(self, paragraphs):
        refs = extract_figure_references(paragraphs, PaperSemanticsConfig())
        assert [ref.figure_id for ref in refs] == ["Figure 2", "Figure 3", "Figure 4", "Figure 5", "Table 1"]
        assert refs[-1].kind == "table"

    def test_caption_length_guard(self):
        long_text = "Figure 9. " + "Very long caption. " * 60  # > 400 chars → not treated as a caption
        records = [ParagraphRecord(paragraph_id="p1", text=long_text.strip(), section_type="results", order=0)]
        refs = extract_figure_references(records, PaperSemanticsConfig())
        assert refs[0].caption_paragraph_id is None
        assert refs[0].mention_paragraph_ids == ["p1"]  # still usable as a plain mention paragraph


class TestInlineCaption:
    def test_pipe_caption_embedded_mid_paragraph(self):
        """Two-column PDFs glue body text onto captions; the pipe form still locates them."""
        text = (
            "To migrate to lymph nodes, DCs must express the CCR7 chemokine receptor. We have previously "
            "Fig. 1 | CCR7 upregulation in immature DCs is shape sensitive. "
            "a, Left, migrating CD11c+ DCs in an ear explant."
        )
        records = [ParagraphRecord(paragraph_id="p1", text=text, section_type="results", order=0)]
        refs = extract_figure_references(records, PaperSemanticsConfig())
        assert len(refs) == 1
        ref = refs[0]
        assert ref.figure_id == "Figure 1"
        assert ref.caption_paragraph_id == "p1"
        assert ref.caption_offset > 0
        assert ref.caption_text.startswith("Fig. 1 | CCR7 upregulation")
        assert ref.mention_paragraph_ids == []  # the caption's own label is not a self-mention

    def test_inline_caption_evidence_uses_sliced_text(self):
        from app.services.paper_semantics import collect_evidence, number_evidence

        text = (
            "Body sentence about dendritic cells. "
            "Fig. 2 | GFP intensity in DCs treated with the cPLA2 inhibitor or control. "
            "Data were analyzed by Mann-Whitney test."
        )
        records = [ParagraphRecord(paragraph_id="p9", text=text, section_type="results", order=0)]
        refs = extract_figure_references(records, PaperSemanticsConfig())
        evidences = number_evidence(collect_evidence(refs[0], records, PaperSemanticsConfig()))
        caption = [e for e in evidences if e.role == "caption"]
        assert len(caption) == 1
        assert caption[0].text.startswith("Fig. 2 | GFP intensity")  # body prefix not part of caption evidence
        assert caption[0].paragraph_id == "p9"
