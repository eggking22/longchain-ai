"""L1 deterministic partition of the Results narrative into per-figure text blocks."""

from __future__ import annotations

import pytest

from app.schemas.paper_semantics import PaperEvidence
from app.services.paper_semantics import (
    PaperSemanticsConfig,
    build_experiment,
    collect_evidence,
    extract_figure_references,
    number_evidence,
    partition_results_by_figures,
)
from app.services.paper_semantics.sections import ParagraphRecord


def _pr(pid: str, text: str, section: str = "results", order: int | None = None) -> ParagraphRecord:
    return ParagraphRecord(paragraph_id=pid, text=text, section_type=section, order=order or 0)


PARAGRAPHS = [
    _pr("r0", "Overall the study examined dendritic cell migration in tissues.", order=0),  # before first anchor
    _pr("r1", "Confinement at 3 um increased CCR7 expression (Fig. 2).", order=1),
    _pr("r2", "These cells also upregulated chemokine receptors.", order=2),  # continuation of Figure 2
    _pr("r3", "We next asked whether motility was affected.", order=3),  # continuation of Figure 2
    _pr("r4", "cPLA2 inhibition abolished the response (Fig. 3 and Fig. 2).", order=4),  # anchor Figure 3, shared with 2
    _pr("r5", "Migration speed was quantified thereafter.", order=5),  # continuation of Figure 3
    _pr("m1", "Cells were divided into control and treatment groups.", section="methods", order=6),  # excluded
    _pr("d1", "These data suggest shape sensing licenses migration.", section="discussion", order=7),  # excluded
]


@pytest.fixture
def refs_and_blocks():
    refs = extract_figure_references(PARAGRAPHS, PaperSemanticsConfig())
    known = {ref.figure_id: ref for ref in refs}
    blocks = partition_results_by_figures(PARAGRAPHS, refs)
    return known, blocks


class TestL1Rules:
    def test_anchor_opens_block_and_inheritance(self, refs_and_blocks):
        _, blocks = refs_and_blocks
        block2 = blocks["Figure 2"]
        assert block2.figure_id == "Figure 2"
        assert block2.anchor_paragraph_ids == ["r1"]
        assert block2.continuation_paragraph_ids == ["r2", "r3"]

    def test_new_anchor_closes_previous_block(self, refs_and_blocks):
        _, blocks = refs_and_blocks
        assert blocks["Figure 3"].anchor_paragraph_ids == ["r4"]
        assert blocks["Figure 3"].continuation_paragraph_ids == ["r5"]

    def test_multi_figure_paragraph_primary_and_shared(self, refs_and_blocks):
        _, blocks = refs_and_blocks
        # "Fig. 3 and Fig. 2": first-cited (Figure 3) owns the anchor, Figure 2 is shared_with
        assert "r4" in blocks["Figure 3"].anchor_paragraph_ids
        assert "Figure 2" in blocks["Figure 3"].shared_with
        assert "r4" not in blocks["Figure 2"].anchor_paragraph_ids

    def test_leading_unassigned_safety_valve(self, refs_and_blocks):
        _, blocks = refs_and_blocks
        assigned = {pid for block in blocks.values() for pid in block.paragraph_ids}
        assert "r0" not in assigned  # before the first anchor → never guessed

    def test_methods_and_discussion_never_partitioned(self, refs_and_blocks):
        _, blocks = refs_and_blocks
        assigned = {pid for block in blocks.values() for pid in block.paragraph_ids}
        assert "m1" not in assigned and "d1" not in assigned

    def test_panel_mention_rolls_up_to_parent(self):
        paragraphs = [
            _pr("p1", "GFP expression decreased in panel Fig. 2B of the assay.", order=0),
            _pr("p2", "Further quantification confirmed the effect.", order=1),
        ]
        refs = extract_figure_references(paragraphs, PaperSemanticsConfig())
        blocks = partition_results_by_figures(paragraphs, refs)
        assert blocks["Figure 2"].anchor_paragraph_ids == ["p1"]
        assert blocks["Figure 2"].continuation_paragraph_ids == ["p2"]

    def test_paragraph_ids_property_is_ordered_union(self, refs_and_blocks):
        _, blocks = refs_and_blocks
        assert blocks["Figure 2"].paragraph_ids == ["r1", "r2", "r3"]


class TestEvidenceWiring:
    def test_anchor_and_continuation_assignments(self, refs_and_blocks):
        known, _ = refs_and_blocks
        block = known_refs_block(known)
        evidences = number_evidence(collect_evidence(known["Figure 2"], PARAGRAPHS, PaperSemanticsConfig(), text_block=block))
        anchors = {e.paragraph_id for e in evidences if e.assignment == "anchor"}
        continuations = {e.paragraph_id for e in evidences if e.assignment == "continuation"}
        assert anchors == {"r1", "r4"}  # r4 cites Fig. 2 explicitly (shared paragraph) → anchor too
        assert continuations == {"r2", "r3"}

    def test_continuation_excluded_from_semantic_extraction(self, refs_and_blocks):
        """Baseline lock: inherited paragraphs widen the bundle but never produce semantics."""
        known, _ = refs_and_blocks
        block = known_refs_block(known)
        paragraphs_with_marker = [
            _pr("r1", "Confinement at 3 um increased CCR7 expression (Fig. 2).", order=1),
            _pr("r2", "Expression of gene Z decreased dramatically.", order=2),  # continuation WITH a marker
        ]
        refs = extract_figure_references(paragraphs_with_marker, PaperSemanticsConfig())
        blocks = partition_results_by_figures(paragraphs_with_marker, refs)
        evidences = number_evidence(
            collect_evidence(refs[0], paragraphs_with_marker, PaperSemanticsConfig(), text_block=blocks["Figure 2"])
        )
        experiment = build_experiment(refs[0], evidences)
        # the marker sentence lives in continuation evidence and must NOT become an observation
        assert all("gene Z" not in o.statement for o in experiment.observations)
        # but the continuation text is still stored in the bundle for downstream use
        assert any("gene Z" in e.text for e in evidences if e.assignment == "continuation")

    def test_shared_paragraph_is_anchor_for_both_figures(self, refs_and_blocks):
        known, _ = refs_and_blocks
        refs_list = [known["Figure 2"], known["Figure 3"]]
        for ref in refs_list:
            evidences = collect_evidence(ref, PARAGRAPHS, PaperSemanticsConfig(), text_block=None)
            assert any(e.paragraph_id == "r4" for e in evidences if e.role == "direct")


def known_refs_block(known):
    from app.services.paper_semantics import partition_results_by_figures as partition

    return partition(PARAGRAPHS, list(known.values()))["Figure 2"]
