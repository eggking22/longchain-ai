"""Evidence collection: roles, priority, provenance, caps."""

from __future__ import annotations

import pytest

from app.services.paper_semantics import (
    PaperSemanticsConfig,
    collect_evidence,
    extract_figure_references,
    flatten_document,
    number_evidence,
)

from .conftest import build_paper_tree


@pytest.fixture
def paragraphs():
    return flatten_document(build_paper_tree())


@pytest.fixture
def refs(paragraphs):
    return {ref.figure_id: ref for ref in extract_figure_references(paragraphs, PaperSemanticsConfig())}


class TestRolesAndPriority:
    def test_figure2_bundle_roles_in_priority_order(self, paragraphs, refs):
        evidences = number_evidence(collect_evidence(refs["Figure 2"], paragraphs, PaperSemanticsConfig()))
        roles = [e.role for e in evidences]
        assert roles[0] == "caption"  # caption first
        assert "direct" in roles  # Results mention
        assert "supporting" in roles  # Methods paragraph (term overlap with caption/results)
        assert "interpretation" in roles  # Discussion mention

    def test_evidence_ids_sequential(self, paragraphs, refs):
        evidences = number_evidence(collect_evidence(refs["Figure 2"], paragraphs, PaperSemanticsConfig()))
        assert [e.evidence_id for e in evidences] == [f"ev_{i:03d}" for i in range(1, len(evidences) + 1)]

    def test_bare_mention_gets_no_methods_pollution(self, paragraphs, refs):
        """Figure 3's only evidence shares no content words with Methods → no supporting evidence."""
        evidences = collect_evidence(refs["Figure 3"], paragraphs, PaperSemanticsConfig())
        assert [e.role for e in evidences] == ["direct"]
        assert evidences[0].paragraph_id == "p-results-3"

    def test_discussion_mention_is_interpretation(self, paragraphs, refs):
        evidences = collect_evidence(refs["Figure 2"], paragraphs, PaperSemanticsConfig())
        interpretation = [e for e in evidences if e.role == "interpretation"]
        assert [e.paragraph_id for e in interpretation] == ["p-disc-1"]


class TestProvenance:
    def test_fields_trace_back_to_paragraph(self, paragraphs, refs):
        evidences = collect_evidence(refs["Figure 2"], paragraphs, PaperSemanticsConfig())
        direct = [e for e in evidences if e.role == "direct"][0]
        assert direct.paragraph_id == "p-results-1"
        assert direct.page_no == 2
        assert direct.section_type == "results"
        assert direct.section_title == "Results"
        assert "Results" in direct.breadcrumb
        assert direct.figure_id == "Figure 2"


class TestCaps:
    def test_max_evidence_per_figure(self, paragraphs, refs):
        config = PaperSemanticsConfig(max_evidence_per_figure=2)
        evidences = collect_evidence(refs["Figure 2"], paragraphs, config)
        assert len(evidences) <= 2
        assert evidences[0].role == "caption"  # the cap keeps the highest-priority evidence

    def test_max_methods_paragraphs_zero(self, paragraphs, refs):
        config = PaperSemanticsConfig(max_methods_paragraphs=0)
        evidences = collect_evidence(refs["Figure 2"], paragraphs, config)
        assert all(e.role != "supporting" for e in evidences)
