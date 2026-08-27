"""Storage readability: background-first figures.json, evidence.jsonl store, report.md."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.paper_semantics import (
    PaperSemanticsConfig,
    build_figures_document,
    build_report_markdown,
    reconstruct_figures,
)

from .conftest import build_paper_tree, write_document_artifact


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    root = tmp_path_factory.mktemp("storage")
    write_document_artifact(build_paper_tree(), root, "storage-paper")
    return reconstruct_figures("storage-paper", root, config=PaperSemanticsConfig())


class TestFiguresDocument:
    def test_background_comes_first(self, report):
        document = build_figures_document(report)
        assert list(document)[:3] == ["doc_id", "background", "summary"]

    def test_summary_block(self, report):
        document = build_figures_document(report)
        summary = document["summary"]
        assert summary["figures"] == {"total": 5, "SUFFICIENT": 3, "PARTIAL": 1, "INSUFFICIENT": 1}
        assert summary["evidence_units"]["total"] > 0
        assert "direct_observation" in summary["evidence_units"]

    def test_figure_blocks_reference_evidence_by_id_only(self, report):
        document = build_figures_document(report)
        figure2 = next(f for f in document["figures"] if f["figure_id"] == "Figure 2")
        assert figure2["evidence"] == [f"ev_f02_{i:03d}" for i in range(1, len(figure2["evidence"]) + 1)]  # figure-scoped ids
        assert "text" not in str(figure2["evidence"])  # ids only, no inline evidence payloads
        assert figure2["conclusions"][0]["statement"] == "Treatment A increases gene X expression."

    def test_text_block_present(self, report):
        document = build_figures_document(report)
        figure2 = next(f for f in document["figures"] if f["figure_id"] == "Figure 2")
        block = figure2["text_block"]
        assert block["anchors"] and block["paragraph_count"] >= 1


class TestEvidenceStore:
    def test_evidence_jsonl_deduplicated_and_resolvable(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "evidence-paper")
        reconstruct_figures("evidence-paper", tmp_path, config=PaperSemanticsConfig())
        units = [json.loads(line) for line in (tmp_path / "paper_semantics" / "evidence-paper" / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
        ids = [u["evidence_id"] for u in units]
        assert len(ids) == len(set(ids))  # no duplicates across figures/panels
        assert all(u["text"] and "evidence_type" in u for u in units)
        # every id referenced by figures.json resolves into the store
        document = json.loads((tmp_path / "paper_semantics" / "evidence-paper" / "figures.json").read_text(encoding="utf-8"))
        referenced = {eid for f in document["figures"] for eid in f["evidence"]}
        assert referenced <= set(ids)


class TestReportMarkdown:
    def test_report_structure(self, report):
        markdown = build_report_markdown(report)
        assert markdown.startswith("# storage-paper — Figure Semantic Reconstruction")
        assert "**Abstract**" in markdown or "**Introduction**" in markdown
        assert "## Figure 2 — SUFFICIENT" in markdown
        assert "- **CON** Treatment A increases gene X expression." in markdown
        assert "**Evidence:**" in markdown

    def test_report_is_deterministic(self, report):
        assert build_report_markdown(report) == build_report_markdown(report)


class TestManifestReadingIndex:
    def test_reading_index_written(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "index-paper")
        reconstruct_figures("index-paper", tmp_path, config=PaperSemanticsConfig())
        manifest = json.loads((tmp_path / "paper_semantics" / "index-paper" / "manifest.json").read_text(encoding="utf-8"))
        index = manifest["reading_index"]
        assert index["section_paragraph_counts"]["results"] >= 1
        assert {item["figure_id"] for item in index["figure_inventory"]} == {
            "Figure 2", "Figure 3", "Figure 4", "Figure 5", "Table 1",
        }
        partition = index["partition"]
        assert partition["method"] == "L1-deterministic"
        assert partition["flow_paragraphs"] == partition["anchors"] + partition["continuations"] + partition["unassigned"]
        assert 0 < partition["coverage"] <= 1
        assert manifest["files"]["evidence"] == "evidence.jsonl"


class TestReproducibility:
    def test_content_files_byte_identical(self, tmp_path):
        write_document_artifact(build_paper_tree(), tmp_path, "repr2-paper")
        reconstruct_figures("repr2-paper", tmp_path, config=PaperSemanticsConfig())
        out = Path(tmp_path) / "paper_semantics" / "repr2-paper"
        first = {name: (out / name).read_bytes() for name in ("figures.json", "evidence.jsonl", "report.md", "experiments.json")}
        reconstruct_figures("repr2-paper", tmp_path, config=PaperSemanticsConfig())
        second = {name: (out / name).read_bytes() for name in first}
        assert first == second  # no timestamps in content files

    def test_figures_json_is_small_and_scannable(self, tmp_path):
        """The semantic index must not inline evidence texts (bulk lives in evidence.jsonl)."""
        write_document_artifact(build_paper_tree(), tmp_path, "size-paper")
        reconstruct_figures("size-paper", tmp_path, config=PaperSemanticsConfig())
        out = Path(tmp_path) / "paper_semantics" / "size-paper"
        figures_size = (out / "figures.json").stat().st_size
        evidence_size = (out / "evidence.jsonl").stat().st_size
        assert evidence_size > 0
        document = json.loads((out / "figures.json").read_text(encoding="utf-8"))
        assert "paragraph_id" not in json.dumps(document["figures"])
