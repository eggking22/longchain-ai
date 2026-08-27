"""End-to-end: synthetic paper PDF → Phase 1 ingest → figure semantic reconstruction.

This exercises the real parser (heading detection on bold English headings,
caption landing in its own paragraph) and proves the module works on genuine
Phase 1 artifacts, not just hand-built trees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.parser import ingest
from app.services.paper_semantics import PaperSemanticsConfig, reconstruct_figures

from .conftest import build_paper_pdf

DOC_ID = "paper-e2e"


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("paper-e2e")
    pdf = root / "paper.pdf"
    build_paper_pdf(pdf)
    stats = ingest(DOC_ID, pdf, artifacts_root=str(root))
    assert stats["paragraphs"] >= 5, "expected several paragraphs from the synthetic paper"
    return root


class TestEndToEnd:
    def test_reconstruction_from_real_parse(self, artifacts):
        report = reconstruct_figures(DOC_ID, artifacts, config=PaperSemanticsConfig())
        assert report.num_figures == 1
        figure = report.figures[0]
        assert figure.figure_id == "Figure 2"
        assert figure.caption.startswith("Figure 2. Relative expression")
        assert figure.reconstruction_status == "SUFFICIENT", figure.missing_information
        experiment = figure.experiment
        assert experiment.intervention == "Treatment A"
        assert "control" in experiment.control_groups
        assert experiment.dependent_variables[0] == "gene X expression"
        assert experiment.observations[0].direction == "increase"
        assert [c.statement for c in experiment.conclusions] == [
            "Treatment A increases gene X expression."
        ]

    def test_artifacts_persisted(self, artifacts):
        reconstruct_figures(DOC_ID, artifacts, config=PaperSemanticsConfig())
        out = Path(artifacts) / "paper_semantics" / DOC_ID
        document = json.loads((out / "figures.json").read_text(encoding="utf-8"))
        assert list(document)[:3] == ["doc_id", "background", "summary"]  # background at the top
        assert document["figures"][0]["figure_id"] == "Figure 2"
        assert document["figures"][0]["status"] == "SUFFICIENT"
        experiments = json.loads((out / "experiments.json").read_text(encoding="utf-8"))
        assert experiments[0]["experiment_id"] == "exp_f02"
        assert (out / "evidence.jsonl").exists()
        assert (out / "report.md").exists()

    def test_phase1_artifacts_untouched_by_reconstruction(self, artifacts):
        """Reconstruction only *reads* structure/ and adds its own directory."""
        structure_dir = Path(artifacts) / "structure" / DOC_ID
        before = {p.name: p.read_bytes() for p in structure_dir.iterdir()}
        reconstruct_figures(DOC_ID, artifacts, config=PaperSemanticsConfig())
        after = {p.name: p.read_bytes() for p in structure_dir.iterdir()}
        assert before == after
        assert (Path(artifacts) / "paper_semantics" / DOC_ID).exists()
