"""API tests for /api/v1/paper-questions (figure-grouped review endpoint)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services.question_generation import DraftConfig, generate_question_drafts
from app.services.question_translation import translate_document


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()

    from tests.paper_semantics.conftest import build_paper_tree, write_document_artifact

    from app.services.paper_semantics import PaperSemanticsConfig, reconstruct_figures
    from app.services.question_blueprint import BlueprintConfig, generate_blueprints

    artifacts = tmp_path / "data"
    write_document_artifact(build_paper_tree(), artifacts, "api-paper")
    reconstruct_figures("api-paper", artifacts, config=PaperSemanticsConfig())  # figures/evidence/report artifacts
    generate_blueprints("api-paper", artifacts, config=BlueprintConfig())
    generate_question_drafts("api-paper", artifacts, config=DraftConfig())
    translate_document("api-paper", artifacts)

    yield TestClient(app)
    get_settings.cache_clear()


def test_list_grouped_by_figure(client):
    response = client.get("/api/v1/paper-questions/api-paper")
    assert response.status_code == 200
    body = response.json()
    assert body["doc_id"] == "api-paper"
    figures = body["figures"]
    assert figures, "expected at least one grouped figure"
    for figure in figures:
        assert figure["question_sets"]
        for question_set in figure["question_sets"]:
            assert question_set["statements"]
            labels = [s["label"] for s in question_set["statements"]]
            assert labels[0] == "A"
            for statement in question_set["statements"]:
                assert statement["statement"] and statement["statement_zh"]
                assert isinstance(statement["is_correct"], bool)
                assert statement["evidence_ids"]


def test_figure_filter(client):
    all_figures = client.get("/api/v1/paper-questions/api-paper").json()["figures"]
    target = all_figures[0]["figure_id"]
    filtered = client.get(f"/api/v1/paper-questions/api-paper?figure_id={target}").json()["figures"]
    assert [f["figure_id"] for f in filtered] == [target]
    # flexible query form ("2" matches "Figure 2" style ids when present)
    short = target.replace("Figure ", "").replace("Table ", "")
    if short != target:
        loosed = client.get(f"/api/v1/paper-questions/api-paper?figure_id={short}").json()["figures"]
        assert [f["figure_id"] for f in loosed] == [target]


def test_experiment_summary_present(client):
    figure = client.get("/api/v1/paper-questions/api-paper").json()["figures"][0]
    assert "figure_id" in figure and "status" in figure
    if figure.get("experiment"):
        experiment = figure["experiment"]
        assert any(key in experiment for key in ("research_question", "groups", "variables"))


def test_set_detail_with_evidence_texts(client):
    listing = client.get("/api/v1/paper-questions/api-paper").json()
    question_set = listing["figures"][0]["question_sets"][0]
    detail = client.get(f"/api/v1/paper-questions/api-paper/{question_set['draft_set_id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["draft_set_id"] == question_set["draft_set_id"]
    assert body["blueprint"]["question_focus"]
    for statement in body["statements"]:
        assert statement["statement_zh"]
        for evidence in statement["evidence"]:
            assert evidence["evidence_id"]
            assert "text" in evidence


def test_404s(client):
    assert client.get("/api/v1/paper-questions/missing-doc").status_code == 404
    assert client.get("/api/v1/paper-questions/api-paper/no-such-set").status_code == 404
    assert client.get("/api/v1/paper-questions/api-paper?figure_id=Figure 99").status_code == 404


def test_review_page_served(client):
    response = client.get("/review")
    assert response.status_code == 200
    assert "Paper Question Review" in response.text
    assert "paper-questions" in response.text  # the page talks to the API
