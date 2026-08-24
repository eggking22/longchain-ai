"""API tests for /api/v1/documents (upload, structure, chunks)."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    yield TestClient(app)
    get_settings.cache_clear()


def test_upload_and_fetch_roundtrip(sample_pdf, client):
    with sample_pdf.open("rb") as f:
        response = client.post(
            "/api/v1/documents", files={"file": ("sample.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    body = response.json()
    doc_id = body["doc_id"]
    assert body["chapters"] == 2
    assert body["paragraphs"] == 5
    assert body["chunks"]["count"] == 4

    structure = client.get(f"/api/v1/documents/{doc_id}/structure")
    assert structure.status_code == 200
    assert structure.json()["node_type"] == "document"
    assert len(structure.json()["children"]) == 2

    chunks = client.get(f"/api/v1/documents/{doc_id}/chunks")
    assert chunks.status_code == 200
    assert chunks.json()["count"] == 4
    assert all(c["breadcrumb"] for c in chunks.json()["chunks"])


def test_unknown_document_returns_404(client):
    assert client.get("/api/v1/documents/missing/structure").status_code == 404
    assert client.get("/api/v1/documents/missing/chunks").status_code == 404


def test_non_pdf_rejected(client, tmp_path):
    fake = tmp_path / "notes.txt"
    fake.write_text("not a pdf", encoding="utf-8")
    with fake.open("rb") as f:
        response = client.post(
            "/api/v1/documents", files={"file": ("notes.txt", f, "text/plain")}
        )
    assert response.status_code == 400
