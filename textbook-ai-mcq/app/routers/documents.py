"""Document upload / structure / chunks endpoints (thin wrappers over ingestion)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.services.parser import ScannedPdfError, ingest

router = APIRouter(tags=["documents"])


@router.post("/documents")
def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")
    settings = get_settings()
    doc_id = uuid4().hex[:12]
    pdf_path = Path(settings.UPLOADS_DIR) / f"{doc_id}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(file.file.read())
    try:
        stats = ingest(doc_id, pdf_path, artifacts_root=settings.ARTIFACTS_DIR)
    except ScannedPdfError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"doc_id": doc_id, **stats}


@router.get("/documents/{doc_id}/structure")
def get_structure(doc_id: str) -> dict:
    path = _artifacts_path("structure", doc_id, "document.json")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/documents/{doc_id}/chunks")
def get_chunks(doc_id: str) -> dict:
    path = _artifacts_path("chunks", doc_id, "chunks.jsonl")
    chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"doc_id": doc_id, "count": len(chunks), "chunks": chunks}


def _artifacts_path(subdir: str, doc_id: str, filename: str) -> Path:
    settings = get_settings()
    path = Path(settings.ARTIFACTS_DIR) / subdir / doc_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"document '{doc_id}' not found")
    return path
