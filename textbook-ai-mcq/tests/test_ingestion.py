"""End-to-end (golden) tests for the full ingestion pipeline."""

import json

import pytest

from app.services.parser import ScannedPdfError, ingest

EXPECTED_STATS = {
    "num_pages": 3,
    "chapters": 2,
    "sections": 4,  # 3 TOC sections + 1 font-detected sub-heading
    "paragraphs": 5,
}


def test_ingestion_golden(sample_pdf, tmp_path):
    stats = ingest("golden", sample_pdf, artifacts_root=tmp_path)

    for key, expected in EXPECTED_STATS.items():
        assert stats[key] == expected, f"{key}: {stats[key]} != {expected}"
    assert stats["heading_rules"] == {"toc": 5, "font": 1}
    assert stats["chunks"]["count"] == 4

    # artifacts all present
    raw_dir = tmp_path / "raw" / "golden"
    struct_dir = tmp_path / "structure" / "golden"
    chunk_file = tmp_path / "chunks" / "golden" / "chunks.jsonl"
    for name in ("lines.json", "toc.json", "fonts.json", "lines.clean.json", "cleaner.debug.json"):
        assert (raw_dir / name).exists()
    for name in ("document.json", "headings.debug.json", "stats.json"):
        assert (struct_dir / name).exists()
    assert chunk_file.exists()

    # golden structure checks
    tree = json.loads((struct_dir / "document.json").read_text(encoding="utf-8"))
    assert tree["node_type"] == "document"
    assert [c["title"] for c in tree["children"]] == ["第1章 走近细胞", "第2章 组成细胞的分子"]

    # chunks: breadcrumb metadata + no chunk crosses a section
    chunks = [json.loads(line) for line in chunk_file.read_text(encoding="utf-8").splitlines()]
    assert len(chunks) == 4
    assert all(c["breadcrumb"] for c in chunks)
    sub_chunk = next(c for c in chunks if len(c["breadcrumb"]) == 3)
    assert sub_chunk["breadcrumb"][-1] == "一、细胞学说"
    assert sub_chunk["pages"] == [1, 2]

    # debug trace records rules and confidence for every heading
    debug = json.loads((struct_dir / "headings.debug.json").read_text(encoding="utf-8"))
    assert len(debug["accepted"]) == 6
    assert all("rule" in h and "confidence" in h for h in debug["accepted"])


def test_ingestion_without_toc(sample_pdf_no_toc, tmp_path):
    stats = ingest("notoc", sample_pdf_no_toc, artifacts_root=tmp_path)
    assert stats["chapters"] == 2
    assert stats["sections"] == 4
    assert stats["paragraphs"] == 5
    assert stats["heading_rules"] == {"font": 6}


def test_ingestion_scanned_pdf(tmp_path):
    import pymupdf as fitz

    pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf))
    doc.close()
    with pytest.raises(ScannedPdfError):
        ingest("blank", pdf, artifacts_root=tmp_path)


def test_ingestion_reuse_raw(sample_pdf, tmp_path):
    stats1 = ingest("reuse", sample_pdf, artifacts_root=tmp_path)
    stats2 = ingest("reuse", sample_pdf, artifacts_root=tmp_path, reuse_raw=True)
    assert stats1 == stats2
