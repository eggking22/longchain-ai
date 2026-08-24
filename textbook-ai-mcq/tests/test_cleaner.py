"""Unit tests for the cleaner (header/footer/page-number removal)."""

from app.services.parser import clean_lines, extract_raw
from tests.conftest import build_sample_pdf


def test_removes_running_header_and_page_numbers(tmp_path):
    pdf = tmp_path / "s.pdf"
    build_sample_pdf(pdf)
    raw = extract_raw(pdf)
    # 22 raw lines: 3 pages * 2 band lines + 16 content lines
    assert len(raw.lines) == 22

    result = clean_lines(raw)

    texts = [l.text for l in result.lines]
    assert len(result.lines) == 16
    assert all("普通高中教科书" not in t for t in texts)
    assert "1" not in texts and "2" not in texts and "3" not in texts
    assert any("第1章 走近细胞" in t for t in texts)

    report = result.report
    assert report["dropped"] == 6
    assert any(g["band"] == "top" for g in report["dropped_groups"])
    # footers "1"/"2"/"3" digit-mask to "N" and drop as a repeated bottom group
    assert any(g["band"] == "bottom" and g["pattern"] == "N" for g in report["dropped_groups"])
    assert report["dropped_page_numbers"] == 0


def test_whitespace_folding(tmp_path):
    pdf = tmp_path / "s.pdf"
    build_sample_pdf(pdf)
    result = clean_lines(extract_raw(pdf))
    for line in result.lines:
        assert line.text == line.text.strip()
        assert "  " not in line.text


def test_empty_pdf_raises_scanned_error(tmp_path):
    import pymupdf as fitz
    import pytest

    from app.services.parser import ScannedPdfError

    pdf = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()  # blank page: no text at all
    doc.save(str(pdf))
    doc.close()
    with pytest.raises(ScannedPdfError):
        extract_raw(pdf)
