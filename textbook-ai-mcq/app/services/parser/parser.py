"""Stage 1 — raw PDF extraction with PyMuPDF.

Produces reading-ordered lines (with span/font info), PDF bookmarks (TOC)
and a font usage histogram. All downstream stages work on these artifacts,
so the PDF itself is only touched once per ingestion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (new canonical import name)


class ScannedPdfError(ValueError):
    """Raised when a PDF contains (almost) no extractable text."""


@dataclass
class SpanInfo:
    text: str
    font: str
    size: float
    flags: int
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SpanInfo":
        return cls(**d)


@dataclass
class LineInfo:
    index: int  # global reading-order index
    page_no: int  # 1-based
    block_no: int
    text: str
    bbox: tuple[float, float, float, float]
    spans: list[SpanInfo] = field(default_factory=list)
    size: float = 0.0  # dominant (char-weighted) span size
    font: str = ""
    bold: bool = False
    page_width: float = 0.0
    page_height: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["spans"] = [s.to_dict() for s in self.spans]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LineInfo":
        d = dict(d)
        d["spans"] = [SpanInfo.from_dict(s) for s in d.get("spans", [])]
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)


@dataclass
class TocEntry:
    level: int  # 1-based depth from the PDF outline
    title: str
    page_no: int  # 1-based


@dataclass
class RawDoc:
    pdf_path: str
    num_pages: int
    lines: list[LineInfo]
    toc: list[TocEntry]
    font_histogram: dict[str, int]  # "font@size" -> visible char count


def extract_raw(pdf_path: str | Path) -> RawDoc:
    doc = fitz.open(str(pdf_path))
    lines: list[LineInfo] = []
    histogram: dict[str, int] = {}
    pages_with_text = 0
    idx = 0
    try:
        for pno, page in enumerate(doc, start=1):
            pw, ph = float(page.rect.width), float(page.rect.height)
            has_text = False
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for ln in block.get("lines", []):
                    spans = [
                        SpanInfo(
                            text=s["text"],
                            font=s["font"],
                            size=round(float(s["size"]), 1),
                            flags=int(s["flags"]),
                            bbox=tuple(round(float(v), 2) for v in s["bbox"]),
                        )
                        for s in ln.get("spans", [])
                    ]
                    text = "".join(s.text for s in spans)
                    if not text.strip():
                        continue
                    dominant = max(spans, key=lambda s: len(s.text.strip()))
                    bold = bool(dominant.flags & 16) or "bold" in dominant.font.lower()
                    lines.append(
                        LineInfo(
                            index=idx,
                            page_no=pno,
                            block_no=int(block.get("number", -1)),
                            text=text,
                            bbox=tuple(round(float(v), 2) for v in ln["bbox"]),
                            spans=spans,
                            size=dominant.size,
                            font=dominant.font,
                            bold=bold,
                            page_width=pw,
                            page_height=ph,
                        )
                    )
                    idx += 1
                    has_text = True
                    for s in spans:
                        key = f"{s.font}@{s.size}"
                        histogram[key] = histogram.get(key, 0) + len(s.text.strip())
            if has_text:
                pages_with_text += 1
        if pages_with_text == 0:
            raise ScannedPdfError(
                f"{pdf_path}: no extractable text on any page "
                "(scanned PDFs need OCR, which is out of scope for this phase)"
            )
        toc = [TocEntry(int(l), t, int(p)) for l, t, p in doc.get_toc(simple=True)]
        return RawDoc(
            pdf_path=str(pdf_path),
            num_pages=doc.page_count,
            lines=lines,
            toc=toc,
            font_histogram=histogram,
        )
    finally:
        doc.close()


def save_raw(raw: RawDoc, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps({"pdf_path": raw.pdf_path, "num_pages": raw.num_pages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "lines.json").write_text(
        json.dumps([l.to_dict() for l in raw.lines], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "toc.json").write_text(
        json.dumps([asdict(t) for t in raw.toc], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "fonts.json").write_text(
        json.dumps(raw.font_histogram, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_raw(in_dir: Path, pdf_path: str = "") -> RawDoc:
    meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
    lines = [LineInfo.from_dict(d) for d in json.loads((in_dir / "lines.json").read_text(encoding="utf-8"))]
    toc = [TocEntry(**t) for t in json.loads((in_dir / "toc.json").read_text(encoding="utf-8"))]
    fonts = json.loads((in_dir / "fonts.json").read_text(encoding="utf-8"))
    return RawDoc(
        pdf_path=pdf_path or meta.get("pdf_path", ""),
        num_pages=meta["num_pages"],
        lines=lines,
        toc=toc,
        font_histogram=fonts,
    )
