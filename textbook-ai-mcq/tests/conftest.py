"""Test fixtures: synthetic textbook PDFs built with PyMuPDF itself.

The sample mirrors a Chinese biology textbook: running header + page
numbers, chapters at 18pt, sections at 14pt, sub-headings at 12pt, body at
10.5pt with two-character first-line indents, and one paragraph that spans
a page break. Expected structure (with_toc=True):

  document
  ├── chapter 第1章 走近细胞            (toc)
  │   ├── section 第1节 细胞是生命活动的基本单位 (toc)
  │   │   ├── paragraph p1 (2 lines)
  │   │   ├── paragraph p2 (1 line)
  │   │   └── section 一、细胞学说       (font, level 3)
  │   │       └── paragraph p3 (pages 1-2, 3 lines)
  │   └── section 第2节 细胞的多样性和统一性 (toc)
  │       └── paragraph p4 (2 lines)
  └── chapter 第2章 组成细胞的分子       (toc)
      └── section 第1节 细胞中的元素和化合物 (toc)
          └── paragraph p5 (2 lines)
"""

from __future__ import annotations

import pymupdf as fitz
import pytest

A4_W, A4_H = 595, 842

BODY_X = 72
INDENT_X = 92
BODY_SIZE = 10.5


def _add(page, x, y, text, size, font="china-s"):
    page.insert_text((x, y), text, fontsize=size, fontname=font)


def build_sample_pdf(path, with_toc: bool = True) -> None:
    doc = fitz.open()

    def new_page(header: str, page_no: int):
        page = doc.new_page(width=A4_W, height=A4_H)
        _add(page, BODY_X, 40, header, 9)  # running header
        _add(page, 300, 815, str(page_no), 9)  # page number footer
        return page

    # page 1
    p1 = new_page("普通高中教科书·生物学（必修一）", 1)
    _add(p1, BODY_X, 110, "第1章 走近细胞", 18)
    _add(p1, BODY_X, 150, "第1节 细胞是生命活动的基本单位", 14)
    _add(p1, INDENT_X, 190, "细胞是生物体结构和功能的基本单位，除病毒外，一切生物体都是由细胞构成的，", BODY_SIZE)
    _add(p1, BODY_X, 220, "病毒没有细胞结构，必须寄生在活的宿主细胞内才能生活和增殖。", BODY_SIZE)
    _add(p1, INDENT_X, 250, "显微镜下的细胞形态多种多样，但都具有相似的基本结构。", BODY_SIZE)
    _add(p1, BODY_X, 290, "一、细胞学说", 12)
    _add(p1, INDENT_X, 330, "细胞学说揭示了细胞的统一性和生物体结构的统一性，主要内容是由施莱登和施旺提出，", BODY_SIZE)
    _add(p1, BODY_X, 360, "并由魏尔肖进行了补充和完善，指出细胞只能通过分裂产生新细胞", BODY_SIZE)
    # page 2
    p2 = new_page("普通高中教科书·生物学（必修一）", 2)
    _add(p2, BODY_X, 110, "从而为生物学的发展奠定了坚实的基础。", BODY_SIZE)
    _add(p2, BODY_X, 160, "第2节 细胞的多样性和统一性", 14)
    _add(p2, INDENT_X, 200, "真核细胞和原核细胞最明显的区别在于有无以核膜为界限的细胞核，", BODY_SIZE)
    _add(p2, BODY_X, 230, "但在微观结构上又具有高度的统一性。", BODY_SIZE)
    # page 3
    p3 = new_page("普通高中教科书·生物学（必修一）", 3)
    _add(p3, BODY_X, 110, "第2章 组成细胞的分子", 18)
    _add(p3, BODY_X, 150, "第1节 细胞中的元素和化合物", 14)
    _add(p3, INDENT_X, 190, "组成细胞的元素大多以离子的形式存在，", BODY_SIZE)
    _add(p3, BODY_X, 220, "化合物是细胞结构和生命活动的物质基础，其中水的含量最多。", BODY_SIZE)

    if with_toc:
        doc.set_toc(
            [
                [1, "第1章 走近细胞", 1],
                [2, "第1节 细胞是生命活动的基本单位", 1],
                [2, "第2节 细胞的多样性和统一性", 2],
                [1, "第2章 组成细胞的分子", 3],
                [2, "第1节 细胞中的元素和化合物", 3],
            ]
        )
    doc.save(str(path))
    doc.close()


def build_english_pdf(path) -> None:
    """No TOC, no size difference — headings detected via bold-only rule."""
    doc = fitz.open()
    page = doc.new_page(width=A4_W, height=A4_H)
    _add(page, BODY_X, 110, "Cell Structure", 10.5, font="hebo")
    _add(page, INDENT_X, 150, "All living things are made of cells.", 10.5, font="helv")
    _add(page, BODY_X, 180, "Cells are the basic units of structure and function", 10.5, font="helv")
    doc.save(str(path))
    doc.close()


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    build_sample_pdf(path, with_toc=True)
    return path


@pytest.fixture
def sample_pdf_no_toc(tmp_path):
    path = tmp_path / "sample_no_toc.pdf"
    build_sample_pdf(path, with_toc=False)
    return path
