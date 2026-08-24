"""Numbering patterns for Chinese biology textbooks (plus common Western styles).

Levels returned by :func:`match_numbering`:
  1 = chapter   (第1章 / Chapter 1)
  2 = section   (第1节 / 1.1)
  3 = sub-block (一、 / （一） / 1、)
"""

from __future__ import annotations

import re

_CN_NUM = "0-9０-９一二三四五六七八九十百零〇两"

CHAPTER_RE = re.compile(rf"^第\s*[{_CN_NUM}]+\s*[章篇部讲]")
SECTION_RE = re.compile(rf"^第\s*[{_CN_NUM}]+\s*[节]")
DOTTED_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})+)(?=[\s、.．:_：]|$)")
SUB_CN_RE = re.compile(r"^[一二三四五六七八九十]{1,3}\s*、")
PAREN_CN_RE = re.compile(r"^（[一二三四五六七八九十]{1,3}）")
ARABIC_LIST_RE = re.compile(r"^\d{1,2}\s*[、.．]\s*\S")
ENGLISH_CHAPTER_RE = re.compile(r"^Chapter\s+\d+", re.IGNORECASE)

# trailing dot leaders / page numbers in TOC titles: "第1章 走近细胞 .... 12"
_TOC_TAIL_RE = re.compile(r"(?:[.·…．]{2,}|[-–—]{1,2})\s*\d{1,4}\s*$")
_WS_RE = re.compile(r"\s+")


def match_numbering(text: str) -> int | None:
    """Return the hierarchy level implied by a leading numbering pattern."""
    t = text.strip()
    if not t:
        return None
    if CHAPTER_RE.match(t) or ENGLISH_CHAPTER_RE.match(t):
        return 1
    if SECTION_RE.match(t):
        return 2
    m = DOTTED_RE.match(t)
    if m:
        return 1 + m.group(1).count(".")  # 1.1 -> 2, 1.1.1 -> 3
    if SUB_CN_RE.match(t) or PAREN_CN_RE.match(t):
        return 3
    if ARABIC_LIST_RE.match(t):
        return 3
    return None


def clean_toc_title(title: str) -> str:
    """Strip dot leaders / trailing page numbers from a bookmark title."""
    t = title.strip()
    t = _TOC_TAIL_RE.sub("", t)
    t = re.sub(r"\s*\d{1,4}\s*$", "", t) if not _CN_TAIL_RE_CHECK(t) else t
    return t.strip()


def _CN_TAIL_RE_CHECK(t: str) -> bool:
    # don't strip digits that are part of the title itself, e.g. "第1章 走近细胞"
    return bool(CHAPTER_RE.match(t) or SECTION_RE.match(t))


def normalize_for_match(text: str) -> str:
    """Whitespace-free form used for TOC-title vs page-line matching."""
    return _WS_RE.sub("", text)
