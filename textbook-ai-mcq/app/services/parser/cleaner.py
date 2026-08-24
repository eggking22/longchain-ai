"""Stage 2 — remove running headers/footers/page numbers, normalize text.

A band text (top or bottom strip of the page) whose digit-masked form
repeats on enough pages is treated as a running header/footer and dropped
entirely. Every removal is recorded in the debug report.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .parser import RawDoc

PAGE_NO_RE = re.compile(
    r"^[\s\-—–·.。]*(?:第\s*[0-9０-９]+\s*页|[0-9０-９]{1,4}|[IVXLCDM]{1,6}|Page\s*[0-9]{1,4})[\s\-—–·.。]*$",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"[ \t\u00a0\u3000]+")


def _digit_mask(text: str) -> str:
    return re.sub(r"[0-9０-９]", "N", text).strip()


@dataclass
class CleanResult:
    lines: list  # list[LineInfo]
    report: dict


def clean_lines(raw: RawDoc, band_ratio: float = 0.09, repeat_ratio: float = 0.3) -> CleanResult:
    num_pages = max(raw.num_pages, 1)
    min_pages = max(2, math.ceil(repeat_ratio * num_pages))

    groups: dict[tuple[str, str], dict] = {}
    for line in raw.lines:
        top_limit = band_ratio * line.page_height
        bottom_limit = (1.0 - band_ratio) * line.page_height
        if line.bbox[1] < top_limit:
            band = "top"
        elif line.bbox[3] > bottom_limit:
            band = "bottom"
        else:
            continue
        key = (band, _digit_mask(line.text))
        g = groups.setdefault(key, {"band": band, "pages": set(), "lines": []})
        g["pages"].add(line.page_no)
        g["lines"].append(line)

    dropped_indexes: set[int] = set()
    dropped_groups = []
    for (band, pattern), g in groups.items():
        if len(g["pages"]) >= min_pages:
            dropped_indexes.update(l.index for l in g["lines"])
            dropped_groups.append(
                {
                    "band": band,
                    "pattern": pattern,
                    "pages": sorted(g["pages"]),
                    "samples": [l.text.strip() for l in g["lines"][:3]],
                    "reason": "repeated_header_footer",
                }
            )

    kept = []
    dropped_page_numbers = 0
    for line in raw.lines:
        if line.index in dropped_indexes:
            continue
        text = _WS_RE.sub(" ", line.text).strip().replace("\u00ad", "")
        if not text:
            continue
        if len(text) <= 10 and PAGE_NO_RE.match(text):
            dropped_page_numbers += 1
            continue
        line.text = text
        kept.append(line)

    report = {
        "kept": len(kept),
        "dropped": len(raw.lines) - len(kept),
        "dropped_page_numbers": dropped_page_numbers,
        "dropped_groups": dropped_groups,
    }
    return CleanResult(kept, report)
