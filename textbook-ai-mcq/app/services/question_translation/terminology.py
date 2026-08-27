"""Unified Terminology Registry for English → Chinese statement translation.

Rules the registry enforces by construction:
- gene/protein/compound names (GFP, CCR7, CK666, AACOF3, cPLA2 ...) are NOT in
  the registry and therefore always pass through verbatim;
- numbers and units are never mapped;
- direction verbs have exactly one Chinese counterpart each (increases→提高,
  decreases→降低) so a translation can never flip or blur a direction;
- relationship strength is preserved phrase-for-phrase (associated with→相关,
  causes→导致), keeping association ≠ causation visible in Chinese too.

Multi-word entries are applied before single-word ones (longest-first).
"""

from __future__ import annotations

import re

# Phrase-level terminology (applied longest-first inside entity translation).
TERMINOLOGY: dict[str, str] = {
    # direction verbs / events — single, fixed counterparts
    "increases": "提高",
    "increase": "提高",
    "increased": "提高",
    "decreases": "降低",
    "decrease": "降低",
    "decreased": "降低",
    "upregulation": "上调",
    "downregulation": "下调",
    "upregulated": "上调",
    "downregulated": "下调",
    # experimental design
    "control": "对照",
    "untreated": "未处理",
    "unstimulated": "未刺激",
    "wild-type": "野生型",
    "wild type": "野生型",
    "treatment": "处理",
    "treated": "处理",
    "inhibitor": "抑制剂",
    "knockout": "敲除",
    "knockdown": "敲低",
    "overexpression": "过表达",
    "group": "组",
    "baseline": "基线",
    # common endpoints / measurements
    "body weight": "体重",
    "expression": "表达",
    "intensity": "强度",
    "migration": "迁移",
    "distance": "距离",
    "speed": "速度",
    "cells": "细胞",
    "weight": "重量",
    "level": "水平",
    "activity": "活性",
    "concentration": "浓度",
    "geometric mean": "几何平均",
}

# Function words dropped during entity translation. Only the exact lowercase
# forms are dropped — capital single letters are group labels ("Treatment A").
_DROP_WORDS = frozenset({"the", "a", "an", "of"})

# Sorted phrase list, longest first, so "body weight" beats "weight".
_PHRASES = sorted(TERMINOLOGY.items(), key=lambda item: -len(item[0]))


def translate_entity(text: str) -> str:
    """Translate a treatment/endpoint/group name with the registry.

    Unknown tokens (gene names, compounds, numbers, units) pass through
    verbatim; known terms map to their single registered Chinese counterpart
    (case-insensitive match so "Treatment"/"treatment" behave identically).
    """
    if not text:
        return ""
    working = text
    for phrase, translation in _PHRASES:
        working = re.sub(re.escape(phrase), translation, working, flags=re.IGNORECASE)
    tokens = [token for token in working.split() if token not in _DROP_WORDS]
    return " ".join(tokens).strip()
