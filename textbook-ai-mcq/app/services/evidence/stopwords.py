"""Minimal Chinese stopword list for evidence term coverage.

Only interrogatives and pure function words — deliberately small so that
content-bearing words (功能/影响/研究…) stay in play. Term coverage is
meant to detect *missing topical information*, not to clean prose.
"""

from __future__ import annotations

STOPWORDS_ZH: frozenset[str] = frozenset(
    {
        # interrogatives
        "什么", "哪些", "哪里", "哪儿", "为何", "为什么", "如何", "怎么",
        "怎样", "怎么样", "多少", "是否", "几", "何",
        # function words / particles
        "的", "了", "是", "在", "和", "与", "及", "或", "对", "对于",
        "由", "从", "被", "把", "着", "过", "也", "都", "就", "会",
        "能", "可", "可以", "其", "该", "各", "每个", "么", "呢", "吗",
        "吧", "有", "无", "一个", "一种", "一样", "中", "等", "以及",
        "并且", "而且", "但是", "因此", "所以", "如果", "虽然", "关于",
        "请问", "请", "告诉我", "说说", "讲讲", "介绍",
        # generic verbs that carry no topical signal in a query
        "进行", "使得", "属于", "成为",
    }
)

# Token-level equivalence classes for evidence matching (lowercase). Chinese
# textbooks write gas names as chemical symbols (O2/CO2) while questions
# ask in words — without this map, perfectly good evidence looks missing.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "氧气": ("o2", "氧"),
    "o2": ("氧气", "氧"),
    "二氧化碳": ("co2",),
    "co2": ("二氧化碳",),
    "血红蛋白": ("hb",),
    "hb": ("血红蛋白",),
    "三磷酸腺苷": ("atp",),
    "atp": ("三磷酸腺苷",),
    "血管紧张素": ("angiotensin",),
    "心电图": ("ecg", "ekg"),
}

