"""Deterministic English → Chinese statement translation.

Statements come from the draft layer's fixed templates, so translation is a
ordered pattern list: each known template maps to a Chinese template with the
same slots (entities translated via the Terminology Registry). Statements that
match no template (e.g. CONDITION_MUTATION raw evidence sentences) fall back
to a registry word pass — rougher grammar, but numbers/genes/direction words
still map through the same fixed registry, and the method is recorded.

Translation never re-judges the science: is_correct, evidence_ids, numbers,
units and gene names are carried over untouched.
"""

from __future__ import annotations

import re

from .terminology import translate_entity

_VERB_UP = {"increases": "提高", "increase": "提高", "increased": "提高"}
_VERB_DOWN = {"decreases": "降低", "decrease": "降低", "decreased": "降低"}

# Ordered template patterns → Chinese renderers (first match wins).
# A space is kept between the direction verb and the endpoint so mixed
# Chinese/Latin text reads naturally ("可提高 GFP 表达。").
_TEMPLATES = [
    # "According to {label}, {rest}." — recursive on the inner statement
    (
        re.compile(r"^According to (?P<label>.+?), (?P<rest>.+)$"),
        lambda m: f"根据{translate_entity(m['label'])}，{translate_statement(m['rest'])[0]}",
    ),
    # "{X} significantly {increases|decreases} {Y}."
    (
        re.compile(r"^(?P<subj>.+?) significantly (?P<verb>increases|decreases) (?P<obj>.+?)\.?$"),
        lambda m: _glue(translate_entity(m["subj"]), f"可显著{_verb(m['verb'])} {translate_entity(m['obj'])}。"),
    ),
    # "{X} does not significantly {increase|decrease} {Y}."
    (
        re.compile(r"^(?P<subj>.+?) does not significantly (?P<verb>increase|decrease) (?P<obj>.+?)\.?$"),
        lambda m: _glue(translate_entity(m["subj"]), f"不会显著{_verb(m['verb'])} {translate_entity(m['obj'])}。"),
    ),
    # "{X} {increases|decreases} {Y}."
    (
        re.compile(r"^(?P<subj>.+?) (?P<verb>increases|decreases) (?P<obj>.+?)\.?$"),
        lambda m: _glue(translate_entity(m["subj"]), f"可{_verb(m['verb'])} {translate_entity(m['obj'])}。"),
    ),
    # "{X} does not {increase|decrease} {Y}."
    (
        re.compile(r"^(?P<subj>.+?) does not (?P<verb>increase|decrease) (?P<obj>.+?)\.?$"),
        lambda m: _glue(translate_entity(m["subj"]), f"不会{_verb(m['verb'])} {translate_entity(m['obj'])}。"),
    ),
    # "{X} is associated with {increased|decreased} {Y}."
    (
        re.compile(r"^(?P<subj>.+?) is associated with (?P<verb>increased|decreased) (?P<obj>.+?)\.?$"),
        lambda m: _glue(translate_entity(m["subj"]), f"与 {translate_entity(m['obj'])} 的{_verb(m['verb'])}相关。"),
    ),
    # "{X} is not associated with {increased|decreased} {Y}."
    (
        re.compile(r"^(?P<subj>.+?) is not associated with (?P<verb>increased|decreased) (?P<obj>.+?)\.?$"),
        lambda m: _glue(translate_entity(m["subj"]), f"与 {translate_entity(m['obj'])} 的{_verb(m['verb'])}不相关。"),
    ),
    # "{X} causes {Y}." (only ever a FALSE perturbation of an association)
    (
        re.compile(r"^(?P<subj>.+?) causes (?P<obj>.+?)\.?$"),
        lambda m: _glue(translate_entity(m["subj"]), f"导致 {translate_entity(m['obj'])}。"),
    ),
    # "The reported value is {V}."
    (
        re.compile(r"^The reported value is (?P<value>.+?)\.?$"),
        lambda m: f"论文报告的数值为 {m['value']}。",
    ),
    # "{Y} would {increase|decrease}, as already observed in the {G} group."
    (
        re.compile(r"^(?P<subj>.+?) would (?P<verb>increase|decrease), as already observed in the (?P<group>.+?) group\.?$"),
        lambda m: _glue(translate_entity(m["subj"]), f"将会{_verb(m['verb'])}，这在{translate_entity(m['group'])}组中已被观察到。"),
    ),
    # ED: "The {C} group provides the baseline level of {Y} against which {X} is compared."
    (
        re.compile(
            r"^The (?P<control>.+?) group provides the baseline level of (?P<obj>.+?) "
            r"against which (?P<treat>.+?) is compared\.?$"
        ),
        lambda m: (
            f"{translate_entity(m['control'])}组提供{translate_entity(m['obj'])}的基线水平，"
            f"用于与{translate_entity(m['treat'])} 进行比较。"
        ),
    ),
    # ED: "The {G} group receives {X} so that its effect on {Y} can be measured against the {C} baseline."
    (
        re.compile(
            r"^The (?P<group>.+?) group receives (?P<treat>.+?) so that its effect on (?P<obj>.+?) "
            r"can be measured against the (?P<control>.+?) baseline\.?$"
        ),
        lambda m: (
            f"{translate_entity(m['group'])}组接受{translate_entity(m['treat'])}处理，"
            f"以便以{translate_entity(m['control'])}基线为参照测量其对{translate_entity(m['obj'])}的影响。"
        ),
    ),
    # ED: "It measures {Y} in the {G} group versus the {C} group."
    (
        re.compile(r"^It measures (?P<obj>.+?) in the (?P<group>.+?) group versus the (?P<control>.+?) group\.?$"),
        lambda m: (
            f"该实验测量{translate_entity(m['group'])}组与{translate_entity(m['control'])}组中"
            f"{translate_entity(m['obj'])}的差异。"
        ),
    ),
]


def _verb(english: str) -> str:
    return _VERB_UP.get(english) or _VERB_DOWN[english]


def _glue(left: str, right: str) -> str:
    """Insert a space at Latin/CJK boundaries so mixed text reads naturally."""
    if not left or not right:
        return left + right
    if re.search(r"[A-Za-z0-9%)\]]$", left) and re.match(r"[\u4e00-\u9fff]", right):
        return f"{left} {right}"
    if re.search(r"[\u4e00-\u9fff]$", left) and re.match(r"[A-Za-z0-9(]", right):
        return f"{left} {right}"
    return left + right


def _term_fallback(text: str) -> str:
    """Registry word pass for unmatched sentences (e.g. raw evidence mutations).

    Grammar stays English-ordered, but every registered term maps through the
    same fixed counterparts, so directions/relationships/numbers are consistent
    with the templated statements.
    """
    return translate_entity(text)


def translate_statement(statement: str) -> tuple[str, str]:
    """Return (chinese, method) for one English statement."""
    for pattern, render in _TEMPLATES:
        match = pattern.match(statement.strip())
        if match:
            return render(match), "template"
    return _term_fallback(statement.strip()), "term_fallback"
