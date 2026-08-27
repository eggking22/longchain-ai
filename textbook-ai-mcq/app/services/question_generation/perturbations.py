"""Controlled perturbations: turn one TRUE statement into minimal-edit FALSE ones.

Every perturbation is a gate + a minimal textual edit whose replacement material
(treatments, endpoints, conditions, numbers, sibling panels) comes exclusively
from the paper's own evidence via deterministic pools. A perturbation that
cannot apply is skipped (counted) — never force-generated. association is only
ever *upgraded to causation as a false statement*; true statements never carry
causal wording for association experiments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.question_blueprint import QuestionBlueprint

# Fixed deterministic order in which perturbations are attempted per set.
PERTURBATION_ORDER = (
    "DIRECTION_FLIP",
    "SIGNIFICANCE_FLIP",
    "GROUP_SWAP",
    "CAUSALITY_UPGRADE",
    "CONCLUSION_FLIP",
    "DV_SWAP",
    "VARIABLE_SWAP",
    "PANEL_MISATTRIBUTION",
    "NUMERIC_MUTATION",
    "CONDITION_MUTATION",
)

# Single-pass word-level map — sequential str.replace would double-flip
# ("would decrease" → "would increase" → "would decrease" again).
_DIRECTION_MAP = {
    "increases": "decreases",
    "decreases": "increases",
    "increased": "decreased",
    "decreased": "increased",
    "increase": "decrease",
    "decrease": "increase",
    "higher": "lower",
    "lower": "higher",
}

_DIRECTION_WORD_RE = re.compile(r"\b(increases|decreases|increased|decreased|increase|decrease|higher|lower)\b")

_SIGNIFICANT_TRUE_RE = re.compile(r"^(?P<subject>.+?)\s+significantly\s+(?P<verb>increases|decreases)\s+(?P<rest>.+)$")

_BASE_VERB = {"increases": "increase", "decreases": "decrease"}


@dataclass
class PerturbationContext:
    """Evidence-derived pools a perturbation may draw from (never external facts)."""

    intervention: str = ""
    dv: str = ""
    experimental_group: str = ""
    control_group: str = ""
    relationship_type: str = "unspecified"
    direction: str = "unspecified"
    evidence_ids: list[str] = field(default_factory=list)
    other_treatments: list[str] = field(default_factory=list)  # same-figure treatments
    other_endpoints: list[str] = field(default_factory=list)  # same-experiment DVs
    sibling_labels: list[str] = field(default_factory=list)  # sibling panels / other figures
    numeric_pool: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # kind -> [(value, evidence_id)]
    condition_findings: list[tuple[str, str, str]] = field(default_factory=list)  # (sentence, value, evidence_id)


def build_true_statement(blueprint: QuestionBlueprint) -> str | None:
    """The TRUE statement, derived only from blueprint-bound content."""
    detail = blueprint.detail
    if blueprint.question_type == "RESULT_INTERPRETATION" and detail.get("significance") == "significant":
        direction = detail.get("direction")
        comparison = detail.get("comparison", {})
        subject = detail.get("intervention") or comparison.get("experimental")
        endpoint = comparison.get("endpoint")
        if direction in ("increase", "decrease") and subject and endpoint:
            verb = "increases" if direction == "increase" else "decreases"
            return f"{subject} significantly {verb} {endpoint}."
    if blueprint.question_type == "DATA_STATEMENT":
        value = detail.get("data_value")
        if value:
            return f"The reported value is {value}."
        return None
    return blueprint.expected_answer or None


def _swap_tokens(text: str, first: str, second: str) -> str | None:
    if not first or not second or first == second:
        return None
    if first not in text or second not in text:
        return None
    return text.replace(first, "\x00").replace(second, first).replace("\x00", second)


def _apply(text: str, statement: str | None) -> str | None:
    """Accept a candidate only if it is a real change different from the truth."""
    if statement is None or statement == text:
        return None
    return statement


# --- individual perturbations ------------------------------------------------------


def direction_flip(text: str, context: PerturbationContext) -> str | None:
    flipped = _DIRECTION_WORD_RE.sub(lambda match: _DIRECTION_MAP[match.group(1)], text)
    return _apply(text, flipped)


def significance_flip(text: str, context: PerturbationContext) -> str | None:
    match = _SIGNIFICANT_TRUE_RE.match(text)
    if match is None:
        return None
    statement = (
        f"{match.group('subject')} does not significantly "
        f"{_BASE_VERB[match.group('verb')]} {match.group('rest')}"
    )
    return _apply(text, statement)


def group_swap(text: str, context: PerturbationContext) -> str | None:
    return _apply(text, _swap_tokens(text, context.experimental_group, context.control_group))


def variable_swap(text: str, context: PerturbationContext) -> str | None:
    if not context.intervention or not context.other_treatments:
        return None
    replacement = context.other_treatments[0]
    if replacement.lower() == context.intervention.lower() or context.intervention not in text:
        return None
    return _apply(text, text.replace(context.intervention, replacement, 1))


def dv_swap(text: str, context: PerturbationContext) -> str | None:
    if not context.dv or not context.other_endpoints:
        return None
    replacement = context.other_endpoints[0]
    if replacement.lower() == context.dv.lower() or context.dv not in text:
        return None
    return _apply(text, text.replace(context.dv, replacement, 1))


def causality_upgrade(text: str, context: PerturbationContext) -> str | None:
    if context.relationship_type not in ("association", "correlation"):
        return None  # only association/correlation can be (wrongly) upgraded — never emitted as truth
    for phrase in ("is associated with", "correlates with"):
        if phrase in text:
            return _apply(text, text.replace(phrase, "causes", 1))
    return None


def conclusion_flip(text: str, context: PerturbationContext) -> str | None:
    negations = (
        ("would increase", "would not increase"),
        ("would decrease", "would not decrease"),
        ("is associated with", "is not associated with"),
        ("increases", "does not increase"),
        ("decreases", "does not decrease"),
        ("provides the baseline", "does not provide the baseline"),
        ("receives", "does not receive"),
        ("measures", "does not measure"),
    )
    flipped = text
    for phrase, negated in negations:
        if phrase in flipped:
            return _apply(text, flipped.replace(phrase, negated, 1))
    return None


def panel_misattribution(text: str, context: PerturbationContext) -> str | None:
    if not context.sibling_labels:
        return None
    other = context.sibling_labels[0]
    return _apply(text, f"According to {other}, {text}")


def numeric_mutation(text: str, context: PerturbationContext, kind: str, current: str) -> str | None:
    pool = [value for value, _eid in context.numeric_pool.get(kind, []) if value != current]
    if not pool:
        return None
    replacement = pool[0]
    return _apply(text, f"The reported value is {replacement}.")


def condition_mutation(text: str, context: PerturbationContext) -> tuple[str, list[str]] | None:
    """Swap a reported condition (concentration/time) for another literal from the paper."""
    for sentence, value, evidence_id in context.condition_findings:
        kind = _kind_of_value(value)
        pool = [v for v, _eid in context.numeric_pool.get(kind, []) if v != value]
        if not pool:
            continue
        mutated = sentence.replace(value, pool[0], 1)
        if mutated != sentence:
            return mutated, [evidence_id]
    return None


def _kind_of_value(value: str) -> str:
    if "%" in value:
        return "percentage"
    if re.search(r"(?i)fold", value):
        return "fold_change"
    if re.search(r"(?i)^\s*p\s*[<=>]", value):
        return "p_value"
    if re.search(r"(?i)(min|hour|hr|day|week|month|h|d)\s*$", value):
        return "time"
    return "concentration"


PERTURBERS = {
    "DIRECTION_FLIP": lambda text, ctx, bp: (direction_flip(text, ctx), []),
    "SIGNIFICANCE_FLIP": lambda text, ctx, bp: (significance_flip(text, ctx), []),
    "GROUP_SWAP": lambda text, ctx, bp: (group_swap(text, ctx), []),
    "CAUSALITY_UPGRADE": lambda text, ctx, bp: (causality_upgrade(text, ctx), []),
    "CONCLUSION_FLIP": lambda text, ctx, bp: (conclusion_flip(text, ctx), []),
    "DV_SWAP": lambda text, ctx, bp: (dv_swap(text, ctx), []),
    "VARIABLE_SWAP": lambda text, ctx, bp: (variable_swap(text, ctx), []),
    "PANEL_MISATTRIBUTION": lambda text, ctx, bp: (panel_misattribution(text, ctx), []),
    "NUMERIC_MUTATION": lambda text, ctx, bp: (
        (
            numeric_mutation(text, ctx, bp.detail.get("kind", ""), bp.detail.get("data_value", "")),
            [],
        )
        if bp.question_type == "DATA_STATEMENT"
        else (None, [])
    ),
    "CONDITION_MUTATION": lambda text, ctx, bp: condition_mutation(text, ctx) or (None, []),
}
