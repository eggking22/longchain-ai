"""Literal quantitative-value extraction from evidence text (DATA_STATEMENT).

Only values literally written in qualifying evidence sentences are extracted:
percentages, fold changes, p-values, concentrations, times. "significantly
increased" yields nothing — numbers are never inferred, and visual information
(curve heights, bar sizes) does not exist in the text pipeline at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.paper_semantics import PaperEvidence

# Numeric finding kinds, in the priority order used for DATA_STATEMENT selection.
KIND_ORDER = ("percentage", "fold_change", "p_value", "concentration", "time")

PERCENTAGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
FOLD_CHANGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*[-\u2013]?\s*fold", re.IGNORECASE)
CONCENTRATION_RE = re.compile(
    # case-sensitive: "25 µM" (molar) matches, "3 µm" (micrometre) must not
    r"\b\d+(?:\.\d+)?\s*(?:µ|μ|u|n|m|p)?(?:M|g(?:/m[lL])?|mol/L)\b"
)
TIME_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:min(?:ute)?s?|h(?:ou?r)?s?|d(?:ay)?s?|weeks?|months?)\b",
    re.IGNORECASE,
)

# Reuse the semantic layer's p-value pattern so DATA_STATEMENT p-values and
# Observation.p_value can never disagree.
from app.services.paper_semantics.patterns import P_VALUE_RE, split_sentences_en  # noqa: E402

# A glued digit + single panel letter ("1g", "2b") is a figure-panel reference,
# not a quantity, when cited like one: followed by )/,/; or preceded by
# "Fig."/"Figure". Real glued quantities ("6h incubation") are followed by a
# space/word and stay extractable; spaced quantities ("1 g") are never panels.
PANEL_GLUE_RE = re.compile(r"^\d{1,3}[a-hA-H]$")
_FIG_CONTEXT_RE = re.compile(r"(?i)fig(?:ure|\.)\s*$")


def _is_panel_reference(value: str, sentence: str) -> bool:
    if re.search(r"\s", value):
        return False  # "1 g" / "25 µM": spaced quantities are never panel labels
    if not PANEL_GLUE_RE.fullmatch(value):
        return False
    index = sentence.find(value)
    if index < 0:
        return False
    after = sentence[index + len(value) : index + len(value) + 1]
    before = sentence[max(0, index - 10) : index]
    return after in (")", ",", ";") or bool(_FIG_CONTEXT_RE.search(before))


@dataclass
class NumericFinding:
    """One literal quantitative value bound to its sentence and evidence."""

    kind: str  # percentage | fold_change | p_value | concentration | time
    value: str  # literal surface form: "30%", "2.5-fold", "p < 0.01", "25 µM", "24 hours"
    sentence: str
    evidence_id: str
    evidence_type: str


def _evidence_allows_data(evidence: PaperEvidence) -> bool:
    """Only reported results/statistics may carry quantitative statements.

    Methods (experimental_design) numbers describe protocols, and continuation
    paragraphs are stored context only — neither is a reported result.
    """
    if evidence.assignment == "continuation":
        return False
    if evidence.role not in ("caption", "direct"):
        return False
    return evidence.evidence_type in ("direct_observation", "statistical_result")


def extract_numeric_findings(evidences: list[PaperEvidence]) -> list[NumericFinding]:
    """Extract literal quantitative findings from qualifying evidence, in order."""
    findings: list[NumericFinding] = []
    for evidence in evidences:
        if not _evidence_allows_data(evidence):
            continue
        for sentence in split_sentences_en(evidence.text):
            for kind, regex in (
                ("percentage", PERCENTAGE_RE),
                ("fold_change", FOLD_CHANGE_RE),
                ("p_value", P_VALUE_RE),
                ("concentration", CONCENTRATION_RE),
                ("time", TIME_RE),
            ):
                for match in regex.finditer(sentence):
                    value = re.sub(r"\s+", " ", match.group(0))
                    if _is_panel_reference(value, sentence):
                        continue  # "(1g)" / "Fig. 2b," are citations, not quantities
                    findings.append(
                        NumericFinding(
                            kind=kind,
                            value=value,
                            sentence=sentence,
                            evidence_id=evidence.evidence_id,
                            evidence_type=evidence.evidence_type,
                        )
                    )
    return findings
