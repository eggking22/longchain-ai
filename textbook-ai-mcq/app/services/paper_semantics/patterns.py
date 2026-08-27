"""English-only lexical patterns for figure semantic reconstruction.

Pure functions and compiled regexes, no I/O. Every semantic decision in this
package must trace back to one of these patterns so that "no text evidence →
no recovered semantics" holds by construction.

Language scope: English scientific papers (per project decision). Chinese
papers fall through to INSUFFICIENT rather than being guessed at.
"""

from __future__ import annotations

import re
from typing import Optional

# --- Figure / table mentions -------------------------------------------------------
# "Figure 1", "Fig. 2B", "figure 3a", "Figures 2 and 3", "Figure 2-4", "Table 1".
# An optional prefix ("Extended Data", "Supplementary") denotes a distinct figure
# namespace and must not roll up into the main numbering.
_LABEL_PREFIX = r"(Extended\s+Data|Supplementary|Suppl\.?)?\s*"
_FIGURE_LABEL = r"(?:Figures|Figure|Figs\.|Fig\.|Figs|Fig)"
_TABLE_LABEL = r"(?:Tables|Table)"
_PANEL = r"([A-Za-z])?(?![A-Za-z0-9])"
_LIST_TAIL = r"(?:(?:\s*(?:and|,|to|[-\u2013]))\s*(\d+)(?![0-9]))?"

FIGURE_MENTION_RE = re.compile(
    rf"\b{_LABEL_PREFIX}{_FIGURE_LABEL}\s*(\d+)\s*{_PANEL}{_LIST_TAIL}",
    re.IGNORECASE,
)
TABLE_MENTION_RE = re.compile(
    rf"\b{_LABEL_PREFIX}{_TABLE_LABEL}\s*(\d+)\s*{_PANEL}{_LIST_TAIL}",
    re.IGNORECASE,
)

# A caption paragraph starts with the label directly followed by sentence
# punctuation or Nature's pipe separator: "Figure 2. ...", "Fig. 1: ...",
# "Table 3.", "Fig. 1 | Title ...". "Figure 2 shows ..." (label + verb) is a
# mention, not a caption.
CAPTION_START_RE = re.compile(
    r"^(?:(Extended\s+Data|Supplementary|Suppl\.?)\s+)?"
    r"(?:Figures|Figure|Figs\.|Fig\.|Tables|Table)\s*(\d+)\s*([A-Za-z])?\s*[.::|\u2014\u2013]\s*",
    re.IGNORECASE,
)
# Inline caption anchor: the Nature pipe form "Fig. 1 | Title". Body prose never
# references figures this way, so a pipe-marked label inside a merged paragraph
# (two-column PDFs often glue body text onto the caption) still identifies a
# caption unambiguously.
INLINE_CAPTION_RE = re.compile(
    r"(?:(Extended\s+Data|Supplementary|Suppl\.?)\s+)?"
    r"(?:Figures|Figure|Figs\.|Fig\.|Tables|Table)\s*(\d+)\s*([A-Za-z])?\s*\|",
    re.IGNORECASE,
)


# --- Direction ----------------------------------------------------------------------
# Verb forms only: agent nouns (inhibitor, promoter, enhancer, suppressor) name
# the treatment, not a change, and must not fabricate a direction.

INCREASE_RE = re.compile(
    r"\b(increas\w*|elevat\w*|up-?regulat\w*|promot(?:e|es|ed|ing)\b|enhanc(?:e|es|ed|ing|ement)\b|"
    r"induc(?:e|es|ed|ing)\b|stimulat\w*|higher|rose|elevated|greater)\b",
    re.IGNORECASE,
)
DECREASE_RE = re.compile(
    r"\b(decreas\w*|reduc\w*|down-?regulat\w*|inhibit(?:s|ed|ing)\b|suppress(?:es|ed|ing|ion)\b|"
    r"attenuat\w*|diminish\w*|lower\w*|less|fewer|drop(?:ped)?)\b",
    re.IGNORECASE,
)
NO_CHANGE_RE = re.compile(
    r"\b(no significant (?:difference|change|alteration)|not significantly (?:different|changed|altered)|"
    r"unchanged|unaffected|comparable to|similar to)\b",
    re.IGNORECASE,
)


# --- Significance ---------------------------------------------------------------------
# "significantly increased" → significance="significant", but p-values are only
# recorded when literally written; they are never inferred from the word "significant".

SIGNIFICANT_RE = re.compile(r"\bsignifican\w*\b", re.IGNORECASE)
NOT_SIGNIFICANT_RE = re.compile(
    r"\b(?:not|no|without)\s+significan\w*\b|\bnot\s+significant\b|\bns\b(?![a-z])",
    re.IGNORECASE,
)
P_VALUE_RE = re.compile(r"\bp\s*(?:<|>|=|\u2264|\u2265)\s*[\d.]+", re.IGNORECASE)


# --- Relationship markers ---------------------------------------------------------------

CORRELATION_RE = re.compile(
    r"\bcorrelat\w*(?:\s+(?:positively|negatively|strongly|weakly|significantly))?\s+with\b",
    re.IGNORECASE,
)
ASSOCIATION_RE = re.compile(
    r"\bassociat\w*(?:\s+(?:positively|negatively|strongly|significantly))?\s+with\b",
    re.IGNORECASE,
)
CAUSAL_RE = re.compile(
    r"\b(?:caus\w+|due to|result(?:ed|s|ing)? in|led to|leads to|driv\w+|mediat\w+|"
    r"necessar\w+ for|sufficient for|because of)\b",
    re.IGNORECASE,
)
COMPARISON_RE = re.compile(  # controlled-design language that licenses a causal reading
    r"\bcompared\s+(?:with|to|against)\b|\bvs\.?\b|\brelative to\b|\bin\s+response\s+to\b",
    re.IGNORECASE,
)
INHIBITION_RE = re.compile(r"\binhibit\w+|suppress\w+|block\w+|attenuat\w+", re.IGNORECASE)
ACTIVATION_RE = re.compile(r"\bactivat\w+|stimulat\w+|potentiat\w+\b", re.IGNORECASE)
KNOCKOUT_RE = re.compile(
    r"\bknock-?out\w*|\bknocked-?out\b|\bko\b(?![a-z])|\bsiRNA\b|\bshRNA\b|\bCRISPR\w*\b", re.IGNORECASE
)
OVEREXPRESSION_RE = re.compile(r"\bover-?express\w*\b|\boverexpression\b", re.IGNORECASE)
DOSE_RESPONSE_RE = re.compile(r"\bdose-?dependent\b|\bdose\s+response\b|\bdoes-?response\b", re.IGNORECASE)
TIME_DEPENDENT_RE = re.compile(
    r"\btime-?(?:dependent|course|point)\b|\bover time\b|\bhours?\s+(?:after|post)\b|\bdays?\s+(?:after|post)\b",
    re.IGNORECASE,
)


# --- Experimental design vocabulary --------------------------------------------------------

CONTROL_TERM_RE = re.compile(
    r"\b(control(?:\s+group)?|untreated|untransfected|unstimulated|wild-?type|wt|vehicle|mock|sham|placebo|"
    r"scrambled)\b",
    re.IGNORECASE,
)
# "Treatment A" (capitalized name), "X-treated cells", "treated with X".
# The capitalized-name form avoids matching generic "treatment group"; the
# "treated with" form skips articles and keeps the agent name.
TREATMENT_SPAN_RE = re.compile(
    r"\b[Tt]reatment\s+[A-Z][\w-]*"
    r"|\b[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){0,2}-treated\b"
    r"|\b[Tt]reated\s+with\s+(?:the\s+|a\s+|an\s+)?[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*)?"
)
GROUP_SPLIT_RE = re.compile(  # "control and treatment groups" style enumeration
    r"\b(?:divided|assigned|randomized|grouped)\s+into\s+(.{3,120}?)\s+groups?\b",
    re.IGNORECASE,
)
TREATED_WITH_PREFIX_RE = re.compile(r"^[Tt]reated\s+with\s+(?:the\s+|a\s+|an\s+)?")

# --- Measured entities -------------------------------------------------------------------
# "expression of gene X", "gene X expression", "body weight". Entity tokens may
# not be verbs/adverbs/prepositions — enforced in the regex via lookaheads so a
# blocked word can never be consumed as (part of) the entity, which keeps
# finditer's non-overlapping scan from swallowing the real entity.

_ENTITY_BLOCK = (
    "significantly|increased|decreased|showed|shown|was|were|is|are|treated|compared|"
    "higher|lower|reduced|elevated|relative|respectively|of|for|in|on|at|to|with|by|from|"
    "and|or|the|a|an|between|after|following|during|when|while|that|which|"
    "promotes|promoted|enhances|enhanced|induces|induced|stimulates|stimulated|"
    "inhibits|inhibited|suppresses|suppressed|reduces|elevates|activates|activated"
)
_ENTITY = rf"(?:(?!{_ENTITY_BLOCK})[A-Za-z][\w/-]*(?:\s+(?!{_ENTITY_BLOCK})[A-Za-z][\w/-]*){{0,2}}?)"

MEASURED_OF_RE = re.compile(
    rf"\b(expression|activity|abundance|concentration|levels?|rate|intensity|phosphorylation|secretion|production|"
    rf"weights?|up-?regulation|down-?regulation)\s+of\s+"
    rf"({_ENTITY})"
    r"(?=\s*(?:was|were|is|are|in|compared|than|between|after|following|[,.;:]|\(|$))",
    re.IGNORECASE,
)
MEASURED_SUFFIX_RE = re.compile(
    rf"\b({_ENTITY})\s+"
    r"(expression|activity|abundance|concentration|intensity|phosphorylation|secretion|production|weight|"
    r"up-?regulation|down-?regulation)\b",
    re.IGNORECASE,
)

# Caption statistics notes ("Data were analyzed by ...; ****P < 0.0001; NS, not
# significant") carry literal p-values but are methods annotations, not results.
STATS_NOTE_RE = re.compile(
    r"^(?:Data\s+(?:were|are)|Statistics\s+(?:were|are)|\*{1,4}\s*P\b|NS\s*[,;.]|\(?\*{1,4}\)|P\s*[<=>])",
    re.IGNORECASE,
)

# Panel boundary inside a caption: Nature style "a, Left, migrating CD11c+ DCs..."
# "b, Expression of ...". A single letter + comma + capitalized/digit content;
# the lookbehind excludes mid-word letters and the lookahead excludes plain
# enumerations like "a, b, c" (lowercase after the comma).
PANEL_BOUNDARY_RE = re.compile(r"(?<![A-Za-z])([A-Ha-h])\s*,\s+(?=[A-Z(\d])")
SUBJECT_RE = re.compile(
    r"\b(?:cells?|mice|rats?|patients?|subjects?|samples?|tissues?|plants?|seedlings?|flies?|fish|"
    r"human\s+\w+|animals?)\b",
    re.IGNORECASE,
)

# --- Utilities ---------------------------------------------------------------------------


def split_sentences_en(text: str) -> list[str]:
    """Split an English paragraph into sentences (period/question/exclamation + space)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]{2,}")
_STOPWORDS = frozenset(
    """the a an and or of in on at to for with by from as is are was were be been being this that these those
    we our it its their his her not no than then more most such can could may might will would should shall
    have has had do does did using used use show showed shown shows figure table fig results result method
    methods respectively however therefore thus whereas while when where which who whom whose what also both
    each between during after before above below over under into out up down about through during""".split()
)


def word_tokens(text: str) -> list[str]:
    """Lowercased English content words (used for Methods relevance scoring)."""
    return [t.lower() for t in _WORD_RE.findall(text) if t.lower() not in _STOPWORDS]
