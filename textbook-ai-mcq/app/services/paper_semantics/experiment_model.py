"""Deterministic experiment-model extraction from a figure's evidence bundle.

Every field is derived by pattern matching over the collected evidence text;
nothing is invented. Sentences carry the evidence_id of the paragraph they
came from, so each observation can be traced to paragraph/page/section.

Relationship typing is conservative:
- "correlated with" / "associated with" never become causal;
- a causal reading additionally requires controlled-design language
  ("compared with control" / "in response to" ...) or an explicit causal verb.
"""

from __future__ import annotations

from app.schemas.paper_semantics import (
    Direction,
    ExperimentModel,
    FigureReference,
    Interpretation,
    Observation,
    PaperEvidence,
    RelationshipType,
    Significance,
)

from .figure_reference import figure_key

from .patterns import (
    ACTIVATION_RE,
    ASSOCIATION_RE,
    CAUSAL_RE,
    COMPARISON_RE,
    CONTROL_TERM_RE,
    CORRELATION_RE,
    DECREASE_RE,
    DOSE_RESPONSE_RE,
    GROUP_SPLIT_RE,
    INCREASE_RE,
    INHIBITION_RE,
    KNOCKOUT_RE,
    MEASURED_OF_RE,
    MEASURED_SUFFIX_RE,
    NO_CHANGE_RE,
    NOT_SIGNIFICANT_RE,
    OVEREXPRESSION_RE,
    P_VALUE_RE,
    SIGNIFICANT_RE,
    STATS_NOTE_RE,
    SUBJECT_RE,
    TIME_DEPENDENT_RE,
    TREATED_WITH_PREFIX_RE,
    TREATMENT_SPAN_RE,
    split_sentences_en,
)


def detect_direction(sentence: str) -> Direction:
    if NO_CHANGE_RE.search(sentence):
        return "no_change"
    increase = INCREASE_RE.search(sentence)
    decrease = DECREASE_RE.search(sentence)
    if increase and decrease:
        return "increase" if increase.start() < decrease.start() else "decrease"
    if increase:
        return "increase"
    if decrease:
        return "decrease"
    return "unspecified"


def detect_significance(sentence: str) -> Significance:
    if NOT_SIGNIFICANT_RE.search(sentence):
        return "not_significant"
    if SIGNIFICANT_RE.search(sentence):
        return "significant"
    return "unspecified"


def classify_relationship(sentence: str) -> RelationshipType:
    """Map a sentence onto a relationship type with explicit precedence."""
    if CORRELATION_RE.search(sentence):
        return "correlation"
    if ASSOCIATION_RE.search(sentence):
        return "association"
    if KNOCKOUT_RE.search(sentence):
        return "knockout"
    if OVEREXPRESSION_RE.search(sentence):
        return "overexpression"
    if DOSE_RESPONSE_RE.search(sentence):
        return "dose_response"
    if TIME_DEPENDENT_RE.search(sentence):
        return "time_dependent"
    if INHIBITION_RE.search(sentence):
        return "inhibition"
    if ACTIVATION_RE.search(sentence):
        return "activation"
    if CAUSAL_RE.search(sentence):
        return "causal"
    if COMPARISON_RE.search(sentence) and detect_direction(sentence) in ("increase", "decrease"):
        return "causal"  # controlled comparison licenses the causal reading
    return "unspecified"


def _is_result_sentence(sentence: str) -> bool:
    """A sentence states a result if it carries direction, significance or a p-value."""
    return (
        detect_direction(sentence) != "unspecified"
        or detect_significance(sentence) != "unspecified"
        or P_VALUE_RE.search(sentence) is not None
    )


def extract_observations(evidences: list[PaperEvidence]) -> list[Observation]:
    observations: list[Observation] = []
    for evidence in evidences:
        if evidence.role not in ("caption", "direct"):
            continue
        for sentence in split_sentences_en(evidence.text):
            if STATS_NOTE_RE.match(sentence):
                continue  # "Data were analyzed by ...; ****P < 0.0001" is a methods note, not a result
            if not _is_result_sentence(sentence):
                continue
            p_value = P_VALUE_RE.search(sentence)
            observations.append(
                Observation(
                    statement=sentence,
                    direction=detect_direction(sentence),
                    significance=detect_significance(sentence),
                    relationship_type=classify_relationship(sentence),
                    p_value=p_value.group(0) if p_value else None,
                    evidence_ids=[evidence.evidence_id],
                )
            )
    return observations


def extract_control_groups(evidences: list[PaperEvidence]) -> list[str]:
    found: list[str] = []
    for evidence in sorted(evidences, key=lambda e: {"direct": 0, "caption": 1, "supporting": 2, "interpretation": 3}[e.role]):
        for match in CONTROL_TERM_RE.finditer(evidence.text):
            term = match.group(0).lower()
            if term not in found:
                found.append(term)
        for match in GROUP_SPLIT_RE.finditer(evidence.text):
            for part in _split_enumeration(match.group(1)):
                if CONTROL_TERM_RE.fullmatch(part.strip()):
                    if part.strip() not in found:
                        found.append(part.strip())
    return found


def _split_enumeration(inner: str) -> list[str]:
    parts = [p.strip() for p in inner.replace(" and ", ",").split(",") if p.strip()]
    return parts


def extract_experimental_groups(evidences: list[PaperEvidence]) -> list[str]:
    found: list[str] = []
    priority = {"direct": 0, "caption": 1, "supporting": 2, "interpretation": 3}

    def push(name: str) -> None:
        name = TREATED_WITH_PREFIX_RE.sub("", name.strip()) or name.strip()  # "treated with the X" → "X"
        if name.lower() not in [f.lower() for f in found]:
            found.append(name)

    for evidence in sorted(evidences, key=lambda e: priority[e.role]):
        for match in TREATMENT_SPAN_RE.finditer(evidence.text):
            push(match.group(0).strip())
        for match in GROUP_SPLIT_RE.finditer(evidence.text):
            for part in _split_enumeration(match.group(1)):
                lowered = part.lower()
                if "treatment" in lowered or "experimental" in lowered or "intervention" in lowered:
                    push(part)
    return found


def extract_intervention(experimental_groups: list[str]) -> str:
    """Prefer a specific treatment span over the generic word 'treatment'."""
    for group in experimental_groups:
        if group.lower() not in ("treatment", "experimental", "experimental group", "treatment group"):
            return group
    return experimental_groups[0] if experimental_groups else ""


def extract_dependent_variables(evidences: list[PaperEvidence]) -> list[str]:
    """Extract measured endpoints; the suffix form ("body weight") is preferred
    over the of-form ("weight of mice" → "mice weight") when both occur."""
    found: list[str] = []
    priority = {"direct": 0, "caption": 1, "supporting": 2, "interpretation": 3}

    def push(variable: str) -> None:
        if variable.lower() not in [f.lower() for f in found]:
            found.append(variable)

    for evidence in sorted(evidences, key=lambda e: priority[e.role]):
        text = evidence.text
        for match in MEASURED_SUFFIX_RE.finditer(text):
            entity, head = match.group(1).strip(), match.group(2).lower()
            push(f"{entity} {head}")
        for match in MEASURED_OF_RE.finditer(text):
            head, entity = match.group(1).lower(), match.group(2).strip()
            push(f"{entity} {head}")
    return found


def extract_subjects(evidences: list[PaperEvidence]) -> list[str]:
    found: list[str] = []
    for evidence in sorted(evidences, key=lambda e: {"supporting": 0, "caption": 1, "direct": 2, "interpretation": 3}[e.role]):
        for match in SUBJECT_RE.finditer(evidence.text):
            term = match.group(0).lower()
            if term not in found:
                found.append(term)
    return found


def extract_statistical_results(evidences: list[PaperEvidence]) -> list[str]:
    found: list[str] = []
    for evidence in evidences:
        if evidence.role not in ("caption", "direct"):
            continue
        for match in P_VALUE_RE.finditer(evidence.text):
            literal = match.group(0).replace(" ", "")
            if literal not in found:
                found.append(literal)
    return found


def extract_interpretations(evidences: list[PaperEvidence]) -> list[Interpretation]:
    """Author claims from interpretation evidence, recorded verbatim.

    These are *attributed* statements (the authors' mechanistic reading), not
    system assertions; relationship typing keeps the authors' own strength —
    association/correlation sentences are never upgraded to causal.
    """
    interpretations: list[Interpretation] = []
    for evidence in evidences:
        if evidence.role != "interpretation":
            continue
        for sentence in split_sentences_en(evidence.text):
            if STATS_NOTE_RE.match(sentence):
                continue
            direction = detect_direction(sentence)
            relationship = classify_relationship(sentence)
            if direction == "unspecified" and relationship == "unspecified":
                continue
            interpretations.append(
                Interpretation(
                    statement=sentence,
                    direction=direction,
                    relationship_type=relationship,
                    evidence_ids=[evidence.evidence_id],
                )
            )
    for index, interpretation in enumerate(interpretations, start=1):
        interpretation.interpretation_id = f"int_{index:03d}"
    return interpretations


def build_experiment(
    ref: FigureReference, evidences: list[PaperEvidence], id_suffix: str = ""
) -> ExperimentModel:
    """Deterministically reconstruct the ExperimentModel behind one figure.

    id_suffix distinguishes panel-level models ("a" → exp_f02a) from the
    figure-level baseline model (exp_f02). Continuation evidence (uncited
    paragraphs inherited from the L1 text block) is stored in the bundle but
    excluded from semantic extraction: only explicit anchors, captions,
    Methods and Discussion text may produce observations, groups or variables.
    """
    extraction_pool = [e for e in evidences if e.assignment != "continuation"]
    observations = extract_observations(extraction_pool)
    experimental_groups = extract_experimental_groups(extraction_pool)
    control_groups = extract_control_groups(extraction_pool)
    intervention = extract_intervention(experimental_groups)
    dependent_variables = extract_dependent_variables(extraction_pool)
    independent_variables = [intervention] if intervention else []

    research_question = ""
    hypothesis = ""
    if intervention and dependent_variables:
        research_question = f"Does {intervention} affect {dependent_variables[0]}?"
        hypothesis = f"{intervention} affects {dependent_variables[0]}."

    prefix_key = figure_key(ref)
    return ExperimentModel(
        experiment_id=f"exp_{prefix_key}{id_suffix}",
        research_question=research_question,
        hypothesis=hypothesis,
        subjects=extract_subjects(extraction_pool),
        independent_variables=independent_variables,
        dependent_variables=dependent_variables,
        experimental_groups=experimental_groups,
        control_groups=control_groups,
        intervention=intervention,
        measurements=list(dependent_variables),
        observations=observations,
        interpretations=extract_interpretations(extraction_pool),
        statistical_results=extract_statistical_results(extraction_pool),
        evidence_ids=[e.evidence_id for e in evidences if e.evidence_id],
    )
