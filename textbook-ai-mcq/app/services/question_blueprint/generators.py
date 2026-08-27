"""Deterministic Question Blueprint generators (no LLM).

Every generator is a pure gate + template over the reconstructed Experiment
Model; the blueprint inherits the semantic layer's relationship strength
(association/correlation are never upgraded to causal) and every claim cites
Evidence Store ids. Evidence gaps mean *no* blueprint — gaps are counted in
the report's skipped summary instead of being papered over.
"""

from __future__ import annotations

from app.schemas.paper_semantics import (
    ExperimentModel,
    FigureSemantic,
    PanelSemantic,
    PaperEvidence,
)
from app.schemas.question_blueprint import QuestionBlueprint

from .config import BlueprintConfig
from .numeric import NumericFinding

TYPE_ABBR = {
    "RESULT_INTERPRETATION": "ri",
    "EXPERIMENTAL_DESIGN": "ed",
    "SIMPLE_PREDICTION": "pred",
    "DATA_STATEMENT": "ds",
}

# Relationships that license a controlled-experiment prediction. association /
# correlation / unspecified / time_dependent are deliberately excluded: a
# prediction built on them would assert more than the paper showed.
CAUSAL_LICENSED = {"causal", "inhibition", "activation", "knockout", "overexpression", "dose_response"}

NON_CAUSAL = {"association", "correlation"}


class SkipTracker:
    """Counted reasons why a blueprint was not generated (transparency, not noise)."""

    def __init__(self) -> None:
        self.reasons: dict[str, dict[str, int]] = {}

    def skip(self, question_type: str, reason: str) -> None:
        by_reason = self.reasons.setdefault(question_type, {})
        by_reason[reason] = by_reason.get(reason, 0) + 1

    def as_dict(self) -> dict:
        return {qtype: dict(by_reason) for qtype, by_reason in sorted(self.reasons.items())}


def _unit_key(experiment_id: str) -> str:
    return experiment_id.removeprefix("exp_")  # exp_f02 → f02, exp_f02a → f02a


class _IdAssigner:
    def __init__(self) -> None:
        self._counters: dict[tuple[str, str], int] = {}

    def next_id(self, experiment_id: str, question_type: str) -> str:
        key = (_unit_key(experiment_id), TYPE_ABBR[question_type])
        self._counters[key] = self._counters.get(key, 0) + 1
        return f"qb_{key[0]}_{key[1]}_{self._counters[key]:03d}"


def _design_evidence_ids(evidences: list[PaperEvidence]) -> list[str]:
    """Evidence units that carry the experimental-design slots (groups, treatments)."""
    return [e.evidence_id for e in evidences if e.evidence_type == "experimental_design" and e.evidence_id]


def _first(properties) -> str:
    return properties[0] if properties else ""


# --- RESULT_INTERPRETATION -----------------------------------------------------------


def generate_result_interpretation(
    figure: FigureSemantic,
    experiment: ExperimentModel,
    *,
    panel: PanelSemantic | None = None,
    config: BlueprintConfig | None = None,
    ids: _IdAssigner | None = None,
    skips: SkipTracker | None = None,
) -> list[QuestionBlueprint]:
    config = config or BlueprintConfig()
    ids = ids or _IdAssigner()
    skips = skips or SkipTracker()
    label = panel.panel_id if panel else figure.figure_id
    confidence = panel.confidence if panel else figure.confidence

    directed = [o for o in experiment.observations if o.direction != "unspecified"]
    if not directed:
        skips.skip("RESULT_INTERPRETATION", "no_directed_observation")
        return []
    if not (experiment.intervention and experiment.dependent_variables):
        skips.skip("RESULT_INTERPRETATION", "missing_intervention_or_endpoint")
        return []

    dv = _first(experiment.dependent_variables)
    control = _first(experiment.control_groups)
    group = _first(experiment.experimental_groups) or experiment.intervention
    design_ids = _design_evidence_ids(panel.evidence if panel else figure.evidence)

    blueprints: list[QuestionBlueprint] = []
    for observation in directed[: config.max_result_interpretation]:
        non_causal = observation.relationship_type in NON_CAUSAL
        if non_causal:
            focus = f"What is the relationship between {experiment.intervention} and {dv} shown in {label}?"
        elif control:
            focus = f"What does {label} show about {dv} in {group} compared with the {control} group?"
        else:
            focus = f"What does {label} show about {dv} after {experiment.intervention}?"

        expected = _matching_conclusion(experiment, observation)
        if expected is None:
            verb = {"increase": "increased", "decrease": "decreased", "no_change": "did not significantly change"}[
                observation.direction
            ]
            comparator = f" relative to the {control} group" if control else ""
            expected = f"{dv} {verb}{comparator}."

        evidence_ids = list(observation.evidence_ids)
        if design_ids:
            evidence_ids.extend(eid for eid in design_ids if eid not in evidence_ids)
        if not evidence_ids:
            skips.skip("RESULT_INTERPRETATION", "no_bound_evidence")
            continue

        blueprints.append(
            QuestionBlueprint(
                blueprint_id=ids.next_id(experiment.experiment_id, "RESULT_INTERPRETATION"),
                question_type="RESULT_INTERPRETATION",
                experiment_id=experiment.experiment_id,
                figure_id=figure.figure_id,
                panel_ids=[panel.panel_id] if panel else [],
                question_focus=focus,
                required_evidence=["direct_observation", "experimental_design"],
                reasoning_operation="comparison" if control else "result_interpretation",
                expected_answer=expected,
                evidence_ids=evidence_ids,
                confidence=confidence,
                detail={
                    "relationship_type": observation.relationship_type,
                    "direction": observation.direction,
                    "significance": observation.significance,
                    "comparison": {"experimental": group, "control": control, "endpoint": dv},
                },
            )
        )
    return blueprints


def _matching_conclusion(experiment: ExperimentModel, observation) -> str | None:
    observation_evidence = set(observation.evidence_ids)
    for conclusion in experiment.conclusions:
        if set(conclusion.evidence_ids) & observation_evidence:
            return conclusion.statement
    return experiment.conclusions[0].statement if experiment.conclusions else None


# --- EXPERIMENTAL_DESIGN ---------------------------------------------------------------


def generate_experimental_design(
    figure: FigureSemantic,
    experiment: ExperimentModel,
    *,
    config: BlueprintConfig | None = None,
    ids: _IdAssigner | None = None,
    skips: SkipTracker | None = None,
) -> list[QuestionBlueprint]:
    config = config or BlueprintConfig()
    ids = ids or _IdAssigner()
    skips = skips or SkipTracker()

    if not (experiment.experimental_groups and experiment.control_groups and experiment.dependent_variables):
        skips.skip("EXPERIMENTAL_DESIGN", "incomplete_design_slots")
        return []
    design_ids = _design_evidence_ids(figure.evidence)
    if not design_ids:
        skips.skip("EXPERIMENTAL_DESIGN", "no_design_evidence")
        return []

    control = _first(experiment.control_groups)
    group = _first(experiment.experimental_groups)
    intervention = experiment.intervention or group
    dv = _first(experiment.dependent_variables)

    plans = [
        (
            f"Why does the experiment in {figure.figure_id} include a {control} group?",
            f"The {control} group provides the baseline level of {dv} against which {intervention} is compared.",
            {"design_element": "control_group", "group": control},
        ),
        (
            f"What is the purpose of the {group} group in {figure.figure_id}?",
            f"The {group} group receives {intervention} so that its effect on {dv} can be measured against the {control} baseline.",
            {"design_element": "experimental_group", "group": group},
        ),
        (
            f"What does the experiment shown in {figure.figure_id} measure?",
            f"It measures {dv} in the {group} group versus the {control} group.",
            {"design_element": "measured_endpoint", "endpoint": dv},
        ),
    ][: config.max_experimental_design]

    return [
        QuestionBlueprint(
            blueprint_id=ids.next_id(experiment.experiment_id, "EXPERIMENTAL_DESIGN"),
            question_type="EXPERIMENTAL_DESIGN",
            experiment_id=experiment.experiment_id,
            figure_id=figure.figure_id,
            panel_ids=[],
            question_focus=focus,
            required_evidence=["experimental_design"],
            reasoning_operation="experimental_design_reasoning",
            expected_answer=expected,
            evidence_ids=list(design_ids),
            confidence=figure.confidence,
            detail=detail,
        )
        for focus, expected, detail in plans
    ]


# --- SIMPLE_PREDICTION -------------------------------------------------------------------


def generate_simple_prediction(
    figure: FigureSemantic,
    experiment: ExperimentModel,
    *,
    config: BlueprintConfig | None = None,
    ids: _IdAssigner | None = None,
    skips: SkipTracker | None = None,
) -> list[QuestionBlueprint]:
    config = config or BlueprintConfig()
    ids = ids or _IdAssigner()
    skips = skips or SkipTracker()

    if figure.reconstruction_status != "SUFFICIENT":
        skips.skip("SIMPLE_PREDICTION", "figure_not_sufficient")
        return []
    if not (experiment.experimental_groups and experiment.control_groups):
        skips.skip("SIMPLE_PREDICTION", "missing_group_comparison")
        return []
    if not (experiment.intervention and experiment.dependent_variables):
        skips.skip("SIMPLE_PREDICTION", "missing_intervention_or_endpoint")
        return []
    directed = [o for o in experiment.observations if o.direction in ("increase", "decrease")]
    licensed = [o for o in directed if o.relationship_type in CAUSAL_LICENSED]
    if not licensed:
        if directed:
            skips.skip("SIMPLE_PREDICTION", "non_causal_relationship")  # association/correlation: no extrapolation
        else:
            skips.skip("SIMPLE_PREDICTION", "no_directed_observation")
        return []

    observation = licensed[0]
    control = _first(experiment.control_groups)
    group = _first(experiment.experimental_groups)
    intervention = experiment.intervention
    dv = _first(experiment.dependent_variables)
    would = {"increase": "increase", "decrease": "decrease"}[observation.direction]

    design_ids = _design_evidence_ids(figure.evidence)
    evidence_ids = list(observation.evidence_ids)
    evidence_ids.extend(eid for eid in design_ids if eid not in evidence_ids)
    if not evidence_ids:
        skips.skip("SIMPLE_PREDICTION", "no_bound_evidence")
        return []

    return [
        QuestionBlueprint(
            blueprint_id=ids.next_id(experiment.experiment_id, "SIMPLE_PREDICTION"),
            question_type="SIMPLE_PREDICTION",
            experiment_id=experiment.experiment_id,
            figure_id=figure.figure_id,
            panel_ids=[],
            # the prediction is grounded, not speculative: applying the treatment to
            # the control reproduces the already-observed treatment-group outcome
            question_focus=f"If {intervention} were applied to the {control} group, what would happen to {dv}?",
            required_evidence=["direct_observation", "experimental_design"],
            reasoning_operation="local_prediction",
            expected_answer=f"{dv} would {would}, as already observed in the {group} group.",
            evidence_ids=evidence_ids,
            confidence=figure.confidence,
            detail={
                "relationship_type": observation.relationship_type,
                "direction": observation.direction,
                "grounded_in": {"experimental_group": group, "control_group": control},
            },
        )
    ][: config.max_simple_prediction]


# --- DATA_STATEMENT ------------------------------------------------------------------------


def generate_data_statements(
    figure: FigureSemantic,
    *,
    panel: PanelSemantic | None = None,
    findings: list[NumericFinding] | None = None,
    experiment: ExperimentModel | None = None,
    config: BlueprintConfig | None = None,
    ids: _IdAssigner | None = None,
    skips: SkipTracker | None = None,
) -> list[QuestionBlueprint]:
    config = config or BlueprintConfig()
    ids = ids or _IdAssigner()
    skips = skips or SkipTracker()
    label = panel.panel_id if panel else figure.figure_id
    confidence = panel.confidence if panel else figure.confidence
    experiment_id = panel.experiment.experiment_id if panel and panel.experiment else (
        experiment.experiment_id if experiment else f"exp_{figure.figure_id.lower().replace(' ', '')}"
    )

    if not findings:
        skips.skip("DATA_STATEMENT", "no_literal_numeric_value")
        return []

    endpoint = _first(experiment.dependent_variables) if experiment else ""
    blueprints: list[QuestionBlueprint] = []
    for finding in findings[: config.max_data_statement]:
        subject = f"the reported change in {endpoint}" if endpoint else "the reported quantitative result"
        blueprints.append(
            QuestionBlueprint(
                blueprint_id=ids.next_id(experiment_id, "DATA_STATEMENT"),
                question_type="DATA_STATEMENT",
                experiment_id=experiment_id,
                figure_id=figure.figure_id,
                panel_ids=[panel.panel_id] if panel else [],
                question_focus=f"According to {label}, what quantitative value is explicitly reported for {subject}?",
                required_evidence=[finding.evidence_type],
                reasoning_operation="quantitative_reading",
                expected_answer=finding.value,
                evidence_ids=[finding.evidence_id],
                confidence=confidence,
                detail={"data_value": finding.value, "kind": finding.kind, "sentence": finding.sentence},
            )
        )
    return blueprints
