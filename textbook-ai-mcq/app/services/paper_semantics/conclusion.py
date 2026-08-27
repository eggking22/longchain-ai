"""Conservative conclusion synthesis from extracted observations.

A conclusion is only emitted when the experiment model already knows the
intervention, the measured variable, and an observed direction — and it only
restates what the cited observations say. Association/correlation never get
causal phrasing ("X is associated with increased Y", never "X increases Y").
"""

from __future__ import annotations

from app.schemas.paper_semantics import Conclusion, ExperimentModel

_VERB_THIRD_PERSON = {"increase": "increases", "decrease": "decreases", "no_change": "does not significantly alter"}
_VERB_PAST = {"increase": "increased", "decrease": "decreased", "no_change": "unchanged"}


def build_conclusions(experiment: ExperimentModel) -> list[Conclusion]:
    conclusions: list[Conclusion] = []
    dependent = experiment.dependent_variables[0] if experiment.dependent_variables else ""
    intervention = experiment.intervention

    for observation in experiment.observations:
        if not intervention or not dependent:
            break  # without IV/DV anchors any sentence would be unbound paraphrase
        if observation.direction == "unspecified":
            continue
        verb = _VERB_THIRD_PERSON[observation.direction]
        past = _VERB_PAST[observation.direction]
        if observation.relationship_type == "association":
            statement = f"{intervention} is associated with {past} {dependent}."
        elif observation.relationship_type == "correlation":
            statement = f"{intervention} correlates with {past} {dependent}."
        else:
            statement = f"{intervention} {verb} {dependent}."
        if any(c.statement == statement for c in conclusions):
            continue
        conclusions.append(
            Conclusion(
                statement=statement,
                relationship_type=observation.relationship_type,
                evidence_ids=list(observation.evidence_ids),
                interpretation_ids=_linked_interpretations(observation, experiment.interpretations),
            )
        )
    return conclusions


def _linked_interpretations(observation, interpretations) -> list[str]:
    """Ids of author interpretations that back the same direction.

    Linking never changes the conclusion text — the interpretation layer is
    recorded separately so descriptive observation, author interpretation and
    synthesized conclusion stay distinguishable.
    """
    linked = []
    for interpretation in interpretations:
        if interpretation.direction in (observation.direction, "unspecified"):
            linked.append(interpretation.interpretation_id)
    return linked
