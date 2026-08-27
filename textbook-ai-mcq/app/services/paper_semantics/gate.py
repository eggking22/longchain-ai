"""Semantic Evidence Gate for figure reconstruction (independent of Phase 3).

Decides whether the collected text evidence suffices to recover a figure's
experiment semantics:

- SUFFICIENT: caption + experimental design (groups, or IV+DV) + result direction;
- PARTIAL: the experiment is identifiable but a core slot (typically the result
  direction or the group comparison) is missing;
- INSUFFICIENT: only figure mentions like "Figure 3 shows the results" with no
  caption, no design, no result statement — reconstruction would be guessing.

This mirrors Phase 3's EvidenceGate *idea* but shares no code with it; the
existing gate keeps its binary query-coverage contract untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.paper_semantics import (
    ExperimentModel,
    FigureReference,
    PaperEvidence,
    ReconstructionStatus,
)

# Weighted core slots; confidence = sum of recovered weights.
_SLOT_WEIGHTS = {
    "caption": 0.2,
    "independent_variable": 0.15,
    "dependent_variable": 0.15,
    "groups": 0.15,
    "result_direction": 0.25,
    "significance": 0.1,
}

_SLOT_LABELS = {
    "caption": "figure caption",
    "independent_variable": "independent variable / intervention",
    "dependent_variable": "dependent variable / measured endpoint",
    "groups": "experimental vs control group comparison",
    "result_direction": "direction of change",
    "significance": "statistical significance",
}


@dataclass
class GateVerdict:
    status: ReconstructionStatus
    missing_information: list[str]
    confidence: float
    slots: dict[str, bool]


class SemanticEvidenceGate:
    """Three-level sufficiency gate over one figure's reconstruction inputs."""

    name = "semantic"

    def evaluate(self, ref: FigureReference, experiment: ExperimentModel, evidences: list[PaperEvidence]) -> GateVerdict:
        slots = {
            "caption": bool(ref.caption_text),
            "independent_variable": bool(experiment.independent_variables or experiment.intervention),
            "dependent_variable": bool(experiment.dependent_variables),
            "groups": bool(experiment.experimental_groups and experiment.control_groups),
            "result_direction": any(o.direction != "unspecified" for o in experiment.observations),
            "significance": any(o.significance != "unspecified" for o in experiment.observations),
        }
        confidence = round(sum(weight for slot, weight in _SLOT_WEIGHTS.items() if slots[slot]), 2)

        has_design = slots["groups"] or (slots["independent_variable"] and slots["dependent_variable"])
        experiment_identifiable = slots["caption"] or has_design or slots["dependent_variable"]

        if slots["caption"] and has_design and slots["result_direction"]:
            status: ReconstructionStatus = "SUFFICIENT"
        elif experiment_identifiable or slots["result_direction"]:
            status = "PARTIAL"
        else:
            status = "INSUFFICIENT"

        missing = [label for slot, label in _SLOT_LABELS.items() if not slots[slot]]
        if status == "SUFFICIENT":
            missing = []
        return GateVerdict(status=status, missing_information=missing, confidence=confidence, slots=slots)
