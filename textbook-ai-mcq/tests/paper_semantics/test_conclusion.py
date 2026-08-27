"""Conclusion synthesis: evidence binding, conservative phrasing."""

from __future__ import annotations

import pytest

from app.schemas.paper_semantics import ExperimentModel, Observation
from app.services.paper_semantics import build_conclusions


def _experiment(**overrides) -> ExperimentModel:
    defaults = dict(
        experiment_id="exp_f02",
        intervention="Treatment A",
        independent_variables=["Treatment A"],
        dependent_variables=["gene X expression"],
        observations=[
            Observation(
                statement="Treatment A significantly increased expression of gene X compared with control.",
                direction="increase",
                significance="significant",
                relationship_type="causal",
                evidence_ids=["ev_002"],
            )
        ],
    )
    defaults.update(overrides)
    return ExperimentModel(**defaults)


class TestConclusionSynthesis:
    def test_canonical_conclusion(self):
        conclusions = build_conclusions(_experiment())
        assert [c.statement for c in conclusions] == ["Treatment A increases gene X expression."]
        assert conclusions[0].evidence_ids == ["ev_002"]
        assert conclusions[0].relationship_type == "causal"

    def test_association_never_becomes_causal(self):
        experiment = _experiment(
            observations=[
                Observation(
                    statement="Treatment C was associated with increased expression of gene Z.",
                    direction="increase",
                    relationship_type="association",
                    evidence_ids=["ev_003"],
                )
            ],
            intervention="Treatment C",
            independent_variables=["Treatment C"],
            dependent_variables=["gene Z expression"],
        )
        conclusions = build_conclusions(experiment)
        assert conclusions[0].statement == "Treatment C is associated with increased gene Z expression."
        assert conclusions[0].relationship_type == "association"
        assert "increases" not in conclusions[0].statement

    def test_correlation_phrasing(self):
        experiment = _experiment(
            observations=[
                Observation(
                    statement="gene A expression correlated positively with gene B expression.",
                    direction="increase",
                    relationship_type="correlation",
                    evidence_ids=["ev_004"],
                )
            ],
            intervention="gene A expression",
            independent_variables=["gene A expression"],
            dependent_variables=["gene B expression"],
        )
        conclusions = build_conclusions(experiment)
        assert conclusions[0].statement == "gene A expression correlates with increased gene B expression."

    def test_no_change_conclusion(self):
        experiment = _experiment(
            observations=[
                Observation(
                    statement="There was no significant difference between the groups.",
                    direction="no_change",
                    significance="not_significant",
                    evidence_ids=["ev_005"],
                )
            ]
        )
        conclusions = build_conclusions(experiment)
        assert conclusions[0].statement == "Treatment A does not significantly alter gene X expression."

    def test_no_conclusion_without_iv_or_dv(self):
        experiment = _experiment(independent_variables=[], dependent_variables=[], intervention="")
        assert build_conclusions(experiment) == []

    def test_unspecified_direction_is_skipped(self):
        experiment = _experiment(
            observations=[Observation(statement="Expression was measured.", evidence_ids=["ev_001"])]
        )
        assert build_conclusions(experiment) == []

    def test_duplicate_statements_are_deduplicated(self):
        observation = Observation(
            statement="Twice.", direction="increase", evidence_ids=["ev_002"]
        )
        experiment = _experiment(observations=[observation, observation.model_copy(deep=True)])
        assert len(build_conclusions(experiment)) == 1
