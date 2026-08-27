"""Knobs for the Statement Draft layer (deterministic, no env vars)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DraftConfig:
    max_sets_per_figure: int = 3  # one per question type (first blueprint each), capped
    max_perturbations_per_set: int = 4  # false statements per set, fixed perturbation order

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")

    def as_dict(self) -> dict:
        return asdict(self)
