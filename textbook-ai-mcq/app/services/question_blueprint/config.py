"""Knobs for the Question Blueprint layer (deterministic, no env vars)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BlueprintConfig:
    """Per-figure caps keep the artifact small; they never weaken the gates."""

    max_result_interpretation: int = 3  # per figure/panel, one per qualifying observation
    max_experimental_design: int = 3  # control purpose / treatment purpose / measurement
    max_simple_prediction: int = 1  # strictly one per experiment (most speculative type)
    max_data_statement: int = 4  # per figure/panel, one per extracted literal value

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")

    def as_dict(self) -> dict:
        return asdict(self)
