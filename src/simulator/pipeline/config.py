"""
Pipeline configuration: how many samples, where to write them, what the
sweep parameters are, where to place inducing points.

No trajectory / hyperprior / TLF settings — every sample is one
`(disorder_field, CSD)` pair and the disorder source is supplied at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InducingPointsConfig(BaseModel):
    """How to place inducing points on the dense disorder field."""

    layout: Literal["grid", "list"] = "grid"
    n_per_side: int = 8
    coords: Optional[list[tuple[int, int]]] = None

    @field_validator("coords")
    @classmethod
    def _check_coords(cls, v):
        if v is None:
            return v
        for c in v:
            if len(c) != 2:
                raise ValueError(f"each inducing-point coord must be (x, y); got {c}")
        return v


class SweepConfig(BaseModel):
    plunger_indices: tuple[int, int]
    plunger_range: tuple[float, float]
    n_points: int = 30
    base_voltage_mV: float = 0.0

    batched: bool = True
    sc_chunk_size: Optional[int] = None
    transport_workers: Optional[int] = None


class PipelineConfig(BaseModel):
    """Top-level pipeline config."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_samples: int = Field(gt=0)
    output_dir: Path
    seed: int = 0
    graph_config_path: Path

    n_workers: int = 1

    inducing_points: InducingPointsConfig
    sweep: SweepConfig

    @field_validator("output_dir", "graph_config_path", mode="before")
    @classmethod
    def _to_path(cls, v):
        return Path(v)


def load_config(path: str | Path) -> PipelineConfig:
    """Load a pipeline.yaml from disk."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
