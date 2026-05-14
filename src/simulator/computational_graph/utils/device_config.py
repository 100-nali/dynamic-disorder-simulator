"""
Pydantic configs for the simulated device, material, and gate layout.
Migrated to pydantic v2.
"""

from __future__ import annotations

from os.path import exists, isdir, join
from typing import Any, Optional

import numpy as np
import skimage  # type: ignore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from simulator.custom_pydantic_numpy.dtype import NDArrayFp64


class Dot(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    center_of_mass: np.ndarray = np.array([0, 0])
    bounding_well: np.ndarray = np.array([0, 0, 0, 0], dtype=int)
    mask: np.ndarray = np.array([[0, 0], [0, 0]], dtype="bool")
    charge_occupation: int = Field(default=0, ge=0)
    is_charge_sensor: bool = False
    gate_group: list = []

    def to_safe_dict(self) -> dict:
        base_dict = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in self.model_dump().items()
        }
        base_dict["charge_occupation"] = int(base_dict["charge_occupation"])
        return base_dict


class MaterialConfig(BaseModel):
    """Config for the semiconducting material."""

    model_config = ConfigDict(from_attributes=True)

    donors: bool
    mean_disorder: float
    relative_permittivity: float = Field(gt=0.0, lt=100.0)
    carrier_charge: int
    effective_mass: float = Field(gt=0.0, lt=2.0)
    degeneracy: int
    surface_potential: float
    scale_factor: int = 1
    atomic_units: bool = False
    exp_charge_fudge_factor: float = 1.0
    temperature: float = Field(gt=0.0, default=0.0)
    resistivity: float = Field(gt=0.0, default=1e-3)
    electron_confiment_range: float = Field(gt=0.0, default=10e-9)

    @field_validator("carrier_charge")
    @classmethod
    def minus_or_plus_one(cls, v: int) -> int:
        if v not in (-1, 1):
            raise ValueError(
                "Input carrier charge must be positive (holes) or negative (electrons)."
            )
        return v

    @field_validator("mean_disorder")
    @classmethod
    def mean_disorder_bounds(cls, v: float) -> float:
        if not (v >= 0.0 and np.abs(v) < 2000.0):
            raise ValueError(f"Mean disorder value: {v} not within bounds of 0<=v<|2000|")
        return v

    @field_validator("degeneracy")
    @classmethod
    def degeneracy_valid_args(cls, v: int) -> int:
        if v not in (2, 4):
            raise ValueError(f"Degeneracy: {v} invalid device degeneracy, should be 2 or 4")
        return v

    @field_validator("surface_potential")
    @classmethod
    def surface_potential_bounds(cls, v: float) -> float:
        if not (v < 0.0 and np.abs(v) < 2000.0):
            raise ValueError(f"Surface potential: {v} not within bounds of v<0 and v<|2000|")
        return v

    @field_validator("scale_factor")
    @classmethod
    def scale_factor_bounds(cls, v: int) -> int:
        if not 0 < v <= 100:
            raise ValueError(f"Scale factor: {v} not within bounds of 0<=v<=100")
        return v


class GateTypeIndices(BaseModel):
    """Indices grouping gates by role."""

    separator_gates: list[int]
    sensor_barrier_gates: list[int]
    sensor_plunger_gates: list[int]
    qubit_barrier_gates: list[int]
    qubit_plunger_gates: list[int]


class GateConfig(BaseModel):
    """Voltage gate configuration; reads the gate layout image lazily."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gate_design_dir: Optional[str] = None
    gate_design_file: Optional[str] = None

    gate_img_array: Optional[NDArrayFp64] = None

    gate_split: Optional[NDArrayFp64] = None
    gate_catalogue_path: Optional[str] = None
    gate_split_path: Optional[str] = None

    charge_sensor_gates: list[int] = []
    plunger_gates: list[int] = []
    barrier_gates: list[int] = []
    splitter_gates: list[int] = []

    @model_validator(mode="after")
    def load_gate_img_if_needed(self) -> "GateConfig":
        """If no image array was passed in, read it from gate_design_dir/file."""
        if self.gate_img_array is None:
            if self.gate_design_dir is None or self.gate_design_file is None:
                raise ValueError(
                    "Must specify either gate_design_dir + gate_design_file, "
                    "or gate_img_array."
                )
            self.gate_img_array = np.array(
                skimage.io.imread(join(self.gate_design_dir, self.gate_design_file))
            )
        return self

    @field_validator("gate_design_dir")
    @classmethod
    def check_gate_design_dir(cls, v: Optional[str]) -> Optional[str]:
        if v and not isdir(v):
            raise ValueError(f"Directory: {v} is not found.")
        return v

    @field_validator("gate_design_file")
    @classmethod
    def check_gate_design_file(cls, v: Optional[str], info) -> Optional[str]:
        if not v:
            return v
        gate_dir = info.data.get("gate_design_dir")
        if not gate_dir:
            raise ValueError("Gate design directory not provided")
        if not exists(join(gate_dir, v)):
            raise FileNotFoundError(f"Gate design file {v} not found.")
        return v

    @property
    def number_of_gates(self) -> Optional[int]:
        return None if self.gate_split is None else self.gate_split.shape[0]


class DeviceConfig(BaseModel):
    """Device-level configuration."""

    model_config = ConfigDict(from_attributes=True)

    width: int = Field(gt=0, le=5000)
    carrier_depth: float = Field(gt=0.0, le=500.0)
    donor_height: float
    donor_density: float = Field(gt=0.0, le=1e18)
    material: str
    source: list[float]
    drain: list[float]
    source_to_drain_bias: float = Field(gt=0.0, le=1000, default=10)
    gate_config: GateConfig
    material_config: MaterialConfig
    fermi_level: float = 0.0

    @field_validator("material")
    @classmethod
    def valid_material(cls, v: str) -> str:
        if v.lower() not in ("sige", "gaas"):
            raise ValueError(f"Input material {v} unrecognized, valid inputs are SiGe or GaAs")
        return v

    @field_validator("drain", "source")
    @classmethod
    def drain_and_source_bounds(cls, v: list[float]) -> list[float]:
        for coord in v:
            if not (coord >= 0.0 or coord <= 1.0):
                raise ValueError(f"Coordinate {v} invalid.")
        return v

    @model_validator(mode="after")
    def drain_not_eq_source(self) -> "DeviceConfig":
        if np.all(np.asarray(self.drain) == np.asarray(self.source)):
            raise ValueError(
                f"Drain {self.drain} and source {self.source} should not be at same location."
            )
        return self

    @model_validator(mode="after")
    def donor_height_le_carrier_depth(self) -> "DeviceConfig":
        donor_height, cd = self.donor_height, self.carrier_depth

        if donor_height <= 0.0 or cd - 5 <= donor_height <= cd + 5:
            raise ValueError(
                f"Donor height {donor_height} must be greater than 0 (top of device) and be at "
                f"least 5nm away from carrier depth {cd}."
            )
        return self

    @property
    def pixel_associated_length(self):
        return self.width / self.gate_config.gate_img_array.shape[0]

    @property
    def pixel_lengths(self):
        return (
            np.array([self.pixel_associated_length, self.pixel_associated_length])
        ) * 1e-9
