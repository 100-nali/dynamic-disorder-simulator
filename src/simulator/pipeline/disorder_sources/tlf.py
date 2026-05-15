"""
`TLFDisorderSource` — `DisorderSource` backed by the qxcl Ge/SiGe charge-trap
prior. Each call to `sample(rng)` returns a single dense `(nx, ny)` disorder
field in mV.

Two construction modes:

1. **Fixed config** (`sampler_config=...`): every draw uses the same
   `TrapSamplerConfig` — same trap-count mean, same depths, same density.
   Only the geometry/occupancy realisation varies between samples.

2. **Hyperprior** (`hyperprior=...`): every draw first re-samples
   `TrapSamplerConfig` from the hyperprior (n_traps_mean, donor_fraction,
   etc.), then draws a disorder field from that config. Gives epistemic
   diversity in the training set.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from simulator.disorder_prior.hyperprior import HyperpriorConfig, draw_sampler_config
from simulator.disorder_prior.trap_sampler import ChargeTrapSampler, TrapSamplerConfig
from simulator.pipeline.disorder_sources.base import DisorderSource


class TLFDisorderSource(DisorderSource):
    """
    DisorderSource produced by the Ge/SiGe charge-trap prior.

    Parameters
    ----------
    grid_shape         : `(nx, ny)` of the field this source produces. Must
                         match the device-grid shape (the runner enforces
                         this).
    physical_width_nm  : physical width of the device in nm. Pinned by gate
                         layout — should match `device_config.width`.
    sampler_config     : a fully-specified `TrapSamplerConfig`. Use this for
                         "every sample looks the same statistically". The
                         `grid_shape` / `physical_width_nm` fields on the
                         passed config are overwritten with the ones above
                         so they can't disagree.
    hyperprior         : a `HyperpriorConfig` to redraw the sampler config
                         per sample. Mutually exclusive with `sampler_config`.
    carrier_depth_nm   : forwarded to `draw_sampler_config`; only used in
                         hyperprior mode. Default 40.0 (well-known from the
                         heterostructure design).
    epsilon_r          : material permittivity. Default 14.85 (SiGe).
    gate_bias_weights  : optional `(nx, ny)` float array. If non-None, the
                         dipole-trap geometry sampling is biased toward
                         pixels with `weights > 0` (the gate regions). The
                         pipeline's runner can populate this from the
                         device's gate mask.
    """

    def __init__(
        self,
        *,
        grid_shape: tuple[int, int],
        physical_width_nm: float,
        sampler_config: Optional[TrapSamplerConfig] = None,
        hyperprior: Optional[HyperpriorConfig] = None,
        carrier_depth_nm: float = 40.0,
        epsilon_r: float = 14.85,
        gate_bias_weights: Optional[np.ndarray] = None,
    ) -> None:
        if (sampler_config is None) == (hyperprior is None):
            raise ValueError(
                "TLFDisorderSource requires exactly one of `sampler_config` or "
                "`hyperprior`."
            )

        self._grid_shape = tuple(grid_shape)
        self._physical_width_nm = float(physical_width_nm)
        self._carrier_depth_nm = float(carrier_depth_nm)
        self._epsilon_r = float(epsilon_r)
        self._gate_bias_weights = (
            np.asarray(gate_bias_weights, dtype=np.float64)
            if gate_bias_weights is not None
            else None
        )

        if sampler_config is not None:
            self._base_cfg = replace(
                sampler_config,
                grid_shape=self._grid_shape,
                physical_width_nm=self._physical_width_nm,
                carrier_depth_nm=self._carrier_depth_nm,
                epsilon_r=self._epsilon_r,
            )
            self._hyperprior = None
        else:
            self._base_cfg = None
            self._hyperprior = hyperprior

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self._grid_shape

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        if self._hyperprior is not None:
            cfg, _drawn = draw_sampler_config(
                self._hyperprior,
                grid_shape=self._grid_shape,
                physical_width_nm=self._physical_width_nm,
                carrier_depth_nm=self._carrier_depth_nm,
                epsilon_r=self._epsilon_r,
                rng=rng,
            )
        else:
            # Re-seed the fixed config from the run-level rng so successive
            # calls don't return the identical field.
            cfg = replace(
                self._base_cfg,  # type: ignore[arg-type]
                seed=int(rng.integers(0, 2**31 - 1)),
            )

        sampler = ChargeTrapSampler(cfg)

        if self._gate_bias_weights is not None:
            geometry = sampler.sample_geometry(gate_bias_weights=self._gate_bias_weights)
            occupancies = sampler.sample_occupancies(geometry)
            return sampler.compute_field(geometry, occupancies)
        return sampler.sample().field_mV
