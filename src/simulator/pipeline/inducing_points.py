"""
Inducing-point helpers.

Inducing points are a fixed set of `N` `(x, y)` pixel locations on the dense
disorder grid. For every generated sample, the dense field is evaluated at
these locations to produce a length-`N` vector (the DT input/target).

The layout is shared across all samples in a run — the DT learns to map CSDs
to disorder values at *these* points.
"""

from __future__ import annotations

import numpy as np

from simulator.pipeline.config import InducingPointsConfig


def make_inducing_coords(
    cfg: InducingPointsConfig,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """
    Build a fixed `(N, 2)` int array of pixel `(x, y)` coordinates.

    `layout="grid"`: lay an `n_per_side x n_per_side` grid uniformly across
    the field, avoiding the boundary pixels.

    `layout="list"`: use the explicit `coords` from the config.
    """
    nx, ny = grid_shape

    if cfg.layout == "list":
        if cfg.coords is None:
            raise ValueError("inducing_points.coords required when layout='list'")
        coords = np.asarray(cfg.coords, dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"inducing_points.coords must be (N, 2); got shape {coords.shape}")
        if (coords[:, 0] < 0).any() or (coords[:, 0] >= nx).any():
            raise ValueError(f"inducing-point x coords must be in [0, {nx})")
        if (coords[:, 1] < 0).any() or (coords[:, 1] >= ny).any():
            raise ValueError(f"inducing-point y coords must be in [0, {ny})")
        return coords

    n = cfg.n_per_side
    if n < 2:
        raise ValueError(f"n_per_side must be >= 2; got {n}")
    xs = np.linspace(0, nx - 1, n + 2, dtype=int)[1:-1]
    ys = np.linspace(0, ny - 1, n + 2, dtype=int)[1:-1]
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.int32)


def sample_field_at(field_mV: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Return `field_mV[coords[:, 0], coords[:, 1]]` as a length-N float32 vector."""
    if field_mV.ndim != 2:
        raise ValueError(f"field_mV must be 2D (nx, ny); got shape {field_mV.shape}")
    return field_mV[coords[:, 0], coords[:, 1]].astype(np.float32)
