"""
Visualization helpers for single-sample HDF5 files.

Accepts either a dict from `storage.read_sample_h5` or a path to an `.h5`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from simulator.pipeline.storage import read_sample_h5


def _maybe_load(data_or_path: Union[dict, str, Path]) -> dict:
    if isinstance(data_or_path, dict):
        return data_or_path
    return read_sample_h5(Path(data_or_path))


def _device_extent(data: dict) -> Optional[list]:
    width = data.get("physical_width_nm")
    if width is not None:
        return [0.0, float(width), 0.0, float(width)]
    return None


def plot_disorder_field(
    data_or_path: Union[dict, str, Path],
    ax: Optional[plt.Axes] = None,
    show_inducing: bool = True,
) -> plt.Axes:
    """Plot the dense disorder field, optionally overlaying inducing points."""
    data = _maybe_load(data_or_path)
    field = data["disorder"]["field_mV"]
    extent = _device_extent(data)

    if ax is None:
        _, ax = plt.subplots()
    vmax = np.abs(field).max()
    im = ax.imshow(
        field.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
    )
    plt.colorbar(im, ax=ax, label="disorder (mV)")

    if show_inducing:
        coords = data["inducing"]["coords"]
        if extent is not None:
            nx = field.shape[0]
            px = coords[:, 0] * extent[1] / nx
            py = coords[:, 1] * extent[3] / field.shape[1]
        else:
            px, py = coords[:, 0], coords[:, 1]
        ax.scatter(px, py, marker="x", s=30, color="black", linewidths=1.0)

    ax.set_title(f"disorder field (sample {data['sample_id']})")
    return ax


def plot_csd(
    data_or_path: Union[dict, str, Path],
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot the CSD over the (vp1, vp2) sweep.

    CSD values are the minimax barrier height (mV) along the source->drain
    transport path. Sign convention: negative = conducting (path below Fermi
    level), positive = blocked. Plotted with a diverging colormap centered on
    0 so the conducting/blocked boundary is the visual zero of the colorbar.
    """
    data = _maybe_load(data_or_path)
    csd = data["csd"]["data"]
    vp1 = data["sweep"]["vp1"]
    vp2 = data["sweep"]["vp2"]

    if ax is None:
        _, ax = plt.subplots()
    finite = csd[np.isfinite(csd)]
    v = float(np.abs(finite).max()) if finite.size else 1.0
    im = ax.imshow(
        csd.T,
        origin="lower",
        cmap="RdBu_r",
        vmin=-v, vmax=v,
        extent=[vp1.min(), vp1.max(), vp2.min(), vp2.max()],
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, label="barrier height (mV)  blue = conducting")
    ax.set_xlabel(f"VP{data['sweep']['plunger_indices'][0]} (mV)")
    ax.set_ylabel(f"VP{data['sweep']['plunger_indices'][1]} (mV)")
    ax.set_title(f"CSD (sample {data['sample_id']})")
    return ax


def plot_sample_overview(
    data_or_path: Union[dict, str, Path],
    figsize: tuple[float, float] = (10, 4),
):
    """Side-by-side disorder field + CSD for one sample."""
    data = _maybe_load(data_or_path)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_disorder_field(data, ax=axes[0])
    plot_csd(data, ax=axes[1])
    fig.tight_layout()
    return fig
