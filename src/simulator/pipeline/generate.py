"""
Generate one `(disorder, CSD)` sample and write it to disk.

Idempotent: returns immediately if the output file already exists.
Reproducible: per-sample RNG derived from `(master_seed, sample_id)`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from simulator.pipeline.config import PipelineConfig
from simulator.pipeline.disorder_sources.base import DisorderSource
from simulator.pipeline.inducing_points import make_inducing_coords, sample_field_at
from simulator.pipeline.storage import (
    manifest_append,
    sample_path,
    write_sample_h5,
)
from simulator.pipeline.sweep import (
    build_graph_with_disorder,
    load_graph_config,
    run_csd_sweep,
    run_csd_sweep_batched,
)


def generate_sample(
    sample_id: int,
    cfg: PipelineConfig,
    source: DisorderSource,
) -> Optional[Path]:
    """
    Generate one (disorder, CSD) sample.

    1. Build the simulation graph (one-shot).
    2. Draw one dense disorder field from `source`.
    3. Inject it into the ExternalDisorderNode.
    4. Run the CSD sweep over the two plunger gates.
    5. Sample the field at the inducing-point coords.
    6. Write everything to a single HDF5 file.
    """
    out_path = sample_path(cfg.output_dir, sample_id)
    if out_path.exists():
        return out_path

    t0 = time.time()
    rng = np.random.default_rng((cfg.seed, sample_id))

    graph_config = load_graph_config(cfg.graph_config_path)
    physical_width_nm = float(graph_config.device_config.width)

    graph = build_graph_with_disorder(
        disorder_field_mV=None,
        config_path=cfg.graph_config_path,
    )
    init_node = graph.components[0]
    nx, ny = init_node.potential_split.shape[1:3]

    if source.grid_shape != (nx, ny):
        raise ValueError(
            f"DisorderSource produces shape {source.grid_shape} but the device grid is "
            f"({nx}, {ny}). Resample your source to match."
        )

    disorder_field = source.sample(rng)
    init_node.set_disorder_field(disorder_field)

    inducing_coords = make_inducing_coords(cfg.inducing_points, (nx, ny))
    inducing_values = sample_field_at(disorder_field, inducing_coords)

    base_voltages = np.full(graph.num_gates, cfg.sweep.base_voltage_mV)

    if cfg.sweep.batched:
        vp1, vp2, csd = run_csd_sweep_batched(
            graph,
            plunger_indices=cfg.sweep.plunger_indices,
            plunger_range=cfg.sweep.plunger_range,
            n_points=cfg.sweep.n_points,
            base_voltages=base_voltages,
            sc_chunk_size=cfg.sweep.sc_chunk_size,
            transport_workers=cfg.sweep.transport_workers,
            verbose=True,
        )
    else:
        vp1, vp2, csd = run_csd_sweep(
            graph,
            plunger_indices=cfg.sweep.plunger_indices,
            plunger_range=cfg.sweep.plunger_range,
            n_points=cfg.sweep.n_points,
            base_voltages=base_voltages,
            verbose=True,
        )

    gate_mask = np.any(init_node.gate_split > 0, axis=0).astype(bool)

    write_sample_h5(
        path=out_path,
        sample_id=sample_id,
        master_seed=cfg.seed,
        disorder_field_mV=disorder_field,
        inducing_coords=inducing_coords,
        inducing_values=inducing_values,
        csd=csd,
        vp1=vp1,
        vp2=vp2,
        plunger_indices=cfg.sweep.plunger_indices,
        base_voltage_mV=cfg.sweep.base_voltage_mV,
        graph_config_path=Path(cfg.graph_config_path),
        gate_mask=gate_mask,
        physical_width_nm=physical_width_nm,
    )
    manifest_append(
        output_dir=cfg.output_dir,
        sample_id=sample_id,
        file_path=out_path,
        duration_sec=time.time() - t0,
    )
    return out_path
