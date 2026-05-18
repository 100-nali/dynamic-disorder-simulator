"""
Reusable graph-construction and CSD-sweep helpers.

Extracted from demo_scripts/generate_csd.py so both the simple driver and the
orchestrated pipeline (pipeline.py) can share one implementation.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

from simulator.computational_graph.abstract_computational_graph import ComputationalGraph
from simulator.computational_graph.graph_config import GraphConfig
from simulator.injection.external_disorder_node import ExternalDisorderNode


# Per-process state for the transport ProcessPoolExecutor.  The transport
# node has per-call mutated state (self.scale, self.minimax) which is unsafe
# under threading; processes get one node each and avoid the race entirely.
_WORKER_TRANSPORT_NODE = None


def _init_transport_worker(transport_node):
    global _WORKER_TRANSPORT_NODE
    _WORKER_TRANSPORT_NODE = transport_node


def _csd_signal_from_transport(result) -> float:
    """
    Extract the per-frame CSD scalar from SemiClassicalTransportNode output.

    The node returns `[path_map, minimax]`:
      - path_map  : (nx, ny) binary mask of the lowest-energy source->drain path
      - minimax   : scalar (mV), max potential energy along that path

    The channel conducts when `minimax < 0` (the entire path lies below the
    Fermi level); it's blocked when `minimax > 0`. Storing `minimax` directly
    keeps the full signed signal — apply any thermal smoothing post-hoc, e.g.
        I_proxy = 1.0 / np.cosh(minimax / kT_meV) ** 2     # cosh^2 peak
        I_proxy = 1.0 / (1.0 + np.exp(minimax / kT_meV))   # Fermi step

    (Earlier versions of this code returned `mean(path_map)` — the path-length
    fraction. That is geometric, not transport, and was the cause of "flat
    CSDs". This is the fix.)
    """
    return float(result[1])


def _run_transport_one(sc_pot: np.ndarray) -> float:
    """Per-frame transport call. Module-level so it pickles cleanly."""
    try:
        result = _WORKER_TRANSPORT_NODE.compute(potential=sc_pot)
        return _csd_signal_from_transport(result)
    except Exception:
        return float("nan")


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "graph.yaml"


def load_graph_config(config_path: Path = DEFAULT_CONFIG_PATH) -> GraphConfig:
    """
    Load a YAML graph config and rewrite gate_design_dir to an absolute
    repo-relative path so the config works from any CWD.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    gate_dir = raw["device_config"]["gate_config"].get("gate_design_dir")
    if gate_dir is not None and not Path(gate_dir).is_absolute():
        raw["device_config"]["gate_config"]["gate_design_dir"] = str(REPO_ROOT / gate_dir)

    return GraphConfig(**raw)


def build_graph_with_disorder(
    disorder_field_mV: Optional[np.ndarray],
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> ComputationalGraph:
    """
    Build the full simulation graph with the disorder field injected
    into component 0 (InitialElectroStaticPotential).
    """
    graph_config  = load_graph_config(config_path)
    node_config_0 = graph_config.node_configs[0]

    patched_node = ExternalDisorderNode(
        node_config_0, disorder_field_mV=disorder_field_mV
    )

    graph = ComputationalGraph(graph_config)
    graph.components[0] = patched_node

    return graph


def run_csd_sweep(
    graph:           ComputationalGraph,
    plunger_indices: Tuple[int, int],
    plunger_range:   Tuple[float, float],
    n_points:        int = 30,
    base_voltages:   Optional[np.ndarray] = None,
    verbose:         bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep two plunger gates and record the transport observable. Sequential.

    Kept for backward compatibility with demo_scripts/generate_csd.py; new
    callers should prefer run_csd_sweep_batched, which exercises the
    GPU/batched path on IterativeSelfConsistentPotentialNode.

    Returns (vp1_vals, vp2_vals, csd) where csd has shape (n_points, n_points).
    """
    n_gates = graph.num_gates
    if base_voltages is None:
        base_voltages = np.zeros(n_gates)

    vp1_idx, vp2_idx = plunger_indices
    v_min,   v_max   = plunger_range

    vp1_vals = np.linspace(v_min, v_max, n_points)
    vp2_vals = np.linspace(v_min, v_max, n_points)
    csd      = np.zeros((n_points, n_points))

    for i, vp1 in enumerate(vp1_vals):
        for j, vp2 in enumerate(vp2_vals):
            voltages           = base_voltages.copy()
            voltages[vp1_idx]  = vp1
            voltages[vp2_idx]  = vp2

            try:
                result = graph.run_component_outputs({"gate_voltages": voltages})
                # SemiClassicalTransportNode's second output (named "current" in
                # the YAML, but it's actually the minimax barrier height in mV).
                # See _csd_signal_from_transport for the convention.
                if "current" in result and np.ndim(result["current"]) == 0:
                    csd[i, j] = float(result["current"])
                elif "minimax" in result and np.ndim(result["minimax"]) == 0:
                    csd[i, j] = float(result["minimax"])
                else:
                    csd[i, j] = np.nan
            except Exception:
                csd[i, j] = np.nan

        if verbose and (i + 1) % 5 == 0:
            print(f"  Row {i+1}/{n_points} done")

    return vp1_vals, vp2_vals, csd


def _find_node(graph: ComputationalGraph, class_name: str):
    """Return the first component whose class matches class_name."""
    for c in graph.components:
        if type(c).__name__ == class_name:
            return c
    raise LookupError(
        f"No {class_name} found in graph.components "
        f"(have: {[type(c).__name__ for c in graph.components]})"
    )


def _build_voltage_batch(
    base_voltages:   np.ndarray,
    plunger_indices: Tuple[int, int],
    vp1_vals:        np.ndarray,
    vp2_vals:        np.ndarray,
) -> np.ndarray:
    """
    Construct (n_pts1 * n_pts2, n_gates). Row k corresponds to the
    (i, j)=(k // n_pts2, k % n_pts2) point on the (VP1, VP2) grid.
    """
    n1, n2  = len(vp1_vals), len(vp2_vals)
    n_gates = len(base_voltages)

    voltage_batch = np.tile(base_voltages, (n1 * n2, 1)).astype(np.float64)
    voltage_batch[:, plunger_indices[0]] = np.repeat(vp1_vals, n2)
    voltage_batch[:, plunger_indices[1]] = np.tile(vp2_vals, n1)
    return voltage_batch


def _initial_potential_batch(init_node, voltage_batch: np.ndarray) -> np.ndarray:
    """
    Vectorised initial-potential computation across a voltage batch.

    Replicates ExternalDisorderNode.compute() but vectorised over the leading
    batch dim. Returns (B, nx, ny). One disorder field broadcast across B.
    """
    # gate-only contribution: einsum over gates
    pot_gate = np.einsum(
        "gxy,bg->bxy",
        init_node.potential_split,
        voltage_batch / 10000.0,
    )  # (B, nx, ny)

    pot_gate = pot_gate * init_node.material_config.scale_factor

    if init_node._external_disorder_field is not None:
        dis = init_node._external_disorder_field  # (nx, ny)  broadcasts
    else:
        dis = 0.0

    return pot_gate + dis + init_node.material_config.surface_potential


def run_csd_sweep_batched(
    graph:             ComputationalGraph,
    plunger_indices:   Tuple[int, int],
    plunger_range:     Tuple[float, float],
    n_points:          int = 30,
    base_voltages:     Optional[np.ndarray] = None,
    verbose:           bool = False,
    sc_chunk_size:     Optional[int] = None,
    transport_workers: Optional[int] = None,
    sc_initial_guess:  Optional[np.ndarray] = None,
    return_sc_pot:     bool = False,
):
    """
    Sweep two plunger gates using the batched/GPU path on the SC node.

    Pipeline per call:
      1. Vectorised initial potential over (n_points^2, nx, ny).
      2. Batched SC iteration  split into chunks of sc_chunk_size along the
         batch dim to bound peak FFT memory. Each chunk is still batched
         internally so the SC GPU/torch wins are preserved.
         sc_chunk_size=None  one call over the full batch (fastest, hungriest).
      3. Sequential transport-node call per frame to extract the CSD scalar
         (the downstream nodes were not updated for batching).

    Returns (vp1_vals, vp2_vals, csd) where csd has shape (n_points, n_points).
    """
    n_gates = graph.num_gates
    if base_voltages is None:
        base_voltages = np.zeros(n_gates)

    v_min, v_max = plunger_range
    vp1_vals = np.linspace(v_min, v_max, n_points)
    vp2_vals = np.linspace(v_min, v_max, n_points)

    voltage_batch = _build_voltage_batch(
        base_voltages, plunger_indices, vp1_vals, vp2_vals
    )  # (B, n_gates) where B = n_points^2

    init_node      = graph.components[0]  # ExternalDisorderNode (or InitialElectroStaticPotential)
    sc_node        = _find_node(graph, "IterativeSelfConsistentPotentialNode")
    transport_node = _find_node(graph, "SemiClassicalTransportNode")

    B = voltage_batch.shape[0]
    chunk = sc_chunk_size if sc_chunk_size is not None else B
    chunk = max(1, min(chunk, B))

    if verbose:
        print(f"  building initial potential batch ({B} voltages)...")
    init_pot_batch = _initial_potential_batch(init_node, voltage_batch)  # (B, nx, ny)

    if verbose:
        n_chunks = (B + chunk - 1) // chunk
        print(f"  running batched SC iteration: {n_chunks} chunk(s) of size {chunk} "
              f"over shape {init_pot_batch.shape}...")

    # If torch is on CUDA, defrag its memory pool between chunks to
    # avoid OOM from accumulated stale allocations (the SC node reassigns
    # self.k each compute() call, fragmenting the pool).
    try:
        import torch
        _torch_cuda_avail = torch.cuda.is_available()
    except ImportError:
        _torch_cuda_avail = False

    sc_pot_parts = []
    for start in range(0, B, chunk):
        stop    = min(start + chunk, B)
        pot_in  = init_pot_batch[start:stop]              # (chunk, nx, ny)
        assert pot_in.ndim == 3, (
            f"expected (batch, nx, ny) tensor but got shape {pot_in.shape}"
        )
        # Warm-start: pass the previous trajectory step's converged SC slice.
        guess = sc_initial_guess[start:stop] if sc_initial_guess is not None else None
        if verbose:
            warm = " (warm-start)" if guess is not None else ""
            print(f"    calling SC.compute with input_potential.shape="
                  f"{tuple(pot_in.shape)}  (batch dim = {pot_in.shape[0]}){warm}")
        pot_out = sc_node.compute(input_potential=pot_in, initial_guess=guess)[0]
        if verbose:
            print(f"    SC returned shape {tuple(pot_out.shape)}  chunk [{start}:{stop}] done")
        sc_pot_parts.append(np.asarray(pot_out))

        # Free stale GPU memory between chunks  cheap (a few ms) and
        # often the difference between fitting and OOM at chunk_size > 50.
        if _torch_cuda_avail:
            torch.cuda.empty_cache()
    sc_pot_batch = np.concatenate(sc_pot_parts, axis=0)      # (B, nx, ny)

    workers = transport_workers if transport_workers is not None else 1
    workers = max(1, min(workers, B))

    if verbose:
        mode = "serial" if workers == 1 else f"{workers}-process pool"
        print(f"  running transport per-frame ({B} frames, {mode})...")
    t_start = time.time()

    if workers == 1:
        csd_flat = np.empty(B, dtype=np.float64)
        for k in range(B):
            try:
                traj = transport_node.compute(potential=sc_pot_batch[k])[0]
                csd_flat[k] = float(np.mean(traj))
            except Exception:
                csd_flat[k] = np.nan
            if verbose and (k + 1) % 50 == 0:
                rate = (k + 1) / (time.time() - t_start)
                print(f"    transport {k+1}/{B} done ({rate:.1f} frames/s)")
    else:
        sc_pots_list = [sc_pot_batch[k] for k in range(B)]
        chunksize    = max(1, B // (workers * 4))
        with ProcessPoolExecutor(
            max_workers = workers,
            initializer = _init_transport_worker,
            initargs    = (transport_node,),
        ) as ex:
            csd_flat = np.fromiter(
                ex.map(_run_transport_one, sc_pots_list, chunksize=chunksize),
                dtype = np.float64,
                count = B,
            )

    if verbose:
        elapsed = time.time() - t_start
        print(f"  transport done in {elapsed:.1f}s ({B/elapsed:.1f} frames/s)")

    csd = csd_flat.reshape(n_points, n_points)
    if return_sc_pot:
        return vp1_vals, vp2_vals, csd, sc_pot_batch
    return vp1_vals, vp2_vals, csd
