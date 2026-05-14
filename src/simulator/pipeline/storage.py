"""
HDF5 storage for a single (disorder, CSD) sample, plus a JSONL run manifest.

One file per sample keeps the working set small (good for sharded parallel
reads) and lets the dispatcher resume by globbing for missing IDs.

HDF5 schema (one file per sample)
---------------------------------
/
    @sample_id        : int     attribute
    @completion_time  : str     attribute (ISO-8601 UTC)
    @graph_config     : str     attribute (basename of graph YAML used)
    @physical_width_nm: float   attribute (optional)
    @master_seed      : int     attribute (pipeline-level seed)

  disorder/
    field_mV          (nx, ny)        float32 — dense disorder field

  inducing/
    coords            (N, 2)          int32   — pixel (x, y) into the field
    values            (N,)            float32 — disorder at each inducing point

  csd/
    data              (n_pts, n_pts)  float32 — CSD image

  sweep/
    vp1               (n_pts,)        float32
    vp2               (n_pts,)        float32
    @plunger_indices  (2,)            int32 attribute
    @base_voltage_mV  float           attribute

  device/  (optional)
    gate_mask         (nx, ny)        uint8
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Optional, Set

import h5py
import numpy as np


def sample_path(output_dir: Path, sample_id: int) -> Path:
    return Path(output_dir) / f"sample_{sample_id:06d}.h5"


def write_sample_h5(
    path: Path,
    sample_id: int,
    master_seed: int,
    disorder_field_mV: np.ndarray,        # (nx, ny)
    inducing_coords: np.ndarray,          # (N, 2)
    inducing_values: np.ndarray,          # (N,)
    csd: np.ndarray,                      # (n_pts, n_pts)
    vp1: np.ndarray,                      # (n_pts,)
    vp2: np.ndarray,                      # (n_pts,)
    plunger_indices: tuple,
    base_voltage_mV: float,
    graph_config_path: Path,
    gate_mask: Optional[np.ndarray] = None,
    physical_width_nm: Optional[float] = None,
) -> None:
    """Atomic write: stage to .tmp then rename so partial files never appear."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(tmp, "w") as f:
        f.attrs["sample_id"] = int(sample_id)
        f.attrs["completion_time"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        f.attrs["graph_config"] = Path(graph_config_path).name
        f.attrs["master_seed"] = int(master_seed)
        if physical_width_nm is not None:
            f.attrs["physical_width_nm"] = float(physical_width_nm)

        d = f.create_group("disorder")
        d.create_dataset("field_mV", data=disorder_field_mV.astype(np.float32), compression="gzip")

        i = f.create_group("inducing")
        i.create_dataset("coords", data=np.asarray(inducing_coords, dtype=np.int32))
        i.create_dataset("values", data=np.asarray(inducing_values, dtype=np.float32))

        c = f.create_group("csd")
        c.create_dataset("data", data=csd.astype(np.float32), compression="gzip")

        s = f.create_group("sweep")
        s.create_dataset("vp1", data=vp1.astype(np.float32))
        s.create_dataset("vp2", data=vp2.astype(np.float32))
        s.attrs["plunger_indices"] = np.asarray(plunger_indices, dtype=np.int32)
        s.attrs["base_voltage_mV"] = float(base_voltage_mV)

        if gate_mask is not None:
            g = f.create_group("device")
            g.create_dataset("gate_mask", data=gate_mask.astype(np.uint8), compression="gzip")

    tmp.replace(path)


def read_sample_h5(path: Path) -> dict:
    """Load one sample HDF5 into a nested dict."""
    out: dict = {}
    with h5py.File(path, "r") as f:
        out["sample_id"] = int(f.attrs["sample_id"])
        out["completion_time"] = str(f.attrs["completion_time"])
        out["graph_config"] = str(f.attrs["graph_config"])
        out["master_seed"] = int(f.attrs["master_seed"])
        if "physical_width_nm" in f.attrs:
            out["physical_width_nm"] = float(f.attrs["physical_width_nm"])

        out["disorder"] = {"field_mV": f["disorder/field_mV"][...]}
        out["inducing"] = {
            "coords": f["inducing/coords"][...],
            "values": f["inducing/values"][...],
        }
        out["csd"] = {"data": f["csd/data"][...]}
        out["sweep"] = {
            "vp1": f["sweep/vp1"][...],
            "vp2": f["sweep/vp2"][...],
            "plunger_indices": tuple(f["sweep"].attrs["plunger_indices"].tolist()),
            "base_voltage_mV": float(f["sweep"].attrs["base_voltage_mV"]),
        }
        if "device" in f and "gate_mask" in f["device"]:
            out["device"] = {"gate_mask": f["device/gate_mask"][...].astype(bool)}
    return out


def manifest_path(output_dir: Path) -> Path:
    return Path(output_dir) / "manifest.jsonl"


def manifest_append(
    output_dir: Path,
    sample_id: int,
    file_path: Path,
    duration_sec: float,
) -> None:
    """Append-only JSONL — one line per completed sample."""
    record = {
        "sample_id": int(sample_id),
        "file": str(Path(file_path).name),
        "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "duration_sec": round(float(duration_sec), 2),
    }
    with open(manifest_path(output_dir), "a") as f:
        f.write(json.dumps(record) + "\n")


def manifest_completed_ids(output_dir: Path) -> Set[int]:
    mp = manifest_path(output_dir)
    if not mp.exists():
        return set()
    ids: Set[int] = set()
    with open(mp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(int(json.loads(line)["sample_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return ids
