"""
Torch `Dataset` for `(inducing_vector, CSD)` training pairs.

On first access of a sample id, runs the simulator to generate and cache it
(`sample_{NNNNNN}.h5` under `cfg.output_dir`). On subsequent accesses (later
epochs, other workers in this process), reads from disk.

Use this as the data source for distribution-transformer training: it
amortises generation cost across all epochs while letting epoch 0 fill the
dataset on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from simulator.pipeline.config import PipelineConfig
from simulator.pipeline.disorder_sources.base import DisorderSource
from simulator.pipeline.generate import generate_sample
from simulator.pipeline.storage import read_sample_h5, sample_path


class SimulatorDataset(Dataset):
    """
    `(inducing_vector, CSD)` pairs, generated and cached on first access.

    Item shape:
        inducing_values : torch.float32 tensor of shape (N,)
        csd             : torch.float32 tensor of shape (n_pts, n_pts)

    Set `return_extras=True` to also receive the dense disorder field, the
    inducing-point coords, and the sweep voltage axes.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        source_factory: Callable[[], DisorderSource],
        return_extras: bool = False,
    ) -> None:
        self.cfg = cfg
        self._source_factory = source_factory
        self._source: DisorderSource | None = None
        self.return_extras = return_extras

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def _ensure_source(self) -> DisorderSource:
        if self._source is None:
            self._source = self._source_factory()
        return self._source

    def __len__(self) -> int:
        return self.cfg.n_samples

    def __getitem__(self, idx: int):
        if not (0 <= idx < self.cfg.n_samples):
            raise IndexError(idx)

        path = sample_path(self.cfg.output_dir, idx)
        if not path.exists():
            generate_sample(idx, self.cfg, self._ensure_source())

        sample = read_sample_h5(path)
        inducing_values = torch.as_tensor(sample["inducing"]["values"], dtype=torch.float32)
        csd = torch.as_tensor(sample["csd"]["data"], dtype=torch.float32)

        if not self.return_extras:
            return inducing_values, csd

        return {
            "inducing_values": inducing_values,
            "csd": csd,
            "inducing_coords": torch.as_tensor(sample["inducing"]["coords"], dtype=torch.int32),
            "disorder_field_mV": torch.as_tensor(
                sample["disorder"]["field_mV"], dtype=torch.float32
            ),
            "vp1": torch.as_tensor(sample["sweep"]["vp1"], dtype=torch.float32),
            "vp2": torch.as_tensor(sample["sweep"]["vp2"], dtype=torch.float32),
            "sample_id": idx,
        }

    def prefill(self, indices: list[int] | None = None) -> None:
        """
        Eagerly generate all (or specified) samples up-front. Useful if you
        want all of epoch 0 ready before training starts, or if you want to
        delegate generation to a separate process.
        """
        ids = range(self.cfg.n_samples) if indices is None else indices
        source = self._ensure_source()
        for idx in ids:
            path = sample_path(self.cfg.output_dir, idx)
            if not path.exists():
                generate_sample(idx, self.cfg, source)
