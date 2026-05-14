"""
Abstract `DisorderSource` interface.

A DisorderSource produces a dense disorder potential as a `(nx, ny)` float
array in millivolts. Implementations decide where the values come from: a
Gaussian random field prior, a learned generator, a fixed dataset, etc.

The pipeline calls `sample(rng)` once per generated sample. The seeded
`np.random.Generator` lets implementations be reproducible without each one
having to wrangle its own seeds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DisorderSource(ABC):
    """Produces a dense `(nx, ny)` disorder field in millivolts."""

    @property
    @abstractmethod
    def grid_shape(self) -> tuple[int, int]:
        """`(nx, ny)` of the fields this source produces."""

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Return one dense disorder field `(nx, ny)` in mV."""
