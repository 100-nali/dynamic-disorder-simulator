"""
Pydantic v2 type aliases for numpy arrays.

Only the dtypes actually consumed by the simulator package are defined here.
"""

from __future__ import annotations

from typing import Annotated, Any

import numpy as np
from pydantic import BeforeValidator


def _as_array(dtype):
    def _coerce(v: Any) -> np.ndarray:
        return np.asarray(v, dtype=dtype)
    return _coerce


NDArrayFp64 = Annotated[np.ndarray, BeforeValidator(_as_array(np.float64))]
NDArrayFp32 = Annotated[np.ndarray, BeforeValidator(_as_array(np.float32))]
NDArrayInt64 = Annotated[np.ndarray, BeforeValidator(_as_array(np.int64))]
NDArrayBool = Annotated[np.ndarray, BeforeValidator(_as_array(bool))]
