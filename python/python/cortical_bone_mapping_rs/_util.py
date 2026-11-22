"""Input conversion shared by the API."""

from __future__ import annotations

from numbers import Real
from typing import Tuple

import numpy as np



Triple = Tuple[float, float, float]
Quint = Tuple[float, float, float, float, float]


def _triple(v, name: str) -> Triple:
    if isinstance(v, Real):
        v = (float(v),) * 3
    v = tuple(float(x) for x in v)
    if len(v) != 3:
        raise ValueError(f"{name} must be a scalar or have 3 values")
    return v  # type: ignore[return-value]


def _as_u8(arr, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got shape {a.shape}")
    if a.dtype in (np.bool_, np.uint8, np.int8):
        return np.ascontiguousarray(a).view(np.uint8)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be boolean or numeric, got {a.dtype}")
    return (a != 0).view(np.uint8)


_IMAGE_DTYPES = (np.float32, np.float64, np.int16, np.uint16, np.uint8, np.int8, np.int32, np.uint32)


def _as_image(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"image must be a 3D array, got shape {a.shape}")
    if a.dtype == np.bool_:
        return a.view(np.uint8)
    if a.dtype.type not in _IMAGE_DTYPES:
        if a.dtype.kind not in "iuf":
            raise TypeError(f"image must be numeric, got {a.dtype}")
        return a.astype(np.float64)
    return a
