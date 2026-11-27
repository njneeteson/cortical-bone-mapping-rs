"""The profile model and independent fits of profiles."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _core
from .options import PARAM_NAMES


def profile_model(x, params, rho_c: float) -> np.ndarray:
    """Evaluate the profile model at ``x`` for parameters ``(m, t, sigma,
    rho_s, rho_b)`` (``params`` may be (5,) or (N, 5); result broadcasts)."""
    from math import erf, sqrt

    x = np.asarray(x, float)
    p = np.asarray(params, float)
    m, t, s, rs, rb = (p[..., k, None] if p.ndim > 1 else p[k] for k in range(5))
    e = np.vectorize(erf, otypes=[float])
    return (
        rs
        + 0.5 * (rho_c - rs) * (1 + e((x - m + t / 2) / (s * sqrt(2))))
        + 0.5 * (rb - rho_c) * (1 + e((x - m - t / 2) / (s * sqrt(2))))
    )


def fit_profiles(
    profiles,
    x,
    *,
    init=(0.0, 1.0, 0.1, 0.0, 0.0),
    rho_c: float = 1000.0,
    residual_boost: float = 10.0,
    lower=(-3.0, 0.0607, 0.01, -200.0, -200.0),
    upper=(6.0, 12.0, 2.0, 400.0, 400.0),
    fixed: Sequence[str] = (),
    max_iterations: int = 100,
    ftol: float = 1e-8,
    gtol: float = 1e-8,
    xtol: float = 1e-3,
) -> dict:
    """Fit the profile model independently to each row of ``profiles``.

    This is the local method without the surface pipeline, e.g. for
    profiles sampled elsewhere. ``init`` is (5,) or (N, 5); parameters named
    in ``fixed`` (from :data:`PARAM_NAMES`) keep their initial value.

    Returns a dict with ``params`` (N, 5), ``cost`` (N,), ``iterations`` (N,)
    and ``converged`` (N,).
    """
    prof = np.ascontiguousarray(np.atleast_2d(profiles), dtype=np.float32)
    x = [float(v) for v in np.asarray(x, float).ravel()]
    ini = np.ascontiguousarray(np.atleast_2d(np.asarray(init, float)))
    bad = set(fixed) - set(PARAM_NAMES)
    if bad:
        raise ValueError(f"unknown parameter(s) {sorted(bad)}; use {PARAM_NAMES}")
    free = [name not in fixed for name in PARAM_NAMES]
    p, c, it, ok = _core.fit_profiles(
        prof, x, ini, float(rho_c), float(residual_boost),
        tuple(map(float, lower)), tuple(map(float, upper)), free,
        int(max_iterations), float(ftol), float(gtol), float(xtol),
    )
    return {"params": p, "cost": c, "iterations": it, "converged": ok}
