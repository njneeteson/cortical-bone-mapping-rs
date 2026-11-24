"""Meshes: rasterisation, sections, curvature, and VTK output."""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import _core
from ._util import _triple


def voxelize(vertices, faces, shape, spacing=1.0, origin=0.0) -> np.ndarray:
    """Rasterise a closed triangle mesh (physical coordinates) into a boolean
    mask of ``shape`` (even-odd rule along axis 0)."""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3:
        raise ValueError("shape must have 3 entries")
    return _core.voxelize(v, f, shape, _triple(spacing, "spacing"), _triple(origin, "origin"))


def mesh_curvature(vertices, faces, measure: str = "mean", scale: Optional[float] = 0.25,
                   neighbours: int = 64, distance: str = "euclidean") -> np.ndarray:
    """The curvature at every vertex of a triangle mesh (physical
    coordinates), as the curvature penalty measures it (1/length): ``"mean"``
    (``|k1 + k2| = 2|H|``) or ``"total"`` (``sqrt(k1^2 + k2^2)``), with a
    Gaussian surface Laplacian at ``scale`` over ``neighbours`` points
    (``distance``: ``"euclidean"`` or ``"surface"``), or the cotangent
    Laplacian (``scale=None``). For choosing ``curvature_kappa`` from
    reference surfaces."""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    return _core.mesh_curvature(v, f, measure, None if scale is None else float(scale), int(neighbours), distance)


def write_vtk(path, vertices, faces, scalars: Optional[dict] = None, vectors: Optional[dict] = None) -> None:
    """Write a triangle mesh with per-vertex arrays as legacy binary VTK."""
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.uint32)
    s = [(k, [float(x) for x in np.asarray(a, float).ravel()]) for k, a in (scalars or {}).items()]
    vec = [(k, np.ascontiguousarray(a, dtype=np.float64)) for k, a in (vectors or {}).items()]
    _core.write_vtk(str(path), v, f, s, vec)


def mesh_section(vertices, faces, axis: int, index: float, spacing=1.0, origin=0.0) -> np.ndarray:
    """The section of a triangle mesh (physical coordinates) by the image
    plane where the voxel index along ``axis`` equals ``index``.

    Returns the line segments, ``(S, 2, 2)``, in the voxel-index coordinates
    of the other two axes in increasing order (e.g. ``(axis 1, axis 2)`` for
    an axial section), ready to draw over ``image.take(index, axis)`` with
    ``segments[..., ::-1]`` as (x, y).
    """
    sp, org = np.array(_triple(spacing, "spacing")), np.array(_triple(origin, "origin"))
    t = ((np.asarray(vertices, float) - org) / sp)[np.asarray(faces)]  # (F, 3, 3)
    d = t[:, :, axis] - index
    edges = ((0, 1), (1, 2), (2, 0))
    cross = np.stack([(d[:, a] < 0) != (d[:, b] < 0) for a, b in edges], 1)  # (F, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        pts = np.stack([t[:, a] + (d[:, a] / (d[:, a] - d[:, b]))[:, None] * (t[:, b] - t[:, a])
                        for a, b in edges], 1)  # (F, 3, 3)
    sel = cross.sum(1) == 2
    first = np.argmax(cross[sel], 1)
    second = 2 - np.argmax(cross[sel][:, ::-1], 1)
    rows = np.arange(sel.sum())
    seg = np.stack([pts[sel][rows, first], pts[sel][rows, second]], 1)  # (S, 2, 3)
    other = [a for a in range(3) if a != axis]
    return seg[:, :, other]
