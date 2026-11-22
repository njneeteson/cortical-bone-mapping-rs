"""Cortical bone mapping and segmentation refinement, and the profile
model."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Optional, Sequence, Union


from . import _core
from ._util import _as_image, _as_u8
from .options import Options, RefineOptions, _make_options
from .results import CbmResult, RefineResult


def cortical_bone_mapping(
    image,
    mask,
    spacing: Union[float, Sequence[float], None] = None,
    *,
    sub_mask=None,
    options: Optional[Options] = None,
    **kwargs,
) -> CbmResult:
    """Measure cortical thickness by cortical bone mapping.

    Parameters
    ----------
    image : array_like, 3D
        Grey-scale (ideally density-calibrated) image. float32/64, int8/16/32
        and uint8/16/32 are used without copying.
    mask : array_like, 3D
        Binary segmentation of the bone (non-zero = inside the periosteal
        surface).
    spacing : float or 3 floats, optional
        Voxel size per array axis (overrides ``options.spacing``).
    sub_mask : array_like, 3D, optional
        Only surface vertices adjacent to a non-zero sub-mask voxel are measured.
    options : Options, optional
        Base options; any field can also be given as a keyword argument.

    Returns
    -------
    CbmResult
    """
    o = _make_options(options, spacing, kwargs)
    img = _as_image(image)
    m = _as_u8(mask, "mask")
    if img.shape != m.shape:
        raise ValueError(f"image {img.shape} and mask {m.shape} shapes differ")
    s = None if sub_mask is None else _as_u8(sub_mask, "sub_mask")
    d = _core.run(img, m, s, o._to_core())
    return CbmResult._from_core(d, o)


def refine_segmentation(
    image,
    mask,
    spacing: Union[float, Sequence[float], None] = None,
    *,
    sub_mask=None,
    exclude=None,
    options: Optional[Options] = None,
    refine: Optional[RefineOptions] = None,
    **kwargs,
) -> RefineResult:
    """Iteratively snap the segmentation's surface to the fitted periosteal
    surface, with under-relaxation, until it stabilises.

    ``exclude`` (same shape, non-zero = excluded) marks voxels that belong
    to something else, e.g. other bones: profile samples there get zero
    weight, and the surface is never moved into them (a hard constraint).

    Keyword arguments may be fields of :class:`Options` or of
    :class:`RefineOptions`.
    """
    rf = {f.name for f in fields(RefineOptions)}
    r = refine if refine is not None else RefineOptions()
    r = replace(r, **{k: v for k, v in kwargs.items() if k in rf})
    o = _make_options(options, spacing, {k: v for k, v in kwargs.items() if k not in rf})
    if not (0.0 < r.relaxation <= 1.0):
        raise ValueError("relaxation must be in (0, 1]")
    img = _as_image(image)
    m = _as_u8(mask, "mask")
    if img.shape != m.shape:
        raise ValueError(f"image {img.shape} and mask {m.shape} shapes differ")
    s = None if sub_mask is None else _as_u8(sub_mask, "sub_mask")
    e = None if exclude is None else _as_u8(exclude, "exclude")
    if e is not None and e.shape != img.shape:
        raise ValueError(f"image {img.shape} and exclude {e.shape} shapes differ")
    d = _core.refine(img, m, s, o._to_core(), r._to_core(), e)
    return RefineResult(
        result=CbmResult._from_core(d["result"], o),
        periosteal_mask=d["periosteal_mask"],
        endosteal_mask=d["endosteal_mask"],
        cortical_mask=d["cortical_mask"],
        endosteal_vertices=d["endosteal_vertices"],
        history=d["history"],
        surfaces=d["surfaces"],
        converged=d["converged"],
        surfaces_faces=d["surfaces_faces"],
        stalled=d["stalled"],
        message=d["message"],
        timings=d["timings"],
    )
