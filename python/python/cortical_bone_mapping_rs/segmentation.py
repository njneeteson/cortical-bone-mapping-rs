"""Segmentation from the image alone: thresholds, bones, starting
segmentations, and Buie's method."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np

from . import _core
from ._util import Triple, _as_image, _triple


@dataclass
class BuieOptions:
    """Options of :func:`buie_segmentation`, with the filter parameters of
    Buie et al. (2007), Fig. 1. Kernels are in voxels per array axis
    ``(z, y, x)`` (the paper's VTK ``(3,3,1)`` is ``(1, 3, 3)`` here) and, as in
    the paper, are not rescaled with resolution.

    periosteal_threshold, endosteal_threshold
        User inputs 1 and 2, in image units. The paper used native scanner
        units (human XtremeCT: 3000 / 5000); 450 / 320 mg HA/cm^3 are commonly
        used for density-calibrated HR-pQCT (Burghardt et al. 2010).
    gaussian_sigma, gaussian_support
        Step 12: sigma = 3 voxels, support (radius factors) (3,3,1).
    smoothed_threshold
        Step 13 "threshold by lower = 100" on 0/127 masks.
    connectivity_2d
        Per-slice connectivity (the paper: usually identical, faster).
    largest_bone_only
        Pre-processing: keep only the largest bone (removes fibula/ulna).
    """

    periosteal_threshold: float = 450.0
    endosteal_threshold: float = 320.0
    median_kernel: Tuple[int, int, int] = (1, 3, 3)
    periosteal_kernel: Tuple[int, int, int] = (1, 15, 15)
    endosteal_kernel: Tuple[int, int, int] = (1, 10, 10)
    gaussian_sigma: float = 3.0
    gaussian_support: Triple = (1.0, 3.0, 3.0)
    smoothed_threshold: float = 100.0 / 127.0
    connectivity_2d: bool = False
    largest_bone_only: bool = True


def buie_segmentation(image, options: Optional[BuieOptions] = None, **kwargs):
    """Dual-threshold cortical/trabecular segmentation of Buie et al. (2007),
    Bone 41:505-515, doi:10.1016/j.bone.2007.07.007. Returns boolean
    ``(periosteal, cortical, trabecular)`` masks. See ``docs/method.md`` for
    how each step of the paper's Fig. 1 is implemented.
    """
    o = replace(options or BuieOptions(), **kwargs)
    d = asdict(o)
    d["median_kernel"] = tuple(int(v) for v in o.median_kernel)
    d["periosteal_kernel"] = tuple(int(v) for v in o.periosteal_kernel)
    d["endosteal_kernel"] = tuple(int(v) for v in o.endosteal_kernel)
    d["gaussian_support"] = tuple(float(v) for v in o.gaussian_support)
    return _core.buie(_as_image(image), d)


def hull_segmentation(image, threshold: float = 320.0, median_kernel=(1, 3, 3),
                      largest_component: bool = True) -> np.ndarray:
    """A conservative starting segmentation for :func:`refine_segmentation`.

    Threshold the image (``image >= threshold``; 320 mg HA/cm^3 is the usual
    bone threshold for calibrated HR-pQCT), remove noise speckle with a binary
    median (``median_kernel``, as Buie's step 3), keep the largest 3D component
    (drops a separate fibula/ulna), and take the convex hull of every slice
    along axis 0. Unlike a morphological closing, the hull cannot leak through
    cortical breaks or a porous cortex into the marrow; its errors are outward
    bulges over concave parts of the periosteum, which refinement removes.
    Returns a boolean mask.
    """
    return _core.hull_segmentation(_as_image(image), float(threshold),
                                   tuple(int(k) for k in median_kernel), bool(largest_component))


def otsu_thresholds(image, floor: Optional[float] = None, classes: int = 2, bins: int = 1024) -> list:
    """Thresholds from the image histogram by (multi-level) Otsu's method.

    Only voxels ``>= floor`` enter the histogram (e.g. ``floor=-500`` to
    exclude the air around the body in Hounsfield units). ``classes`` is 2
    (one threshold) or 3 (two). Returns the thresholds in ascending order.

    For non-calibrated CT, the two-class threshold of the body voxels
    separates soft tissue from bone and is a reasonable bone threshold for
    :func:`label_bones` (the histogram has fat and soft-tissue peaks but no
    bone peak, only a long tail).
    """
    return _core.otsu_thresholds(_as_image(image), None if floor is None else float(floor), int(classes), int(bins))


def label_bones(image, threshold: Union[float, Literal["otsu"]], median_kernel=(1, 3, 3), min_size: int = 1000,
                max_bones: Optional[int] = None, otsu_floor: Optional[float] = None) -> Tuple[np.ndarray, list]:
    """Separate the bones of an image: ``image >= threshold``, binary median
    (``median_kernel``), 6-connected components of at least ``min_size``
    voxels, at most ``max_bones``.

    ``threshold`` is in the image's units (e.g. 320 mg HA/cm^3 for calibrated
    HR-pQCT, or a Hounsfield value), or ``"otsu"`` for the two-class
    :func:`otsu_thresholds` of the voxels ``>= otsu_floor`` (for images
    without a known bone threshold, such as non-calibrated clinical CT).

    Returns ``(labels, sizes)``: uint16 labels (0 background, ``k`` the
    ``k``-th largest bone) and the size in voxels of each bone. Bones joined
    at the threshold (blur, noise or motion across a thin gap) become one
    label.
    """
    img = _as_image(image)
    if isinstance(threshold, str):
        if threshold != "otsu":
            raise ValueError('threshold must be a number or "otsu"')
        t = otsu_thresholds(img, floor=otsu_floor)
        if not t:
            raise ValueError("no voxels above otsu_floor")
        threshold = t[0]
    return _core.label_bones(img, float(threshold), tuple(int(k) for k in median_kernel),
                             int(min_size), None if max_bones is None else int(max_bones))


def soft_tissue_threshold(image, high: float, floor: float = -500.0, k: float = 4.0, bin_width: float = 2.0) -> float:
    """A low threshold for :func:`grow_bones` from the histogram: the
    soft-tissue peak plus ``k`` standard deviations of soft tissue.

    The soft-tissue peak is the mode of the voxels between the lower
    three-class Otsu threshold of the voxels above ``floor`` (which separates
    fat from soft tissue) and the bone threshold ``high``. Its spread is
    taken from the half-width at half maximum on its lower side, so the bone
    tail does not inflate it (``sd = HWHM / 1.1774``). On a clinical knee CT:
    peak 57 HU, sd 22 HU, threshold 146 HU (``k`` = 4).
    """
    img = _as_image(image)
    t3 = otsu_thresholds(img, floor=floor, classes=3)
    lo = t3[0] if t3 else floor
    v = img[(img >= lo) & (img < high)]
    if v.size == 0:
        raise ValueError("no voxels between the fat/soft-tissue threshold and high")
    h, e = np.histogram(v, bins=np.arange(lo, high + bin_width, bin_width))
    c = 0.5 * (e[1:] + e[:-1])
    i = int(np.argmax(h))
    j = i
    while j > 0 and h[j] > 0.5 * h[i]:
        j -= 1
    sd = max(c[i] - c[j], bin_width) / 1.1774
    return float(min(c[i] + k * sd, high))


def grow_bones(image, labels, low_threshold: float, median_kernel=(1, 3, 3)) -> np.ndarray:
    """Hysteresis thresholding: grow the bones of ``labels`` (from
    :func:`label_bones` or :func:`separate_bones` at a high threshold) into
    the connected voxels ``>= low_threshold`` (after a binary median).

    Thin cortices that partial volume pushes below the high threshold
    (metaphyses, condylar rims, tibial spines in clinical CT) are recovered,
    while the bones are still told apart at the high threshold: a voxel
    reachable from several bones goes to the one fewest 6-connected steps
    away. Labels are unchanged. Returns uint16 labels.
    """
    img = _as_image(image)
    lab = np.ascontiguousarray(labels, dtype=np.uint16)
    if lab.shape != img.shape:
        raise ValueError(f"image {img.shape} and labels {lab.shape} shapes differ")
    return _core.grow_bones(img, lab, float(low_threshold), tuple(int(k) for k in median_kernel))


def bone_hulls(labels, spacing: Union[float, Sequence[float]] = 1.0,
               max_distance: Optional[float] = None,
               closing_radius: Optional[float] = None) -> np.ndarray:
    """Per-bone starting segmentations from :func:`label_bones`: the convex
    hull of every slice (axis 0) of each bone, kept apart.

    Where hulls overlap (a concave surface facing another bone, e.g. the
    femoral trochlea and the patella), each voxel goes to the nearest bone
    (physical distance, ``spacing`` per axis); hull voxels touching another
    bone's region are then cleared, so every label is its own closed surface
    and still contains its thresholded bone.

    ``max_distance`` (physical units) first trims each hull to within that
    distance of its bone and fills the holes of every slice: the marrow and
    gaps narrower than twice the distance (cortical breaks) stay closed,
    while a wide open concavity deeper than the profiles reach (the femoral
    intercondylar notch) is no longer bridged.

    ``closing_radius`` (physical units) replaces each bone's per-slice hull
    by its morphological closing with a ball of that radius, with slice holes
    filled: a "local convex hull" that fills concavities narrower than about
    twice the radius (pits, small cortical gaps) and keeps wider ones (the
    notch, with a few mm), following the bone's curvature at larger scales.
    Returns uint16 labels.
    """
    lab = np.ascontiguousarray(labels, dtype=np.uint16)
    if lab.ndim != 3:
        raise ValueError("labels must be 3D")
    return _core.bone_hulls(lab, _triple(spacing, "spacing"),
                            None if max_distance is None else float(max_distance),
                            None if closing_radius is None else float(closing_radius))
