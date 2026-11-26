"""Several bones in one image: naming, separating, and mapping or refining
one bone at a time."""

from __future__ import annotations

import logging
import sys
from dataclasses import fields, replace
from typing import Literal, Optional, Sequence, Union

import numpy as np

from ._util import _as_image, _triple
from .mapping import cortical_bone_mapping, refine_segmentation
from .options import Options, RefineOptions, _make_options
from .results import RefineResult
from .segmentation import label_bones, otsu_thresholds


def name_bones(labels, landmarks: dict, spacing: Union[float, Sequence[float]] = 1.0,
               origin: Union[float, Sequence[float]] = 0.0) -> dict:
    """Name labelled bones from landmarks.

    ``landmarks`` maps a name to a point (physical coordinates, array-axis
    order, as everywhere in this package) in or near that bone, e.g.
    ``{"femur": (120, 60, 40), "patella": (140, 25, 45)}``. Each name goes to
    the bone containing its point, or else to the bone nearest to it.
    Returns ``{label: name}``. Two names on one bone means the segmentation
    has joined two distinct bones, and is an error (see
    :func:`separate_bones`, which splits them).
    """
    lab = np.asarray(labels)
    if not (lab > 0).any():
        raise ValueError("no labelled bones")
    where = _landmark_bones(lab, landmarks, np.asarray(_triple(spacing, "spacing")),
                            np.asarray(_triple(origin, "origin")))
    out = {}
    for name, b in where.items():
        if b in out:
            raise ValueError(f"landmarks {out[b]!r} and {name!r} both fall on bone {b}: the threshold has "
                             "joined two bones (separate_bones raises it locally until they split)")
        out[b] = str(name)
    return out


def _landmark_bones(lab, landmarks, sp, org):
    """{name: label} for each landmark: the bone containing it, else the
    nearest bone (0 if there are none)."""
    idx = None
    out = {}
    for name, pt in landmarks.items():
        q = (np.asarray(pt, float) - org) / sp
        i = tuple(int(round(v)) for v in q)
        b = int(lab[i]) if all(0 <= a < n for a, n in zip(i, lab.shape)) else 0
        if b == 0:
            if idx is None:
                idx = np.argwhere(lab > 0)
            b = 0 if len(idx) == 0 else int(lab[tuple(idx[np.argmin((((idx - q) * sp) ** 2).sum(1))])])
        out[name] = b
    return out


def _relabel_by_size(lab):
    counts = np.bincount(lab.ravel().astype(np.int64))
    ids = [k for k in np.argsort(-counts[1:], kind="stable") + 1 if counts[k] > 0]
    lut = np.zeros(len(counts), np.uint16)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[lab], [int(counts[k]) for k in ids]


def separate_bones(image, threshold: Union[float, Literal["otsu"]], landmarks: dict,
                   spacing: Union[float, Sequence[float]] = 1.0, origin: Union[float, Sequence[float]] = 0.0, *,
                   median_kernel=(1, 3, 3), min_size: int = 1000, max_bones: Optional[int] = None,
                   otsu_floor: Optional[float] = None, step: Optional[float] = None,
                   max_threshold: Optional[float] = None, verbose: bool = True):
    """:func:`label_bones` checked against landmarks, splitting bones that
    the threshold has joined.

    ``landmarks`` maps a bone name to a point in it (physical coordinates,
    array-axis order). If two landmarks fall on one labelled bone, the
    threshold did not separate two distinct bones (blur, noise or motion
    across a thin soft-tissue gap): within that bone only, the threshold is
    raised by ``step`` (default 5% of the threshold) until the landmarks fall
    on different pieces, up to ``max_threshold`` (default: the 99th
    percentile of that bone's values), which is an error. Bones that were
    already separate keep the original threshold; pieces of a split bone
    smaller than ``min_size`` are dropped.

    Returns ``(labels, names, thresholds)``: uint16 labels ordered by size
    (1 = largest), ``{label: name}`` for the landmarked bones, and
    ``{name: threshold}`` used for each landmarked bone.
    """
    img = _as_image(image)
    sp = np.asarray(_triple(spacing, "spacing"))
    org = np.asarray(_triple(origin, "origin"))
    t0 = _threshold(img, threshold, otsu_floor)
    step = float(step) if step is not None else 0.05 * abs(t0)
    if not step > 0:
        raise ValueError("step must be positive")
    lab = label_bones(img, t0, median_kernel, min_size)[0].astype(np.uint32)
    used = {name: t0 for name in landmarks}
    while (merged := _first_merged(lab, landmarks, sp, org)) is not None:
        b, ns = merged
        comp = lab == b
        tmax = float(max_threshold) if max_threshold is not None else float(np.percentile(img[comp], 99))
        box, pieces, t = _split(img, comp, {n: landmarks[n] for n in ns}, sp, org, t0, step, tmax,
                                median_kernel, min_size)
        _log(verbose, f"separate_bones: {', '.join(ns)} shared one bone at {t0:g}; separated at {t:g}")
        _replace_bone(lab, comp, box, pieces)
        used.update({n: t for n in ns})
    lab, sizes = _relabel_by_size(lab)
    if max_bones is not None and len(sizes) > max_bones:
        lab[lab > max_bones] = 0
        sizes = sizes[:max_bones]
    names = name_bones(lab, landmarks, sp, org)
    _log(verbose, f"separate_bones: {len(sizes)} bone(s): " + ", ".join(
        f"{k}{' (' + names[k] + ')' if k in names else ''} {s} voxels" for k, s in enumerate(sizes, 1)))
    return lab, names, used


def _threshold(img, threshold, otsu_floor):
    """A number, or the two-class Otsu threshold for ``"otsu"``."""
    if isinstance(threshold, str):
        if threshold != "otsu":
            raise ValueError('threshold must be a number or "otsu"')
        threshold = otsu_thresholds(img, floor=otsu_floor)[0]
    return float(threshold)


def _first_merged(lab, landmarks, sp, org):
    """The first bone with several landmarks on it, as ``(label, names)``,
    or None."""
    shared = {}
    for name, b in _landmark_bones(lab, landmarks, sp, org).items():
        shared.setdefault(b, []).append(name)
    return next(((b, ns) for b, ns in shared.items() if b > 0 and len(ns) > 1), None)


def _split(img, comp, marks, sp, org, t0, step, tmax, median_kernel, min_size):
    """Raise the threshold within the bone ``comp`` from ``t0`` by ``step``
    until each landmark of ``marks`` is on its own piece. Returns the bone's
    bounding box, the labelled pieces in it, and the threshold."""
    idx = np.argwhere(comp)
    box = tuple(slice(int(a), int(c) + 1) for a, c in zip(idx.min(0), idx.max(0)))
    sub_img = np.where(comp[box], img[box], -np.inf).astype(np.float32)
    sub_org = org + np.array([s.start for s in box]) * sp
    t = t0
    while True:
        t += step
        if t > tmax:
            raise ValueError(f"landmarks {list(marks)} stay on one bone up to threshold {tmax:g}; "
                             "the bones could not be separated")
        pieces, _ = label_bones(sub_img, t, median_kernel, min_size)
        w = _landmark_bones(pieces, marks, sp, sub_org)
        if len(set(w.values())) == len(marks) and 0 not in w.values():
            return box, pieces, t


def _replace_bone(lab, comp, box, pieces):
    """Replace the bone ``comp`` in ``lab`` (in place) by its ``pieces``
    (labelled within ``box``), under new labels."""
    next_id = int(lab.max()) + 1
    region = lab[box]
    region[comp[box]] = 0
    for k in range(1, int(pieces.max()) + 1):
        region[pieces == k] = next_id
        next_id += 1


_logger = logging.getLogger(__name__)


def _log(verbose, message):
    _logger.info(message)
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _bone_ids(lab, bones=None):
    """The labels to fit (default: every label present), checked, from the
    smallest bone to the largest; and the voxel count per label."""
    counts = np.bincount(lab.ravel().astype(np.int64))
    ids = [int(b) for b in (bones if bones is not None else np.flatnonzero(counts[1:]) + 1)]
    for b in ids:
        if b <= 0 or b >= len(counts) or counts[b] == 0:
            raise ValueError(f"no voxels with label {b}")
    # smallest bone first: small bones are quick, and problems show early
    ids.sort(key=lambda b: (counts[b], b))
    return ids, counts


def _list_bones(ids, counts, voxel_volume, names):
    print(f"{len(ids)} bone(s), fitted from smallest to largest:", file=sys.stderr)
    for b in ids:
        nm = f" ({names[b]})" if names and b in names else ""
        print(f"  bone {b}{nm}: {counts[b]} voxels, {counts[b] * voxel_volume:.0f} mm^3", file=sys.stderr)
    sys.stderr.flush()


def _per_bone(fn, image, labels, spacing, bones, options, per_bone, crop, verbose, kwargs, names=None,
              exclude_of=None, mask_of=None, report=True):
    """``fn`` (mapping or refinement) on each bone of ``bones``, with its
    overrides from ``per_bone``; ``mask_of`` / ``exclude_of`` give each bone's
    start and excluded region (default: its label, nothing)."""
    lab = np.asarray(labels)
    img = np.asarray(image)
    if img.shape != lab.shape:
        raise ValueError(f"image {img.shape} and labels {lab.shape} shapes differ")
    ids, counts = _bone_ids(lab, bones)
    if verbose and report:
        _list_bones(ids, counts, np.prod(_make_options(options, spacing, {}).spacing), names)
    out = {}
    for b in ids:
        kw = dict(kwargs)
        kw.update((per_bone or {}).get(b, {}))
        o = _make_options(options, spacing, {k: v for k, v in kw.items() if k in _OPTION_FIELDS})
        rest = {k: v for k, v in kw.items() if k not in _OPTION_FIELDS}
        if fn is refine_segmentation:
            if "label" not in rest and not (rest.get("refine") and rest["refine"].label):
                rest["label"] = names[b] if names and b in names else f"bone {b}"  # progress prefix
            rest.setdefault("verbose", verbose)
        m = lab == b if mask_of is None else mask_of(b)
        ex = None if exclude_of is None else exclude_of(b)
        out[b] = (_fit_cropped if crop else _fit_whole)(fn, img, m, ex, o, rest)
    return out


def _fit_whole(fn, img, m, ex, o, rest):
    if ex is not None:
        rest = dict(rest, exclude=ex)
    return fn(img, m, options=o, **rest)


def _fit_cropped(fn, img, m, ex, o, rest):
    """``fn`` on the bounding box of ``m`` plus the sampling range; results
    in the full image."""
    sp = np.asarray(o.spacing, float)
    margin = np.ceil((o.outside + o.inside) / sp).astype(int) + 2
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + margin + 1, img.shape)
    sl = tuple(slice(int(a), int(c)) for a, c in zip(lo, hi))
    oc = replace(o, origin=tuple(float(v) for v in np.asarray(o.origin, float) + lo * sp))
    if ex is not None:
        rest = dict(rest, exclude=np.ascontiguousarray(ex[sl]))
    r = fn(np.ascontiguousarray(img[sl]), m[sl], options=oc, **rest)
    return _uncrop(r, sl, img.shape, o)


def _uncrop(r, sl, shape, options):
    """Results of a cropped fit in the full image: masks pasted back at the
    full shape, options with the full image's origin (points are physical
    and unaffected)."""
    if isinstance(r, RefineResult):
        def full(a):
            f = np.zeros(shape, dtype=a.dtype)
            f[sl] = a
            return f

        return replace(r, result=replace(r.result, options=options), periosteal_mask=full(r.periosteal_mask),
                       endosteal_mask=full(r.endosteal_mask), cortical_mask=full(r.cortical_mask))
    return replace(r, options=options)


_OPTION_FIELDS = {f.name for f in fields(Options)}


def shared_scanner(image, labels, spacing=None, *, bones=None, options: Optional[Options] = None,
                   verbose: bool = True, **kwargs) -> dict:
    """The scanner parameters shared by the bones of one scan: one fit of all
    the bones' surfaces together (one mesh component per bone) estimates the
    cortical density (if ``rho_c="estimate"``) and, for the volumetric model,
    the PSF (unless given). Returns the option overrides that hold them in
    every bone's fit, e.g. ``{"rho_c": 1270.0, "vol_sigma_xy": 0.57,
    "vol_sigma_z": 0.80}`` (empty if nothing is estimated).

    The bones share the scanner, so they share its blur and its apparent
    cortical density; a joint estimate also sees more thick cortex than any
    one bone's (the cortical density can only be measured where the cortex
    is thicker than the blur).
    """
    o = _make_options(options, spacing, {k: v for k, v in kwargs.items() if k in _OPTION_FIELDS})
    volumetric = o.method == "volumetric"
    psf_free = volumetric and (o.vol_sigma_xy is None or o.vol_sigma_z in ("fit", "tied"))
    if o.rho_c != "estimate" and not psf_free:
        return {}
    lab = np.asarray(labels)
    ids, _ = _bone_ids(lab, bones)
    r = cortical_bone_mapping(np.asarray(image), np.isin(lab, ids), options=o)
    out = {}
    if o.rho_c == "estimate":
        out["rho_c"] = float(r.rho_c)
    if psf_free:
        out["vol_sigma_xy"] = float(r.volumetric["sigma_xy"])
        out["vol_sigma_z"] = float(r.volumetric["sigma_z"])
    _log(verbose, "scanner parameters of all bones: " + ", ".join(f"{k} {v:.4g}" for k, v in out.items()))
    return out


def map_bones(image, labels, spacing: Union[float, Sequence[float], None] = None, *, bones=None,
              options: Optional[Options] = None, per_bone: Optional[dict] = None, crop: bool = True,
              verbose: bool = True, names: Optional[dict] = None, shared: bool = True, **kwargs) -> dict:
    """:func:`cortical_bone_mapping` on each labelled bone separately.

    Every bone is its own closed surface and its own fit, with its own global
    initialisation and cortical density estimate: bones differ in size,
    cortex and convergence. ``labels`` is e.g. :func:`bone_hulls`; ``bones``
    selects labels (default all); ``per_bone`` maps a label to option
    overrides for that bone; ``crop`` fits each bone in its bounding box plus
    the sampling range (results are in the full image's coordinates).
    Bones are fitted from the smallest to the largest; with ``verbose`` the
    bones found are listed first (stderr), with ``names`` ({label: name},
    e.g. from :func:`name_bones`) if given. With ``shared`` (default) and
    more than one bone, the scanner parameters (an estimated cortical
    density, the volumetric model's PSF) are estimated once for all the bones
    (:func:`shared_scanner`) and held in every bone's fit. Returns
    ``{label: CbmResult}``.
    """
    kwargs = _with_shared(image, labels, spacing, bones, options, verbose, kwargs, shared)
    return _per_bone(cortical_bone_mapping, image, labels, spacing, bones, options, per_bone, crop, verbose,
                     kwargs, names)


def _with_shared(image, labels, spacing, bones, options, verbose, kwargs, shared):
    """``kwargs`` with the shared scanner parameters, if ``shared`` and there
    is more than one bone (explicit keyword arguments win)."""
    ids, _ = _bone_ids(np.asarray(labels), bones)
    if not shared or len(ids) < 2:
        return kwargs
    opts = {k: v for k, v in kwargs.items() if k in _OPTION_FIELDS}
    return {**shared_scanner(image, labels, spacing, bones=bones, options=options, verbose=verbose, **opts),
            **kwargs}


def refine_bones(image, labels, spacing: Union[float, Sequence[float], None] = None, *, bones=None,
                 options: Optional[Options] = None, refine: Optional[RefineOptions] = None,
                 per_bone: Optional[dict] = None, crop: bool = True, verbose: bool = True,
                 names: Optional[dict] = None, exclude_others: bool = True, rounds: int = 1,
                 round_tolerance: float = 0.005, shared: bool = True, **kwargs) -> dict:
    """:func:`refine_segmentation` on each labelled bone separately (see
    :func:`map_bones`); each bone converges at its own rate. Keyword arguments
    may be fields of :class:`Options` or :class:`RefineOptions`. Masks are
    returned at the full image shape. Bones are refined from the smallest to
    the largest; with ``verbose`` the bones found are listed first and every
    refinement iteration prints a line, prefixed by the bone's name in
    ``names`` ({label: name}, e.g. from :func:`name_bones`) or ``bone <label>``.

    With ``exclude_others`` (default) every other bone's region is excluded
    while a bone is refined: profile samples inside it get zero weight and
    the surface never moves into it. The regions are the other bones'
    current segmentations: their labels at first, their refined periosteal
    masks once refined (bones refined later in a round see the earlier ones'
    refined regions).

    ``rounds`` > 1 repeats the pass over all bones, each starting from its
    own refined region and excluding the others' refined regions, until no
    bone's region changes by more than ``round_tolerance`` (fraction of its
    voxels).

    With ``shared`` (default) and more than one bone, the scanner parameters
    are estimated once on all the starting surfaces (:func:`shared_scanner`)
    and held in every bone's refinement. Returns ``{label: RefineResult}``
    from the last round.
    """
    kwargs = _with_shared(image, labels, spacing, bones, options, verbose, kwargs, shared)
    kw = dict(kwargs, refine=refine) if refine is not None else dict(kwargs)
    lab = np.asarray(labels)
    ids, counts = _bone_ids(lab, bones)
    if verbose:
        _list_bones(ids, counts, np.prod(_make_options(options, spacing, {}).spacing), names)
    regions = {b: lab == b for b in ids}
    exclude_of = (lambda b: _others(regions, b, lab.shape)) if exclude_others else None
    rounds = max(1, int(rounds))
    for rnd in range(rounds):
        if rounds > 1:
            _log(verbose, f"round {rnd + 1} of at most {rounds}")
        prev = {b: r.copy() for b, r in regions.items()}
        out = {}
        # one bone at a time, so that later bones see earlier refined regions
        for b in sorted(ids, key=lambda b: (int(prev[b].sum()), b)):
            out.update(_per_bone(refine_segmentation, image, lab, spacing, [b], options, per_bone, crop, verbose,
                                 kw, names, exclude_of=exclude_of, mask_of=prev.get, report=False))
            regions[b] = out[b].periosteal_mask
        change = {b: float((regions[b] ^ prev[b]).sum()) / max(1, int(prev[b].sum())) for b in ids}
        if rounds > 1:
            _log(verbose, "round %d: region change per bone %s" % (rnd + 1, {
                (names.get(b, b) if names else b): round(v, 4) for b, v in change.items()}))
        if rnd > 0 and max(change.values()) < round_tolerance:
            break
    return out


def _others(regions, b, shape):
    """The union of every region but bone ``b``'s."""
    ex = np.zeros(shape, dtype=np.uint8)
    for c, r in regions.items():
        if c != b:
            ex |= r.astype(np.uint8)
    return ex

