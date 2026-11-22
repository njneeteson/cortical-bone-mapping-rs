"""Several bones in one image: naming, separating, and mapping or refining
one bone at a time."""

from __future__ import annotations

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
    percentile of that bone's values), which is an error. Bones that were already separate keep the original
    threshold; pieces of a split bone smaller than ``min_size`` are dropped.

    Returns ``(labels, names, thresholds)``: uint16 labels ordered by size
    (1 = largest), ``{label: name}`` for the landmarked bones, and
    ``{name: threshold}`` used for each landmarked bone.
    """
    import sys

    img = _as_image(image)
    sp = np.asarray(_triple(spacing, "spacing"))
    org = np.asarray(_triple(origin, "origin"))
    if isinstance(threshold, str):
        if threshold != "otsu":
            raise ValueError('threshold must be a number or "otsu"')
        threshold = otsu_thresholds(img, floor=otsu_floor)[0]
    t0 = float(threshold)
    step = float(step) if step is not None else 0.05 * abs(t0)
    if not step > 0:
        raise ValueError("step must be positive")
    lab, _ = label_bones(img, t0, median_kernel, min_size)
    lab = lab.astype(np.uint32)
    used = {name: t0 for name in landmarks}
    next_id = int(lab.max()) + 1
    while True:
        where = _landmark_bones(lab, landmarks, sp, org)
        shared = {}
        for name, b in where.items():
            shared.setdefault(b, []).append(name)
        merged = [(b, ns) for b, ns in shared.items() if b > 0 and len(ns) > 1]
        if not merged:
            break
        b, ns = merged[0]
        comp = lab == b
        box = tuple(slice(int(a), int(c) + 1) for a, c in zip(np.argwhere(comp).min(0), np.argwhere(comp).max(0)))
        sub_img = np.where(comp[box], img[box], -np.inf).astype(np.float32)
        sub_org = org + np.array([s.start for s in box]) * sp
        tmax = float(max_threshold) if max_threshold is not None else float(np.percentile(img[comp], 99))
        t = t0
        while True:
            t += step
            if t > tmax:
                raise ValueError(f"landmarks {ns} stay on one bone up to threshold {tmax:g}; "
                                 "the bones could not be separated")
            pieces, _ = label_bones(sub_img, t, median_kernel, min_size)
            w = _landmark_bones(pieces, {n: landmarks[n] for n in ns}, sp, sub_org)
            if len(set(w.values())) == len(ns) and 0 not in w.values():
                break
        if verbose:
            print(f"separate_bones: {', '.join(ns)} shared one bone at {t0:g}; separated at {t:g}", file=sys.stderr)
        region = lab[box]
        region[comp[box]] = 0
        for k in range(1, int(pieces.max()) + 1):
            region[pieces == k] = next_id
            next_id += 1
        for n in ns:
            used[n] = t
    lab, sizes = _relabel_by_size(lab)
    if max_bones is not None and len(sizes) > max_bones:
        lab[lab > max_bones] = 0
        sizes = sizes[:max_bones]
    names = name_bones(lab, landmarks, sp, org)
    if verbose:
        print(f"separate_bones: {len(sizes)} bone(s): " + ", ".join(
            f"{k}{' (' + names[k] + ')' if k in names else ''} {s} voxels" for k, s in enumerate(sizes, 1)),
            file=sys.stderr)
    return lab, names, used


def _per_bone(fn, image, labels, spacing, bones, options, per_bone, crop, verbose, kwargs, names=None,
              exclude_of=None, mask_of=None, report=True):
    lab = np.asarray(labels)
    img = np.asarray(image)
    if img.shape != lab.shape:
        raise ValueError(f"image {img.shape} and labels {lab.shape} shapes differ")
    counts = np.bincount(lab.ravel().astype(np.int64))
    ids = [int(b) for b in (bones if bones is not None else np.flatnonzero(counts[1:]) + 1)]
    for b in ids:
        if b <= 0 or b >= len(counts) or counts[b] == 0:
            raise ValueError(f"no voxels with label {b}")
    # smallest bone first: small bones are quick, and problems show early
    ids.sort(key=lambda b: (counts[b], b))
    sp0 = np.asarray(_make_options(options, spacing, {}).spacing, float)
    if verbose and report:
        import sys

        print(f"{len(ids)} bone(s), fitted from smallest to largest:", file=sys.stderr)
        for b in ids:
            nm = f" ({names[b]})" if names and b in names else ""
            print(f"  bone {b}{nm}: {counts[b]} voxels, {counts[b] * sp0.prod():.0f} mm^3", file=sys.stderr)
        sys.stderr.flush()
    out = {}
    for b in ids:
        kw = dict(kwargs)
        kw.update((per_bone or {}).get(b, {}))
        m = lab == b if mask_of is None else mask_of(b)
        ex = None if exclude_of is None else exclude_of(b)
        o = _make_options(options, spacing, {k: v for k, v in kw.items() if k in _OPTION_FIELDS})
        rest = {k: v for k, v in kw.items() if k not in _OPTION_FIELDS}
        if fn is refine_segmentation:
            if "label" not in rest and not (rest.get("refine") and rest["refine"].label):
                rest["label"] = names[b] if names and b in names else f"bone {b}"  # progress prefix
            rest.setdefault("verbose", verbose)
        if crop:
            sp = np.asarray(o.spacing, float)
            margin = np.ceil((o.outside + o.inside) / sp).astype(int) + 2
            idx = np.argwhere(m)
            lo = np.maximum(idx.min(0) - margin, 0)
            hi = np.minimum(idx.max(0) + margin + 1, lab.shape)
            sl = tuple(slice(int(a), int(c)) for a, c in zip(lo, hi))
            oc = replace(o, origin=tuple(float(v) for v in np.asarray(o.origin, float) + lo * sp))
            if ex is not None:
                rest["exclude"] = np.ascontiguousarray(ex[sl])
            r = fn(np.ascontiguousarray(img[sl]), m[sl], options=oc, **rest)
            out[b] = _uncrop(r, sl, lab.shape, o)
        else:
            if ex is not None:
                rest["exclude"] = ex
            out[b] = fn(img, m, options=o, **rest)
    return out


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


def map_bones(image, labels, spacing: Union[float, Sequence[float], None] = None, *, bones=None,
              options: Optional[Options] = None, per_bone: Optional[dict] = None, crop: bool = True,
              verbose: bool = True, names: Optional[dict] = None, **kwargs) -> dict:
    """:func:`cortical_bone_mapping` on each labelled bone separately.

    Every bone is its own closed surface and its own fit, with its own global
    initialisation and cortical density estimate: bones differ in size,
    cortex and convergence. ``labels`` is e.g. :func:`bone_hulls`; ``bones``
    selects labels (default all); ``per_bone`` maps a label to option
    overrides for that bone; ``crop`` fits each bone in its bounding box plus
    the sampling range (results are in the full image's coordinates).
    Bones are fitted from the smallest to the largest; with ``verbose`` the
    bones found are listed first (stderr), with ``names`` ({label: name},
    e.g. from :func:`name_bones`) if given. Returns ``{label: CbmResult}``.
    """
    return _per_bone(cortical_bone_mapping, image, labels, spacing, bones, options, per_bone, crop, verbose,
                     kwargs, names)


def refine_bones(image, labels, spacing: Union[float, Sequence[float], None] = None, *, bones=None,
                 options: Optional[Options] = None, refine: Optional[RefineOptions] = None,
                 per_bone: Optional[dict] = None, crop: bool = True, verbose: bool = True,
                 names: Optional[dict] = None, exclude_others: bool = True, rounds: int = 1,
                 round_tolerance: float = 0.005, **kwargs) -> dict:
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
    voxels). Returns ``{label: RefineResult}`` from the last round.
    """
    import sys

    kw = dict(kwargs)
    if refine is not None:
        kw["refine"] = refine
    lab = np.asarray(labels)
    ids = [int(b) for b in (bones if bones is not None else np.flatnonzero(np.bincount(lab.ravel())[1:]) + 1)]
    regions = {b: lab == b for b in ids}

    def others(b):
        if not exclude_others:
            return None
        ex = np.zeros(lab.shape, dtype=np.uint8)
        for c, r in regions.items():
            if c != b:
                ex |= r.astype(np.uint8)
        return ex

    out = {}
    for rnd in range(max(1, int(rounds))):
        if verbose and rounds > 1:
            print(f"round {rnd + 1} of at most {rounds}", file=sys.stderr, flush=True)
        prev = {b: r.copy() for b, r in regions.items()}

        def mask_of(b):
            return prev[b] if rnd > 0 else lab == b

        def exclude_of(b):
            return others(b)

        res = {}
        # run the bones one at a time, so that later bones see earlier refined regions
        order = sorted(ids, key=lambda b: (int(prev[b].sum()), b))
        for b in order:
            res.update(_per_bone(refine_segmentation, image, labels, spacing, [b], options, per_bone, crop,
                                 verbose, kw, names, exclude_of=exclude_of, mask_of=mask_of,
                                 report=(rnd == 0 and b == order[0])))
            regions[b] = res[b].periosteal_mask
        out = res
        change = {b: float((regions[b] ^ prev[b]).sum()) / max(1, int(prev[b].sum())) for b in ids}
        if verbose and rounds > 1:
            print("round %d: region change per bone %s" % (rnd + 1, {
                (names.get(b, b) if names else b): round(v, 4) for b, v in change.items()}), file=sys.stderr,
                flush=True)
        if rnd > 0 and max(change.values()) < round_tolerance:
            break
    return out
