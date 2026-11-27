"""Boolean regions stored as the crop of their bounding box.

Several bones in one large image (an HR-pQCT knee holds about a billion
voxels) would otherwise each keep a full-size mask, and every bone's
exclusion would be a full-size union of the others. A :class:`Region` keeps
only its bounding box, and the operations below touch only the voxels that
can matter.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

Box = tuple  # a tuple of three slices


def bounding_box(mask: np.ndarray) -> Optional[Box]:
    """The bounding box of the true voxels of ``mask``, or ``None``."""
    box = []
    for axis in range(mask.ndim):
        any_ = np.any(mask, axis=tuple(a for a in range(mask.ndim) if a != axis))
        idx = np.flatnonzero(any_)
        if idx.size == 0:
            return None
        box.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return tuple(box)


def grow(box: Box, margin: Sequence[int], shape: Sequence[int]) -> Box:
    """``box`` grown by ``margin`` voxels per axis, clipped to ``shape``."""
    return tuple(slice(max(s.start - m, 0), min(s.stop + m, n)) for s, m, n in zip(box, margin, shape))


def _overlap(a: Box, b: Box) -> Optional[tuple]:
    """The intersection of ``a`` and ``b`` as slices into each, or ``None``."""
    lo = [max(x.start, y.start) for x, y in zip(a, b)]
    hi = [min(x.stop, y.stop) for x, y in zip(a, b)]
    if any(h <= l for l, h in zip(lo, hi)):
        return None
    in_a = tuple(slice(l - x.start, h - x.start) for l, h, x in zip(lo, hi, a))
    in_b = tuple(slice(l - y.start, h - y.start) for l, h, y in zip(lo, hi, b))
    return in_a, in_b


class Region:
    """A boolean mask of an image of ``shape``, kept as the crop ``mask`` of
    its bounding box ``box`` (``box`` is ``None`` for an empty region)."""

    def __init__(self, shape, box: Optional[Box], mask: Optional[np.ndarray]):
        self.shape = tuple(shape)
        self.box = box
        self.mask = mask

    @classmethod
    def of(cls, full: np.ndarray) -> "Region":
        """The region of a full-size boolean array."""
        box = bounding_box(full)
        return cls(full.shape, box, None if box is None else np.ascontiguousarray(full[box]))

    def count(self) -> int:
        return 0 if self.box is None else int(self.mask.sum())

    def within(self, box: Box) -> np.ndarray:
        """The region's voxels in ``box`` (a crop of the image)."""
        out = np.zeros(tuple(s.stop - s.start for s in box), dtype=bool)
        if self.box is not None:
            ov = _overlap(box, self.box)
            if ov is not None:
                out[ov[0]] |= self.mask[ov[1]]
        return out

    def full(self) -> np.ndarray:
        """The region as a full-size array."""
        return self.within(tuple(slice(0, n) for n in self.shape))


def others(regions: dict, b, box: Box) -> np.ndarray:
    """The union of every region but ``b``'s, within ``box`` (``uint8``)."""
    out = np.zeros(tuple(s.stop - s.start for s in box), dtype=np.uint8)
    for c, r in regions.items():
        if c != b:
            out |= r.within(box)
    return out


def changed(a: Region, b: Region) -> int:
    """Number of voxels in exactly one of ``a`` and ``b``."""
    if a.box is None or b.box is None:
        return a.count() + b.count()
    box = tuple(slice(min(x.start, y.start), max(x.stop, y.stop)) for x, y in zip(a.box, b.box))
    return int((a.within(box) ^ b.within(box)).sum())
