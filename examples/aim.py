"""A minimal reader for Scanco AIM files (HR-pQCT), for the example scripts
only: the library takes and returns arrays, and file formats are the user's
business.

Reads AIM versions 020 and 030 with 16-bit images (``short``) and
binary-compressed masks (``bin compressed``, as written by IPL for masks and
segmentations). Arrays come back in (z, y, x) order, with the position of
voxel (0, 0, 0) in the scanner's voxel grid, so that an image and its masks
can be aligned.

    image, pos, spacing, density = read_aim("SCAN.AIM")
    mask, mpos, _, _ = read_aim("SCAN_CORT_MASK.AIM")
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

SHORT, BIN_COMPRESSED = 0x00020002, 0x00150001


def read_aim(path):
    """``(array, position, spacing, density)``: the array in (z, y, x)
    order, the (z, y, x) position of its first voxel, the voxel size in mm
    from the processing log (``None`` if not given) and the linear map
    ``(slope, intercept)`` from stored values to mg HA/cm³ (``None`` if not
    calibrated)."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(64)
        if head.startswith(b"AIMDATA_V030"):
            word, start = "q", 16
        else:
            word, start = "i", 0
        size = struct.calcsize(word)
        f.seek(start)
        pre_size, struct_size, log_size, data_size = struct.unpack("<4" + word, f.read(4 * size))
        f.seek(pre_size)  # the pre-header size counts from the start of the file
        fields = struct.unpack(f"<{struct_size // size}{word}", f.read(struct_size))
        log = f.read(log_size).decode("latin1")
        offset = pre_size + struct_size + log_size
    kind = fields[5] if word == "i" else fields[3] >> 32  # 030: the type is the upper half of the 4th word
    if word == "i":
        pos, dim = fields[6:9], fields[9:12]
    else:
        pos, dim = fields[4:7], fields[7:10]
    shape = tuple(int(n) for n in dim[::-1])
    if kind == SHORT:
        data = np.fromfile(path, np.int16, count=int(np.prod(shape)), offset=offset).reshape(shape)
    elif kind == BIN_COMPRESSED:
        data = _bin_decompress(np.fromfile(path, np.uint8, count=data_size, offset=offset), shape)
    else:
        raise ValueError(f"{path.name}: AIM type {kind:#x} is not supported")
    return data, tuple(int(p) for p in pos[::-1]), _spacing(log, fields, word), _density(log)


def _bin_decompress(raw, shape):
    """IPL's binary compression: a 4-byte length, the two values (background,
    object), then run lengths of alternating values; a byte of 255 is a run
    of 254 that continues with the same value."""
    values = raw[4:6]
    runs = raw[6:].astype(np.int64)
    toggles = np.concatenate([[0], np.cumsum(runs[:-1] != 255)])
    out = np.repeat(values[toggles % 2], np.where(runs == 255, 254, runs))
    n = int(np.prod(shape))
    if out.size != n:
        raise ValueError(f"decompressed {out.size} voxels, expected {n}")
    return out.reshape(shape)


def _spacing(log, fields, word):
    if word == "q":  # version 030: element size in nm
        el = fields[25:28] if len(fields) > 27 else None
        if el and all(el):
            return tuple(float(e) * 1e-6 for e in el[::-1])
    m = re.search(r"Orig-ISQ-Dim-p\s+(\d+)\s+(\d+)\s+(\d+)\s*\n\s*Orig-ISQ-Dim-um\s+(\d+)\s+(\d+)\s+(\d+)", log)
    if m:
        p = [int(v) for v in m.groups()[:3]]
        um = [int(v) for v in m.groups()[3:]]
        return tuple(u / q * 1e-3 for u, q in zip(um[::-1], p[::-1]))
    return None


def _density(log):
    scale = re.search(r"Mu_Scaling\s+([-\d.e+]+)", log)
    slope = re.search(r"Density: slope\s+([-\d.e+]+)", log)
    intercept = re.search(r"Density: intercept\s+([-\d.e+]+)", log)
    if not (scale and slope and intercept):
        return None
    s = float(scale.group(1))
    return float(slope.group(1)) / s, float(intercept.group(1))
