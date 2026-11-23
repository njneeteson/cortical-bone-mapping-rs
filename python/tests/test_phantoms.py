"""Site phantoms: the presets, the proximal femur and the knee's notch."""

from dataclasses import replace

import numpy as np
import pytest

import cortical_bone_mapping_rs as cbm


def test_site_presets():
    kinds = {"distal_radius": cbm.BonePhantom, "distal_tibia": cbm.BonePhantom,
             "proximal_femur": cbm.FemurPhantom, "knee": cbm.JointPhantom}
    for name in cbm.SITES:
        ph = cbm.site_phantom(name, seed=3)
        assert isinstance(ph, kinds[name]) and ph.seed == 3
    radius = cbm.site_phantom("distal_radius")
    assert 220 < np.pi * radius.radius ** 2 < 360 and radius.neighbour is not None
    knee = cbm.site_phantom("knee")
    assert knee.notch is not None and knee.femur_plate_thickness < knee.plate_thickness
    with pytest.raises(ValueError, match="unknown site"):
        cbm.site_phantom("skull")


def test_femur_phantom_truth():
    ph = cbm.FemurPhantom(dims=(56, 52, 70), spacing=(2.0, 1.5, 1.5), noise=0.0, supersample=1)
    img, peri, endo = ph.generate()
    assert img.shape == (56, 52, 70) and peri.dtype == np.uint8
    assert ((peri == 1).sum() > (endo == 1).sum() > 0) and (peri == 2).any()
    # lines straight down through the top of the volume meet the acetabulum
    # or the femoral head, then their marrow
    zs = np.full(5, (ph.dims[0] - 1) * ph.spacing[0])
    pts = np.stack([zs, np.full(5, 39.0), np.linspace(60, 80, 5)], 1)
    e = ph.edges_along(pts, np.tile([-1.0, 0, 0], (5, 1)), 0.0, 60.0)
    hit = np.isfinite(e[:, 0])
    assert hit.any() and (e[hit, 1] > e[hit, 0]).all() and set(e[hit, 2]) <= {1.0, 2.0}


def test_knee_notch_removes_bone_between_the_condyles():
    ph = cbm.site_phantom("knee")
    small = replace(ph, dims=(57, 72, 72), spacing=(1.25, 1.3, 1.3), supersample=1, noise=0.0)
    _, peri, _ = small.generate()
    roof = ph.joint_height() + ph.joint_space + ph.notch[1]
    mid = (small.dims[2] - 1) // 2
    k_low = int((roof - 6.0) / small.spacing[0])
    # on the mid-line below the roof, posteriorly: no femur; beside it: femur
    post = slice(small.dims[1] // 2 + 4, small.dims[1] - 8)
    assert (peri[k_low, post, mid] != 2).all()
    side = mid + int(18.0 / small.spacing[2])
    assert (peri[k_low, post, side] == 2).any()
