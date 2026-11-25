"""Site phantoms: the presets and the proximal femur."""

import numpy as np
import pytest

import cortical_bone_mapping_rs as cbm


def test_site_presets():
    kinds = {"distal_radius": cbm.BonePhantom, "distal_tibia": cbm.BonePhantom,
             "proximal_femur": cbm.FemurPhantom}
    for name in cbm.SITES:
        ph = cbm.site_phantom(name, seed=3)
        assert isinstance(ph, kinds[name]) and ph.seed == 3
    radius = cbm.site_phantom("distal_radius")
    assert 220 < np.pi * radius.radius ** 2 < 360 and radius.neighbour is not None
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
