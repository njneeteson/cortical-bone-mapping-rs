"""The realistic bone phantom, Buie's dual-threshold segmentation, and CBM
correction of a Buie segmentation."""

import numpy as np
import pytest

import cortical_bone_mapping_rs as cbm

SMALL = dict(dims=(16, 200, 200), noise=80.0, seed=3)
CBM = dict(spacing=0.0607, outside=1.0, inside=3.0, normal_constraint="in_plane",
           method="regularized", reg_neighbours=64, reg_sigma=1.0)


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


@pytest.fixture(scope="module")
def osteoporotic():
    ph = cbm.BonePhantom(thickness_mean=0.45, thickness_min=0.1, bvtv=0.12, endo_bvtv=0.35,
                         porosity=0.06, endo_porosity=0.35, rho_bone=800.0, **SMALL)
    return ph, *ph.generate()


def test_phantom_truth_is_consistent(osteoporotic):
    ph, img, peri, endo = osteoporotic
    assert img.dtype == np.float32 and peri.dtype == bool
    assert (endo <= peri).all()
    # radial lines from far outside towards the centre hit the true masks' boundaries
    c = (np.array(img.shape) - 1) / 2 * 0.0607
    k = img.shape[0] // 2
    pts = np.array([[k * 0.0607, c[1] + 6 * np.cos(a), c[2] + 6 * np.sin(a)] for a in np.linspace(0, 6, 12)])
    nrm = np.column_stack([np.zeros(12), -np.cos(np.linspace(0, 6, 12)), -np.sin(np.linspace(0, 6, 12))])
    e = ph.edges_along(pts, nrm, 0.0, 6.0)
    assert np.isfinite(e).all() and (e[:, 1] > e[:, 0]).all()
    for (xp, xe), p, n in zip(e, pts, nrm):
        for x, m in ((xp, peri), (xe, endo)):
            inside = p + (x + 0.1) * n
            outside = p + (x - 0.1) * n
            ii = tuple(np.round(inside / 0.0607).astype(int))
            oo = tuple(np.round(outside / 0.0607).astype(int))
            assert m[ii] and not m[oo]


def test_buie_partitions_and_finds_the_bone(osteoporotic):
    ph, img, peri, endo = osteoporotic
    bp, bc, bt = cbm.buie_segmentation(img)
    assert (bp == (bc | bt)).all() and not (bc & bt).any()
    assert dice(bp, peri) > 0.99
    # 2D connectivity gives the same result here (as the paper reports)
    bp2, bc2, _ = cbm.buie_segmentation(img, connectivity_2d=True)
    assert dice(bc2, bc) > 0.99


def test_cbm_corrects_buie_endosteal_surface_on_porous_thin_cortex(osteoporotic):
    ph, img, peri, endo = osteoporotic
    cort = peri & ~endo
    bp, bc, bt = cbm.buie_segmentation(img)
    fit = cbm.cortical_bone_mapping(img, bp, rho_c=800.0, **CBM)
    _, _, c = fit.masks(img.shape)
    s = slice(3, -3)
    d_buie, d_cbm = dice(bc[s], cort[s]), dice(c[s], cort[s])
    assert d_buie < 0.85 and d_cbm > 0.9, (d_buie, d_cbm)
    t = ph.true_thickness(fit)
    ok = np.isfinite(t)
    assert abs((fit.thickness[ok] - t[ok]).mean()) < 0.05
