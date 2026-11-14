"""Methods, sigma modes, curvature penalties and segmentation refinement."""

import numpy as np
import pytest

import cortical_bone_mapping_rs as cbm
from conftest import CYL_OPTS, dice, thickness_error

METHODS = {
    "local": dict(method="local"),
    "multigrid": dict(method="multigrid", multigrid_spacings=(4.0, 2.0, 1.0)),
    "regularized": dict(method="regularized", reg_neighbours=64, reg_sigma=1.0),
    "curvature-change": dict(method="regularized", reg_neighbours=64, reg_sigma=1.0, curvature="change"),
    "curvature-excess": dict(method="regularized", reg_neighbours=64, reg_sigma=1.0,
                             curvature="excess", curvature_kappa=0.5),
    "curvature-cotangent": dict(method="regularized", reg_lambda=0.0, curvature="change",
                                curvature_scale=None, curvature_weight=0.1),
}


@pytest.mark.parametrize("name", METHODS)
def test_methods_are_accurate(cylinder, name):
    ph, image, mask = cylinder
    r = cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS, **METHODS[name])
    bias, rmse = thickness_error(ph, r)
    assert abs(bias) < 0.01, bias
    assert rmse < 0.03, rmse
    assert np.sqrt((ph.radial_offset(r.periosteal_points) ** 2).mean()) < 0.02


def test_regularisation_improves_precision_under_noise():
    ph = cbm.Phantom(noise=120.0, seed=5)
    image, mask = ph.generate()
    e = {name: thickness_error(ph, cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS, **kw))[1]
         for name, kw in METHODS.items() if name != "curvature-cotangent"}
    for name in ("multigrid", "regularized", "curvature-change"):
        assert e[name] < 0.6 * e["local"], e


def test_estimated_psf_is_anisotropic():
    ph = cbm.Phantom(shape="sphere", dims=(64, 72, 72), radius=2.8, thickness_mean=0.9,
                     thickness_amplitude=0.2, thickness_lobes=1, blur=(0.2, 0.1, 0.1), noise=10)
    image, mask = ph.generate()
    r = cbm.cortical_bone_mapping(image, mask, ph.spacing, outside=1.5, inside=2.5,
                                  sigma="estimate_psf", psf_model="axial", psf_axis=0)
    assert r.psf[0] > r.psf[1] + 0.04 and r.psf[1] == r.psf[2]
    assert abs(thickness_error(ph, r)[0]) < 0.03
    f = cbm.cortical_bone_mapping(image, mask, ph.spacing, outside=1.5, inside=2.5, sigma="fixed", sigma_value=0.15)
    assert (f.sigma == 0.15).all()


def test_refinement_snaps_bad_segmentation():
    ph = cbm.Phantom(mask_offset=0.35, mask_bump=0.25, noise=30)
    image, bad = ph.generate()
    truth = ph.true_mask()
    rr = cbm.refine_segmentation(image, bad, ph.spacing, **CYL_OPTS, method="regularized",
                                 reg_neighbours=64, reg_sigma=1.0, keep_history=True)
    assert rr.converged
    assert dice(bad, truth) < 0.92 < 0.995 < dice(rr.periosteal_mask, truth)
    rms = [h["rms_displacement"] for h in rr.history]
    assert all(b < a for a, b in zip(rms, rms[1:]))
    assert len(rr.surfaces) == len(rr.history) + 1
    assert all(s.shape == rr.result.surface_vertices.shape for s in rr.surfaces)
    assert (rr.cortical_mask == (rr.periosteal_mask & ~rr.endosteal_mask)).all()
    with pytest.raises(ValueError, match="relaxation"):
        cbm.refine_segmentation(image, bad, ph.spacing, relaxation=1.5)
