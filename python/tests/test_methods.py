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
    "curvature-absolute": dict(method="regularized", reg_neighbours=64, reg_sigma=1.0,
                               curvature="absolute", curvature_kappa=0.5),
    "curvature-total": dict(method="regularized", reg_neighbours=64, reg_sigma=1.0,
                            curvature="absolute", curvature_measure="total", curvature_kappa=0.5),
    "curvature-cotangent": dict(method="regularized", reg_lambda=0.0, curvature="change",
                                curvature_scale="cotangent", curvature_weight=0.1),
    # a shaft: every normal is in-plane, so the axial blur is not identified
    "volumetric": dict(method="volumetric", vol_sigma_z="tied"),
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
    for name in ("multigrid", "regularized", "curvature-change", "volumetric"):
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


def test_volumetric_outputs(cylinder):
    ph, image, mask = cylinder
    r = cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS, method="volumetric", vol_sigma_z="tied")
    v = r.volumetric
    n = len(r.thickness)
    assert v["sigma_xy"] == v["sigma_z"] and abs(v["sigma_xy"] - 0.12) < 0.02
    assert abs(v["noise_sd"] / ph.noise - 1) < 0.1
    assert v["std_error_t"].shape == (n,) and v["misfit"].shape == (n,) and v["data_share"].shape == (n, 4)
    share = v["data_share"][np.isfinite(v["data_share"])]
    assert ((share >= 0) & (share <= 1)).all()
    assert abs(np.median(v["misfit"]) - 1) < 0.15
    assert r.volumetric is not None and cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS).volumetric is None
    f = cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS, method="volumetric", vol_sigma_z=0.2,
                                  vol_global_densities=True, vol_curvature=None, vol_voxel_stride=2)
    assert f.volumetric["sigma_z"] == 0.2
    assert f.volumetric["voxels"] < v["voxels"] / 4
    for bad in (dict(vol_sigma_z="free"), dict(vol_scanner_axis=3), dict(vol_sigma_xy=-1.0),
                dict(rho_c="profile_max")):
        with pytest.raises(ValueError):
            cbm.cortical_bone_mapping(image, mask, ph.spacing, method="volumetric", **bad)


def test_volumetric_refinement(cylinder):
    ph = cbm.Phantom(mask_offset=0.35, mask_bump=0.25, noise=30)
    image, bad = ph.generate()
    rr = cbm.refine_segmentation(image, bad, ph.spacing, **CYL_OPTS, method="volumetric", vol_sigma_z="tied")
    assert rr.converged and rr.result.volumetric is not None
    assert dice(rr.periosteal_mask, ph.true_mask()) > 0.995


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
    # one history entry per fit (on the input surface and on every moved one)
    assert len(rr.surfaces) == len(rr.history) == len(rr.surfaces_faces)
    assert rr.history[0]["full_fit"] and rr.history[-1]["full_fit"]
    # the level set (default) extracts a new mesh every move
    assert all(f.max() < len(s) for s, f in zip(rr.surfaces, rr.surfaces_faces))
    assert rr.surfaces[-1].shape == rr.result.surface_vertices.shape
    assert (rr.cortical_mask == (rr.periosteal_mask & ~rr.endosteal_mask)).all()
    # moving the same mesh instead keeps its vertices
    same = cbm.refine_segmentation(image, bad, ph.spacing, **CYL_OPTS, method="regularized", reg_neighbours=64,
                                   reg_sigma=1.0, keep_history=True, level_set=False, freeze=False,
                                   refine_iterations=3, verbose=False)
    assert all(v.shape == same.surfaces[0].shape for v in same.surfaces)
    with pytest.raises(ValueError, match="relaxation"):
        cbm.refine_segmentation(image, bad, ph.spacing, relaxation=1.5)
