"""Independent reference checks.

* The local fitter against SciPy's bounded least squares
  (``scipy.optimize.least_squares``, trust-region reflective) on the same
  weighted problem.
* The full pipeline on a phantom built here with NumPy/SciPy (not with the
  crate's own phantom generator): a tilted cylinder whose cortex is not
  aligned with the voxel grid, blurred with ``scipy.ndimage``.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.special import erf

import cortical_bone_mapping_rs as cbm

RHO_C = 1000.0
LOWER = np.array([-3.0, 0.05, 0.01, -200.0, -200.0])
UPPER = np.array([6.0, 12.0, 2.0, 600.0, 900.0])


def model(x, p):
    m, t, s, rs, rb = p
    return (
        rs
        + 0.5 * (RHO_C - rs) * (1 + erf((x - m + t / 2) / (s * np.sqrt(2))))
        + 0.5 * (rb - RHO_C) * (1 + erf((x - m - t / 2) / (s * np.sqrt(2))))
    )


def weights(x, boost):
    return np.maximum(boost + (1 - boost) * np.abs(x) / np.abs(x).max(), 0)


def scipy_fit(y, x, w, p0):
    sw = np.sqrt(w / w.sum())
    r = least_squares(
        lambda p: sw * (model(x, p) - y),
        np.clip(p0, LOWER, UPPER),
        bounds=(LOWER, UPPER),
        x_scale=np.array([0.1, 0.1, 0.05, 50.0, 50.0]),
        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000,
    )
    return r.x, 0.5 * (r.fun**2).sum()


def test_profile_model_matches_scipy_erf():
    x = np.linspace(-2, 3, 101)
    p = np.array([0.4, 0.9, 0.2, 80.0, 300.0])
    np.testing.assert_allclose(cbm.profile_model(x, p, RHO_C), model(x, p), atol=1e-9)


def test_local_fit_matches_scipy_on_noisy_profiles():
    rng = np.random.default_rng(0)
    x = np.linspace(-1.5, 2.5, 81)
    w = weights(x, 5.0)
    n = 200
    truths = np.column_stack([
        rng.uniform(0.2, 0.8, n), rng.uniform(0.3, 1.5, n), rng.uniform(0.1, 0.3, n),
        rng.uniform(0, 150, n), rng.uniform(150, 400, n),
    ])
    profiles = np.stack([model(x, p) for p in truths]) + rng.normal(0, 40, (n, len(x)))
    init = np.array([0.0, 1.0, 0.1, 0.0, 0.0])
    ours = cbm.fit_profiles(profiles, x, init=init, rho_c=RHO_C, residual_boost=5.0,
                            lower=LOWER, upper=UPPER, ftol=1e-12, gtol=1e-12, xtol=1e-12, max_iterations=500)
    worse = 0
    for i in range(n):
        ref_p, ref_c = scipy_fit(profiles[i].astype(np.float32).astype(float), x, w, init)
        # same weighted cost, normalised the same way
        if ours["cost"][i] > ref_c * (1 + 1e-6) + 1e-9:
            worse += 1
            # a different local minimum is acceptable only if both are close
            assert ours["cost"][i] < 1.05 * ref_c, (i, ours["cost"][i], ref_c)
        else:
            np.testing.assert_allclose(ours["params"][i, 1], ref_p[1], atol=5e-3)
    # almost always the same optimum
    assert worse <= n // 50


@settings(max_examples=60, deadline=None)
@given(
    m=st.floats(0.0, 1.0), t=st.floats(0.25, 2.0), s=st.floats(0.08, 0.3),
    rs=st.floats(-50, 200), rb=st.floats(100, 500),
)
def test_noise_free_profiles_are_recovered(m, t, s, rs, rb):
    x = np.linspace(-2.0, 4.0, 121)
    truth = np.array([m, t, s, rs, rb])
    y = model(x, truth)
    out = cbm.fit_profiles(y, x, rho_c=RHO_C, residual_boost=3.0, lower=LOWER, upper=UPPER,
                           ftol=1e-14, gtol=1e-12, max_iterations=300)
    np.testing.assert_allclose(out["params"][0, :3], truth[:3], atol=2e-3)


def tilted_cylinder(noise, seed=0):
    """Cylinder along a tilted axis, cortex thickness varying with angle,
    supersampled ×3, blurred with an anisotropic Gaussian PSF in SciPy."""
    shape, h, ss = (40, 90, 90), 0.1, 3
    axis = np.array([1.0, 0.25, -0.15])
    axis /= np.linalg.norm(axis)
    c = (np.array(shape) - 1) / 2 * h
    g = [(np.arange(n * ss) + 0.5) / ss - 0.5 for n in shape]
    zz, yy, xx = np.meshgrid(*g, indexing="ij")
    p = np.stack([zz, yy, xx], -1) * h - c
    along = p @ axis
    radial = p - along[..., None] * axis
    r = np.linalg.norm(radial, axis=-1)
    e1 = np.cross(axis, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    theta = np.arctan2(radial @ e2, radial @ e1)
    R = 3.2
    t = 0.7 + 0.25 * np.cos(2 * theta)
    dens = np.where(r > R, 40.0, np.where(r > R - t, RHO_C, 220.0))
    dens = dens.reshape(shape[0], ss, shape[1], ss, shape[2], ss).mean(axis=(1, 3, 5))
    img = ndimage.gaussian_filter(dens, sigma=np.array([0.15, 0.1, 0.1]) / h, mode="nearest")
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    # mask from voxel centres
    gi = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1) * h - c
    ra = gi - (gi @ axis)[..., None] * axis
    mask = np.linalg.norm(ra, axis=-1) <= R

    def truth(points):
        q = points - c
        rad = q - (q @ axis)[:, None] * axis
        th = np.arctan2(rad @ e2, rad @ e1)
        return 0.7 + 0.25 * np.cos(2 * th)

    return img.astype(np.float32), mask, truth, h


@pytest.mark.parametrize("method", ["local", "multigrid", "regularized"])
def test_independent_tilted_phantom(method):
    img, mask, truth, h = tilted_cylinder(noise=30)
    r = cbm.cortical_bone_mapping(
        img, mask, h, outside=1.5, inside=2.5, method=method,
        multigrid_spacings=(4.0, 2.0, 1.0), reg_neighbours=64, reg_sigma=1.0,
    )
    e = r.thickness - truth(r.points)
    assert abs(e.mean()) < 0.02, e.mean()
    assert np.sqrt((e**2).mean()) < 0.05
