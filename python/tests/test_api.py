"""Argument handling, dtypes, layouts, options, results and errors."""

import doctest

import numpy as np
import pytest

import cortical_bone_mapping_rs as cbm
from conftest import CYL_OPTS, thickness_error


def test_version_and_docstrings():
    assert cbm.__version__.count(".") == 2
    r = doctest.testmod(cbm, verbose=False)
    assert r.attempted > 0 and r.failed == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.uint16, np.int32, np.float16])
def test_image_dtypes(clean_cylinder, dtype):
    ph, image, mask = clean_cylinder
    ref = cbm.cortical_bone_mapping(np.round(image).astype(np.float64), mask, ph.spacing, **CYL_OPTS)
    img = np.round(image).astype(dtype)
    r = cbm.cortical_bone_mapping(img, mask, ph.spacing, **CYL_OPTS)
    tol = 1e-3 if dtype == np.float16 else 1e-9
    np.testing.assert_allclose(r.thickness, ref.thickness, atol=tol)


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int8, np.int64, np.float32])
def test_mask_dtypes(clean_cylinder, dtype):
    ph, image, mask = clean_cylinder
    ref = cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS)
    r = cbm.cortical_bone_mapping(image, (mask * 3).astype(dtype) if dtype != np.bool_ else mask.astype(bool), ph.spacing, **CYL_OPTS)
    np.testing.assert_array_equal(r.thickness, ref.thickness)


@pytest.mark.parametrize(
    "view", [lambda a: a[::-1].copy()[::-1], np.asfortranarray, lambda a: a.transpose(1, 2, 0).transpose(2, 0, 1)],
    ids=["reversed-twice", "fortran", "transposed-back"],
)
def test_memory_layouts(clean_cylinder, view):
    ph, image, mask = clean_cylinder
    ref = cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS)
    r = cbm.cortical_bone_mapping(view(image), view(mask), ph.spacing, **CYL_OPTS)
    np.testing.assert_array_equal(r.thickness, ref.thickness)


def test_inputs_not_modified(clean_cylinder):
    ph, image, mask = clean_cylinder
    i0, m0 = image.copy(), mask.copy()
    cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS)
    np.testing.assert_array_equal(image, i0)
    np.testing.assert_array_equal(mask, m0)


def test_errors(clean_cylinder):
    ph, image, mask = clean_cylinder
    with pytest.raises(ValueError, match="empty"):
        cbm.cortical_bone_mapping(image, np.zeros_like(mask), ph.spacing)
    with pytest.raises(ValueError, match="shapes differ"):
        cbm.cortical_bone_mapping(image, mask[:-1], ph.spacing)
    with pytest.raises(ValueError, match="3D"):
        cbm.cortical_bone_mapping(image[0], mask[0], ph.spacing)
    with pytest.raises(TypeError, match="unknown option"):
        cbm.cortical_bone_mapping(image, mask, ph.spacing, thicknes=1)
    with pytest.raises(ValueError, match="spacing"):
        cbm.cortical_bone_mapping(image, mask, (0.1, 0.1))
    with pytest.raises(ValueError, match="method"):
        cbm.cortical_bone_mapping(image, mask, ph.spacing, method="global")
    with pytest.raises(ValueError, match="curvature"):
        cbm.cortical_bone_mapping(image, mask, ph.spacing, curvature="change", method="local")
    with pytest.raises(ValueError):
        cbm.cortical_bone_mapping(image, mask, ph.spacing, lower=(0, 0, 0, 0))
    with pytest.raises(ValueError, match="threads"):
        cbm.cortical_bone_mapping(image, mask, ph.spacing, threads=0)
    with pytest.raises(ValueError, match="no surface nodes"):
        cbm.cortical_bone_mapping(image, mask, ph.spacing, sub_mask=np.zeros_like(mask))
    with pytest.raises(TypeError):
        cbm.cortical_bone_mapping(image.astype(complex), mask, ph.spacing)


def test_options_object_and_overrides(clean_cylinder):
    ph, image, mask = clean_cylinder
    o = cbm.Options(spacing=ph.spacing, **CYL_OPTS)
    a = cbm.cortical_bone_mapping(image, mask, options=o)
    b = cbm.cortical_bone_mapping(image, mask, options=o, threads=1)
    np.testing.assert_allclose(a.thickness, b.thickness, atol=1e-9)
    assert a.options.threads is None and b.options.threads == 1
    # scalar spacing broadcast
    c = cbm.cortical_bone_mapping(image, mask, 0.1, **CYL_OPTS)
    np.testing.assert_array_equal(c.thickness, a.thickness)


def test_result_contents(clean_cylinder, tmp_path):
    ph, image, mask = clean_cylinder
    r = cbm.cortical_bone_mapping(image, mask, ph.spacing, keep_profiles=True, **CYL_OPTS)
    n = r.n
    assert r.points.shape == r.normals.shape == (n, 3)
    assert r.params.shape == (n, 5) and r.residual.shape == (n,)
    np.testing.assert_allclose(np.linalg.norm(r.normals, axis=1), 1.0)
    np.testing.assert_allclose(r.normals[:, 0], 0.0, atol=1e-12)  # in-plane constraint
    assert r.faces.max() < n and r.faces.dtype == np.uint32
    np.testing.assert_array_equal(r.surface_vertices[r.vertex_ids], r.points)
    assert r.profiles.shape == (n, len(r.profile_x))
    # the model reproduces the profile up to the residual
    i = n // 3
    resid = r.model(i) - r.profiles[i]
    assert np.sqrt((resid**2).mean()) < 3 * r.residual[i] + 1.0
    # edges straddle the node's normal line
    d = r.endosteal_points - r.periosteal_points
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), r.thickness, rtol=1e-9)
    assert set(r.info["timings"]) >= {"surface", "sampling", "global fit", "local fit"}
    # VTK
    p = tmp_path / "res.vtk"
    r.save_vtk(p)
    data = p.read_bytes()
    assert data.startswith(b"# vtk DataFile") and b"SCALARS thickness" in data


def test_masks_from_result(clean_cylinder):
    ph, image, mask = clean_cylinder
    r = cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS)
    peri, endo, cort = r.masks(image.shape)
    assert peri.dtype == bool and peri.shape == image.shape
    assert (cort == (peri & ~endo)).all()
    inter = (peri & mask.astype(bool)).sum()
    assert 2 * inter / (peri.sum() + mask.sum()) > 0.99


def test_voxelize_roundtrip_of_extracted_surface(clean_cylinder):
    ph, image, mask = clean_cylinder
    r = cbm.cortical_bone_mapping(image, mask, ph.spacing, taubin_iterations=0, **CYL_OPTS)
    vox = cbm.voxelize(r.surface_vertices, r.surface_faces, mask.shape, ph.spacing)
    # an unsmoothed surface-nets surface encloses exactly the mask voxels
    np.testing.assert_array_equal(vox, mask.astype(bool))
    with pytest.raises(ValueError):
        cbm.voxelize(r.surface_vertices, r.surface_faces + 10**9 // 1000, mask.shape)


def test_sub_mask(cylinder):
    ph, image, mask = cylinder
    sub = np.zeros_like(mask)
    sub[:, : mask.shape[1] // 2] = 1
    full = cbm.cortical_bone_mapping(image, mask, ph.spacing, **CYL_OPTS)
    half = cbm.cortical_bone_mapping(image, mask, ph.spacing, sub_mask=sub, **CYL_OPTS)
    assert 0.4 * full.n < half.n < 0.6 * full.n
    assert thickness_error(ph, half)[1] < 0.03


def test_fit_profiles_api():
    x = np.linspace(-1, 2, 61)
    truth = np.array([0.5, 0.8, 0.15, 60.0, 300.0])
    y = cbm.profile_model(x, truth, 1000.0)
    out = cbm.fit_profiles(np.stack([y, y]), x, residual_boost=1.0)
    np.testing.assert_allclose(out["params"], np.stack([truth, truth]), atol=1e-4)
    assert out["converged"].all()
    fixed = cbm.fit_profiles(y, x, init=(0, 1, 0.2, 0, 0), fixed=["sigma"])
    assert fixed["params"][0, 2] == 0.2
    with pytest.raises(ValueError):
        cbm.fit_profiles(y, x, fixed=["nope"])
