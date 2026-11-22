"""Realistic phantoms, the starting segmentations (Buie's dual threshold and
the threshold hull), and automatic segmentation with CBM from the image alone."""

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


def test_hull_contains_the_bone(osteoporotic):
    ph, img, peri, endo = osteoporotic
    h = cbm.hull_segmentation(img)
    assert (h & peri).sum() / peri.sum() > 0.998  # a superset of the bone ...
    assert dice(h, peri) > 0.97  # ... and a close one


def test_hull_then_refinement_beats_buie_on_a_porous_thin_cortex(osteoporotic):
    """From the image alone, without the true density: hull, then CBM
    refinement. Buie classifies the porous inner cortex as marrow."""
    ph, img, peri, endo = osteoporotic
    cort = peri & ~endo
    s = slice(3, -3)
    _, bc, _ = cbm.buie_segmentation(img)
    rr = cbm.refine_segmentation(img, cbm.hull_segmentation(img), rho_c="estimate", **CBM)
    assert rr.converged
    assert 650 < rr.result.rho_c < 850  # the porous cortex's apparent density
    d_buie, d_cbm = dice(bc[s], cort[s]), dice(rr.cortical_mask[s], cort[s])
    assert d_buie < 0.85 and d_cbm > 0.9, (d_buie, d_cbm)
    assert dice(rr.periosteal_mask[s], peri[s]) > 0.99
    t = ph.true_thickness(rr.result)
    ok = np.isfinite(t)
    assert abs((rr.result.thickness[ok] - t[ok]).mean()) < 0.08


def test_rho_c_modes(osteoporotic):
    ph, img, peri, endo = osteoporotic
    bp, _, _ = cbm.buie_segmentation(img)
    pm = cbm.cortical_bone_mapping(img, bp, rho_c="profile_max", **CBM)
    assert pm.rho_c_node is not None and pm.rho_c_node.shape == (pm.n,)
    with pytest.raises(ValueError):
        cbm.cortical_bone_mapping(img, bp, rho_c="max", **CBM)


def test_presets():
    o = cbm.Options.preset("radius_tibia")
    assert (o.outside, o.inside, o.profile_smoothing, o.normal_constraint) == (5.0, 15.0, 3.0, "in_plane")
    assert o.reg_neighbours == 500 and abs(o.reg_sigma - 32 * 0.0607) < 1e-3
    k = cbm.Options.preset("knee", outside=1.0)
    assert k.normal_constraint == "along_axis" and k.reg_neighbours == 1000 and k.outside == 1.0
    assert np.allclose(np.array(k.multigrid_spacings) / 0.0607, [256, 128, 64, 32, 16])
    with pytest.raises(ValueError):
        cbm.Options.preset("hip")


def test_neighbour_and_breaks_phantom():
    ph = cbm.BonePhantom(dims=(8, 240, 240), noise=0.0, neighbour=(1.5, 0.5, 0.5, np.pi / 4),
                         breaks=[(2.0, 1.0, 0.24, 0.3)])
    img, peri, endo = ph.generate()
    lab = ph.labels()
    assert (lab == 1).any() and not (peri & (lab == 1)).any()
    assert (lab == 2).any() and (peri[lab == 2]).all() and not endo[lab == 2].any()
    cortex = peri & ~endo & (lab == 0)
    assert img[lab == 2].mean() < 0.5 * img[cortex].mean()  # soft tissue in the break
    # the paper's largest-bone pre-processing keeps the neighbour out at a 0.5 mm gap,
    # and so does the hull's largest-component step
    bp, _, _ = cbm.buie_segmentation(img)
    assert bp[lab == 1].mean() < 0.05
    assert cbm.hull_segmentation(img)[lab == 1].mean() < 0.05
    # the hull spans the break; Buie may not
    assert cbm.hull_segmentation(img)[lab == 2].all()


def test_otsu_label_and_hull_two_bones():
    ph = cbm.BonePhantom(dims=(12, 240, 240), noise=40.0, neighbour=(1.5, 0.5, 0.5, np.pi / 4))
    img, peri, _ = ph.generate()
    t = cbm.otsu_thresholds(img)
    assert len(t) == 1 and 100 < t[0] < 900
    t3 = cbm.otsu_thresholds(img, classes=3)
    assert len(t3) == 2 and t3[0] < t3[1]
    lab, sizes = cbm.label_bones(img, t[0], min_size=5000)
    lab2, _ = cbm.label_bones(img, "otsu", min_size=5000)
    assert (lab2 == lab).all()
    assert lab.dtype == np.uint16 and len(sizes) == 2 and sizes[0] > sizes[1]
    truth = ph.labels()
    nb = lab[truth == 1]
    assert (nb[nb > 0] == 2).mean() > 0.99  # the neighbour (bone voxels) is the smaller bone
    hulls = cbm.bone_hulls(lab, ph.spacing)
    # names from landmarks: a point in the main marrow (inside the hull, not the bone) and one on the neighbour
    sp = np.asarray(ph.spacing)
    centre = (np.array(img.shape) - 1) / 2 * sp
    on_nb = np.argwhere(lab == 2)[0] * sp
    assert cbm.name_bones(hulls, {"radius": centre, "ulna": on_nb}, ph.spacing) == {1: "radius", 2: "ulna"}
    far = centre + np.array([0.0, 100.0, 0.0])  # outside the image: nearest bone
    assert cbm.name_bones(lab, {"x": far}, ph.spacing) in ({1: "x"}, {2: "x"})
    with pytest.raises(ValueError):
        cbm.name_bones(lab, {"a": centre, "b": centre}, ph.spacing)
    assert ((lab == 0) | (hulls == lab)).all()  # each hull contains its bone
    assert (hulls[peri] == 1).mean() > 0.97  # and the main hull is the main bone
    # per-bone fits: cropping to each bone gives the same result as the full image
    o = cbm.Options(spacing=ph.spacing, outside=0.4, inside=1.5, method="regularized", reg_neighbours=64,
                    rho_c="estimate")
    full = cbm.map_bones(img, hulls, options=o, crop=False)
    crop = cbm.map_bones(img, hulls, options=o)
    assert set(full) == set(crop) == {1, 2}
    for b in (1, 2):
        d = np.abs(crop[b].thickness - full[b].thickness)  # same surface; float rounding + solver tolerance
        assert crop[b].n == full[b].n and np.median(d) < 1e-4 and d.mean() < 2e-3
        assert crop[b].options.origin == o.origin
    assert full[1].n > full[2].n
    # refinement per bone, with a per-bone override; masks at the full shape
    rr = cbm.refine_bones(img, hulls, options=o, bones=[2], refine_iterations=2,
                          per_bone={2: dict(rho_c=1000.0)})
    assert list(rr) == [2] and rr[2].periosteal_mask.shape == img.shape
    assert rr[2].result.rho_c == 1000.0
    assert not rr[2].periosteal_mask[peri].any()


def test_separate_bones_raises_the_threshold_locally():
    # a neighbour 0.15 mm away, blurred: the two bones join at a low threshold
    ph = cbm.BonePhantom(dims=(8, 240, 240), noise=20.0, neighbour=(1.5, 0.5, 0.15, np.pi / 4), blur=(0.12,) * 3)
    img, peri, _ = ph.generate()
    sp = np.asarray(ph.spacing)
    truth = ph.labels()
    marks = {"main": np.argwhere(peri & (img > 600))[0] * sp, "neighbour": np.argwhere((truth == 1) & (img > 600))[0] * sp}
    lab, sizes = cbm.label_bones(img, 150.0, min_size=500)
    with pytest.raises(ValueError, match="joined two bones"):
        cbm.name_bones(lab, marks, ph.spacing)
    lab, names, used = cbm.separate_bones(img, 150.0, marks, ph.spacing, min_size=500, verbose=False)
    assert sorted(names.values()) == ["main", "neighbour"]
    assert used["main"] == used["neighbour"] > 150.0
    inv = {v: k for k, v in names.items()}
    assert inv["main"] == 1 and (lab[truth == 1] == inv["main"]).mean() < 0.01
    with pytest.raises(ValueError, match="could not be separated"):
        cbm.separate_bones(img, 150.0, marks, ph.spacing, min_size=500, max_threshold=160.0, verbose=False)


def test_joint_phantom_truth():
    ph = cbm.JointPhantom(dims=(100, 120, 120), radius=3.0, joint_space=2.0, joint_level=0.4, noise=0.0)
    img, peri, endo = ph.generate()
    assert set(np.unique(peri)) == {0, 1, 2}
    assert ((endo > 0) <= (peri > 0)).all()
    # vertical lines through the joint space: down to the tibia, up to the femur
    c = (np.array(img.shape) - 1) / 2 * 0.0607
    z = ph.joint_height() + 0.5 * ph.joint_space
    pts = np.array([[z, c[1] + ph.contact_offset[0] + dy, c[2] + ph.contact_offset[1]] for dy in (-0.5, 0, 0.5)])
    down = ph.edges_along(pts, np.tile([-1.0, 0, 0], (3, 1)), 0.0, 2.0)
    up = ph.edges_along(pts, np.tile([1.0, 0, 0], (3, 1)), 0.0, 2.0)
    assert (down[:, 2] == cbm.JointPhantom.TIBIA).all() and (up[:, 2] == cbm.JointPhantom.FEMUR).all()
    plate = down[:, 1] - down[:, 0]
    assert (plate > ph.plate_min - 1e-6).all() and (plate < 1.0).all()
