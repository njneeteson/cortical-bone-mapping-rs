use super::hull::{convex_hull, fill_convex};
use super::morphology::radii;
use super::*;
use ndarray::Array2;
use ndarray::{Array3, Axis};

fn brute_dilate(m: &Array3<bool>, k: [usize; 3]) -> Array3<bool> {
    let r = radii(k);
    let (d0, d1, d2) = m.dim();
    let ri = r.map(|v| v.floor() as isize);
    Array3::from_shape_fn(m.dim(), |(i, j, l)| {
        for a in -ri[0]..=ri[0] {
            for b in -ri[1]..=ri[1] {
                for c in -ri[2]..=ri[2] {
                    let e = [a, b, c];
                    let s: f64 = (0..3)
                        .map(|x| if r[x] > 0.0 { (e[x] as f64 / r[x]).powi(2) } else { 0.0 })
                        .sum();
                    if s > 1.0 + 1e-6 {
                        continue;
                    }
                    let (ii, jj, ll) = (i as isize + a, j as isize + b, l as isize + c);
                    if ii >= 0
                        && jj >= 0
                        && ll >= 0
                        && (ii as usize) < d0
                        && (jj as usize) < d1
                        && (ll as usize) < d2
                        && m[[ii as usize, jj as usize, ll as usize]]
                    {
                        return true;
                    }
                }
            }
        }
        false
    })
}

fn random_mask(seed: u64, dims: (usize, usize, usize), p: f64) -> Array3<bool> {
    let mut r = crate::util::Rng::new(seed);
    Array3::from_shape_fn(dims, |_| r.uniform() < p)
}

#[test]
fn dilation_matches_brute_force_ellipsoid() {
    for (seed, k) in [(1, [1, 5, 5]), (2, [3, 7, 4]), (3, [1, 15, 15]), (4, [5, 1, 2])] {
        let m = random_mask(seed, (9, 21, 23), 0.01);
        assert_eq!(dilate(&m, k), brute_dilate(&m, k), "kernel {k:?}");
    }
}

#[test]
fn erosion_is_dual_and_median_majority() {
    let m = random_mask(5, (6, 15, 15), 0.7);
    let e = erode(&m, [1, 5, 5]);
    assert!(e.iter().zip(m.iter()).all(|(&a, &b)| !a || b));
    let mut one = Array3::from_elem((1, 5, 5), false);
    one[[0, 2, 2]] = true;
    assert!(!binary_median(&one, [1, 3, 3]).iter().any(|&b| b));
    let full = Array3::from_elem((1, 5, 5), true);
    assert!(binary_median(&full, [1, 3, 3]).iter().all(|&b| b));
}

#[test]
fn components_outer_region_and_gaussian() {
    let mut m = Array3::from_elem((3, 10, 10), false);
    for j in 0..5 {
        m[[1, j, 1]] = true;
    }
    m[[1, 8, 8]] = true;
    let l = largest_component(&m, false);
    assert_eq!(l.iter().filter(|&&b| b).count(), 5);
    assert!(!l[[1, 8, 8]]);
    // a ring's outside is outer, its enclosed hole is not
    let ring = Array3::from_shape_fn((2, 12, 12), |(_, j, k)| {
        let r = ((j as f64 - 5.5).powi(2) + (k as f64 - 5.5).powi(2)).sqrt();
        r > 2.5 && r < 4.5
    });
    let bg = ring.mapv(|b| !b);
    let out = outer_region(&bg, false);
    assert!(out[[0, 0, 0]] && !out[[0, 6, 6]]);
    assert_eq!(out, outer_region(&bg, true));
    let g = gaussian_truncated(&Array3::from_elem((2, 6, 6), true), 3.0, [3, 9, 9]);
    assert!(g.iter().all(|&v| (v - 1.0).abs() < 1e-6));
}

#[test]
fn buie_on_a_ring_with_trabeculae_and_a_second_bone() {
    // cortex ring 1000 (r 24..30), marrow 100 with bright trabecular rods,
    // a separate small "fibula" disc, outside 0
    let (d1, d2) = (110usize, 100usize);
    let img = Array3::from_shape_fn((4, d1, d2), |(_, j, k)| {
        let r = ((j as f64 - 39.5).powi(2) + (k as f64 - 49.5).powi(2)).sqrt();
        let rf = ((j as f64 - 95.0).powi(2) + (k as f64 - 49.5).powi(2)).sqrt();
        if rf < 6.0 {
            1000.0f32
        } else if r > 30.0 {
            0.0
        } else if r > 24.0 {
            1000.0
        } else if (j % 9 == 0) && r < 20.0 {
            800.0
        } else {
            100.0
        }
    });
    let res = buie(img.view(), &BuieOptions::default());
    let n = |m: &Array3<bool>| m.iter().filter(|&&b| b).count() as f64;
    let peri_area = std::f64::consts::PI * 30.0f64.powi(2) * 4.0;
    assert!(
        (n(&res.periosteal) / peri_area - 1.0).abs() < 0.05,
        "{}",
        n(&res.periosteal) / peri_area
    );
    // the second bone is removed
    assert!(!res.periosteal[[1, 95, 49]]);
    // Step 12–13 (σ = 3, re-threshold at 100/127) moves the endosteal
    // boundary into the cortex by ≈ Φ⁻¹(100/127)·σ ≈ 0.8σ = 2.4 voxels on
    // a straight edge (a little less on this concave one).
    let r_in = (30.0f64.powi(2) - n(&res.cortical) / 4.0 / std::f64::consts::PI).sqrt();
    assert!(r_in > 24.5 && r_in < 26.9, "endosteal radius {r_in} (true 24)");
    // trabeculae are trabecular, not cortex
    assert!(!res.cortical[[1, 36, 49]] && res.trabecular[[1, 36, 49]]);
    // compartments partition the periosteal mask
    assert!(res
        .periosteal
        .iter()
        .zip(res.cortical.iter().zip(res.trabecular.iter()))
        .all(|(&p, (&c, &t))| p == (c || t) && !(c && t)));
}

#[test]
fn hull_contains_the_bone_and_closes_breaks() {
    // a ring with a wide gap, a speckle, a separate small disc
    let (n0, n1, n2) = (3, 80, 80);
    let img = Array3::from_shape_fn((n0, n1, n2), |(_, j, k)| {
        let (y, x) = (j as f64 - 35.0, k as f64 - 35.0);
        let r = (y * y + x * x).sqrt();
        let ring = (20.0..25.0).contains(&r) && !(x > 15.0 && y.abs() < 8.0);
        let disc = ((j as f64 - 72.0).powi(2) + (k as f64 - 72.0).powi(2)).sqrt() < 4.0;
        let speckle = j == 2 && k == 70;
        if ring || disc || speckle {
            1000.0f32
        } else {
            0.0
        }
    });
    let h = hull_segmentation(img.view(), &HullOptions::default());
    let thr = img.mapv(|v| v >= 320.0);
    for (t, (j, k)) in thr.indexed_iter().filter(|(_, &t)| t).map(|((_, j, k), t)| (t, (j, k))) {
        let in_ring = ((j as f64 - 35.0).powi(2) + (k as f64 - 35.0).powi(2)).sqrt() < 26.0;
        assert!(!*t || !in_ring || h[[1, j, k]], "ring pixel ({j},{k}) outside hull");
    }
    // the gap and the marrow are filled; the disc and the speckle are not included
    assert!(h[[1, 35, 35]] && h[[1, 35, 57]]);
    assert!(!h[[1, 72, 72]] && !h[[1, 2, 70]]);
    // the hull of a disc is about the disc
    let filled = h.index_axis(Axis(0), 1).iter().filter(|&&b| b).count() as f64;
    let area = std::f64::consts::PI * 24.5f64.powi(2);
    assert!((filled - area).abs() < 0.06 * area, "{filled} vs {area}");
}

#[test]
fn convex_hull_of_a_square_with_interior_points() {
    let pts = vec![(0.0, 0.0), (0.0, 4.0), (4.0, 0.0), (4.0, 4.0), (2.0, 2.0), (0.0, 2.0)];
    let h = convex_hull(&pts);
    assert_eq!(h.len(), 4);
    let mut out = Array2::from_elem((6, 6), false);
    fill_convex(&h, &mut out.view_mut());
    assert_eq!(out.iter().filter(|&&b| b).count(), 25);
}

#[test]
fn otsu_separates_a_trimodal_histogram() {
    // air (excluded by the floor), soft tissue ~0, bone ~1000 (fewer voxels)
    let mut r = crate::util::Rng::new(3);
    let img = Array3::from_shape_fn((4, 60, 60), |(_, j, k)| {
        let base = if j < 10 {
            -1000.0
        } else if k < 15 {
            1000.0
        } else if k < 35 {
            -100.0
        } else {
            40.0
        };
        (base + 20.0 * r.normal()) as f32
    });
    let o = OtsuOptions {
        floor: Some(-500.0),
        ..OtsuOptions::default()
    };
    let t2 = otsu_thresholds(img.view(), &o);
    assert_eq!(t2.len(), 1);
    assert!(t2[0] > 100.0 && t2[0] < 900.0, "{t2:?}");
    let t3 = otsu_thresholds(img.view(), &OtsuOptions { classes: 3, ..o });
    assert_eq!(t3.len(), 2);
    assert!(
        t3[0] > -80.0 && t3[0] < 20.0 && t3[1] > 100.0 && t3[1] < 900.0,
        "{t3:?}"
    );
    // nothing above the floor
    let none = otsu_thresholds(
        img.view(),
        &OtsuOptions {
            floor: Some(5000.0),
            ..OtsuOptions::default()
        },
    );
    assert!(none.is_empty());
}

/// Two "bones" facing each other: a C-shaped bone whose concavity holds
/// a small disc (as the trochlea holds the patella), and a speckle.
fn two_bones() -> Array3<f32> {
    Array3::from_shape_fn((4, 80, 80), |(_, j, k)| {
        let (y, x) = (j as f64 - 40.0, k as f64 - 35.0);
        let r = (y * y + x * x).sqrt();
        let c_shape = (15.0..22.0).contains(&r) && !(x > 5.0 && y.abs() < 12.0);
        let disc = ((j as f64 - 40.0).powi(2) + (k as f64 - 50.0).powi(2)).sqrt() < 5.0;
        let speckle = j == 3 && k == 3;
        if c_shape || disc || speckle {
            1000.0
        } else {
            0.0
        }
    })
}

#[test]
fn bones_are_labelled_by_size() {
    let img = two_bones();
    let b = label_bones(
        img.view(),
        &BoneLabelOptions {
            threshold: 500.0,
            min_size: 20,
            ..BoneLabelOptions::default()
        },
    );
    assert_eq!(b.sizes.len(), 2);
    assert!(b.sizes[0] > b.sizes[1]);
    assert_eq!(b.labels[[1, 40, 50]], 2); // the disc
    assert_eq!(b.labels[[1, 40, 15]], 1); // the C
    assert_eq!(b.labels[[1, 3, 3]], 0); // speckle removed (median and size)
    let one = label_bones(
        img.view(),
        &BoneLabelOptions {
            threshold: 500.0,
            min_size: 20,
            max_bones: Some(1),
            ..BoneLabelOptions::default()
        },
    );
    assert_eq!(one.sizes.len(), 1);
    assert_eq!(one.labels[[1, 40, 50]], 0);
}

#[test]
fn hysteresis_recovers_faint_cortex_and_keeps_bones_apart() {
    // two dense blocks (600) joined by a faint bridge (200); block 1 has
    // a faint rim (200); soft tissue 50
    let img = Array3::from_shape_fn((4, 30, 60), |(_, j, k)| {
        if (5..25).contains(&j) && ((5..25).contains(&k) || (35..55).contains(&k)) {
            600.0f32
        } else if ((12..18).contains(&j) && (25..35).contains(&k)) || ((2..5).contains(&j) && (5..25).contains(&k)) {
            200.0 // the bridge, and the rim of block 1
        } else {
            50.0
        }
    });
    let seeds = label_bones(
        img.view(),
        &BoneLabelOptions {
            threshold: 400.0,
            median_kernel: [1, 1, 1],
            min_size: 10,
            ..BoneLabelOptions::default()
        },
    );
    assert_eq!(seeds.sizes.len(), 2);
    let g = grow_bones(img.view(), seeds.labels.view(), 150.0, [1, 1, 1]);
    let (a, b) = (seeds.labels[[1, 15, 10]], seeds.labels[[1, 15, 45]]);
    assert!(a != 0 && b != 0 && a != b);
    assert_eq!(g[[1, 3, 10]], a); // the faint rim joins its bone
    assert_eq!(g[[1, 15, 26]], a); // the bridge is split between the bones
    assert_eq!(g[[1, 15, 33]], b);
    assert_eq!(g[[1, 28, 10]], 0); // soft tissue is not grown into
                                   // nothing at or above the low threshold is left unlabelled where connected
    assert!(g.iter().zip(img.iter()).all(|(&l, &v)| v < 150.0 || l != 0));
}

#[test]
fn bone_hulls_contain_their_bones_and_stay_apart() {
    let img = two_bones();
    let b = label_bones(
        img.view(),
        &BoneLabelOptions {
            threshold: 500.0,
            min_size: 20,
            ..BoneLabelOptions::default()
        },
    );
    let h = bone_hulls(b.labels.view(), [1.0, 1.0, 1.0], None, None);
    // every bone voxel is in its own hull
    assert!(b.labels.iter().zip(h.iter()).all(|(&l, &hv)| l == 0 || hv == l));
    // the C's hull would cover the disc; the disc keeps it, and the C's
    // hull still closes the C's own marrow
    assert_eq!(h[[1, 40, 35]], 1);
    assert_eq!(h[[1, 40, 50]], 2);
    // trimmed to 3 voxels from the bone: the C's wide mouth (24 voxels) is
    // an open concavity, so its inside opens up; the bone stays in
    let t = bone_hulls(b.labels.view(), [1.0, 1.0, 1.0], Some(3.0), None);
    assert!(b.labels.iter().zip(t.iter()).all(|(&l, &hv)| l == 0 || hv == l));
    assert_eq!(t[[1, 40, 35]], 0);
    // a ring with a narrow (4 voxel) break keeps its marrow
    let ring = Array3::from_shape_fn((4, 60, 60), |(_, j, k)| {
        let (y, x) = (j as f64 - 30.0, k as f64 - 30.0);
        let r = (y * y + x * x).sqrt();
        ((15.0..20.0).contains(&r) && !(x > 5.0 && y.abs() < 2.0)) as u16
    });
    let tr = bone_hulls(ring.view(), [1.0, 1.0, 1.0], Some(3.0), None);
    assert_eq!(tr[[1, 30, 30]], 1);
    assert_eq!(tr[[1, 30, 47]], 1); // the break is closed
    assert_eq!(tr[[1, 2, 2]], 0);
    // closing by a 3-voxel ball: the C's 24-voxel mouth stays open, the
    // ring's 4-voxel break closes and its marrow is filled
    let c = bone_hulls(b.labels.view(), [1.0, 1.0, 1.0], None, Some(3.0));
    assert!(b.labels.iter().zip(c.iter()).all(|(&l, &cv)| l == 0 || cv == l));
    assert_eq!(c[[1, 40, 35]], 0);
    let cr = bone_hulls(ring.view(), [1.0, 1.0, 1.0], None, Some(3.0));
    assert_eq!(cr[[1, 30, 47]], 1);
    assert_eq!(cr[[1, 30, 30]], 1);
    assert_eq!(cr[[1, 2, 2]], 0);
    // no voxel of one hull (or closing; a 6-voxel closing of the C would
    // reach the disc) 6-touches another
    let wide = bone_hulls(b.labels.view(), [1.0, 1.0, 1.0], None, Some(6.0));
    assert!(b.labels.iter().zip(wide.iter()).all(|(&l, &cv)| l == 0 || cv == l));
    for h in [&h, &c, &wide] {
        let (d0, d1, d2) = h.dim();
        for ((i, j, k), &v) in h.indexed_iter() {
            if v == 0 {
                continue;
            }
            for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                let (a, bb, cc) = (i + di, j + dj, k + dk);
                if a < d0 && bb < d1 && cc < d2 {
                    let w = h[[a, bb, cc]];
                    assert!(w == 0 || w == v, "labels {v} and {w} touch at ({i},{j},{k})");
                }
            }
        }
    }
}
