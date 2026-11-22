//! Model-driven segmentation refinement.

mod common;
use common::*;
use cortical_bone_mapping::synthetic::ShellPhantom;
use cortical_bone_mapping::*;

fn bad_phantom(
    offset: f64,
    bump: f64,
) -> (
    ShellPhantom,
    ndarray::Array3<f32>,
    ndarray::Array3<u8>,
    ndarray::Array3<u8>,
) {
    let ph = ShellPhantom {
        mask_offset: offset,
        mask_bump: bump,
        noise: 30.0,
        ..ShellPhantom::default()
    };
    let (img, bad) = ph.generate();
    let truth = ShellPhantom {
        mask_offset: 0.0,
        mask_bump: 0.0,
        ..ph.clone()
    }
    .generate()
    .1;
    (ph, img, bad, truth)
}

#[test]
fn snaps_dilated_and_bumpy_segmentation_to_the_periosteum() {
    let (ph, img, bad, truth) = bad_phantom(0.35, 0.25);
    let o = options(&ph, regularized(64, None));
    let r = refine_segmentation(img.view(), bad.view(), NO_SUB, &o, &RefineOptions::default()).unwrap();
    assert!(r.converged);
    let before = dice(&bad.mapv(|v| v != 0), &truth);
    let after = dice(&r.periosteal_mask, &truth);
    assert!(before < 0.92 && after > 0.995, "dice {before} -> {after}");
    assert!(periosteal_error(&ph, &r.result) < 0.02);
    assert!(thickness_error(&ph, &r.result).1 < 0.03);
    // under-relaxed fixed-point iteration: displacement shrinks every step
    for w in r.history.windows(2) {
        assert!(w[1].rms_displacement < w[0].rms_displacement, "{:?}", r.history);
    }
    // cortical area per slice (away from the caps, where the endosteal
    // surface closes through the unmoved cap) matches the true annulus:
    // (1/2)∫(2R t − t²) dθ = π (2R t̄ − t̄² − a²/2)
    let (d0, _, _) = ph.dims;
    let interior = 4..d0 - 4;
    let cort: usize = interior
        .clone()
        .map(|i| {
            r.cortical_mask
                .index_axis(ndarray::Axis(0), i)
                .iter()
                .filter(|&&b| b)
                .count()
        })
        .sum();
    let (rr, t, a) = (ph.radius, ph.thickness_mean, ph.thickness_amplitude);
    let area = std::f64::consts::PI * (2.0 * rr * t - t * t - 0.5 * a * a);
    let want = area / (ph.spacing[1] * ph.spacing[2]) * interior.len() as f64;
    assert!(
        (cort as f64 / want - 1.0).abs() < 0.02,
        "cortical voxels {cort} vs {want}"
    );
}

#[test]
fn level_set_refinement_remeshes_and_snaps_to_the_periosteum() {
    let (ph, img, bad, truth) = bad_phantom(0.35, 0.25);
    let o = options(&ph, regularized(64, None));
    let ro = RefineOptions {
        level_set: true,
        keep_history: true,
        verbose: false,
        ..RefineOptions::default()
    };
    let r = refine_segmentation(img.view(), bad.view(), NO_SUB, &o, &ro).unwrap();
    assert!(r.converged, "{:?}", r.history);
    let after = dice(&r.periosteal_mask, &truth);
    assert!(after > 0.995, "dice {after}");
    assert!(periosteal_error(&ph, &r.result) < 0.025);
    assert!(thickness_error(&ph, &r.result).1 < 0.035);
    // every iteration has its own closed mesh
    for s in &r.surfaces {
        assert!(s.is_closed_manifold());
    }
    let counts: std::collections::HashSet<usize> = r.surfaces.iter().map(|s| s.vertices.len()).collect();
    assert!(counts.len() > 1, "the mesh was never re-extracted");
}

#[test]
fn regional_convergence_freezes_still_regions_and_refits_all_at_the_end() {
    let (ph, img, bad, truth) = bad_phantom(0.35, 0.25);
    let o = options(&ph, regularized(64, None));
    for level_set in [false, true] {
        let ro = RefineOptions {
            level_set,
            verbose: false,
            freeze: Some(FreezeOptions {
                patience: 1,
                displacement: Some(0.02),
                thickness: Some(0.02),
                ..FreezeOptions::default()
            }),
            ..RefineOptions::default()
        };
        let r = refine_segmentation(img.view(), bad.view(), NO_SUB, &o, &ro).unwrap();
        assert!(r.converged, "{:?}", r.history);
        assert!(
            r.history.iter().any(|h| h.moving_nodes < h.nodes),
            "nothing froze: {:?}",
            r.history
        );
        // the final fit covers the whole surface
        assert_eq!(r.result.n(), r.history.last().unwrap().nodes);
        let after = dice(&r.periosteal_mask, &truth);
        assert!(after > 0.99, "level set {level_set}: dice {after}");
        assert!(periosteal_error(&ph, &r.result) < 0.03);
        assert!(thickness_error(&ph, &r.result).1 < 0.04);
    }
}

#[test]
fn refinement_never_enters_an_excluded_region() {
    // a dilated start next to an excluded slab (e.g. another bone) that cuts
    // into the bone along one side: the surface must stay out of it
    let (ph, img, bad, truth) = bad_phantom(0.35, 0.0);
    let o = options(&ph, regularized(64, None));
    let (_, d1, _) = ph.dims;
    let ex = ndarray::Array3::from_shape_fn(ph.dims, |(_, j, _)| (j >= d1 * 3 / 4) as u8);
    let bad = {
        let mut b = bad.clone();
        b.zip_mut_with(&ex, |v, &e| {
            if e != 0 {
                *v = 0
            }
        });
        b
    };
    for level_set in [true, false] {
        let ro = RefineOptions {
            level_set,
            verbose: false,
            ..RefineOptions::default()
        };
        let r = refine_segmentation_excluding(img.view(), bad.view(), NO_SUB, Some(ex.view()), &o, &ro).unwrap();
        let inside = r
            .periosteal_mask
            .iter()
            .zip(ex.iter())
            .filter(|(&p, &e)| p && e != 0)
            .count();
        assert_eq!(
            inside, 0,
            "level set {level_set}: {inside} voxels inside the excluded region"
        );
        // away from the slab the surface is still right
        let free = |a: &ndarray::Array3<bool>| a.indexed_iter().filter(|((_, j, _), &v)| v && *j < d1 / 2).count();
        let (got, want) = (free(&r.periosteal_mask), free(&truth.mapv(|v| v != 0)));
        assert!(
            (got as f64 / want as f64 - 1.0).abs() < 0.01,
            "level set {level_set}: {got} vs {want} voxels"
        );
    }
}

#[test]
fn eroded_segmentation_grows_outward() {
    let (ph, img, bad, truth) = bad_phantom(-0.3, 0.0);
    let o = options(&ph, Method::Local);
    let r = refine_segmentation(img.view(), bad.view(), NO_SUB, &o, &RefineOptions::default()).unwrap();
    assert!(dice(&r.periosteal_mask, &truth) > 0.995);
}

#[test]
fn history_is_kept_on_request_and_full_steps_also_work() {
    let (ph, img, bad, _) = bad_phantom(0.3, 0.0);
    let o = options(&ph, Method::Local);
    let ro = RefineOptions {
        keep_history: true,
        verbose: false,
        relaxation: 1.0,
        max_iterations: 8,
        ..Default::default()
    };
    let r = refine_segmentation(img.view(), bad.view(), NO_SUB, &o, &ro).unwrap();
    // one history entry per fit: the input surface and every moved one
    assert_eq!(r.surfaces.len(), r.history.len());
    assert!(r.history[0].full_fit);
    assert!(!r.converged || r.history.last().unwrap().full_fit);
    assert!(periosteal_error(&ph, &r.result) < 0.03);
    let bad_relax = RefineOptions {
        relaxation: 0.0,
        ..Default::default()
    };
    assert!(refine_segmentation(img.view(), bad.view(), NO_SUB, &o, &bad_relax).is_err());
}

#[test]
fn masks_from_a_single_fit() {
    let ph = cylinder(20.0, 5);
    let (img, mask) = ph.generate();
    let o = options(&ph, Method::Local);
    let r = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
    let [peri, endo, cort] = cortical_masks(&r, img.dim(), &o);
    assert!(dice(&peri, &mask) > 0.99);
    let n = |m: &ndarray::Array3<bool>| m.iter().filter(|&&b| b).count();
    assert_eq!(n(&cort), n(&peri) - n(&endo));
}

#[test]
fn lobed_bone_with_gross_bumps_converges() {
    // Steep 7-lobed segmentation bumps on a 3-lobed bone: some profiles on the
    // bump flanks miss the cortex; refinement must not trust those fits.
    let ph = ShellPhantom {
        dims: (30, 128, 128),
        radius: 4.0,
        radius_bump: 0.5,
        radius_lobes: 3.0,
        mask_offset: 0.45,
        mask_bump: 0.35,
        mask_lobes: 7.0,
        noise: 40.0,
        ..ShellPhantom::default()
    };
    let (img, bad) = ph.generate();
    let truth = ShellPhantom {
        mask_offset: 0.0,
        mask_bump: 0.0,
        ..ph.clone()
    }
    .generate()
    .1;
    let o = options(&ph, regularized(64, None));
    let r = refine_segmentation(img.view(), bad.view(), NO_SUB, &o, &RefineOptions::default()).unwrap();
    assert!(r.converged, "{:?} {:?}", r.history, r.result.info);
    let after = dice(&r.periosteal_mask, &truth);
    assert!(after > 0.995, "dice {after}");
    for w in r.history.windows(2) {
        assert!(w[1].rms_displacement < w[0].rms_displacement, "{:?}", r.history);
    }
}
