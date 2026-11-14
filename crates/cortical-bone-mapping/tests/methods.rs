//! Accuracy of every method on phantoms with known cortical geometry.

mod common;
use common::*;
use cortical_bone_mapping::*;

#[test]
fn local_method_is_accurate_on_a_noise_free_cylinder() {
    let ph = cylinder(0.0, 0);
    let (img, mask) = ph.generate();
    let r = run(img.view(), mask.view(), NO_SUB, &options(&ph, Method::Local)).unwrap();
    let (bias, rmse) = thickness_error(&ph, &r);
    assert!(bias.abs() < 0.01 && rmse < 0.015, "bias {bias} rmse {rmse}");
    assert!(periosteal_error(&ph, &r) < 0.01);
    assert!(r.info.converged > 0.99);
    // blur is recovered (PSF 0.12 plus trilinear/partial-volume blur)
    let s = r.param(model::SIGMA);
    let ms = s.iter().sum::<f64>() / s.len() as f64;
    assert!((ms - 0.12).abs() < 0.03, "mean sigma {ms}");
    // densities
    let rs = r.param(model::RHO_S);
    let rb = r.param(model::RHO_B);
    assert!((rs.iter().sum::<f64>() / rs.len() as f64 - 50.0).abs() < 10.0);
    assert!((rb.iter().sum::<f64>() / rb.len() as f64 - 250.0).abs() < 10.0);
}

#[test]
fn all_methods_are_accurate_on_a_noisy_cylinder() {
    let ph = cylinder(40.0, 1);
    let (img, mask) = ph.generate();
    let cases = [
        ("local", Method::Local, 0.03),
        (
            "multigrid",
            Method::Multigrid {
                spacings: vec![4.0, 2.0, 1.0],
                neighbours: 5,
                seed: 0,
            },
            0.03,
        ),
        ("regularized", regularized(100, None), 0.02),
        (
            "curvature-change",
            regularized(100, Some(CurvatureOptions::default())),
            0.02,
        ),
        (
            "curvature-excess",
            regularized(
                100,
                Some(CurvatureOptions {
                    mode: CurvatureMode::Excess { kappa: 0.5 },
                    ..Default::default()
                }),
            ),
            0.02,
        ),
    ];
    for (name, method, tol) in cases {
        let r = run(img.view(), mask.view(), NO_SUB, &options(&ph, method)).unwrap();
        let (bias, rmse) = thickness_error(&ph, &r);
        assert!(bias.abs() < 0.01, "{name}: bias {bias}");
        assert!(rmse < tol, "{name}: rmse {rmse}");
    }
}

#[test]
fn spatial_constraints_improve_precision_under_heavy_noise() {
    let ph = cylinder(120.0, 2);
    let (img, mask) = ph.generate();
    let local = run(img.view(), mask.view(), NO_SUB, &options(&ph, Method::Local)).unwrap();
    let reg = run(img.view(), mask.view(), NO_SUB, &options(&ph, regularized(100, None))).unwrap();
    let curv = run(
        img.view(),
        mask.view(),
        NO_SUB,
        &options(&ph, regularized(100, Some(CurvatureOptions::default()))),
    )
    .unwrap();
    let mg = run(
        img.view(),
        mask.view(),
        NO_SUB,
        &options(
            &ph,
            Method::Multigrid {
                spacings: vec![4.0, 2.0, 1.0],
                neighbours: 5,
                seed: 0,
            },
        ),
    )
    .unwrap();
    let e_local = thickness_error(&ph, &local).1;
    for (name, r) in [("regularized", &reg), ("curvature", &curv), ("multigrid", &mg)] {
        let e = thickness_error(&ph, r).1;
        assert!(e < 0.6 * e_local, "{name}: rmse {e} vs local {e_local}");
    }
}

#[test]
fn sphere_without_normal_constraint() {
    let ph = sphere(30.0);
    let (img, mask) = ph.generate();
    for method in [Method::Local, regularized(64, None)] {
        let r = run(img.view(), mask.view(), NO_SUB, &options(&ph, method.clone())).unwrap();
        let (bias, rmse) = thickness_error(&ph, &r);
        assert!(
            bias.abs() < 0.02 && rmse < 0.05,
            "{}: bias {bias} rmse {rmse}",
            method.name()
        );
        assert!(periosteal_error(&ph, &r) < 0.03);
    }
}

#[test]
fn fixed_sigma_and_psf_modes() {
    let ph = synthetic::ShellPhantom {
        blur: [0.2, 0.1, 0.1],
        dims: (64, 72, 72),
        shape: synthetic::Shape::Sphere,
        radius: 2.8,
        thickness_mean: 0.9,
        thickness_amplitude: 0.2,
        thickness_lobes: 1.0,
        noise: 10.0,
        ..Default::default()
    };
    let (img, mask) = ph.generate();
    let mut o = options(&ph, Method::Local);
    o.model.sigma = SigmaMode::EstimatePsf {
        model: PsfModel::Axial(0),
        min_ratio: 3.0,
    };
    let r = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
    let psf = r.psf.unwrap();
    // effective blur includes ~0.03 of partial-volume/interpolation blur
    assert!(psf[0] > psf[1] + 0.04, "{psf:?}");
    assert!((psf[0] - 0.2).abs() < 0.05 && (psf[1] - 0.1).abs() < 0.04, "{psf:?}");
    // σ is then fixed per node from the PSF
    let s = r.param(model::SIGMA);
    for (sv, n) in s.iter().zip(&r.normals) {
        assert!((sv - psf::sigma_along(psf, *n)).abs() < 1e-12);
    }
    let (bias, rmse) = thickness_error(&ph, &r);
    assert!(bias.abs() < 0.03 && rmse < 0.05, "bias {bias} rmse {rmse}");
    // a fixed scalar σ is honoured exactly
    o.model.sigma = SigmaMode::Fixed(0.15);
    let r = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
    assert!(r.param(model::SIGMA).iter().all(|&v| v == 0.15));
    // the regularized method with a PSF
    o.model.sigma = SigmaMode::Psf(psf);
    o.method = regularized(64, None);
    let r = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
    assert!(thickness_error(&ph, &r).1 < 0.05);
}

#[test]
fn excess_curvature_with_huge_threshold_is_a_no_op() {
    let ph = cylinder(40.0, 3);
    let (img, mask) = ph.generate();
    let a = run(img.view(), mask.view(), NO_SUB, &options(&ph, regularized(64, None))).unwrap();
    let b = run(
        img.view(),
        mask.view(),
        NO_SUB,
        &options(
            &ph,
            regularized(
                64,
                Some(CurvatureOptions {
                    mode: CurvatureMode::Excess { kappa: 1e6 },
                    weight: 100.0,
                    ..Default::default()
                }),
            ),
        ),
    )
    .unwrap();
    assert_eq!(b.info.curvature_cost, 0.0);
    let d = a
        .thickness()
        .iter()
        .zip(b.thickness())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    // identical optimum; only the preconditioner differs (solver tolerance)
    assert!(d < 1e-4, "max difference {d}");
}

#[test]
fn curvature_penalty_reduces_surface_roughness() {
    let ph = cylinder(150.0, 4);
    let (img, mask) = ph.generate();
    let roughness = |r: &CbmResult| -> f64 {
        // RMS of the endosteal offset minus its neighbourhood mean
        let mesh = Mesh {
            vertices: r.points.clone(),
            faces: r.faces.clone(),
        };
        let adj = mesh.adjacency();
        let d: Vec<f64> = r.params.iter().map(|p| p[model::M] + 0.5 * p[model::T]).collect();
        let mut s = 0.0;
        for i in 0..d.len() {
            let ring = adj.of(i);
            if ring.is_empty() {
                continue;
            }
            let mean = ring.iter().map(|&j| d[j as usize]).sum::<f64>() / ring.len() as f64;
            s += (d[i] - mean).powi(2);
        }
        (s / d.len() as f64).sqrt()
    };
    let local = run(img.view(), mask.view(), NO_SUB, &options(&ph, Method::Local)).unwrap();
    let (r0, e0) = (roughness(&local), thickness_error(&ph, &local).1);
    // Gaussian (default scale) and cotangent Laplacians, curvature penalty only
    for (scale, weight) in [(Some(0.25), 1.0), (None, 0.1)] {
        let c = CurvatureOptions {
            weight,
            scale,
            ..Default::default()
        };
        let mut o = options(&ph, regularized(1, Some(c)));
        if let Method::Regularized { lambda, .. } = &mut o.method {
            *lambda = 0.0;
        }
        let r = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
        let (rough, (bias, rmse)) = (roughness(&r), thickness_error(&ph, &r));
        assert!(rough < 0.5 * r0, "scale {scale:?}: roughness {rough} vs local {r0}");
        assert!(rmse < 0.5 * e0, "scale {scale:?}: rmse {rmse} vs local {e0}");
        assert!(bias.abs() < 0.01);
    }
}
