//! Inputs, options, errors, determinism and sub-masks.

mod common;
use common::*;
use cortical_bone_mapping::*;
use ndarray::Array3;

#[test]
fn errors() {
    let ph = cylinder(0.0, 0);
    let (img, mask) = ph.generate();
    let o = options(&ph, Method::Local);
    let empty = Array3::<u8>::zeros(mask.dim());
    assert_eq!(run(img.view(), empty.view(), NO_SUB, &o).unwrap_err(), Error::EmptyMask);
    let small = Array3::<u8>::zeros((2, 2, 2));
    assert!(matches!(
        run(img.view(), small.view(), NO_SUB, &o),
        Err(Error::ShapeMismatch(_))
    ));
    assert!(matches!(
        run(img.view(), mask.view(), Some(small.view()), &o),
        Err(Error::ShapeMismatch(_))
    ));
    let mut bad = o.clone();
    bad.geometry.spacing[1] = 0.0;
    assert!(matches!(
        run(img.view(), mask.view(), NO_SUB, &bad),
        Err(Error::InvalidOption(_))
    ));
    let mut bad = o.clone();
    bad.model.lower[model::T] = 5.0;
    bad.model.upper[model::T] = 1.0;
    assert!(matches!(
        run(img.view(), mask.view(), NO_SUB, &bad),
        Err(Error::InvalidOption(_))
    ));
    let mut bad = o.clone();
    bad.threads = Some(0);
    assert!(run(img.view(), mask.view(), NO_SUB, &bad).is_err());
    // an empty sub-mask leaves nothing to fit
    assert_eq!(
        run(img.view(), mask.view(), Some(empty.view()), &o).unwrap_err(),
        Error::NoNodes
    );
}

#[test]
fn integer_images_match_float_images() {
    let ph = cylinder(20.0, 1);
    let (img, mask) = ph.generate();
    let as_i16 = img.mapv(|v| v.round() as i16);
    let as_f32 = as_i16.mapv(|v| v as f32);
    let o = options(&ph, Method::Local);
    let a = run(as_i16.view(), mask.view(), NO_SUB, &o).unwrap();
    let b = run(as_f32.view(), mask.mapv(|v| v != 0).view(), NO_SUB, &o).unwrap();
    assert_eq!(a.thickness(), b.thickness());
}

#[test]
fn results_are_deterministic_and_thread_count_independent() {
    let ph = cylinder(40.0, 2);
    let (img, mask) = ph.generate();
    for method in [
        Method::Local,
        Method::Multigrid {
            spacings: vec![3.0, 1.5],
            neighbours: 5,
            seed: 7,
        },
        regularized(32, None),
        Method::Volumetric(VolumetricOptions {
            sigma_z: AxialBlur::Tied,
            ..Default::default()
        }),
    ] {
        let mut o = options(&ph, method.clone());
        let a = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
        let b = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
        o.threads = Some(1);
        let c = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
        let maxdiff = |x: &CbmResult, y: &CbmResult| {
            x.thickness()
                .iter()
                .zip(y.thickness())
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max)
        };
        assert!(maxdiff(&a, &b) < 1e-9, "{} not deterministic", method.name());
        // parallel reductions may reorder floating-point sums
        assert_eq!(
            maxdiff(&a, &c),
            0.0,
            "{}: result depends on the thread count",
            method.name()
        );
    }
}

#[test]
fn sub_mask_restricts_the_measured_region() {
    let ph = cylinder(20.0, 3);
    let (img, mask) = ph.generate();
    let (_, d1, _) = mask.dim();
    // only the half with axis-1 index < d1/2
    let sub = Array3::from_shape_fn(mask.dim(), |(_, j, _)| (j < d1 / 2) as u8);
    let o = options(&ph, regularized(32, None));
    let full = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
    let half = run(img.view(), mask.view(), Some(sub.view()), &o).unwrap();
    assert!(half.n() < full.n() * 6 / 10 && half.n() > full.n() * 4 / 10);
    let lim = (d1 as f64 / 2.0 + 1.0) * ph.spacing[1];
    assert!(half.points.iter().all(|p| p[1] < lim));
    assert!(thickness_error(&ph, &half).1 < 0.03);
    // faces index the fitted points
    assert!(half.faces.iter().all(|f| f.iter().all(|&i| (i as usize) < half.n())));
}

#[test]
fn border_caps_are_excluded_and_profiles_kept_on_request() {
    let ph = cylinder(0.0, 0);
    let (img, mask) = ph.generate();
    let mut o = options(&ph, Method::Local);
    o.keep_profiles = true;
    o.sampling.normal_constraint = NormalConstraint::None;
    let r = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
    let (x, prof) = r.profiles.as_ref().unwrap();
    assert_eq!(prof.n, r.n());
    assert_eq!(prof.m, x.len());
    // no node lies on a cap (outside the image along axis 0)
    let len = (ph.dims.0 as f64 - 1.0) * ph.spacing[0];
    assert!(r.points.iter().all(|p| p[0] >= 0.0 && p[0] <= len));
    o.surface.exclude_border = false;
    let r2 = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
    assert!(r2.n() > r.n());
}

#[test]
fn vtk_output() {
    let ph = cylinder(0.0, 0);
    let (img, mask) = ph.generate();
    let r = run(img.view(), mask.view(), NO_SUB, &options(&ph, Method::Local)).unwrap();
    let path = std::env::temp_dir().join("cbm_result_test.vtk");
    r.write_vtk(&path).unwrap();
    let text = std::fs::read(&path).unwrap();
    let s = String::from_utf8_lossy(&text);
    assert!(s.contains("SCALARS thickness double") && s.contains("VECTORS normals double"));
    let _ = std::fs::remove_file(path);
}
