//! The solvers' stopping tests do not depend on the units of the image
//! values (nor where they put their zero) or of lengths: the same problem in
//! other units takes the same iterations to the same solution.
//!
//! Lengths are scaled by 1024, which floating point represents exactly, so
//! the solutions agree to rounding; the few absolute constants (floors such
//! as 10⁻⁶) leave differences below 10⁻⁵ mm in the regularised solve, which
//! stops at a step of 10⁻³ of the noise. (Scaled by 1000, its results differ
//! by about 10⁻³ mm: round-off in the inputs, amplified by the solve.)
//!
//! The penalties of the regularised and volumetric methods are weighted by
//! the contrast `ρc − ρs`, not by `ρc` (as in Neeteson & Boyd 2025), so an
//! offset of the image values changes nothing either.

mod common;
use common::*;
use cortical_bone_mapping::model::{M, RHO_B, RHO_S, SIGMA, T};
use cortical_bone_mapping::*;

fn max_thickness_difference(a: &CbmResult, b: &CbmResult, scale: f64) -> f64 {
    a.params
        .iter()
        .zip(&b.params)
        .map(|(p, q)| (p[T] - q[T] / scale).abs())
        .fold(0.0, f64::max)
}

#[test]
fn an_intensity_offset_changes_nothing() {
    let ph = cylinder(40.0, 3);
    let (img, mask) = ph.generate();
    let c = 1000.0;
    let shifted = img.mapv(|v| v + c as f32);
    for (name, method) in methods() {
        let o = options(&ph, method);
        let mut os = o.clone();
        os.model.rho_c += c;
        for p in [RHO_S, RHO_B] {
            os.model.init[p] += c;
            os.model.lower[p] += c;
            os.model.upper[p] += c;
        }
        let a = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
        let b = run(shifted.view(), mask.view(), NO_SUB, &os).unwrap();
        assert_eq!(a.info.iterations, b.info.iterations, "{name}: iterations");
        let dt = max_thickness_difference(&a, &b, 1.0);
        assert!(dt < 1e-4, "{name}: thickness differs by {dt}");
    }
}

fn methods() -> [(&'static str, Method); 3] {
    [
        ("local", Method::Local),
        ("regularized", regularized(64, None)),
        ("volumetric", volumetric_tied()),
    ]
}

#[test]
fn other_length_units_change_nothing() {
    let ph = cylinder(40.0, 4);
    let (img, mask) = ph.generate();
    let k = 1024.0;
    for (name, method) in methods() {
        let o = options(&ph, method.clone());
        let mut os = o.clone();
        os.geometry.spacing = o.geometry.spacing.map(|s| s * k);
        os.geometry.origin = o.geometry.origin.map(|s| s * k);
        os.sampling.outside *= k;
        os.sampling.inside *= k;
        for p in [M, T, SIGMA] {
            os.model.init[p] *= k;
            os.model.lower[p] *= k;
            os.model.upper[p] *= k;
        }
        os.method = match method {
            Method::Regularized {
                lambda,
                sigma,
                neighbours,
                cutoff,
                init_from_local,
                curvature,
            } => Method::Regularized {
                lambda,
                sigma: sigma * k,
                neighbours,
                cutoff,
                init_from_local,
                curvature,
            },
            Method::Volumetric(v) => Method::Volumetric(VolumetricOptions {
                sigma: v.sigma * k,
                curvature: v.curvature.map(|c| CurvatureOptions {
                    length: c.length.map(|l| l * k),
                    ..c
                }),
                ..v
            }),
            m => m,
        };
        let a = run(img.view(), mask.view(), NO_SUB, &o).unwrap();
        let b = run(img.view(), mask.view(), NO_SUB, &os).unwrap();
        assert_eq!(a.info.iterations, b.info.iterations, "{name}: iterations");
        let dt = max_thickness_difference(&a, &b, k);
        assert!(dt < 1e-5, "{name}: thickness differs by {dt} mm");
    }
}
