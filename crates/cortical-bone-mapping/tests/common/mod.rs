#![allow(dead_code)]

use cortical_bone_mapping::phantoms::shell::{Shape, ShellPhantom};
use cortical_bone_mapping::*;
use ndarray::ArrayView3;

pub const NO_SUB: Option<ArrayView3<'static, u8>> = None;

pub fn cylinder(noise: f64, seed: u64) -> ShellPhantom {
    ShellPhantom {
        noise,
        seed,
        ..ShellPhantom::default()
    }
}

pub fn sphere(noise: f64) -> ShellPhantom {
    ShellPhantom {
        dims: (80, 80, 80),
        shape: Shape::Sphere,
        radius: 3.0,
        thickness_mean: 0.7,
        thickness_amplitude: 0.2,
        thickness_lobes: 1.0,
        noise,
        ..ShellPhantom::default()
    }
}

pub fn options(ph: &ShellPhantom, method: Method) -> Options {
    let constraint = match ph.shape {
        Shape::Cylinder => NormalConstraint::InPlane(0),
        Shape::Sphere => NormalConstraint::None,
    };
    Options {
        geometry: ph.geometry(),
        sampling: SamplingOptions {
            outside: 1.5,
            inside: 2.5,
            normal_constraint: constraint,
            ..Default::default()
        },
        method,
        ..Default::default()
    }
}

pub fn regularized(neighbours: usize, curvature: Option<CurvatureOptions>) -> Method {
    Method::Regularized {
        lambda: 1.0,
        sigma: 1.0,
        neighbours,
        cutoff: 3.0,
        init_from_local: false,
        curvature,
    }
}

/// (bias, rmse) of fitted thickness against the phantom truth.
pub fn thickness_error(ph: &ShellPhantom, r: &CbmResult) -> (f64, f64) {
    let e: Vec<f64> = r
        .points
        .iter()
        .zip(r.thickness())
        .map(|(p, t)| t - ph.true_thickness(*p))
        .collect();
    let n = e.len() as f64;
    (
        e.iter().sum::<f64>() / n,
        (e.iter().map(|v| v * v).sum::<f64>() / n).sqrt(),
    )
}

/// RMS error of the fitted periosteal edge position (radially).
pub fn periosteal_error(ph: &ShellPhantom, r: &CbmResult) -> f64 {
    let p = r.periosteal_points();
    (p.iter().map(|q| ph.radial_offset(*q).powi(2)).sum::<f64>() / p.len() as f64).sqrt()
}

pub fn dice(a: &ndarray::Array3<bool>, b: &ndarray::Array3<u8>) -> f64 {
    let (mut i, mut sa, mut sb) = (0usize, 0usize, 0usize);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let y = y != 0;
        i += (x && y) as usize;
        sa += x as usize;
        sb += y as usize;
    }
    2.0 * i as f64 / (sa + sb) as f64
}
