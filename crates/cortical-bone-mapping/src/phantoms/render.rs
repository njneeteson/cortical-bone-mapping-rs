//! Rendering shared by the phantoms: smooth random fields, supersampled
//! voxel integration with a Gaussian PSF and noise, and root finding along
//! lines for the ground truth.

use ndarray::Array3;
use rayon::prelude::*;

use crate::util::{Rng, V3};
use crate::volume::{gaussian_filter, Geometry};

/// Inverse of erf for the level-set widths (Giles' approximation).
pub(crate) fn erfinv(x: f64) -> f64 {
    let w = -((1.0 - x) * (1.0 + x)).ln();
    if w < 5.0 {
        let w = w - 2.5;
        let mut p = 2.810_226_36e-08;
        for c in [
            3.432_739_39e-07,
            -3.523_387_7e-06,
            -4.391_506_54e-06,
            0.000_218_580_87,
            -0.001_253_725_03,
            -0.004_177_681_64,
            0.246_640_727,
            1.501_409_41,
        ] {
            p = c + p * w;
        }
        p * x
    } else {
        let w = w.sqrt() - 3.0;
        let mut p = -0.000_200_214_257;
        for c in [
            0.000_100_950_558,
            0.001_349_343_22,
            -0.003_673_428_44,
            0.005_739_507_73,
            -0.007_622_461_3,
            0.009_438_870_47,
            1.001_674_06,
            2.832_976_82,
        ] {
            p = c + p * w;
        }
        p * x
    }
}

pub(crate) fn waves(rng: &mut Rng, n: usize, wavelength: f64, axial_stretch: f64) -> Vec<(V3, f64)> {
    let k = 2.0 * std::f64::consts::PI / wavelength;
    (0..n)
        .map(|_| {
            let v = [rng.normal() / axial_stretch, rng.normal(), rng.normal()];
            let l = crate::util::norm(v).max(1e-12);
            (crate::util::scale(v, k / l), 2.0 * std::f64::consts::PI * rng.uniform())
        })
        .collect()
}

#[inline]
pub(crate) fn field(w: &[(V3, f64)], p: V3) -> f64 {
    let s: f64 = w.iter().map(|(k, ph)| (crate::util::dot(*k, p) + ph).cos()).sum();
    s * (2.0 / w.len() as f64).sqrt() // unit variance
}

/// Render a phantom: integrate `density` over each voxel by supersampling,
/// blur with a Gaussian PSF (`blur` per axis, physical units) and add
/// Gaussian noise of standard deviation `noise`.
pub(crate) fn render(
    dims: (usize, usize, usize),
    g: &Geometry,
    blur: V3,
    noise: f64,
    seed: u64,
    supersample: usize,
    density: impl Fn(V3) -> f64 + Sync,
) -> Array3<f32> {
    let ss = supersample.max(1);
    let (d0, d1, d2) = dims;
    let mut truth = Array3::<f64>::zeros(dims);
    truth
        .as_slice_mut()
        .unwrap()
        .par_iter_mut()
        .enumerate()
        .for_each(|(lin, v)| {
            let (i, j, k) = (lin / (d1 * d2), (lin / d2) % d1, lin % d2);
            let off = |q: usize| (q as f64 + 0.5) / ss as f64 - 0.5;
            let mut acc = 0.0;
            for a in 0..ss {
                for b in 0..ss {
                    for e in 0..ss {
                        acc += density(g.to_physical([i as f64 + off(a), j as f64 + off(b), k as f64 + off(e)]));
                    }
                }
            }
            *v = acc / (ss * ss * ss) as f64;
        });
    let sig = [0, 1, 2].map(|a| blur[a] / g.spacing[a]);
    let blurred = gaussian_filter(&truth, sig);
    drop(truth);
    let n = d0 * d1 * d2;
    let mut eps = vec![0f32; n];
    eps.par_chunks_mut(65536).enumerate().for_each(|(ci, ch)| {
        let mut r = Rng::new(seed ^ 0xABCD ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        ch.iter_mut().for_each(|v| *v = r.normal() as f32);
    });
    Array3::from_shape_vec(
        dims,
        blurred
            .iter()
            .zip(eps)
            .map(|(&v, e)| (v + noise * e as f64) as f32)
            .collect(),
    )
    .unwrap()
}

/// The first `x` in `(from, x1]` where `f` becomes negative, by stepping `h`
/// and bisection; NaN if `f(from) < 0` (already inside) or never.
pub(crate) fn first_entry(f: &dyn Fn(f64) -> f64, from: f64, x1: f64, h: f64) -> f64 {
    let mut a = from;
    if f(a) < 0.0 {
        return f64::NAN;
    }
    while a < x1 {
        let b = (a + h).min(x1);
        if f(b) < 0.0 {
            let (mut lo, mut hi) = (a, b);
            for _ in 0..50 {
                let m = 0.5 * (lo + hi);
                if f(m) < 0.0 {
                    hi = m
                } else {
                    lo = m
                }
            }
            return 0.5 * (lo + hi);
        }
        a = b;
    }
    f64::NAN
}
