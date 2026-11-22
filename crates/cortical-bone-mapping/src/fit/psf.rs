//! Imaging blur (point-spread function) handling.
//!
//! Treece et al. (2012, 2015) hold the blur `σ` fixed while fitting, using an
//! estimate of the scanner's point-spread function (PSF). For a Gaussian PSF
//! with standard deviations `s = (s0, s1, s2)` along the array axes, the blur
//! seen along a line with unit direction `n` is
//! `σ(n) = sqrt(Σ_a n_a² s_a²)`, so it depends on the profile direction
//! (CT blur is usually larger along the scanner axis).
//!
//! [`estimate_psf`] estimates `s` from a first fit with `σ` free, by least
//! squares regression of `σᵢ²` on `nᵢ²` over nodes where `σ` is well
//! determined (cortex thick relative to the blur, `σ` not on a bound), with
//! robust outlier rejection.

use super::{local, local_settings, Data};
use crate::error::Error;
use crate::model::{Params, SIGMA, T};
use crate::util::{median, V3};

/// How the PSF is parametrised.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PsfModel {
    /// One standard deviation for all directions.
    Isotropic,
    /// One value in the plane perpendicular to the axis and one along it.
    Axial(usize),
    /// One value per array axis.
    Anisotropic,
}

/// Blur along the unit direction `n` for a PSF with standard deviations `psf`.
#[inline]
pub fn sigma_along(psf: [f64; 3], n: V3) -> f64 {
    (0..3).map(|a| (n[a] * psf[a]).powi(2)).sum::<f64>().sqrt()
}

/// Estimate the PSF from per-node fits with free `σ`. Uses nodes with
/// `t ≥ min_ratio · σ` whose `σ` lies strictly inside `(sigma_lo, sigma_hi)`.
/// Returns `None` if fewer than 10 nodes qualify.
pub fn estimate_psf(
    normals: &[V3],
    params: &[Params],
    model: PsfModel,
    min_ratio: f64,
    sigma_lo: f64,
    sigma_hi: f64,
) -> Option<[f64; 3]> {
    let margin = 1e-3 * (sigma_hi - sigma_lo);
    let mut rows: Vec<(V3, f64)> = normals
        .iter()
        .zip(params)
        .filter(|(_, p)| p[T] >= min_ratio * p[SIGMA] && p[SIGMA] > sigma_lo + margin && p[SIGMA] < sigma_hi - margin)
        .map(|(n, p)| (*n, p[SIGMA] * p[SIGMA]))
        .collect();
    if rows.len() < 10 {
        return None;
    }
    let mut est = fit(&rows, model);
    for _ in 0..2 {
        // reject outliers beyond 3 robust standard deviations
        let res: Vec<f64> = rows.iter().map(|(n, s2)| s2 - predict(est, *n)).collect();
        let med = median(&res);
        let mad = median(&res.iter().map(|r| (r - med).abs()).collect::<Vec<_>>()) * 1.4826;
        if !(mad > 0.0) {
            break;
        }
        let keep: Vec<(V3, f64)> = rows
            .iter()
            .zip(&res)
            .filter(|(_, r)| (*r - med).abs() <= 3.0 * mad)
            .map(|(row, _)| *row)
            .collect();
        if keep.len() < 10 || keep.len() == rows.len() {
            break;
        }
        rows = keep;
        est = fit(&rows, model);
    }
    Some(est.map(|v| v.max(0.0).sqrt()))
}

fn predict(s2: [f64; 3], n: V3) -> f64 {
    (0..3).map(|a| n[a] * n[a] * s2[a]).sum()
}

/// Non-negative least squares for `σ² ≈ Σ_a n_a² s_a²` under `model`,
/// returning `s²` per axis.
fn fit(rows: &[(V3, f64)], model: PsfModel) -> [f64; 3] {
    // feature map
    let feats = |n: V3| -> Vec<f64> {
        match model {
            PsfModel::Isotropic => vec![1.0],
            PsfModel::Axial(a) => vec![1.0 - n[a] * n[a], n[a] * n[a]],
            PsfModel::Anisotropic => vec![n[0] * n[0], n[1] * n[1], n[2] * n[2]],
        }
    };
    let k = feats([0.0, 0.0, 1.0]).len();
    let mut ata = vec![0.0; k * k];
    let mut atb = vec![0.0; k];
    for (n, s2) in rows {
        let f = feats(*n);
        for i in 0..k {
            atb[i] += f[i] * s2;
            for j in 0..k {
                ata[i * k + j] += f[i] * f[j];
            }
        }
    }
    // enumerate active sets (k ≤ 3) and keep the best feasible solution
    let mut best: Option<(f64, Vec<f64>)> = None;
    for subset in 1u32..(1 << k) {
        let idx: Vec<usize> = (0..k).filter(|i| subset & (1 << i) != 0).collect();
        let m = idx.len();
        let mut a = vec![0.0; m * m];
        let mut b = vec![0.0; m];
        for (p, &i) in idx.iter().enumerate() {
            b[p] = atb[i];
            for (q, &j) in idx.iter().enumerate() {
                a[p * m + q] = ata[i * k + j];
            }
        }
        crate::util::cholesky_solve(&mut a, &mut b, m);
        if b.iter().any(|&v| v < 0.0) {
            continue;
        }
        let mut sol = vec![0.0; k];
        for (p, &i) in idx.iter().enumerate() {
            sol[i] = b[p];
        }
        let sse: f64 = rows
            .iter()
            .map(|(n, s2)| {
                let f = feats(*n);
                (s2 - f.iter().zip(&sol).map(|(a, b)| a * b).sum::<f64>()).powi(2)
            })
            .sum();
        if best.as_ref().is_none_or(|(e, _)| sse < *e) {
            best = Some((sse, sol));
        }
    }
    let sol = best.map(|b| b.1).unwrap_or_else(|| vec![0.0; k]);
    match model {
        PsfModel::Isotropic => [sol[0]; 3],
        PsfModel::Axial(a) => {
            let mut s = [sol[0]; 3];
            s[a] = sol[1];
            s
        }
        PsfModel::Anisotropic => [sol[0], sol[1], sol[2]],
    }
}

/// [`SigmaMode::EstimatePsf`](crate::options::SigmaMode::EstimatePsf): the
/// PSF from local fits with `σ` free (from `starts`), clamped to the bounds of
/// `σ`.
pub(crate) fn estimate_from_fits(
    data: &Data,
    normals: &[V3],
    starts: &[Params],
    model: PsfModel,
    min_ratio: f64,
) -> Result<[f64; 3], Error> {
    let o = data.options;
    let fits = local::fit_all(
        data.profiles,
        data.x,
        data.weights,
        data.rho_c,
        starts,
        &local_settings(o, true),
    );
    let ps: Vec<Params> = fits.iter().map(|f| f.params).collect();
    let (lo, hi) = (o.model.lower[SIGMA], o.model.upper[SIGMA]);
    let est = estimate_psf(normals, &ps, model, min_ratio, lo, hi)
        .ok_or_else(|| Error::InvalidOption("too few nodes with a thick cortex to estimate the PSF".into()))?;
    Ok(est.map(|v| v.clamp(lo, hi)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Rng;

    #[test]
    fn recovers_axial_psf() {
        let psf = [0.4, 0.15, 0.15];
        let mut rng = Rng::new(1);
        let mut normals = Vec::new();
        let mut params = Vec::new();
        for _ in 0..500 {
            let v = [rng.normal(), rng.normal(), rng.normal()];
            let l = crate::util::norm(v);
            let n = crate::util::scale(v, 1.0 / l);
            let s = sigma_along(psf, n) * (1.0 + 0.02 * rng.normal());
            normals.push(n);
            params.push([0.0, 1.5, s, 0.0, 0.0]);
        }
        // a few outliers and thin-cortex nodes that must be ignored
        for i in 0..20 {
            params[i][SIGMA] = 1.5;
            params[i + 20][T] = 0.1;
        }
        let est = estimate_psf(&normals, &params, PsfModel::Axial(0), 3.0, 0.01, 2.0).unwrap();
        assert!((est[0] - 0.4).abs() < 0.02, "{est:?}");
        assert!((est[1] - 0.15).abs() < 0.01 && est[1] == est[2]);
        let iso = estimate_psf(&normals, &params, PsfModel::Isotropic, 3.0, 0.01, 2.0).unwrap();
        assert!(iso[0] > 0.15 && iso[0] < 0.4);
        let full = estimate_psf(&normals, &params, PsfModel::Anisotropic, 3.0, 0.01, 2.0).unwrap();
        assert!((full[0] - 0.4).abs() < 0.03 && (full[2] - 0.15).abs() < 0.02);
    }
}
