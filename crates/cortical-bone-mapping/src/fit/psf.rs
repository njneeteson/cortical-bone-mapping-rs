//! Imaging blur (point-spread function) handling.
//!
//! Treece et al. (2012, 2015) hold the blur `σ` fixed while fitting, using an
//! estimate of the scanner's point-spread function (PSF). For a separable
//! Gaussian PSF with standard deviations `s = (s0, s1, s2)` along the array
//! axes, the line-spread function along a line with unit direction `n` is the
//! projection of the PSF onto that line, a Gaussian with variance
//!
//! ```text
//! σ(n)² = nᵀ diag(s²) n = Σ_a n_a² s_a²
//! ```
//!
//! so the blur depends on the profile direction (CT blur is usually larger
//! along the scanner axis). [`sigma_along`] evaluates it.
//!
//! **PSF estimation** ([`estimate_psf`], ours). From a first local fit with
//! `σ` free, `s²` is estimated by linear least squares of `σ_i²` on the
//! squared normal components:
//!
//! ```text
//! σ_i² ≈ Σ_a n_ia² s_a²       (anisotropic: 3 unknowns)
//! σ_i² ≈ (1 − n_ia²) s_⊥² + n_ia² s_∥²    (axial: in-plane and along axis a)
//! σ_i² ≈ s²                  (isotropic)
//! ```
//!
//! using only nodes where `σ` is identifiable: `t ≥ min_ratio · σ` (on a thin
//! cortex `σ` and `t` trade off and `σ` is unreliable) and `σ` strictly inside
//! its bounds. The variances must be non-negative, so the problem is
//! non-negative least squares (Lawson & Hanson 1974); with at most three
//! unknowns it is solved exactly by enumerating the `2³ − 1` active sets and
//! keeping the best feasible one. Two rounds reject outliers beyond 3 robust
//! standard deviations, estimated as 1.4826 × the median absolute deviation
//! (the consistency factor for normally distributed residuals; Rousseeuw &
//! Croux 1993).
//!
//! # References
//!
//! * G. M. Treece, K. E. S. Poole, A. H. Gee, *Imaging the femoral cortex:
//!   thickness, density and mass from clinical CT*, Medical Image Analysis
//!   16(5):952–965, 2012.
//! * G. M. Treece, A. H. Gee, *Independent measurement of femoral cortical
//!   thickness and cortical bone density using clinical CT*, Medical Image
//!   Analysis 20(1):249–264, 2015.
//! * C. L. Lawson, R. J. Hanson, *Solving Least Squares Problems*,
//!   Prentice-Hall, 1974.
//! * P. J. Rousseeuw, C. Croux, *Alternatives to the median absolute
//!   deviation*, Journal of the American Statistical Association
//!   88(424):1273–1283, 1993.

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

/// The blur along the unit direction `n` for a Gaussian PSF with standard
/// deviations `psf`: `σ(n) = √(Σ_a n_a² s_a²)` (the projection of the PSF onto
/// the line).
#[inline]
pub fn sigma_along(psf: [f64; 3], n: V3) -> f64 {
    (0..3).map(|a| (n[a] * psf[a]).powi(2)).sum::<f64>().sqrt()
}

/// Estimate the PSF from per-node fits with free `σ` (module documentation):
/// least squares of `σ_i²` on `n_i²` under `model` over the nodes with
/// `t ≥ min_ratio · σ` and `σ` strictly inside `(sigma_lo, sigma_hi)`, then
/// two rounds rejecting residuals beyond 3 × 1.4826 × MAD from the median.
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

/// `Σ_a n_a² s2_a`: the predicted `σ²` along `n` for per-axis variances `s2`.
fn predict(s2: [f64; 3], n: V3) -> f64 {
    (0..3).map(|a| n[a] * n[a] * s2[a]).sum()
}

/// Non-negative least squares (Lawson & Hanson 1974) for
/// `σ² ≈ features(n) · β` under `model`, returning `s²` per axis: every
/// subset of the features is solved by the normal equations and the best
/// feasible (all `β ≥ 0`) subset wins, exact for at most three features.
fn fit(rows: &[(V3, f64)], model: PsfModel) -> [f64; 3] {
    let k = features(model, [0.0, 0.0, 1.0]).len();
    let (ata, atb) = normal_equations(rows, model, k);
    // enumerate active sets (k ≤ 3) and keep the best feasible solution
    let mut best: Option<(f64, Vec<f64>)> = None;
    for subset in 1u32..(1 << k) {
        let Some(sol) = solve_on_subset(&ata, &atb, k, subset) else {
            continue;
        };
        let sse: f64 = rows
            .iter()
            .map(|(n, s2)| {
                let f = features(model, *n);
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

/// The regression features of a normal `n` under `model`.
fn features(model: PsfModel, n: V3) -> Vec<f64> {
    match model {
        PsfModel::Isotropic => vec![1.0],
        PsfModel::Axial(a) => vec![1.0 - n[a] * n[a], n[a] * n[a]],
        PsfModel::Anisotropic => vec![n[0] * n[0], n[1] * n[1], n[2] * n[2]],
    }
}

/// `AᵀA` (`k × k`, row-major) and `Aᵀb` of the regression.
fn normal_equations(rows: &[(V3, f64)], model: PsfModel, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut ata = vec![0.0; k * k];
    let mut atb = vec![0.0; k];
    for (n, s2) in rows {
        let f = features(model, *n);
        for i in 0..k {
            atb[i] += f[i] * s2;
            for j in 0..k {
                ata[i * k + j] += f[i] * f[j];
            }
        }
    }
    (ata, atb)
}

/// The least-squares solution with only the features in `subset` (a bit
/// mask) non-zero; `None` if it is not non-negative.
fn solve_on_subset(ata: &[f64], atb: &[f64], k: usize, subset: u32) -> Option<Vec<f64>> {
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
        return None;
    }
    let mut sol = vec![0.0; k];
    for (p, &i) in idx.iter().enumerate() {
        sol[i] = b[p];
    }
    Some(sol)
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
    let est = estimate_psf(normals, &ps, model, min_ratio, lo, hi).ok_or_else(|| {
        log::warn!("PSF estimate: too few nodes with a thick cortex");
        Error::InvalidOption("too few nodes with a thick cortex to estimate the PSF".into())
    })?;
    log::debug!("PSF estimate ({model:?}): {est:.3?} from {} local fits", ps.len());
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
