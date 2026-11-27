//! Reliability of the volumetric fit.
//!
//! * **Noise.** The voxels in the outer half of the outside band sample soft
//!   tissue, where the model is the soft-tissue level, so their residuals
//!   measure the noise: its standard deviation `s`, and the correlation `ρ_a`
//!   of neighbouring residuals along each axis (CT noise is correlated over a
//!   few voxels by the reconstruction kernel). For noise with separable
//!   first-order (exponential) correlation, the variance of a mean over a
//!   block of voxels is larger than for independent voxels by
//!   `Π_a (1 + ρ_a)/(1 − ρ_a)`, the factor by which the voxels overstate the
//!   number of independent measurements (reported for reference: it holds
//!   for a mean over a block long along every axis). Correlations are
//!   clamped to `[0, 0.95]`.
//! * **Local standard error of `t`.** From the node's own voxels: with the
//!   data-only Gauss–Newton block `B = Σ w_i g_i g_iᵀ` of its parameters
//!   (`g_i` the gradient of voxel `i`'s model value) and the noise
//!   covariance of the weighted gradient `C = Σ_ij ρ_ij (w_i g_i)(w_j g_j)ᵀ`,
//!   `ρ_ij = Π_a ρ_a^{|Δ_a|}` the separable correlation of voxels `i` and
//!   `j`, the covariance of the weighted least-squares estimate is the
//!   sandwich `s² B⁻¹ C B⁻¹`. The correlation enters through the actual
//!   voxel pairs: a node's voxels form a column about one voxel across, for
//!   which the block factor above would overstate the variance by the
//!   in-plane factors. It ignores the neighbours sharing the voxels and the
//!   regulariser, so it is the error of a node fitted from its own voxels
//!   alone, the honest answer to "what did the data say here". Inverting
//!   the block (rather than using `1/B_tt`) accounts for the trade-off of
//!   `t` with the node's other parameters.
//! * **Data share.** `B_cc / (B_cc + P_cc)` per node and field parameter, `P`
//!   the penalties' Gram diagonal: near one the data decided the value, near
//!   zero the regulariser filled it in.
//! * **Misfit.** The node's RMS residual, `√(Σ β w r² / Σ β w)` over the
//!   voxels of its faces (barycentric weights `β`), relative to `s`: where
//!   the model does not hold (porosity, a plate within the blur of its other
//!   cortex, a surface bridging a gap).

use super::problem::{Correlation, Prepared, M, RHO_B, RHO_S, T};
use crate::fit::global::LeastSquares;
use crate::result::VolumetricInfo;
use crate::util::cholesky_solve;

/// Per-node residuals (for the result) and the reliability outputs.
pub(super) struct Reliability {
    pub residual: Vec<f64>,
    pub info: VolumetricInfo,
}

/// Only the per-node residuals, for the inexact inner fits of a refinement,
/// which use nothing else: the reliability outputs are NaN (the noise,
/// standard errors and data shares cost about a tenth of a fit).
pub(super) fn residuals_only(prep: &Prepared, x: &[f64], psf: [f64; 2], axis: usize) -> Reliability {
    let n = prep.lay.nodes;
    Reliability {
        residual: node_rms(prep, x),
        info: VolumetricInfo {
            sigma_xy: psf[0],
            sigma_z: psf[1],
            scanner_axis: axis,
            voxels: prep.problem.band.len(),
            noise_sd: f64::NAN,
            noise_correlation: [f64::NAN; 3],
            noise_factor: f64::NAN,
            std_error_t: vec![f64::NAN; n],
            data_share: vec![[f64::NAN; 4]; n],
            misfit: vec![f64::NAN; n],
            reach: Vec::new(),
        },
    }
}

/// Each node's RMS residual `√(Σ β w r² / Σ β w)` over the voxels of its
/// faces at `x`.
fn node_rms(prep: &Prepared, x: &[f64]) -> Vec<f64> {
    let (wr2, w) = prep.node_residuals(x);
    wr2.iter().zip(&w).map(|(a, b)| (a / b.max(1e-300)).sqrt()).collect()
}

/// Assess the fit `x` of `prep` (PSF `psf` along `axis`).
pub(super) fn assess(prep: &Prepared, x: &[f64], psf: [f64; 2], axis: usize) -> Reliability {
    let band = prep.problem.band;
    let st = crate::stage!("noise");
    let (noise_sd, rho, step) = noise(prep, x);
    crate::done!(
        st,
        "sd {noise_sd:.2}, lag-one correlation per axis {rho:.3?} (grid step {step}), from {} soft-tissue voxels",
        band.soft.len()
    );
    let factor = rho.iter().map(|&r| (1.0 + r) / (1.0 - r)).product::<f64>();
    let st = crate::stage!("node moments (standard errors)");
    let cor = Correlation { rho, step };
    let moments = prep.node_moments(x, &cor);
    crate::done!(
        st,
        "{} nodes; correlated pairs within {:?} grid steps",
        prep.lay.nodes,
        cor.reach()
    );
    let st = crate::stage!("data share");
    let (_, _, lin) = prep.linearise(x);
    let share = prep.data_share(&lin);
    st.end();
    let lay = &prep.lay;
    let (n, nfc) = (lay.nodes, lay.fields.len());
    let t_field = lay.fields.iter().position(|&c| c == T);
    let residual = node_rms(prep, x);
    let std_error_t = (0..n)
        .map(|v| match t_field {
            Some(ti) => {
                let bs = nfc * nfc;
                let var = sandwich(
                    &moments.b[v * bs..(v + 1) * bs],
                    &moments.c[v * bs..(v + 1) * bs],
                    nfc,
                    ti,
                );
                noise_sd * var.sqrt()
            }
            None => f64::NAN,
        })
        .collect();
    let data_share = (0..n)
        .map(|v| {
            [M, T, RHO_S, RHO_B].map(|c| match lay.fields.iter().position(|&f| f == c) {
                Some(fi) => share[v * nfc + fi],
                None => f64::NAN,
            })
        })
        .collect();
    let misfit = residual.iter().map(|r| r / noise_sd.max(1e-300)).collect();
    Reliability {
        residual,
        info: VolumetricInfo {
            sigma_xy: psf[0],
            sigma_z: psf[1],
            scanner_axis: axis,
            voxels: band.len(),
            noise_sd,
            noise_correlation: rho,
            noise_factor: factor,
            std_error_t,
            data_share,
            misfit,
            reach: Vec::new(),
        },
    }
}

/// The noise standard deviation, the lag-one correlation of the residuals
/// along each axis (from the soft-tissue voxels), and the band's grid step.
fn noise(prep: &Prepared, x: &[f64]) -> (f64, [f64; 3], u32) {
    let band = prep.problem.band;
    let positions: Vec<u32> = band.soft.iter().map(|s| s.1).collect();
    let model = prep.values_at(x, &positions);
    let r: Vec<f64> = positions
        .iter()
        .zip(&model)
        .map(|(&e, &m)| band.value[e as usize] as f64 - m)
        .collect();
    if r.is_empty() {
        return (f64::NAN, [0.0; 3], 1);
    }
    let mean = r.iter().sum::<f64>() / r.len() as f64;
    let sd = (r.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
    // the band is subsampled with a stride: neighbours are a stride apart
    let grid: Vec<[u32; 3]> = band.soft.iter().map(|s| s.0).collect();
    let step = neighbour_step(&grid);
    let rho = [0, 1, 2].map(|a| {
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (i, g) in grid.iter().enumerate() {
            let mut h = *g;
            h[a] += step;
            if let Ok(j) = grid.binary_search(&h) {
                let (p, q) = (r[i] - mean, r[j] - mean);
                sxy += p * q;
                sxx += p * p;
                syy += q * q;
            }
        }
        if sxx > 0.0 && syy > 0.0 {
            (sxy / (sxx * syy).sqrt()).clamp(0.0, 0.95)
        } else {
            0.0
        }
    });
    (sd, rho, step)
}

/// The grid step between neighbouring band voxels (the stride).
fn neighbour_step(grid: &[[u32; 3]]) -> u32 {
    grid.windows(2)
        .filter(|w| w[0][0] == w[1][0] && w[0][1] == w[1][1])
        .map(|w| w[1][2] - w[0][2])
        .min()
        .unwrap_or(1)
        .max(1)
}

/// `[B⁻¹ C B⁻¹]_tt` for `n × n` blocks `B`, `C` (row-major), by solving
/// `B u = e_t` and taking `uᵀ C u`.
fn sandwich(b: &[f64], c: &[f64], n: usize, t: usize) -> f64 {
    let mut a = b.to_vec();
    let mut u = vec![0.0; n];
    u[t] = 1.0;
    cholesky_solve(&mut a, &mut u, n);
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..n {
            s += u[i] * c[i * n + j] * u[j];
        }
    }
    s
}
