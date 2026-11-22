//! The local method: an independent bounded least-squares fit per node.
//!
//! This is the classic CBM formulation (Treece et al. 2010), the "local method"
//! of our paper (Neeteson & Boyd 2025). Each profile is fitted by minimising
//!
//! ```text
//! F(p) = ½ Σ_j w_j r_j²,    r_j = f(x_j; p) − f_j,    Σ_j w_j = 1
//! ```
//!
//! over at most five parameters `p` within box bounds `l ≤ p ≤ u`, with a
//! projected Levenberg–Marquardt iteration (Levenberg 1944; Marquardt 1963):
//!
//! 1. **Normal equations.** With `J` the Jacobian (from the analytic gradient
//!    of [`crate::model::eval_grad`]), form `H = JᵀWJ` (5×5, packed) and
//!    `g = JᵀWr` directly; `H` is the Gauss–Newton approximation of the
//!    Hessian of `F`.
//! 2. **Active set.** Variables at a bound whose gradient points out of the
//!    box (`p_a = l_a, g_a > 0` or `p_a = u_a, g_a < 0`) are held for this step
//!    (a projected Newton step; Bertsekas 1982).
//! 3. **Damped step.** Solve `(H + μD) δ = −g` on the active variables by
//!    Cholesky, with Marquardt's scaling `D = diag(H)` (floored at
//!    `10⁻¹² max D`), which makes the iteration invariant to the very different
//!    scales of positions (mm) and densities (hundreds); then project:
//!    `p' = clamp(p + δ, l, u)`.
//! 4. **Gain ratio.** `ρ = (F(p) − F(p')) / L(s)`, with `s = p' − p` the actual
//!    (projected) step and `L(s) = −gᵀs − ½ sᵀHs` the decrease predicted by
//!    the quadratic model.
//! 5. **Damping update** (Nielsen 1999; Madsen et al. 2004): if `ρ > 10⁻⁴`
//!    accept `p'` and set `μ ← μ · max(1/3, 1 − (2ρ − 1)³)`, `ν ← 2`;
//!    otherwise `μ ← μ ν`, `ν ← 2ν`. Unlike the classic ×10/÷10 rule this
//!    changes `μ` smoothly with the quality of the model.
//!
//! Stopping (as in MINPACK; Moré 1978): the scaled gradient
//! `max_a |g_a| / (‖J_a‖ ‖r‖)` over the active variables, the cosine of the
//! angle between the residual and a Jacobian column, below `gtol`; an accepted
//! step reducing `F` by less than `ftol` (relative); the step, scaled by the
//! column norms `‖J_a‖ = √H_aa`, below `xtol` relative to `p`; `μ > 10¹⁶`; or
//! `max_iterations` trial steps.
//!
//! Every node is independent, so nodes are fitted in parallel with no shared
//! state.
//!
//! # References
//!
//! * G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
//!   cortical bone thickness measurement from clinical CT data*, Medical Image
//!   Analysis 14(3):276–290, 2010, doi:10.1016/j.media.2010.01.003.
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.
//! * K. Levenberg, *A method for the solution of certain non-linear problems in
//!   least squares*, Quarterly of Applied Mathematics 2(2):164–168, 1944.
//! * D. W. Marquardt, *An algorithm for least-squares estimation of nonlinear
//!   parameters*, Journal of the Society for Industrial and Applied
//!   Mathematics 11(2):431–441, 1963.
//! * J. J. Moré, *The Levenberg–Marquardt algorithm: implementation and
//!   theory*, in G. A. Watson (ed.), Numerical Analysis, Lecture Notes in
//!   Mathematics 630, Springer, 1978, pp. 105–116.
//! * H. B. Nielsen, *Damping parameter in Marquardt's method*, Technical
//!   Report IMM-REP-1999-05, Technical University of Denmark, 1999.
//! * K. Madsen, H. B. Nielsen, O. Tingleff, *Methods for non-linear least
//!   squares problems*, 2nd ed., Informatics and Mathematical Modelling,
//!   Technical University of Denmark, 2004.
//! * D. P. Bertsekas, *Projected Newton methods for optimization problems with
//!   simple constraints*, SIAM Journal on Control and Optimization
//!   20(2):221–246, 1982.

use rayon::prelude::*;

use super::profiles::Profiles;
use super::{local_settings, Data, Start};
use crate::model::{eval, eval_grad, Params, NPARAM};
use crate::result::FitInfo;
use crate::util::{cholesky_solve, sym5};

/// Box bounds and stopping criteria for a fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitSettings {
    pub lower: Params,
    pub upper: Params,
    /// Which parameters are optimised (the others keep their initial value).
    pub free: [bool; NPARAM],
    pub max_iterations: usize,
    /// Stop when the relative cost decrease of an accepted step is below this.
    pub ftol: f64,
    /// Stop when the largest scaled projected gradient component
    /// `|gᵢ| / (‖Jᵢ‖ ‖r‖)` (cosine of the angle between residual and Jacobian
    /// column, as in MINPACK) is below this.
    pub gtol: f64,
    /// Stop when the scaled step is below this.
    pub xtol: f64,
}

/// Why a fit stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ftol,
    Gtol,
    Xtol,
    MaxIterations,
}

impl Status {
    /// Whether the fit stopped on a convergence test (not on its budget).
    pub fn converged(self) -> bool {
        self != Status::MaxIterations
    }
}

/// Result of fitting one profile.
#[derive(Clone, Copy, Debug)]
pub struct LocalFit {
    pub params: Params,
    /// `½ Σ wj rj²`, with the weights normalised to sum to one.
    pub cost: f64,
    pub iterations: u32,
    pub status: Status,
}

struct Normal {
    cost: f64,
    g: Params,
    h: [f64; 15],
}

/// The cost `F = ½ Σ_j w_j r_j²`, the gradient `g = Jᵀ W r` and the packed
/// Gauss–Newton matrix `H = Jᵀ W J` at `p` (step 1 of the module
/// documentation).
fn normal_eq(y: &[f32], x: &[f64], w: &[f64], rho_c: f64, p: &Params) -> Normal {
    let mut cost = 0.0;
    let mut g = [0.0; NPARAM];
    let mut h = [0.0; 15];
    for j in 0..x.len() {
        let (f, d) = eval_grad(x[j], p, rho_c);
        let r = f - y[j] as f64;
        let wj = w[j];
        cost += 0.5 * wj * r * r;
        for a in 0..NPARAM {
            let wd = wj * d[a];
            g[a] += wd * r;
            for b in a..NPARAM {
                h[sym5(a, b)] += wd * d[b];
            }
        }
    }
    Normal { cost, g, h }
}

/// The cost `F = ½ Σ_j w_j r_j²` alone, for trial points.
fn cost_only(y: &[f32], x: &[f64], w: &[f64], rho_c: f64, p: &Params) -> f64 {
    let mut c = 0.0;
    for j in 0..x.len() {
        let r = eval(x[j], p, rho_c) - y[j] as f64;
        c += 0.5 * w[j] * r * r;
    }
    c
}

/// Fit one profile `y` sampled at `x` with weights `w` (normalised by the
/// caller), starting from `init` clamped into the bounds: the projected
/// Levenberg–Marquardt iteration of the module documentation (steps 2–5 per
/// trial step, with Nielsen's damping update).
pub fn fit_profile(y: &[f32], x: &[f64], w: &[f64], rho_c: f64, init: &Params, s: &FitSettings) -> LocalFit {
    let mut p = *init;
    for a in 0..NPARAM {
        if s.free[a] {
            p[a] = p[a].clamp(s.lower[a], s.upper[a]);
        }
    }
    let mut ne = normal_eq(y, x, w, rho_c, &p);
    let mut mu = 1e-3;
    let mut nu = 2.0;
    let mut status = Status::MaxIterations;
    let mut it = 0;
    while it < s.max_iterations {
        it += 1;
        let active = Active::new(&p, &ne, s);
        if active.n == 0 || active.gmax <= s.gtol || ne.cost == 0.0 {
            status = Status::Gtol;
            break;
        }
        let pn = active.trial_point(&p, &ne, mu, s);
        let step: Params = std::array::from_fn(|a| pn[a] - p[a]);
        if negligible(&ne, &p, &step, s.xtol) {
            status = Status::Xtol;
            break;
        }
        let pred = predicted_decrease(&ne, &step);
        let cn = cost_only(y, x, w, rho_c, &pn);
        let actual = ne.cost - cn;
        let rho = if pred > 0.0 { actual / pred } else { -1.0 };
        if rho > 1e-4 {
            let prev = ne.cost;
            p = pn;
            ne = normal_eq(y, x, w, rho_c, &p);
            mu *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
            nu = 2.0;
            if actual <= s.ftol * prev {
                status = Status::Ftol;
                break;
            }
        } else {
            mu *= nu;
            nu *= 2.0;
            if mu > 1e16 {
                status = Status::Xtol;
                break;
            }
        }
    }
    LocalFit {
        params: p,
        cost: ne.cost,
        iterations: it as u32,
        status,
    }
}

/// The free variables not held at a bound by the gradient, and the largest
/// scaled gradient `|gᵢ| / (‖Jᵢ‖ ‖r‖)` over them.
struct Active {
    vars: [usize; NPARAM],
    n: usize,
    gmax: f64,
}

impl Active {
    /// The active set at `p` (step 2 of the module documentation: free variables
    /// not held at a bound by the gradient) and the MINPACK scaled gradient
    /// `max_a |g_a| / (√H_aa ‖r‖)` over it, `‖r‖ = √(2F)`.
    fn new(p: &Params, ne: &Normal, s: &FitSettings) -> Self {
        let mut vars = [0usize; NPARAM];
        let mut n = 0;
        let mut gmax: f64 = 0.0;
        let rnorm = (2.0 * ne.cost).sqrt();
        for a in 0..NPARAM {
            if !s.free[a] {
                continue;
            }
            let at_lo = p[a] <= s.lower[a] && ne.g[a] > 0.0;
            let at_hi = p[a] >= s.upper[a] && ne.g[a] < 0.0;
            if at_lo || at_hi {
                continue;
            }
            let jn = ne.h[sym5(a, a)].sqrt();
            if jn > 0.0 && rnorm > 0.0 {
                gmax = gmax.max(ne.g[a].abs() / (jn * rnorm));
            }
            vars[n] = a;
            n += 1;
        }
        Active { vars, n, gmax }
    }

    /// Solve `(H + μ D) δ = −g` on the active variables by Cholesky, with
    /// Marquardt's scaling `D_aa = max(H_aa, 10⁻¹² max H)`, and project `p + δ`
    /// onto the bounds (step 3).
    fn trial_point(&self, p: &Params, ne: &Normal, mu: f64, s: &FitSettings) -> Params {
        let (vars, nv) = (&self.vars, self.n);
        let mut a_mat = [0.0; 25];
        let mut b = [0.0; NPARAM];
        let dmax = (0..nv).map(|i| ne.h[sym5(vars[i], vars[i])]).fold(0.0, f64::max);
        for i in 0..nv {
            for j in 0..nv {
                a_mat[i * nv + j] = ne.h[sym5(vars[i], vars[j])];
            }
            let d = ne.h[sym5(vars[i], vars[i])].max(1e-12 * dmax).max(1e-300);
            a_mat[i * nv + i] += mu * d;
            b[i] = -ne.g[vars[i]];
        }
        cholesky_solve(&mut a_mat, &mut b, nv);
        let mut pn = *p;
        for i in 0..nv {
            let a = vars[i];
            pn[a] = (p[a] + b[i]).clamp(s.lower[a], s.upper[a]);
        }
        pn
    }
}

/// The step, scaled by the Jacobian column norms `√H_aa`, is below `xtol`
/// relative to `p` in the same norm: `‖D s‖ ≤ xtol (‖D p‖ + xtol)`.
fn negligible(ne: &Normal, p: &Params, step: &Params, xtol: f64) -> bool {
    let mut snorm = 0.0;
    let mut pnorm = 0.0;
    for a in 0..NPARAM {
        let d = ne.h[sym5(a, a)].max(1e-300).sqrt();
        snorm += (d * step[a]).powi(2);
        pnorm += (d * p[a]).powi(2);
    }
    snorm.sqrt() <= xtol * (pnorm.sqrt() + xtol)
}

/// The cost decrease the Gauss–Newton model predicts for the actual step `s`,
/// `L(s) = −gᵀs − ½ sᵀHs` (the denominator of the gain ratio, step 4).
fn predicted_decrease(ne: &Normal, step: &Params) -> f64 {
    let mut pred = 0.0;
    for a in 0..NPARAM {
        pred -= ne.g[a] * step[a];
        for bb in 0..NPARAM {
            pred -= 0.5 * step[a] * ne.h[sym5(a, bb)] * step[bb];
        }
    }
    pred
}

/// Fit every profile independently in parallel. `init` may be per node
/// (length n) or a single value (length 1). The weights are normalised to sum
/// to one per profile, after setting unused samples to zero, so that every
/// profile's cost is a weighted mean square whatever samples it lost.
pub fn fit_all(
    profiles: &Profiles,
    x: &[f64],
    weights: &[f64],
    rho_c: f64,
    init: &[Params],
    settings: &FitSettings,
) -> Vec<LocalFit> {
    let wsum: f64 = weights.iter().sum();
    let w: Vec<f64> = weights.iter().map(|v| v / wsum).collect();
    (0..profiles.n)
        .into_par_iter()
        .map(|i| {
            let p0 = if init.len() == 1 { &init[0] } else { &init[i] };
            match profiles.used_row(i) {
                None => fit_profile(profiles.row(i), x, &w, rho_c, p0, settings),
                Some(_) => {
                    // ignored samples get weight 0; renormalise the rest
                    let wr = profiles.row_weights(i, weights);
                    let s: f64 = wr.iter().sum::<f64>().max(1e-300);
                    let wr: Vec<f64> = wr.iter().map(|v| v / s).collect();
                    fit_profile(profiles.row(i), x, &wr, rho_c, p0, settings)
                }
            }
        })
        .collect()
}

/// The local method: fit every node's profile independently from its start.
pub(crate) fn fit_nodes(data: &Data, start: &Start, info: &mut FitInfo) -> Vec<Params> {
    let free_sigma = start.sigma_fixed.is_none();
    let fits = fit_all(
        data.profiles,
        data.x,
        data.weights,
        data.rho_c,
        &start.nodes,
        &local_settings(data.options, free_sigma),
    );
    let n = fits.len();
    info.converged = fits.iter().filter(|f| f.status.converged()).count() as f64 / n as f64;
    info.iterations = fits.iter().map(|f| f.iterations as usize).max().unwrap_or(0);
    fits.iter().map(|f| f.params).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{residual_weights, RHO_B, RHO_S, SIGMA, T};
    use crate::util::Rng;

    fn settings() -> FitSettings {
        FitSettings {
            lower: [-3.0, 0.05, 0.01, -200.0, -200.0],
            upper: [6.0, 12.0, 2.0, 600.0, 900.0],
            free: [true; 5],
            max_iterations: 200,
            ftol: 1e-10,
            gtol: 1e-10,
            xtol: 1e-12,
        }
    }

    #[test]
    fn recovers_noise_free_parameters() {
        let x: Vec<f64> = (0..121).map(|j| -2.0 + j as f64 * 0.05).collect();
        let w = vec![1.0 / 121.0; 121];
        let truth = [0.8, 1.1, 0.25, 60.0, 280.0];
        let y: Vec<f32> = x.iter().map(|&xj| eval(xj, &truth, 1000.0) as f32).collect();
        let fit = fit_profile(&y, &x, &w, 1000.0, &[0.0, 1.0, 0.1, 0.0, 0.0], &settings());
        assert!(fit.status.converged(), "{:?}", fit.status);
        for a in 0..5 {
            let tol = if a >= RHO_S { 0.05 } else { 1e-3 };
            assert!((fit.params[a] - truth[a]).abs() < tol, "{a}: {:?}", fit.params);
        }
    }

    #[test]
    fn respects_bounds_and_fixed_parameters() {
        let x: Vec<f64> = (0..81).map(|j| -1.0 + j as f64 * 0.05).collect();
        let w = residual_weights(&x, 3.0);
        let truth = [0.5, 0.9, 0.2, 50.0, 300.0];
        let y: Vec<f32> = x.iter().map(|&xj| eval(xj, &truth, 1000.0) as f32).collect();
        let mut s = settings();
        s.upper[T] = 0.6; // thickness forced below the truth
        s.free[SIGMA] = false;
        let fit = fit_profile(&y, &x, &w, 1000.0, &[0.0, 0.3, 0.2, 0.0, 0.0], &s);
        assert!(fit.params[T] <= 0.6 + 1e-12 && (fit.params[T] - 0.6).abs() < 1e-9);
        assert_eq!(fit.params[SIGMA], 0.2);
        let mut s2 = settings();
        s2.free[SIGMA] = false;
        let f2 = fit_profile(&y, &x, &w, 1000.0, &[0.0, 0.3, 0.2, 0.0, 0.0], &s2);
        assert!((f2.params[T] - 0.9).abs() < 1e-4);
        assert!(fit.cost > f2.cost);
    }

    #[test]
    fn noisy_fits_are_unbiased() {
        let x: Vec<f64> = (0..121).map(|j| -2.0 + j as f64 * 0.05).collect();
        let w = residual_weights(&x, 2.0);
        let truth = [0.6, 0.7, 0.2, 80.0, 250.0];
        let mut rng = Rng::new(11);
        let mut ts = Vec::new();
        for _ in 0..200 {
            let y: Vec<f32> = x
                .iter()
                .map(|&xj| (eval(xj, &truth, 1000.0) + 30.0 * rng.normal()) as f32)
                .collect();
            let wsum: f64 = w.iter().sum();
            let wn: Vec<f64> = w.iter().map(|v| v / wsum).collect();
            let f = fit_profile(&y, &x, &wn, 1000.0, &[0.0, 1.0, 0.1, 0.0, 0.0], &settings());
            ts.push(f.params[T]);
        }
        let mean = ts.iter().sum::<f64>() / ts.len() as f64;
        assert!((mean - 0.7).abs() < 0.02, "mean thickness {mean}");
        let _ = RHO_B;
    }
}
