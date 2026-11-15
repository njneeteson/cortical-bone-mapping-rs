//! The cortical density profile model and its analytic gradient.
//!
//! Along a line that starts outside the bone and runs inward along the
//! surface normal, the true density is modelled as three constant levels
//! separated by two step edges (soft tissue → cortex → trabecular/marrow).
//! The imaging system blurs these steps with a Gaussian of standard deviation
//! `σ`, giving (Treece et al. 2010, 2012; Neeteson & Boyd 2025):
//!
//! ```text
//! f(x) = ρs + (ρc − ρs)/2 · [1 + erf((x − m + t/2) / (σ√2))]
//!           + (ρb − ρc)/2 · [1 + erf((x − m − t/2) / (σ√2))]
//! ```
//!
//! `x` is the signed distance along the profile (negative outside the bone,
//! positive inside), `m` the position of the cortical mid-line, `t` the
//! cortical thickness, `ρs`, `ρc`, `ρb` the soft tissue, cortical and
//! trabecular densities. The periosteal edge is at `m − t/2` and the
//! endosteal edge at `m + t/2`. `ρc` is always held fixed, since `ρc` and `σ`
//! cannot both be estimated from a thin cortex.

/// Index of the cortical mid-line position `m` in a parameter vector.
pub const M: usize = 0;
/// Index of the cortical thickness `t`.
pub const T: usize = 1;
/// Index of the blur standard deviation `σ`.
pub const SIGMA: usize = 2;
/// Index of the soft-tissue density `ρs`.
pub const RHO_S: usize = 3;
/// Index of the trabecular (endocortical) density `ρb`.
pub const RHO_B: usize = 4;
/// Number of model parameters.
pub const NPARAM: usize = 5;

/// Parameter names, in index order.
pub const PARAM_NAMES: [&str; NPARAM] = ["m", "t", "sigma", "rho_s", "rho_b"];

/// A full parameter vector `[m, t, σ, ρs, ρb]`.
pub type Params = [f64; NPARAM];

const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// 1 / sqrt(2π)
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Model density at distance `x`.
#[inline]
pub fn eval(x: f64, p: &Params, rho_c: f64) -> f64 {
    let (m, t, s, rs, rb) = (p[M], p[T], p[SIGMA], p[RHO_S], p[RHO_B]);
    let e1 = libm::erf((x - m + 0.5 * t) / s * FRAC_1_SQRT_2);
    let e2 = libm::erf((x - m - 0.5 * t) / s * FRAC_1_SQRT_2);
    rs + 0.5 * (rho_c - rs) * (1.0 + e1) + 0.5 * (rb - rho_c) * (1.0 + e2)
}

/// Model density at `x` and its gradient with respect to `[m, t, σ, ρs, ρb]`.
#[inline]
pub fn eval_grad(x: f64, p: &Params, rho_c: f64) -> (f64, Params) {
    let (m, t, s, rs, rb) = (p[M], p[T], p[SIGMA], p[RHO_S], p[RHO_B]);
    let inv_s = 1.0 / s;
    let u1 = (x - m + 0.5 * t) * inv_s;
    let u2 = (x - m - 0.5 * t) * inv_s;
    let e1 = libm::erf(u1 * FRAC_1_SQRT_2);
    let e2 = libm::erf(u2 * FRAC_1_SQRT_2);
    let phi1 = INV_SQRT_2PI * (-0.5 * u1 * u1).exp();
    let phi2 = INV_SQRT_2PI * (-0.5 * u2 * u2).exp();
    let a1 = rho_c - rs; // height of the periosteal step
    let a2 = rb - rho_c; // height of the endosteal step
    let f = rs + 0.5 * a1 * (1.0 + e1) + 0.5 * a2 * (1.0 + e2);
    let g1 = a1 * phi1 * inv_s; // ∂/∂(edge position) of step 1, negated
    let g2 = a2 * phi2 * inv_s;
    let grad = [
        -g1 - g2,             // ∂f/∂m
        0.5 * (g1 - g2),      // ∂f/∂t
        -(g1 * u1 + g2 * u2), // ∂f/∂σ
        0.5 * (1.0 - e1),     // ∂f/∂ρs
        0.5 * (1.0 + e2),     // ∂f/∂ρb
    ];
    (f, grad)
}

/// Residual weights along the profile (Treece et al. 2010):
/// `γj = γ0 + (1 − γ0) · |xj| / max|x|`, so residuals at the surface node
/// (`x = 0`) weigh `γ0` times as much as those at the far end of the line,
/// decreasing linearly both outwards and inwards (as in the reference
/// implementation, `bonelab`). This discourages the model from fitting its
/// edges to trabeculae. Weights are clamped at zero (relevant only for
/// `γ0 > 1` lines that are longer on one side, never in practice, or `γ0 < 0`).
pub fn residual_weights(x: &[f64], boost: f64) -> Vec<f64> {
    let xmax = x.iter().map(|v| v.abs()).fold(0.0, f64::max);
    if !(xmax > 0.0) {
        return vec![boost.max(0.0); x.len()];
    }
    x.iter()
        .map(|&xj| (boost + (1.0 - boost) * xj.abs() / xmax).max(0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let p = [0.4, 0.9, 0.2, 80.0, 300.0];
        let rho_c = 1000.0;
        for &x in &[-1.0, -0.2, 0.0, 0.1, 0.4, 0.85, 1.3, 2.0] {
            let (f, g) = eval_grad(x, &p, rho_c);
            assert!((f - eval(x, &p, rho_c)).abs() < 1e-12);
            for k in 0..NPARAM {
                let h = 1e-6 * p[k].abs().max(1.0);
                let mut pp = p;
                let mut pm = p;
                pp[k] += h;
                pm[k] -= h;
                let fd = (eval(x, &pp, rho_c) - eval(x, &pm, rho_c)) / (2.0 * h);
                assert!(
                    (fd - g[k]).abs() < 1e-5 * (1.0 + fd.abs()),
                    "param {k} at x={x}: fd={fd} analytic={}",
                    g[k]
                );
            }
        }
    }

    #[test]
    fn levels_far_from_edges() {
        let p = [1.0, 1.0, 0.1, 50.0, 250.0];
        assert!((eval(-5.0, &p, 1000.0) - 50.0).abs() < 1e-9);
        assert!((eval(1.0, &p, 1000.0) - 1000.0).abs() < 1e-3);
        assert!((eval(6.0, &p, 1000.0) - 250.0).abs() < 1e-9);
        // Each edge is at half height.
        let half = eval(0.5, &[1.0, 1.0, 0.1, 0.0, 1000.0], 1000.0);
        assert!((half - 500.0).abs() < 1e-9);
    }

    #[test]
    fn weights() {
        let x = [-1.0, 0.0, 1.0, 2.0];
        let w = residual_weights(&x, 5.0);
        assert!((w[1] - 5.0).abs() < 1e-12);
        assert!((w[3] - 1.0).abs() < 1e-12);
        // symmetric about the node
        assert!((w[0] - w[2]).abs() < 1e-12 && w[0] < w[1]);
        assert_eq!(residual_weights(&x, 1.0), vec![1.0; 4]);
    }
}
