//! The spatial smoothing penalty `S = ½ Σ_c κ_c ‖ω x_c‖²` on node-space
//! fields, shared by the profile and the volumetric problems.
//!
//! For channels `x_c` with weights `κ_c` (Neeteson & Boyd 2025): the cost,
//! its gradient `κ_c ωᵀω x_c`, and the (exact) Gauss–Newton product
//! `κ_c ωᵀω δx_c`. All channels go through the operator in one pass
//! ([`SmoothingOperator::apply_many`]).

use rayon::prelude::*;

use crate::operators::SmoothingOperator;
use crate::util::det_sum;

/// `½ Σ_c κ_c ‖ω x_c‖²`; with `grads`, `κ_c ωᵀω x_c` is added to `grads[c]`.
pub(crate) fn smoothing_cost(
    op: &SmoothingOperator,
    xs: &[&[f64]],
    coefs: &[f64],
    grads: Option<&mut [&mut Vec<f64>]>,
) -> f64 {
    let n = op.rows.nrows();
    let us = smooth_each(op, xs, n, false);
    let mut cost = 0.0;
    for (u, &k) in us.iter().zip(coefs) {
        cost += 0.5 * k * det_sum(n, |i| u[i] * u[i]);
    }
    if let Some(grads) = grads {
        let ys: Vec<&[f64]> = us.iter().map(|u| u.as_slice()).collect();
        let ts = smooth_each(op, &ys, n, true);
        for ((t, &k), g) in ts.iter().zip(coefs).zip(grads.iter_mut()) {
            add_scaled(g, t, k);
        }
    }
    cost
}

/// Add `κ_c ωᵀω δx_c` to `ys[c]`.
pub(crate) fn smoothing_product(op: &SmoothingOperator, dxs: &[&[f64]], coefs: &[f64], ys: &mut [&mut Vec<f64>]) {
    let n = op.rows.nrows();
    let us = smooth_each(op, dxs, n, false);
    let refs: Vec<&[f64]> = us.iter().map(|u| u.as_slice()).collect();
    let ts = smooth_each(op, &refs, n, true);
    for ((t, &k), y) in ts.iter().zip(coefs).zip(ys.iter_mut()) {
        add_scaled(y, t, k);
    }
}

/// `ω x` (or `ωᵀ x`) for each of `xs`, in one pass over the operator.
fn smooth_each(op: &SmoothingOperator, xs: &[&[f64]], n: usize, transpose: bool) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n]; xs.len()];
    if transpose {
        op.apply_t_many(xs, &mut out);
    } else {
        op.apply_many(xs, &mut out);
    }
    out
}

/// `dst += c src`.
pub(crate) fn add_scaled(dst: &mut [f64], src: &[f64], c: f64) {
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(a, &b)| *a += c * b);
}
