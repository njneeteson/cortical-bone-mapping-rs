//! The spatial regularisation operator `ω` of our paper (Neeteson & Boyd
//! 2025).
//!
//! ```text
//! (ω x)_i = x_i − Σ_{k ∈ N(i)} w_ik x_k,   w_ik = exp(−d_ik²/2σ²) / Σ_{l ∈ N(i)} exp(−d_il²/2σ²)
//! ```
//!
//! with `N(i)` the `k` nearest nodes of `i` (excluding `i`) and `d_ik` their
//! distances. `ω x` is the deviation of each value from a Gaussian-weighted
//! local mean: it vanishes for a constant field, so `‖ω x‖²` penalises
//! roughness but not level. As a difference of a value and a local average it
//! is a discrete (graph) Laplacian up to a scale factor.
//!
//! Two departures from the paper, both deliberate: the node is not its own
//! neighbour (a self-weight only rescales `ω` by `1 − w_ii`, which varies from
//! node to node and so weights the penalty unevenly), and neighbours farther
//! than `cutoff · σ` can be dropped (the nearest is always kept). With
//! `k` nearest neighbours the operator acts over the radius they cover,
//! `≈ √(k/π) ×` the node spacing, so at fine node spacing the Gaussian is nearly
//! flat and `k`, not `σ`, sets the length scale; `cutoff = ∞` keeps the
//! paper's operator.
//!
//! # References
//!
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.

use rayon::prelude::*;

use super::neighbours::Neighbours;
use super::{gaussian_weights, SparseRows};

/// The spatial regularisation operator `ω` with the off-diagonal
/// weights stored as `f32`: `(ω x)_i = x_i − Σ_k w_ik x_k`, `Σ_k w_ik = 1`.
pub struct SmoothingOperator {
    pub rows: SparseRows<f32>,
}

impl SmoothingOperator {
    /// Build `ω` from the `k` nearest neighbours of each point (excluding the
    /// point itself, by the distance of `neighbours`) with normalised Gaussian weights of standard deviation `σ`
    /// (`gaussian_weights`), keeping only neighbours within `cutoff · σ`
    /// of the point but always at least the nearest one. `cutoff = ∞` reproduces
    /// the plain k-nearest-neighbour operator of our paper.
    ///
    /// Memory: 8 bytes per stored neighbour (`u32` index + `f32` weight; single
    /// precision is ample for averaging weights). The matrix is filled in place
    /// at a fixed stride of `k` and then compacted, so construction never holds a
    /// second copy.
    pub fn new(neighbours: &Neighbours, k: usize, sigma: f64, cutoff: f64) -> Self {
        let n = neighbours.points().len();
        let k = k.max(1).min(n.saturating_sub(1));
        let r2 = (cutoff * sigma).powi(2);
        if k == 0 {
            return SmoothingOperator {
                rows: SparseRows {
                    offsets: vec![0; n + 1],
                    cols: Vec::new(),
                    vals: Vec::new(),
                    ncols: n,
                },
            };
        }
        let s = crate::stage!("smoothing operator");
        let mut cols = vec![0u32; n * k];
        let mut vals = vec![0f32; n * k];
        let mut counts = vec![0usize; n];
        cols.par_chunks_mut(k)
            .zip(vals.par_chunks_mut(k))
            .zip(counts.par_iter_mut())
            .enumerate()
            .for_each_init(
                || neighbours.scratch(),
                |s, (i, ((c, v), cnt))| {
                    let mut nn = neighbours.nearest(i, k, s);
                    let keep = nn.iter().filter(|x| x.1 <= r2).count().max(1);
                    nn.truncate(keep);
                    let d2: Vec<f64> = nn.iter().map(|x| x.1).collect();
                    let w = gaussian_weights(&d2, sigma);
                    for (e, (&(j, _), wj)) in nn.iter().zip(w).enumerate() {
                        c[e] = j;
                        v[e] = wj as f32;
                    }
                    *cnt = nn.len();
                },
            );
        // compact in place
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let mut w = 0;
        for (i, &cnt) in counts.iter().enumerate() {
            cols.copy_within(i * k..i * k + cnt, w);
            vals.copy_within(i * k..i * k + cnt, w);
            w += cnt;
            offsets.push(w);
        }
        cols.truncate(w);
        vals.truncate(w);
        if w < (n * k) * 3 / 4 {
            cols.shrink_to_fit();
            vals.shrink_to_fit();
        }
        s.done(format_args!(
            "{n} nodes, {k} neighbours within {:.3} (sigma {sigma}): {w} weights, {:.1} per node, {:.1} MB",
            r2.sqrt(),
            w as f64 / n.max(1) as f64,
            (w * 8) as f64 / 1e6
        ));
        SmoothingOperator {
            rows: SparseRows {
                offsets,
                cols,
                vals,
                ncols: n,
            },
        }
    }

    /// `out = ω x = x − W x`.
    pub fn apply(&self, x: &[f64], out: &mut [f64]) {
        self.rows.apply(x, out);
        out.par_iter_mut().zip(x.par_iter()).for_each(|(o, &xi)| *o = xi - *o);
    }

    /// `out = ωᵀ y = y − Wᵀ y`.
    pub fn apply_t(&self, y: &[f64], out: &mut [f64]) {
        self.rows.apply_t(y, out);
        out.par_iter_mut().zip(y.par_iter()).for_each(|(o, &yi)| *o = yi - *o);
    }

    /// [`Self::apply`] for up to 8 vectors in one pass over the operator.
    pub fn apply_many(&self, xs: &[&[f64]], outs: &mut [Vec<f64>]) {
        assert!(xs.len() <= 8 && outs.len() >= xs.len());
        self.rows.apply_many(xs, outs);
        for (o, x) in outs.iter_mut().zip(xs) {
            o.par_iter_mut().zip(x.par_iter()).for_each(|(o, &xi)| *o = xi - *o);
        }
    }

    /// [`Self::apply_t`] for up to 8 vectors in one pass over the operator.
    pub fn apply_t_many(&self, ys: &[&[f64]], outs: &mut [Vec<f64>]) {
        assert!(ys.len() <= 8 && outs.len() >= ys.len());
        self.rows.apply_t_many(ys, outs);
        for (o, y) in outs.iter_mut().zip(ys) {
            o.par_iter_mut().zip(y.par_iter()).for_each(|(o, &yi)| *o = yi - *o);
        }
    }

    /// Diagonal of `ωᵀω` (for the block-Jacobi preconditioner):
    /// `(ωᵀω)_ii = Σ_l ω_li² = 1 + Σ_l w_li²`, since `ω = I − W` and the cross
    /// term `−2 w_ii` is zero (points are not their own neighbours).
    pub fn gram_diag(&self) -> Vec<f64> {
        let ones = vec![1.0; self.rows.nrows()];
        let mut d = self.rows.col_sq_weighted(&ones);
        d.iter_mut().for_each(|v| *v += 1.0);
        d
    }

    /// Bytes used by the operator.
    pub fn memory_bytes(&self) -> usize {
        self.rows.nnz() * (4 + 4) + self.rows.offsets.len() * 8
    }
}
