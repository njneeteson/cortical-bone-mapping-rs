//! The spatial regularisation operator `ω` of Neeteson & Boyd (2025).

use rayon::prelude::*;

use super::{gaussian_weights, SparseRows};
use crate::kdtree::KdTree;
use crate::util::V3;

/// The spatial regularisation operator `ω` with the off-diagonal
/// weights stored as `f32`: `(ω x)_i = x_i − Σ_k w_ik x_k`, `Σ_k w_ik = 1`.
pub struct SmoothingOperator {
    pub rows: SparseRows<f32>,
}

impl SmoothingOperator {
    /// Build from the `k` nearest neighbours of each point (excluding the
    /// point itself), keeping only neighbours within `cutoff · σ` of the
    /// point but always at least the nearest one. `cutoff = ∞` reproduces the
    /// plain k-nearest-neighbour operator of the paper.
    ///
    /// Memory: 8 bytes per stored neighbour (`u32` index + `f32` weight). The
    /// matrix is filled in place at a fixed stride of `k` and then compacted,
    /// so construction never holds a second copy.
    pub fn new(points: &[V3], k: usize, sigma: f64, cutoff: f64) -> Self {
        let n = points.len();
        let tree = KdTree::new(points);
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
        let mut cols = vec![0u32; n * k];
        let mut vals = vec![0f32; n * k];
        let mut counts = vec![0usize; n];
        cols.par_chunks_mut(k)
            .zip(vals.par_chunks_mut(k))
            .zip(counts.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((c, v), cnt))| {
                let mut nn = tree.knn(points[i], k, Some(i as u32));
                let keep = nn.iter().filter(|x| x.1 <= r2).count().max(1);
                nn.truncate(keep);
                let d2: Vec<f64> = nn.iter().map(|x| x.1).collect();
                let w = gaussian_weights(&d2, sigma);
                for (e, (&(j, _), wj)) in nn.iter().zip(w).enumerate() {
                    c[e] = j;
                    v[e] = wj as f32;
                }
                *cnt = nn.len();
            });
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
        SmoothingOperator {
            rows: SparseRows {
                offsets,
                cols,
                vals,
                ncols: n,
            },
        }
    }

    pub fn apply(&self, x: &[f64], out: &mut [f64]) {
        self.rows.apply(x, out);
        out.par_iter_mut().zip(x.par_iter()).for_each(|(o, &xi)| *o = xi - *o);
    }

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

    /// Diagonal of `ωᵀω`: `1 + Σ_l w_li²` (the cross term `−2 w_ii` is zero
    /// since points are not their own neighbours).
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
