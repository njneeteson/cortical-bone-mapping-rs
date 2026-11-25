//! Sparse linear operators used by the global solver:
//!
//! * [`SparseRows`]: a row-compressed sparse matrix (used for the multigrid
//!   interpolation matrix `A` and the operators below).
//! * [`SmoothingOperator`]: the spatial regularisation matrix `ω` of our paper
//!   (Neeteson & Boyd 2025), `(ω x)_i = x_i − Σ_k g(d_ik) x_k / Σ_k g(d_ik)`
//!   over the `k` nearest neighbours, `g` a Gaussian.
//! * [`CurvatureOperator`]: curvature penalties on the fitted periosteal and
//!   endosteal surfaces (an extension in this crate).
//! * [`select_control_points`] and [`interpolation_matrix`]: the multigrid
//!   control points and their interpolation to the nodes (Neeteson & Boyd
//!   2025).
//!
//! Transposed products `Mᵀ y` scatter into one accumulation buffer per fixed
//! chunk of rows, which are then added in chunk order, instead of storing the
//! transpose: this halves the memory of the large neighbourhood matrices, and
//! because the chunking does not depend on the number of threads the result is
//! bitwise reproducible.
//!
//! # References
//!
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.

use rayon::prelude::*;

mod curvature;
mod hessian;
mod multigrid;
mod neighbours;
mod smoothing;
#[cfg(test)]
mod tests;
mod variation;

pub use curvature::{
    mesh_curvature, CurvatureMeasure, CurvatureMode, CurvatureOperator, CurvatureState, SurfaceLaplacian,
};
pub use hessian::HeightHessian;
pub use multigrid::{interpolation_matrix, select_control_points};
pub use neighbours::{Distance, Neighbours, QueryScratch};
pub use smoothing::SmoothingOperator;
pub use variation::CurvatureGradient;

/// Row-compressed sparse matrix with `u32` column indices.
#[derive(Clone, Debug)]
pub struct SparseRows<V> {
    pub offsets: Vec<usize>,
    pub cols: Vec<u32>,
    pub vals: Vec<V>,
    pub ncols: usize,
}

impl<V: Copy + Into<f64> + Send + Sync> SparseRows<V> {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.offsets.len() - 1
    }
    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.cols.len()
    }

    /// Assemble from per-row entry lists produced in parallel.
    pub fn from_rows(rows: Vec<Vec<(u32, V)>>, ncols: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        let mut total = 0;
        for r in &rows {
            total += r.len();
            offsets.push(total);
        }
        let mut cols = Vec::with_capacity(total);
        let mut vals = Vec::with_capacity(total);
        for r in rows {
            for (c, v) in r {
                cols.push(c);
                vals.push(v);
            }
        }
        SparseRows {
            offsets,
            cols,
            vals,
            ncols,
        }
    }

    /// `out = M x`.
    pub fn apply(&self, x: &[f64], out: &mut [f64]) {
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let mut s = 0.0;
            for e in self.offsets[i]..self.offsets[i + 1] {
                s += self.vals[e].into() * x[self.cols[e] as usize];
            }
            *o = s;
        });
    }

    /// `out = Mᵀ y` (scatter into one buffer per worker, then reduce).
    pub fn apply_t(&self, y: &[f64], out: &mut [f64]) {
        let n = self.nrows();
        // A fixed number of chunks (independent of the thread count) keeps
        // the result bitwise reproducible; memory is <= 16 buffers of ncols.
        let chunk = n.div_ceil(16).max(4096);
        let nchunks = n.div_ceil(chunk).max(1);
        if nchunks == 1 {
            out.iter_mut().for_each(|o| *o = 0.0);
            for i in 0..n {
                for e in self.offsets[i]..self.offsets[i + 1] {
                    out[self.cols[e] as usize] += self.vals[e].into() * y[i];
                }
            }
            return;
        }
        let parts: Vec<Vec<f64>> = (0..nchunks)
            .into_par_iter()
            .map(|c| {
                let mut acc = vec![0.0; self.ncols];
                for i in c * chunk..((c + 1) * chunk).min(n) {
                    let yi = y[i];
                    if yi == 0.0 {
                        continue;
                    }
                    for e in self.offsets[i]..self.offsets[i + 1] {
                        acc[self.cols[e] as usize] += self.vals[e].into() * yi;
                    }
                }
                acc
            })
            .collect();
        out.par_iter_mut().enumerate().for_each(|(k, o)| {
            *o = parts.iter().map(|p| p[k]).sum();
        });
    }

    /// `out[c] = M x[c]` for several vectors in one pass over the matrix
    /// (the matrix is read once instead of once per vector; each result is
    /// bitwise identical to [`Self::apply`]).
    pub fn apply_many(&self, xs: &[&[f64]], outs: &mut [Vec<f64>]) {
        let nc = xs.len();
        let n = self.nrows();
        let rows: Vec<[f64; 8]> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut s = [0.0; 8];
                for e in self.offsets[i]..self.offsets[i + 1] {
                    let w: f64 = self.vals[e].into();
                    let j = self.cols[e] as usize;
                    for c in 0..nc {
                        s[c] += w * xs[c][j];
                    }
                }
                s
            })
            .collect();
        for (c, o) in outs.iter_mut().enumerate().take(nc) {
            o.par_iter_mut().zip(rows.par_iter()).for_each(|(v, r)| *v = r[c]);
        }
    }

    /// `out[c] = Mᵀ y[c]` for several vectors in one pass (bitwise identical
    /// to [`Self::apply_t`] per vector).
    pub fn apply_t_many(&self, ys: &[&[f64]], outs: &mut [Vec<f64>]) {
        let nc = ys.len();
        let n = self.nrows();
        let m = self.ncols;
        let chunk = n.div_ceil(16).max(4096);
        let nchunks = n.div_ceil(chunk).max(1);
        if nchunks == 1 {
            for (c, o) in outs.iter_mut().enumerate().take(nc) {
                self.apply_t(ys[c], o);
            }
            return;
        }
        // one interleaved accumulator per chunk: acc[k * nc + c]
        let parts: Vec<Vec<f64>> = (0..nchunks)
            .into_par_iter()
            .map(|ch| {
                let mut acc = vec![0.0; m * nc];
                for i in ch * chunk..((ch + 1) * chunk).min(n) {
                    for e in self.offsets[i]..self.offsets[i + 1] {
                        let w: f64 = self.vals[e].into();
                        let k = self.cols[e] as usize * nc;
                        for c in 0..nc {
                            let yi = ys[c][i];
                            if yi != 0.0 {
                                acc[k + c] += w * yi;
                            }
                        }
                    }
                }
                acc
            })
            .collect();
        for (c, o) in outs.iter_mut().enumerate().take(nc) {
            o.par_iter_mut().enumerate().for_each(|(k, v)| {
                *v = parts.iter().map(|p| p[k * nc + c]).sum();
            });
        }
    }

    /// `Σ_i M_ik² w_i` for every column `k`.
    pub fn col_sq_weighted(&self, w: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.ncols];
        for i in 0..self.nrows() {
            for e in self.offsets[i]..self.offsets[i + 1] {
                let v: f64 = self.vals[e].into();
                out[self.cols[e] as usize] += v * v * w[i];
            }
        }
        out
    }
}

/// Normalised Gaussian weights `w_k = g(d_k|σ) / Σ_l g(d_l|σ)` from squared
/// distances `d_k²`, computed as `exp(−(d_k² − d_0²)/2σ²)` relative to the
/// nearest neighbour `d_0`: the common factor cancels in the normalisation,
/// and the nearest weight is 1, so the weights never all underflow to zero
/// for far-away points.
pub(crate) fn gaussian_weights(d2: &[f64], sigma: f64) -> Vec<f64> {
    let d0 = d2.iter().cloned().fold(f64::INFINITY, f64::min);
    let inv = 1.0 / (2.0 * sigma * sigma);
    let w: Vec<f64> = d2.iter().map(|&d| (-(d - d0) * inv).exp()).collect();
    let s: f64 = w.iter().sum();
    w.into_iter().map(|v| v / s).collect()
}
