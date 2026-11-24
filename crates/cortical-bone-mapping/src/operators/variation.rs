//! The curvature-variation penalty: how fast the curvature of a fitted
//! surface changes along it (ours, after the minimum-variation surfaces of
//! Moreton & Séquin 1992).
//!
//! A bump left by noise, or a dent, changes the curvature over a short
//! distance, while bone, however curved, changes its curvature slowly. So
//! instead of the curvature `K = κ₁ + κ₂` itself, this penalises its
//! gradient, with the neighbours and weights `a_ik ≥ 0` of the Gaussian
//! surface Laplacian (rows normalised so that `Σ_k a_ik (f_k − f_i)²
//! ≈ 2 |∇f|²` for a smooth `f`):
//!
//! ```text
//! E = ½ Σ_i ½ Σ_k a_ik (K_i − K_k)² ≈ ½ ∫ |∇K|² = ½ Kᵀ G K,
//! G = ½ (D_r + D_c − A − Aᵀ),
//! ```
//!
//! `D_r`, `D_c` the row and column sums of `A`. `G` is symmetric positive
//! semi-definite and annihilates constants: spheres, cylinders and their
//! parallel surfaces cost nothing. The signed `K_i(S) = 2 n_i · ½ (L S)_i` is
//! linear in the offsets, so the penalty is exactly quadratic in `(m, t)`.
//! Pairs touching a node without a valid Laplacian row are left out.
//!
//! # References
//!
//! * H. P. Moreton, C. H. Séquin, *Functional optimization for fair surface
//!   design*, Computer Graphics (Proceedings of SIGGRAPH 92) 26(2):167–176,
//!   1992.

use rayon::prelude::*;

use super::curvature::SurfaceLaplacian;
use super::SparseRows;

/// The operator `G` (module documentation).
pub struct CurvatureGradient {
    weights: SparseRows<f64>,
    /// `½ (row sum + column sum)`, the diagonal of `G`.
    diag: Vec<f64>,
}

impl CurvatureGradient {
    /// From the neighbours of `lap` (negative entries, e.g. of a cotangent
    /// Laplacian on obtuse triangles, are dropped).
    pub fn new(lap: &SurfaceLaplacian) -> Self {
        let r = &lap.rows;
        let n = r.nrows();
        let rows: Vec<Vec<(u32, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                if !lap.interior[i] {
                    return Vec::new();
                }
                (r.offsets[i]..r.offsets[i + 1])
                    .filter(|&e| r.vals[e] > 0.0 && lap.interior[r.cols[e] as usize] && r.cols[e] as usize != i)
                    .map(|e| (r.cols[e], r.vals[e]))
                    .collect()
            })
            .collect();
        let weights = SparseRows::from_rows(rows, n);
        let mut diag = vec![0.0; n];
        for i in 0..n {
            for e in weights.offsets[i]..weights.offsets[i + 1] {
                let w = weights.vals[e];
                diag[i] += 0.5 * w;
                diag[weights.cols[e] as usize] += 0.5 * w;
            }
        }
        CurvatureGradient { weights, diag }
    }

    /// `out = G k`.
    pub fn apply(&self, k: &[f64], out: &mut [f64]) {
        let n = k.len();
        let mut ak = vec![0.0; n];
        let mut atk = vec![0.0; n];
        self.weights.apply(k, &mut ak);
        self.weights.apply_t(k, &mut atk);
        out.par_iter_mut()
            .enumerate()
            .for_each(|(i, o)| *o = self.diag[i] * k[i] - 0.5 * (ak[i] + atk[i]));
    }

    /// The diagonal of `G`.
    pub fn diagonal(&self) -> &[f64] {
        &self.diag
    }
}
