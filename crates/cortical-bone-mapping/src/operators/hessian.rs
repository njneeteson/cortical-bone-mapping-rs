//! The second fundamental form of a fitted surface, for the total-curvature
//! penalty (ours).
//!
//! Near node `i`, a surface is the graph of a height function over its
//! tangent plane, and the Hessian of that height at `i` is the second
//! fundamental form `II_i` (do Carmo 1976, §3-3): its eigenvalues are the
//! principal curvatures `κ₁, κ₂`, so `½ tr II = H` (mean curvature) and
//! `‖II‖²_F = κ₁² + κ₂²` (total curvature). We estimate it by a weighted
//! least-squares fit of the quadric
//!
//! ```text
//! f(u, v) − f(0) ≈ b u + c v + ½ (p u² + 2 q uv + r v²)
//! ```
//!
//! to the heights `f_j` of the neighbours `j` of node `i` over its tangent
//! plane (`u`, `v` their tangent coordinates), with the neighbours and
//! weights of the Gaussian surface Laplacian (the same physical scale as the
//! mean curvature), or the one-ring of the mesh, equally weighted. The fit is linear in the heights: `(p, q, r) = Σ_j G_ij (f_j −
//! f_i)`, with `G_ij` precomputed per node.
//!
//! For a fitted surface `S = V + d n`, the height of the point of node `j`
//! over node `i`'s tangent plane is `n_i · (V_j − V_i) + d_j (n_i · n_j)`, and
//! that of node `i` is `d_i`; to first order in `d` (ignoring the tangential
//! shift of the points) the vector `w_i = (p, √2 q, r)` of `S` is
//!
//! ```text
//! w_i = B_i + Σ_j G_ij (n_i · n_j) d_j − (Σ_j G_ij) d_i,
//! ```
//!
//! linear in `d`, with `B_i` that of `V`. Then `|w_i| = ‖II‖_F = √(κ₁² +
//! κ₂²)`, the total curvature, which, unlike the mean curvature, is not zero
//! at a saddle (√2 times the curvedness of Koenderink & van Doorn 1992).
//!
//! # References
//!
//! * M. P. do Carmo, *Differential Geometry of Curves and Surfaces*,
//!   Prentice-Hall, 1976.
//! * J. J. Koenderink, A. J. van Doorn, *Surface shape and curvature scales*,
//!   Image and Vision Computing 10(8):557–564, 1992.

use rayon::prelude::*;

use super::curvature::SurfaceLaplacian;
use super::SparseRows;
use crate::util::{cholesky_solve, cross, dot, norm, scale, sub, V3};

/// The linear map `d ↦ w` (module documentation) of a set of nodes.
pub struct HeightHessian {
    /// `3n × n`: row `3i + c` gives component `c` of `w_i − B_i`.
    rows: SparseRows<f64>,
    /// `B_i`, the second fundamental form of the input surface.
    pub base: Vec<V3>,
    /// The quadric fit at the node is well posed.
    pub valid: Vec<bool>,
}

/// One node's entries, its base `B_i` and whether its fit is well posed.
type NodeRows = (Vec<(u32, f64)>, V3, bool);

/// Unknowns of the quadric fit: `b, c, p, q, r`.
const NQ: usize = 5;

impl HeightHessian {
    /// From the node positions and unit normals, using the neighbours of
    /// `lap`, weighted by its entries if it is a weighted (Gaussian) one and
    /// equally otherwise (the one-ring of the cotangent Laplacian, whose
    /// coefficients vanish on right angles).
    pub fn new(lap: &SurfaceLaplacian, points: &[V3], normals: &[V3]) -> Self {
        let n = points.len();
        let r = &lap.rows;
        let fits: Vec<NodeRows> = (0..n)
            .into_par_iter()
            .map(|i| {
                let nbrs: Vec<(usize, f64)> = (r.offsets[i]..r.offsets[i + 1])
                    .map(|e| (r.cols[e] as usize, if lap.weighted { r.vals[e].abs() } else { 1.0 }))
                    .filter(|&(j, w)| j != i && w > 0.0)
                    .collect();
                node_rows(i, &nbrs, points, normals)
            })
            .collect();
        let mut entries = Vec::with_capacity(3 * n);
        let mut base = Vec::with_capacity(n);
        let mut valid = Vec::with_capacity(n);
        for (rows, b, ok) in fits {
            // split the interleaved per-node entries into the three rows
            for c in 0..3 {
                entries.push(rows.iter().skip(c).step_by(3).copied().collect::<Vec<_>>());
            }
            base.push(b);
            valid.push(ok);
        }
        HeightHessian {
            rows: SparseRows::from_rows(entries, n),
            base,
            valid,
        }
    }

    /// `out_i = Σ_j K_ij d_j` (the linear part of `w_i`).
    pub fn apply(&self, d: &[f64], out: &mut [V3]) {
        let r = &self.rows;
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            for (c, oc) in o.iter_mut().enumerate() {
                let row = 3 * i + c;
                let mut s = 0.0;
                for e in r.offsets[row]..r.offsets[row + 1] {
                    s += r.vals[e] * d[r.cols[e] as usize];
                }
                *oc = s;
            }
        });
    }

    /// `out_k = Σ_i K_ikᵀ e_i` (the adjoint of [`Self::apply`]).
    pub fn apply_t(&self, e: &[V3], out: &mut [f64]) {
        let flat: Vec<f64> = e.iter().flat_map(|v| v.iter().copied()).collect();
        self.rows.apply_t(&flat, out);
    }

    /// `Σ_i |K_ik|²` over the valid rows, for every column `k`.
    pub fn col_sq(&self) -> Vec<f64> {
        let w: Vec<f64> = self.valid.iter().flat_map(|&v| [v as u8 as f64; 3]).collect();
        self.rows.col_sq_weighted(&w)
    }
}

/// Node `i`'s entries (interleaved `(column, value)` for the components
/// `p, √2 q, r` of each neighbour and of itself), its base `B_i`, and whether
/// the fit is well posed (at least `NQ` neighbours, a non-singular system).
fn node_rows(i: usize, nbrs: &[(usize, f64)], points: &[V3], normals: &[V3]) -> NodeRows {
    let ni = normals[i];
    let (e1, e2) = tangent_frame(ni);
    let design: Vec<[f64; NQ]> = nbrs
        .iter()
        .map(|&(j, _)| {
            let dp = sub(points[j], points[i]);
            let (u, v) = (dot(dp, e1), dot(dp, e2));
            [u, v, 0.5 * u * u, u * v, 0.5 * v * v]
        })
        .collect();
    let mut a = [0.0; NQ * NQ];
    for (x, &(_, w)) in design.iter().zip(nbrs) {
        for p in 0..NQ {
            for q in 0..NQ {
                a[p * NQ + q] += w * x[p] * x[q];
            }
        }
    }
    // rows 2..5 of A⁻¹ (A is symmetric, so they are its columns)
    let mut inv = [[0.0; NQ]; 3];
    let mut ok = nbrs.len() >= NQ;
    for (c, row) in inv.iter_mut().enumerate() {
        let mut m = a;
        row[c + 2] = 1.0;
        ok &= cholesky_solve(&mut m, row, NQ);
    }
    let factor = [1.0, std::f64::consts::SQRT_2, 1.0];
    let mut entries = Vec::with_capacity(3 * (nbrs.len() + 1));
    let mut own = [0.0; 3];
    let mut base = [0.0; 3];
    for (x, &(j, w)) in design.iter().zip(nbrs) {
        let h = dot(ni, sub(points[j], points[i]));
        let along = dot(ni, normals[j]);
        for c in 0..3 {
            let g = factor[c] * w * (0..NQ).map(|p| inv[c][p] * x[p]).sum::<f64>();
            entries.push((j as u32, g * along));
            own[c] -= g;
            base[c] += g * h;
        }
    }
    entries.extend(own.iter().map(|&g| (i as u32, g)));
    if !ok {
        entries.clear();
        base = [0.0; 3];
    }
    (entries, base, ok)
}

/// Two unit tangent vectors completing `n` to a right-handed frame.
fn tangent_frame(n: V3) -> (V3, V3) {
    let a = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1 = cross(n, a);
    let e1 = scale(e1, 1.0 / norm(e1));
    (e1, cross(n, e1))
}
