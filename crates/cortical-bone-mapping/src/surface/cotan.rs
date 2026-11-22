//! The cotangent Laplace–Beltrami operator of a mesh (Pinkall & Polthier
//! 1993; Meyer et al. 2003).
//!
//! ```text
//! (L x)_i = Σ_{k ∈ N(i)} (cot α_ik + cot β_ik) / (2 A_i) · (x_k − x_i)
//! ```
//!
//! with `α_ik`, `β_ik` the angles opposite edge `ik` in its two faces and `A_i`
//! the area associated with vertex `i`. Applied to the vertex positions,
//! `½ L x = H n` is the mean-curvature normal (`H` the mean curvature, `n` the
//! inward normal for an outward oriented mesh), so `|½ L x| = 1/R` on a sphere
//! of radius `R`. Each cotangent is computed as `cos/sin = (e₁ · e₂) / |e₁ × e₂|`.
//!
//! Two simplifications, both standard and both safe on the near-uniform
//! meshes of surface nets: `A_i` is the barycentric area (a third of the area
//! of the incident faces) rather than the mixed Voronoi area of Meyer et al.,
//! and negative weights (edges opposite obtuse angles) are clamped to zero,
//! which keeps `−L` positive semi-definite and the penalty convex at the cost
//! of a small bias where triangles are obtuse. Boundary vertices (on an open
//! sub-mesh) get no valid row.
//!
//! # References
//!
//! * U. Pinkall, K. Polthier, *Computing discrete minimal surfaces and their
//!   conjugates*, Experimental Mathematics 2(1):15–36, 1993.
//! * M. Meyer, M. Desbrun, P. Schröder, A. H. Barr, *Discrete
//!   differential-geometry operators for triangulated 2-manifolds*, in
//!   H.-C. Hege, K. Polthier (eds.), Visualization and Mathematics III,
//!   Springer, 2003, pp. 35–57.

use std::collections::HashMap;

use rayon::prelude::*;

use super::mesh::Mesh;
use crate::util::{add, cross, dot, norm, scale, sub, V3};

/// A cotangent Laplace–Beltrami operator on a mesh, stored row-wise with the
/// transposed value alongside each entry so that both `L x` and `Lᵀ y` are
/// gather operations (parallel, no atomics).
///
/// `(L x)_i = Σ_k L_ik (x_k − x_i)` with `L_ik = (cot α_ik + cot β_ik) / (2 A_i)`
/// (negative cotangents clamped to zero) and `A_i` one third of the area of the
/// faces around `i`. For a surface, `½ L x = H n` is the mean-curvature
/// normal (`H` the mean curvature, `n` the inward normal for an outward
/// oriented mesh), so `|½ L x| = 1/R` on a sphere of radius `R`.
pub struct CotanLaplacian {
    pub offsets: Vec<usize>,
    pub cols: Vec<u32>,
    /// `L_ik` for the entry `(i, k)` of row `i`.
    pub vals: Vec<f64>,
    /// `L_ki` for the same entry (the transposed value).
    pub tvals: Vec<f64>,
    /// `L_ii = −Σ_k L_ik`.
    pub diag: Vec<f64>,
    /// Rows that are valid (interior vertices with a non-degenerate ring).
    pub interior: Vec<bool>,
}

impl CotanLaplacian {
    /// Assemble the operator: per face, add `½ cot` of each angle to the weight of
    /// the opposite edge (`cot = e₁·e₂ / |e₁ × e₂|`) and a third of the face area
    /// to each vertex; then `L_ik = max(w_ik, 0) / A_i` (module documentation for
    /// the clamping and the barycentric areas).
    pub fn new(mesh: &Mesh) -> Self {
        let n = mesh.vertices.len();
        let mut area = vec![0.0f64; n];
        let mut w: HashMap<(u32, u32), f64> = HashMap::with_capacity(mesh.faces.len() * 3 / 2);
        for f in &mesh.faces {
            let p = f.map(|i| mesh.vertices[i as usize]);
            let a2 = norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
            for &i in f {
                area[i as usize] += a2 / 6.0;
            }
            if a2 <= 0.0 {
                continue;
            }
            for k in 0..3 {
                // angle at vertex k, opposite edge (k+1, k+2)
                let (i, j) = (f[(k + 1) % 3], f[(k + 2) % 3]);
                let e1 = sub(p[(k + 1) % 3], p[k]);
                let e2 = sub(p[(k + 2) % 3], p[k]);
                let cot = dot(e1, e2) / a2;
                *w.entry((i.min(j), i.max(j))).or_insert(0.0) += 0.5 * cot;
            }
        }
        let adj = mesh.adjacency();
        let boundary = mesh.boundary_vertices();
        let mut cols = Vec::with_capacity(adj.neighbours.len());
        let mut vals = Vec::with_capacity(adj.neighbours.len());
        let mut tvals = Vec::with_capacity(adj.neighbours.len());
        let mut diag = vec![0.0; n];
        let mut interior = vec![false; n];
        for i in 0..n {
            let ai = area[i].max(1e-300);
            for &k in adj.of(i) {
                let wik = w
                    .get(&((i as u32).min(k), (i as u32).max(k)))
                    .copied()
                    .unwrap_or(0.0)
                    .max(0.0);
                let ak = area[k as usize].max(1e-300);
                cols.push(k);
                vals.push(wik / ai);
                tvals.push(wik / ak);
                diag[i] -= wik / ai;
            }
            interior[i] = !boundary[i] && !adj.of(i).is_empty() && area[i] > 0.0;
        }
        CotanLaplacian {
            offsets: adj.offsets,
            cols,
            vals,
            tvals,
            diag,
            interior,
        }
    }

    /// `out_i = ½ (L x)_i` for 3-vectors.
    pub fn half_apply3(&self, x: &[V3], out: &mut [V3]) {
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let mut acc = scale(x[i], self.diag[i]);
            for e in self.offsets[i]..self.offsets[i + 1] {
                acc = add(acc, scale(x[self.cols[e] as usize], self.vals[e]));
            }
            *o = scale(acc, 0.5);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cotan_mean_curvature_of_sphere() {
        let r = 3.0;
        let m = Mesh::uv_sphere(r, 80, 60);
        let lap = CotanLaplacian::new(&m);
        let mut h = vec![[0.0; 3]; m.vertices.len()];
        lap.half_apply3(&m.vertices, &mut h);
        // away from the poles, |H| ≈ 1/R and H points inwards
        for (i, p) in m.vertices.iter().enumerate() {
            if p[2].abs() > 0.8 * r {
                continue;
            }
            let hn = norm(h[i]);
            assert!((hn * r - 1.0).abs() < 0.02, "H·R = {}", hn * r);
            assert!(dot(h[i], *p) < 0.0);
        }
        assert!(lap.interior.iter().all(|&b| b));
    }

    #[test]
    fn transposed_values_are_consistent() {
        let m = Mesh::uv_sphere(1.0, 12, 9);
        let lap = CotanLaplacian::new(&m);
        for i in 0..m.vertices.len() {
            for e in lap.offsets[i]..lap.offsets[i + 1] {
                let k = lap.cols[e] as usize;
                let back = (lap.offsets[k]..lap.offsets[k + 1])
                    .find(|&f| lap.cols[f] as usize == i)
                    .unwrap();
                assert!((lap.tvals[e] - lap.vals[back]).abs() < 1e-12);
            }
        }
    }
}
