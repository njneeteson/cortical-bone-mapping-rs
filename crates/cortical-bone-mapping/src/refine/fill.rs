//! Moving the vertices without a usable fit: each connected region of them
//! moves as a membrane spanned by the displacements of the trusted vertices
//! around it (ours).
//!
//! The displacements `u` of the unknown vertices solve the discrete Laplace
//! equation `u_i = mean_{j ∈ N(i)} u_j`, with the known displacements as
//! Dirichlet boundary values. This harmonic extension is the smoothest one (it
//! minimises the Dirichlet energy `Σ_edges |u_i − u_j|²`), so a region follows
//! its boundary instead of tearing away from it, and by the maximum principle
//! no vertex moves further than the boundary does. It is solved exactly, by
//! conjugate gradients on the (symmetric positive definite) system
//!
//! ```text
//! deg_i u_i − Σ_{j ∈ N(i), unknown} u_j = Σ_{j ∈ N(i), known} u_j,
//! ```
//!
//! with the degrees as preconditioner. Jacobi iteration spreads the boundary
//! values by about one ring per sweep and needs on the order of `R²` sweeps
//! for a region `R` rings across; a fixed number of sweeps left the middle of
//! a wide region (e.g. a surface bridging a notch) behind its rim. A region
//! with no known vertex around it has no boundary values and stays put.

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::surface::mesh::Adjacency;
use crate::util::{add, det_sum, dot, scale, sub, V3};

/// Relative residual at which the solve stops.
const TOLERANCE: f64 = 1e-8;

/// Harmonic extension of the known displacements `vals` (where `known`) to
/// the other vertices (module documentation).
pub(super) fn harmonic_fill(adj: &Adjacency, known: &[bool], vals: &mut [V3]) {
    let anchored = anchored_regions(adj, known);
    let unknown: Vec<usize> = (0..known.len()).filter(|&i| !known[i]).collect();
    for &i in &unknown {
        vals[i] = [0.0; 3];
    }
    let free: Vec<usize> = unknown.into_iter().filter(|&i| anchored[i]).collect();
    if free.is_empty() {
        return;
    }
    let mut index = vec![usize::MAX; known.len()];
    for (k, &i) in free.iter().enumerate() {
        index[i] = k;
    }
    let system = Membrane {
        adj,
        free: &free,
        index: &index,
    };
    let rhs: Vec<V3> = free
        .par_iter()
        .map(|&i| {
            adj.of(i)
                .iter()
                .filter(|&&j| known[j as usize])
                .fold([0.0; 3], |acc, &j| add(acc, vals[j as usize]))
        })
        .collect();
    let u = system.solve(&rhs);
    for (k, &i) in free.iter().enumerate() {
        vals[i] = u[k];
    }
}

/// Whether each unknown vertex lies in a connected region of unknown
/// vertices with at least one known vertex next to it.
fn anchored_regions(adj: &Adjacency, known: &[bool]) -> Vec<bool> {
    let n = known.len();
    let mut region = vec![usize::MAX; n];
    let mut anchored_region = Vec::new();
    let mut queue = VecDeque::new();
    for seed in 0..n {
        if known[seed] || region[seed] != usize::MAX {
            continue;
        }
        let id = anchored_region.len();
        let mut anchored = false;
        region[seed] = id;
        queue.push_back(seed);
        while let Some(i) = queue.pop_front() {
            for &j in adj.of(i) {
                let j = j as usize;
                if known[j] {
                    anchored = true;
                } else if region[j] == usize::MAX {
                    region[j] = id;
                    queue.push_back(j);
                }
            }
        }
        anchored_region.push(anchored);
    }
    (0..n).map(|i| !known[i] && anchored_region[region[i]]).collect()
}

/// The membrane system over the free vertices (`index[i]` is the position of
/// vertex `i` among them, `usize::MAX` for the others).
struct Membrane<'a> {
    adj: &'a Adjacency,
    free: &'a [usize],
    index: &'a [usize],
}

impl Membrane<'_> {
    /// `out = A x`.
    fn apply(&self, x: &[V3], out: &mut [V3]) {
        out.par_iter_mut().enumerate().for_each(|(k, o)| {
            let ring = self.adj.of(self.free[k]);
            let mut acc = scale(x[k], ring.len() as f64);
            for &j in ring {
                let kj = self.index[j as usize];
                if kj != usize::MAX {
                    acc = sub(acc, x[kj]);
                }
            }
            *o = acc;
        });
    }

    /// Solve `A u = b` for the three coordinates at once by preconditioned
    /// conjugate gradients (one search per coordinate, sharing the products).
    fn solve(&self, b: &[V3]) -> Vec<V3> {
        let n = b.len();
        let inv_deg: Vec<f64> = self.free.iter().map(|&i| 1.0 / self.adj.of(i).len() as f64).collect();
        let precondition = |r: &[V3]| -> Vec<V3> { r.par_iter().zip(&inv_deg).map(|(v, &d)| scale(*v, d)).collect() };
        let dots = |a: &[V3], c: &[V3]| -> V3 { [0, 1, 2].map(|axis| det_sum(n, |k| a[k][axis] * c[k][axis])) };
        let mut u = vec![[0.0; 3]; n];
        let mut r = b.to_vec();
        let mut z = precondition(&r);
        let mut p = z.clone();
        let mut rz = dots(&r, &z);
        let norm_b = dots(b, b);
        let mut ap = vec![[0.0; 3]; n];
        for _ in 0..n.max(1) {
            let rr = dots(&r, &r);
            if (0..3).all(|a| rr[a] <= TOLERANCE * TOLERANCE * norm_b[a].max(1e-300)) {
                break;
            }
            self.apply(&p, &mut ap);
            let pap = dots(&p, &ap);
            let alpha = [0, 1, 2].map(|a| if pap[a] > 0.0 { rz[a] / pap[a] } else { 0.0 });
            u.par_iter_mut()
                .zip(&p)
                .for_each(|(v, pk)| *v = add(*v, mul3(alpha, *pk)));
            r.par_iter_mut()
                .zip(&ap)
                .for_each(|(v, a)| *v = sub(*v, mul3(alpha, *a)));
            z = precondition(&r);
            let rz_new = dots(&r, &z);
            let beta = [0, 1, 2].map(|a| if rz[a] > 0.0 { rz_new[a] / rz[a] } else { 0.0 });
            p.par_iter_mut()
                .zip(&z)
                .for_each(|(v, zk)| *v = add(*zk, mul3(beta, *v)));
            rz = rz_new;
        }
        debug_assert!(dot(rz, rz).is_finite());
        u
    }
}

/// Component-wise product.
fn mul3(a: V3, b: V3) -> V3 {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::mesh::Mesh;

    /// A strip of `w × h` vertices on a grid, triangulated.
    fn grid(w: usize, h: usize) -> Mesh {
        let vertices = (0..h)
            .flat_map(|y| (0..w).map(move |x| [x as f64, y as f64, 0.0]))
            .collect();
        let mut faces = Vec::new();
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                let i = (y * w + x) as u32;
                let w = w as u32;
                faces.push([i, i + 1, i + w]);
                faces.push([i + 1, i + w + 1, i + w]);
            }
        }
        Mesh { vertices, faces }
    }

    #[test]
    fn a_wide_region_moves_with_its_boundary() {
        // a 61-vertex-wide unknown band between two columns moved by 1:
        // the exact harmonic fill moves all of it by 1 (Jacobi with 50
        // sweeps left its middle far behind)
        let (w, h) = (63, 5);
        let mesh = grid(w, h);
        let adj = mesh.adjacency();
        let known: Vec<bool> = (0..w * h).map(|i| i % w == 0 || i % w == w - 1).collect();
        let mut vals: Vec<V3> = (0..w * h)
            .map(|i| if known[i] { [0.0, 0.0, 1.0] } else { [0.0; 3] })
            .collect();
        harmonic_fill(&adj, &known, &mut vals);
        for v in &vals {
            assert!((v[2] - 1.0).abs() < 1e-6, "{v:?}");
        }
    }

    #[test]
    fn linear_boundary_values_are_interpolated_linearly() {
        // known on the whole border (the umbrella average of a linear
        // function is linear only where the ring is symmetric)
        let (w, h) = (21, 6);
        let mesh = grid(w, h);
        let adj = mesh.adjacency();
        let known: Vec<bool> = (0..w * h)
            .map(|i| i % w == 0 || i % w == w - 1 || i / w == 0 || i / w == h - 1)
            .collect();
        let mut vals: Vec<V3> = (0..w * h)
            .map(|i| if known[i] { [(i % w) as f64, 0.0, 0.0] } else { [0.0; 3] })
            .collect();
        harmonic_fill(&adj, &known, &mut vals);
        for (i, v) in vals.iter().enumerate() {
            assert!((v[0] - (i % w) as f64).abs() < 1e-6);
        }
    }

    #[test]
    fn a_region_without_boundary_stays_put() {
        let mesh = grid(6, 6);
        let adj = mesh.adjacency();
        let known = vec![false; 36];
        let mut vals = vec![[1.0, 2.0, 3.0]; 36];
        harmonic_fill(&adj, &known, &mut vals);
        assert!(vals.iter().all(|v| *v == [0.0; 3]));
    }
}
