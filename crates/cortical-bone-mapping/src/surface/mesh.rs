//! Triangle meshes: normals, adjacency, Taubin smoothing, sub-meshes and
//! topology checks. Rasterisation ([`super::voxelize`]), VTK output
//! ([`super::vtk`]) and the cotangent Laplacian ([`super::cotan`]) are in
//! their own modules.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::util::{add, cross, dot, norm, scale, sub, V3};

/// An indexed triangle mesh. Faces are wound counter-clockwise seen from the
/// outside, so face normals point outwards.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<V3>,
    pub faces: Vec<[u32; 3]>,
}

/// Compressed vertex adjacency (one-ring neighbours).
pub struct Adjacency {
    pub offsets: Vec<usize>,
    pub neighbours: Vec<u32>,
}

impl Adjacency {
    #[inline]
    pub fn of(&self, i: usize) -> &[u32] {
        &self.neighbours[self.offsets[i]..self.offsets[i + 1]]
    }
}

impl Mesh {
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Area-weighted vertex normals (unit length, pointing outwards).
    pub fn vertex_normals(&self) -> Vec<V3> {
        let mut acc = vec![[0.0; 3]; self.vertices.len()];
        for f in &self.faces {
            let [a, b, c] = f.map(|i| self.vertices[i as usize]);
            let n = cross(sub(b, a), sub(c, a)); // |n| = 2 · area
            for &i in f {
                acc[i as usize] = add(acc[i as usize], n);
            }
        }
        acc.into_par_iter()
            .map(|n| {
                let l = norm(n);
                if l > 0.0 {
                    scale(n, 1.0 / l)
                } else {
                    [0.0; 3]
                }
            })
            .collect()
    }

    /// One-ring neighbours of every vertex (sorted, de-duplicated).
    pub fn adjacency(&self) -> Adjacency {
        let n = self.vertices.len();
        let mut deg = vec![0usize; n + 1];
        for f in &self.faces {
            for &i in f {
                deg[i as usize] += 2;
            }
        }
        let mut offsets = vec![0usize; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + deg[i];
        }
        let mut fill = offsets.clone();
        let mut nb = vec![0u32; offsets[n]];
        for f in &self.faces {
            for k in 0..3 {
                let (i, j, l) = (f[k], f[(k + 1) % 3], f[(k + 2) % 3]);
                nb[fill[i as usize]] = j;
                nb[fill[i as usize] + 1] = l;
                fill[i as usize] += 2;
            }
        }
        // sort + dedup each ring, then compact
        let mut out_off = vec![0usize; n + 1];
        let mut out = Vec::with_capacity(nb.len() / 2);
        for i in 0..n {
            let ring = &mut nb[offsets[i]..offsets[i + 1]];
            ring.sort_unstable();
            let mut last = u32::MAX;
            for &j in ring.iter() {
                if j != last {
                    out.push(j);
                    last = j;
                }
            }
            out_off[i + 1] = out.len();
        }
        Adjacency {
            offsets: out_off,
            neighbours: out,
        }
    }

    /// Taubin λ|μ smoothing (Taubin 1995) with the uniform umbrella operator:
    /// alternating shrinking (`λ > 0`) and inflating (`μ < 0`) Laplacian steps.
    /// `μ` is derived from the pass-band frequency `k_pb` via
    /// `1/λ + 1/μ = k_pb`. Low-pass filters the surface without shrinking it.
    /// Vertices with `pinned[i]` stay put.
    pub fn taubin_smooth(&mut self, iterations: usize, lambda: f64, passband: f64, pinned: Option<&[bool]>) {
        if iterations == 0 || self.vertices.is_empty() {
            return;
        }
        let mu = 1.0 / (passband - 1.0 / lambda);
        let adj = self.adjacency();
        let mut next = self.vertices.clone();
        for _ in 0..iterations {
            for &w in &[lambda, mu] {
                let cur = &self.vertices;
                next.par_iter_mut().enumerate().for_each(|(i, out)| {
                    let ring = adj.of(i);
                    if ring.is_empty() || pinned.is_some_and(|p| p[i]) {
                        *out = cur[i];
                        return;
                    }
                    let mut c = [0.0; 3];
                    for &j in ring {
                        c = add(c, cur[j as usize]);
                    }
                    let c = scale(c, 1.0 / ring.len() as f64);
                    *out = add(cur[i], scale(sub(c, cur[i]), w));
                });
                std::mem::swap(&mut self.vertices, &mut next);
            }
        }
    }

    /// A closed UV sphere of radius `r` centred at the origin with `nu`
    /// longitudinal and `nv` latitudinal divisions (useful for tests and demos).
    pub fn uv_sphere(r: f64, nu: usize, nv: usize) -> Mesh {
        let mut vertices = vec![[0.0, 0.0, r]];
        for i in 1..nv {
            let th = std::f64::consts::PI * i as f64 / nv as f64;
            for j in 0..nu {
                let ph = 2.0 * std::f64::consts::PI * j as f64 / nu as f64;
                vertices.push([r * th.sin() * ph.cos(), r * th.sin() * ph.sin(), r * th.cos()]);
            }
        }
        vertices.push([0.0, 0.0, -r]);
        let last = vertices.len() as u32 - 1;
        let id = |i: usize, j: usize| (1 + (i - 1) * nu + j % nu) as u32;
        let mut faces = Vec::new();
        for j in 0..nu {
            faces.push([0, id(1, j), id(1, j + 1)]);
            faces.push([last, id(nv - 1, j + 1), id(nv - 1, j)]);
        }
        for i in 1..nv - 1 {
            for j in 0..nu {
                faces.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
                faces.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
            }
        }
        Mesh { vertices, faces }
    }

    /// Vertices on a boundary edge (an edge used by exactly one face).
    pub fn boundary_vertices(&self) -> Vec<bool> {
        let mut count: HashMap<(u32, u32), u32> = HashMap::with_capacity(self.faces.len() * 3 / 2);
        for f in &self.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        let mut out = vec![false; self.vertices.len()];
        for ((a, b), c) in count {
            if c == 1 {
                out[a as usize] = true;
                out[b as usize] = true;
            }
        }
        out
    }

    /// The sub-mesh of the selected vertices (faces with all three vertices
    /// selected) and, for each new vertex, its index in `self`.
    pub fn submesh(&self, keep: &[bool]) -> (Mesh, Vec<u32>) {
        let mut map = vec![u32::MAX; self.vertices.len()];
        let mut back = Vec::new();
        for (i, &k) in keep.iter().enumerate() {
            if k {
                map[i] = back.len() as u32;
                back.push(i as u32);
            }
        }
        let faces = self
            .faces
            .iter()
            .filter(|f| f.iter().all(|&i| keep[i as usize]))
            .map(|f| f.map(|i| map[i as usize]))
            .collect();
        let vertices = back.iter().map(|&i| self.vertices[i as usize]).collect();
        (Mesh { vertices, faces }, back)
    }

    /// `true` if every edge is shared by exactly two faces with opposite
    /// orientation.
    pub fn is_closed_manifold(&self) -> bool {
        let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
        for f in &self.faces {
            for k in 0..3 {
                *directed.entry((f[k], f[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
        directed
            .iter()
            .all(|(&(a, b), &c)| c == 1 && directed.get(&(b, a)) == Some(&1))
    }

    /// V − E + F over the vertices referenced by faces.
    pub fn euler_characteristic(&self) -> i64 {
        let mut edges = std::collections::HashSet::new();
        let mut used = vec![false; self.vertices.len()];
        for f in &self.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                edges.insert((a.min(b), a.max(b)));
                used[a as usize] = true;
            }
        }
        let v = used.iter().filter(|&&u| u).count() as i64;
        v - edges.len() as i64 + self.faces.len() as i64
    }

    /// Enclosed volume (positive for an outward-oriented closed mesh).
    pub fn signed_volume(&self) -> f64 {
        crate::util::det_sum(self.faces.len(), |k| {
            let [a, b, c] = self.faces[k].map(|i| self.vertices[i as usize]);
            dot(a, cross(b, c)) / 6.0
        })
    }

    /// Mean edge length.
    pub fn mean_edge_length(&self) -> f64 {
        if self.faces.is_empty() {
            return 0.0;
        }
        let s: f64 = self
            .faces
            .par_iter()
            .map(|f| {
                (0..3)
                    .map(|k| {
                        norm(sub(
                            self.vertices[f[k] as usize],
                            self.vertices[f[(k + 1) % 3] as usize],
                        ))
                    })
                    .sum::<f64>()
            })
            .sum();
        s / (3 * self.faces.len()) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_mesh_properties() {
        let m = Mesh::uv_sphere(2.0, 40, 30);
        assert!(m.is_closed_manifold());
        assert_eq!(m.euler_characteristic(), 2);
        let v = m.signed_volume();
        assert!((v / (4.0 / 3.0 * std::f64::consts::PI * 8.0) - 1.0).abs() < 0.02);
        // outward normals
        for (p, n) in m.vertices.iter().zip(m.vertex_normals()) {
            assert!(dot(*p, n) > 0.9 * norm(*p));
        }
    }

    #[test]
    fn taubin_reduces_noise_without_shrinking() {
        let mut m = Mesh::uv_sphere(5.0, 60, 40);
        let mut rng = crate::util::Rng::new(4);
        for v in m.vertices.iter_mut() {
            let s = 1.0 + 0.03 * rng.normal();
            *v = scale(*v, s);
        }
        let spread = |m: &Mesh| {
            let r: Vec<f64> = m.vertices.iter().map(|v| norm(*v)).collect();
            let mean = r.iter().sum::<f64>() / r.len() as f64;
            (
                mean,
                (r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / r.len() as f64).sqrt(),
            )
        };
        let (_, s0) = spread(&m);
        m.taubin_smooth(30, 0.5, 0.1, None);
        let (mean1, s1) = spread(&m);
        assert!(s1 < 0.3 * s0, "{s0} -> {s1}");
        assert!((mean1 - 5.0).abs() < 0.1, "mean radius {mean1}");
    }

    #[test]
    fn submesh_and_boundary() {
        let m = Mesh::uv_sphere(1.0, 12, 9);
        let keep: Vec<bool> = m.vertices.iter().map(|v| v[2] > 0.0).collect();
        let (s, back) = m.submesh(&keep);
        assert!(!s.is_closed_manifold());
        assert_eq!(back.len(), s.vertices.len());
        let b = s.boundary_vertices();
        assert!(b.iter().any(|&x| x) && b.iter().any(|&x| !x));
    }
}
