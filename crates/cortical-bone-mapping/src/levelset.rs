//! Level-set embedding of a closed surface, for refinement with remeshing
//! (extension).
//!
//! The surface is the zero level of a signed distance `φ` (negative inside)
//! on a box of the image grid, clamped to `±band`. Moving the surface by a
//! displacement `s` along its inward normal is `φ ← φ + s_ext`, where `s_ext`
//! extends the per-vertex displacements to the grid; for a signed distance
//! with `s_ext` constant along the normals this moves the zero level by
//! exactly `s`. The surface is then extracted again from `φ` (surface nets),
//! which keeps the node spacing at the grid spacing however much the surface
//! stretches, and cannot fold or self-intersect. Finally `φ` is reset to the
//! distance to the new mesh where it changed, so it stays a signed distance.

use ndarray::{Array3, Zip};
use rayon::prelude::*;

use crate::kdtree::KdTree;
use crate::mesh::Mesh;
use crate::surface::{extract_level_set, Extracted};
use crate::util::{add, dot, norm, scale, sub, V3};
use crate::volume::Geometry;

pub struct LevelSet {
    /// Signed distance (physical units, negative inside), clamped to ±band.
    pub phi: Array3<f32>,
    /// Image index of `phi[[0, 0, 0]]`.
    pub lo: [usize; 3],
    /// Shape of the image.
    pub dims: [usize; 3],
    pub band: f64,
    pub geom: Geometry,
}

impl LevelSet {
    /// The signed distance to a closed mesh on the image voxels within
    /// `margin` (physical units) of its bounding box.
    pub fn from_mesh(mesh: &Mesh, dims: [usize; 3], geom: &Geometry, margin: f64, band: f64) -> LevelSet {
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        for &p in &mesh.vertices {
            let q = geom.to_index(p);
            for a in 0..3 {
                let m = (margin / geom.spacing[a]).ceil();
                let l = (q[a] - m).floor().max(0.0) as usize;
                let h = ((q[a] + m).ceil().max(0.0) as usize).min(dims[a] - 1);
                lo[a] = lo[a].min(l);
                hi[a] = hi[a].max(h);
            }
        }
        let shape = (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1);
        let sub_geom = Geometry {
            spacing: geom.spacing,
            origin: geom.to_physical(lo.map(|v| v as f64)),
        };
        let inside = mesh.voxelize(shape, &sub_geom);
        let mut ls = LevelSet {
            phi: inside.mapv(|b| if b { -band as f32 } else { band as f32 }),
            lo,
            dims,
            band,
            geom: *geom,
        };
        ls.redistance(mesh, None);
        ls
    }

    /// Physical position of grid voxel `(i, j, k)`.
    #[inline]
    fn position(&self, i: usize, j: usize, k: usize) -> V3 {
        self.geom.to_physical([
            (self.lo[0] + i) as f64,
            (self.lo[1] + j) as f64,
            (self.lo[2] + k) as f64,
        ])
    }

    /// The zero level as a mesh.
    pub fn extract(&self) -> Extracted {
        extract_level_set(&self.phi.view(), self.lo, self.dims, &self.geom)
    }

    /// Move the surface: `s[v]` is the displacement of mesh vertex `v` along
    /// the inward normal (physical units; negative = outwards). It is
    /// extended to the grid by Gaussian-weighted averaging over the
    /// `neighbours` nearest vertices (width `width`). Returns the voxels
    /// that changed.
    pub fn advance(&mut self, points: &[V3], s: &[f64], width: f64, neighbours: usize) -> Array3<bool> {
        let tree = KdTree::new(points);
        let band = self.band as f32;
        let inv = 1.0 / (2.0 * width * width);
        let lo = self.lo;
        let geom = self.geom;
        let mut changed = Array3::from_elem(self.phi.dim(), false);
        Zip::indexed(&mut self.phi)
            .and(&mut changed)
            .par_for_each(|(i, j, k), phi, ch| {
                if phi.abs() >= band {
                    return;
                }
                let q = geom.to_physical([(lo[0] + i) as f64, (lo[1] + j) as f64, (lo[2] + k) as f64]);
                let nn = tree.knn(q, neighbours, None);
                let d0 = nn.first().map(|n| n.1).unwrap_or(0.0);
                let (mut ws, mut acc) = (0.0, 0.0);
                for &(v, d2) in &nn {
                    // relative to the nearest, so far voxels still get weights
                    let w = (-(d2 - d0) * inv).exp();
                    ws += w;
                    acc += w * s[v as usize];
                }
                if ws > 0.0 {
                    let e = acc / ws;
                    if e.abs() > 1e-9 {
                        *phi = (*phi as f64 + e).clamp(-(band as f64), band as f64) as f32;
                        *ch = true;
                    }
                }
            });
        changed
    }

    /// Reset `φ` to the signed distance to `mesh` (the sign is kept) on the
    /// voxels `only` (all if `None`).
    pub fn redistance(&mut self, mesh: &Mesh, only: Option<&Array3<bool>>) {
        if mesh.vertices.is_empty() {
            return;
        }
        let tree = KdTree::new(&mesh.vertices);
        let vf = VertexFaces::new(mesh);
        let band = self.band;
        let reach = band + 2.0 * self.geom.spacing.iter().cloned().fold(0.0, f64::max);
        let (_, s1, s2) = self.phi.dim();
        let new: Vec<Option<f32>> = (0..self.phi.len())
            .into_par_iter()
            .map(|f| {
                let (i, j, k) = (f / (s1 * s2), (f / s2) % s1, f % s2);
                if let Some(o) = only {
                    if !o[[i, j, k]] {
                        return None;
                    }
                }
                let q = self.position(i, j, k);
                let sign = if self.phi[[i, j, k]] < 0.0 { -1.0 } else { 1.0 };
                let nn = tree.knn(q, 4, None);
                if nn.is_empty() || nn[0].1.sqrt() > reach {
                    return Some((sign * band) as f32);
                }
                let mut best = f64::INFINITY;
                for &(v, _) in &nn {
                    for &f in vf.of(v as usize) {
                        let t = mesh.faces[f as usize].map(|x| mesh.vertices[x as usize]);
                        best = best.min(norm(sub(q, closest_on_triangle(q, t[0], t[1], t[2]))));
                    }
                }
                Some((sign * best.min(band)) as f32)
            })
            .collect();
        let phi = self.phi.as_slice_mut().expect("phi is in standard layout");
        for (p, v) in phi.iter_mut().zip(new) {
            if let Some(v) = v {
                *p = v;
            }
        }
    }
}

/// Faces incident to each vertex.
struct VertexFaces {
    offsets: Vec<usize>,
    faces: Vec<u32>,
}

impl VertexFaces {
    fn new(mesh: &Mesh) -> Self {
        let n = mesh.vertices.len();
        let mut offsets = vec![0usize; n + 1];
        for f in &mesh.faces {
            for &v in f {
                offsets[v as usize + 1] += 1;
            }
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut fill = offsets.clone();
        let mut faces = vec![0u32; offsets[n]];
        for (fi, f) in mesh.faces.iter().enumerate() {
            for &v in f {
                faces[fill[v as usize]] = fi as u32;
                fill[v as usize] += 1;
            }
        }
        VertexFaces { offsets, faces }
    }
    fn of(&self, v: usize) -> &[u32] {
        &self.faces[self.offsets[v]..self.offsets[v + 1]]
    }
}

/// Closest point to `p` on triangle `abc` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
pub fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> V3 {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add(a, add(scale(ab, v), scale(ac, w)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::extract_surface;

    fn ball(n: usize, r: f64) -> Array3<u8> {
        let c = (n as f64 - 1.0) / 2.0;
        Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            let d = ((i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2)).sqrt();
            (d < r) as u8
        })
    }

    fn mean_radius(m: &Mesh, c: f64) -> (f64, f64) {
        let r: Vec<f64> = m.vertices.iter().map(|&p| norm(sub(p, [c; 3]))).collect();
        let mean = r.iter().sum::<f64>() / r.len() as f64;
        let sd = (r.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
        (mean, sd)
    }

    #[test]
    fn closest_point_regions() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(norm(sub(closest_on_triangle([0.2, 0.2, 1.0], a, b, c), [0.2, 0.2, 0.0])) < 1e-12);
        assert_eq!(closest_on_triangle([-1.0, -1.0, 0.0], a, b, c), a);
        assert_eq!(closest_on_triangle([0.5, -1.0, 0.0], a, b, c), [0.5, 0.0, 0.0]);
        let p = closest_on_triangle([1.0, 1.0, 0.0], a, b, c);
        assert!((p[0] - 0.5).abs() < 1e-12 && (p[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn sphere_round_trip_and_uniform_move() {
        let n = 40;
        let c = (n as f64 - 1.0) / 2.0;
        let g = Geometry::new([1.0; 3]);
        let mut mesh = extract_surface(&ball(n, 12.0).view(), &g, 1.0).mesh;
        mesh.taubin_smooth(50, 0.5, 0.1, None);
        let (r0, _) = mean_radius(&mesh, c);
        let mut ls = LevelSet::from_mesh(&mesh, [n; 3], &g, 4.0, 4.0);
        // re-extraction reproduces the surface
        let e = ls.extract();
        assert!(e.mesh.is_closed_manifold());
        let (r1, sd1) = mean_radius(&e.mesh, c);
        assert!((r1 - r0).abs() < 0.05, "{r0} {r1}");
        assert!(sd1 < 0.1, "{sd1}");
        // move 1.5 inwards everywhere
        let s = vec![1.5; e.mesh.vertices.len()];
        let ch = ls.advance(&e.mesh.vertices, &s, 1.0, 8);
        let e2 = ls.extract();
        ls.redistance(&e2.mesh, Some(&ch));
        let (r2, sd2) = mean_radius(&e2.mesh, c);
        assert!((r2 - (r1 - 1.5)).abs() < 0.05, "{r1} {r2}");
        assert!(sd2 < 0.1);
        // and a second, outward move after re-distancing
        let s = vec![-2.0; e2.mesh.vertices.len()];
        ls.advance(&e2.mesh.vertices, &s, 1.0, 8);
        let (r3, _) = mean_radius(&ls.extract().mesh, c);
        assert!((r3 - (r2 + 2.0)).abs() < 0.08, "{r2} {r3}");
    }
}
