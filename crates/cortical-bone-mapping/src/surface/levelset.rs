//! Level-set embedding of a closed surface, for refinement with remeshing
//! (extension).
//!
//! The surface is the zero level of a signed distance `φ` (negative inside)
//! on a box of the image grid, clamped to `±band` (a narrow band: only values
//! near the surface matter; Osher & Sethian 1988). Moving the surface by a
//! displacement `s` along its inward normal is
//!
//! ```text
//! φ ← φ + s_ext
//! ```
//!
//! where `s_ext` extends the per-vertex displacements to the grid. For a
//! signed distance (`|∇φ| = 1`) and `s_ext` constant along the normals, this
//! moves the zero level by exactly `s`: a point `y` at distance `s` inside the
//! surface has `φ(y) = −s`, and after the update `φ(y) = 0`. The surface is then
//! extracted again from `φ` ([`super::nets`]), which keeps the node spacing at
//! the grid spacing however much the surface stretches, and cannot fold or
//! self-intersect, unlike moving mesh vertices.
//!
//! **Extension** (ours). The standard construction solves `∇s_ext · ∇φ = 0`
//! from the surface outwards (Adalsteinsson & Sethian 1999). Here each grid
//! point instead takes the Gaussian-weighted mean of the displacements of its
//! `k` nearest vertices (width about the mean edge length, weights relative to
//! the nearest vertex so they never underflow). Within the band this is
//! nearly constant along the normals, as required, and it also smooths the
//! displacements over about one edge length, which the noisy per-node fits
//! need anyway.
//!
//! **Re-distancing**. After a move `φ` is no longer a distance where the
//! displacement varied; it is reset to the exact distance to the new mesh
//! (the closest point over the faces around the 4 nearest vertices;
//! Ericson 2005) on the voxels that changed, keeping its sign. Exact distances
//! avoid the slow drift of PDE-based re-initialisation, at the cost of a
//! nearest-vertex search per voxel, which only the changed band pays.
//!
//! # References
//!
//! * S. Osher, J. A. Sethian, *Fronts propagating with curvature-dependent
//!   speed: algorithms based on Hamilton–Jacobi formulations*, Journal of
//!   Computational Physics 79(1):12–49, 1988.
//! * D. Adalsteinsson, J. A. Sethian, *The fast construction of extension
//!   velocities in level set methods*, Journal of Computational Physics
//!   148(1):2–22, 1999.
//! * C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2005,
//!   §5.1.5.

use ndarray::{Array3, ArrayView3, Zip};
use rayon::prelude::*;

use crate::kdtree::KdTree;
use crate::surface::closest::MeshPoints;
use crate::surface::mesh::Mesh;
use crate::surface::nets::{extract_level_set_coarse, Extracted};
use crate::surface::voxelize::index_box;
use crate::util::V3;
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
    /// The surface is extracted on every `step`-th voxel
    /// ([`SurfaceOptions::grid_step`](crate::options::SurfaceOptions::grid_step)).
    pub step: usize,
}

impl LevelSet {
    /// The signed distance to a closed mesh on the image voxels within `margin`
    /// (physical units) of its bounding box: the sign from rasterising the mesh
    /// (even–odd rule), the magnitude from the exact distance to the mesh,
    /// clamped to `band`.
    pub fn from_mesh(mesh: &Mesh, dims: [usize; 3], geom: &Geometry, margin: f64, band: f64, step: usize) -> LevelSet {
        let (lo, hi) = index_box(mesh, dims, geom, margin);
        let shape = (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1);
        log::debug!(
            "level set: grid {shape:?} at {lo:?} of an image {dims:?} ({} voxels), band {band:.3}",
            shape.0 * shape.1 * shape.2
        );
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
            step,
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
        extract_level_set_coarse(&self.phi.view(), self.lo, self.dims, &self.geom, self.step)
    }

    /// Move the surface: `s[v]` is the displacement of mesh vertex `v` along the
    /// inward normal (physical units; negative = outwards). It is extended to the
    /// grid by Gaussian-weighted averaging over the `neighbours` nearest vertices
    /// (width `width`; module documentation) and added to `φ` within the band,
    /// `φ ← clamp(φ + s_ext, ±band)`. Returns the voxels that changed.
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

    /// Keep the voxels marked in `forbidden` (image-shaped) outside the surface,
    /// `φ ← max(φ, h/2)` there (`h` the smallest spacing): half a voxel keeps the
    /// extracted surface between the forbidden voxel and its neighbours. Changed
    /// voxels are marked in `changed`.
    pub fn forbid(&mut self, forbidden: &ArrayView3<bool>, changed: &mut Array3<bool>) {
        let h = 0.5 * self.geom.min_spacing() as f32;
        let lo = self.lo;
        Zip::indexed(&mut self.phi)
            .and(changed)
            .par_for_each(|(i, j, k), phi, ch| {
                if forbidden[[lo[0] + i, lo[1] + j, lo[2] + k]] && *phi < h {
                    *phi = h;
                    *ch = true;
                }
            });
    }

    /// Reset `φ` to the signed distance to `mesh` on the voxels `only` (all if
    /// `None`), keeping the sign: the distance to the closest point on the faces
    /// around the 4 nearest vertices (a closest point on a mesh lies on a face of
    /// a nearby vertex; 4 guards against the nearest vertex not owning the
    /// closest face), clamped to `band`; voxels farther than the band from every
    /// vertex get `±band`.
    pub fn redistance(&mut self, mesh: &Mesh, only: Option<&Array3<bool>>) {
        if mesh.vertices.is_empty() {
            return;
        }
        let points = MeshPoints::new(mesh);
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
                let d = points.closest(q, reach).map_or(band, |c| c.dist.min(band));
                Some((sign * d) as f32)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::nets::extract_surface;
    use crate::util::{norm, sub};

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
    fn sphere_round_trip_and_uniform_move() {
        let n = 40;
        let c = (n as f64 - 1.0) / 2.0;
        let g = Geometry::new([1.0; 3]);
        let mut mesh = extract_surface(&ball(n, 12.0).view(), &g, 1.0).mesh;
        mesh.taubin_smooth(50, 0.5, 0.1, None);
        let (r0, _) = mean_radius(&mesh, c);
        let mut ls = LevelSet::from_mesh(&mesh, [n; 3], &g, 4.0, 4.0, 1);
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
