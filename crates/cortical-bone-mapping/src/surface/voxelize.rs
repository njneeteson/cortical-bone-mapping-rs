//! Rasterising the region enclosed by a closed mesh into a mask.
//!
//! A voxel is inside if a ray from its centre along array axis 0 crosses the
//! surface an odd number of times (the even–odd or parity rule of the Jordan
//! curve theorem; for voxelising meshes see Nooruddin & Turk 2003). All rays
//! share a direction, so each triangle is intersected only with the rays of
//! the voxel columns `(j, k)` inside its projection: for column `(y, z)` the
//! barycentric coordinates `(u, v)` of the projected triangle are solved from
//! the 2×2 system of its edges, and the crossing is at axis-0 coordinate
//! `x = p₀ + u (q₀ − p₀) + v (r₀ − p₀)` if `u, v ≥ 0, u + v ≤ 1`. Sorting the
//! crossings by column and position, each column is filled between
//! successive pairs.
//!
//! Rays through a vertex or an edge would be counted twice or not at all; the
//! rays are shifted by tiny irrational offsets (`≈ 10⁻⁷` voxel) so that they
//! never pass exactly through the lattice points where surface-nets vertices
//! can lie. Works for any closed mesh, including self-intersecting ones, where
//! the parity rule decides. Memory is the output plus 8 bytes per crossing.
//!
//! # References
//!
//! * F. S. Nooruddin, G. Turk, *Simplification and repair of polygonal models
//!   using volumetric techniques*, IEEE Transactions on Visualization and
//!   Computer Graphics 9(2):191–205, 2003.

use ndarray::Array3;
use rayon::prelude::*;

use super::mesh::Mesh;
use crate::util::V3;
use crate::volume::Geometry;

impl Mesh {
    /// Rasterise the region enclosed by the mesh into a mask of `shape`
    /// voxels: a voxel is inside if a ray from its centre along axis 0 crosses
    /// the surface an odd number of times (even–odd rule). Works for any
    /// closed mesh, including self-intersecting ones (where the parity rule
    /// decides). Memory is the output plus 8 bytes per ray crossing.
    pub fn voxelize(&self, shape: (usize, usize, usize), geom: &Geometry) -> Array3<bool> {
        let (_, d1, d2) = shape;
        let tri_idx: Vec<[V3; 3]> = self
            .faces
            .iter()
            .map(|f| f.map(|i| geom.to_index(self.vertices[i as usize])))
            .collect();
        let mut crossings: Vec<(u32, f32)> = tri_idx
            .par_iter()
            .flat_map_iter(|t| ray_crossings(t, d1, d2).into_iter())
            .collect();
        crossings.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        fill_columns(&crossings, shape)
    }
}

/// Tiny irrational offsets so rays never pass exactly through vertices or
/// edges lying on the integer lattice.
const EPS1: f64 = 1.234_567e-7;
const EPS2: f64 = 7.654_321e-8;

/// Where the axis-0 rays through the voxel columns `(j, k)` (`d1 × d2` of
/// them) cross the triangle `t` (index coordinates): (column id, axis-0
/// coordinate).
fn ray_crossings(t: &[V3; 3], d1: usize, d2: usize) -> Vec<(u32, f32)> {
    let mut out = Vec::new();
    let (min1, max1) = minmax(t[0][1], t[1][1], t[2][1]);
    let (min2, max2) = minmax(t[0][2], t[1][2], t[2][2]);
    let j0 = (min1 - EPS1).ceil().max(0.0) as isize;
    let j1 = ((max1 - EPS1).floor() as isize).min(d1 as isize - 1);
    let k0 = (min2 - EPS2).ceil().max(0.0) as isize;
    let k1 = ((max2 - EPS2).floor() as isize).min(d2 as isize - 1);
    let (p, q, r) = (t[0], t[1], t[2]);
    let det = (q[1] - p[1]) * (r[2] - p[2]) - (r[1] - p[1]) * (q[2] - p[2]);
    if det.abs() < 1e-300 {
        return out;
    }
    for j in j0..=j1 {
        for k in k0..=k1 {
            let y = j as f64 + EPS1;
            let z = k as f64 + EPS2;
            let u = ((y - p[1]) * (r[2] - p[2]) - (r[1] - p[1]) * (z - p[2])) / det;
            let v = ((q[1] - p[1]) * (z - p[2]) - (y - p[1]) * (q[2] - p[2])) / det;
            if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
                let x = p[0] + u * (q[0] - p[0]) + v * (r[0] - p[0]);
                out.push(((j as usize * d2 + k as usize) as u32, x as f32));
            }
        }
    }
    out
}

/// Fill each column between successive pairs of its (sorted) crossings.
fn fill_columns(crossings: &[(u32, f32)], shape: (usize, usize, usize)) -> Array3<bool> {
    let (d0, _, d2) = shape;
    let mut out = Array3::from_elem(shape, false);
    for xs in crossings.chunk_by(|a, b| a.0 == b.0) {
        let (j, k) = (xs[0].0 as usize / d2, xs[0].0 as usize % d2);
        for [first, second] in xs.as_chunks::<2>().0 {
            let a = (first.1 as f64).ceil().max(0.0) as usize;
            let b = (second.1 as f64).floor();
            if b < 0.0 {
                continue;
            }
            let b = (b as usize).min(d0 - 1);
            for i in a..=b {
                out[[i, j, k]] = true;
            }
        }
    }
    out
}

#[inline]
fn minmax(a: f64, b: f64, c: f64) -> (f64, f64) {
    (a.min(b).min(c), a.max(b).max(c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::add;

    #[test]
    fn voxelize_sphere() {
        let mut m = Mesh::uv_sphere(8.0, 64, 48);
        for v in m.vertices.iter_mut() {
            *v = add(*v, [12.0, 12.0, 12.0]);
        }
        let mask = m.voxelize((25, 25, 25), &Geometry::default());
        let count = mask.iter().filter(|&&b| b).count() as f64;
        let want = 4.0 / 3.0 * std::f64::consts::PI * 512.0;
        assert!((count / want - 1.0).abs() < 0.03, "{count} vs {want}");
        assert!(mask[[12, 12, 12]] && !mask[[0, 0, 0]] && !mask[[12, 12, 21]]);
    }
}
