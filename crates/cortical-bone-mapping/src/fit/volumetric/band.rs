//! The band of voxels the volumetric model fits, from a fixed surface.
//!
//! Every voxel `x` within `−outside ≤ d(x) ≤ inside` of the surface, where
//! `d` is the signed distance (positive inside, from the even–odd
//! rasterisation of the mesh for the sign and the exact closest point for the
//! magnitude), is a sample of the model. It belongs to the face of its
//! closest point `c(x)`, with that point's barycentric coordinates `β` on the
//! face, and its inward normal is `n(x) = (x − c(x)) / d(x)`, the gradient of
//! the distance (the face normal for a voxel on the surface). These are fixed
//! for a fit (section 2 of `docs/volumetric_model.md`).
//!
//! The caps closing a mask cut by the image edge are not bone surface, so
//! closest points are taken on the rest of the mesh: a voxel inside the bone
//! near the image edge belongs to the lateral surface it lies under, as a
//! profile there would sample it, instead of to the nearer cap. A voxel is
//! dropped if its closest point lies on the rim the caps leave (it is beyond
//! the edge of the surface), if its closest face is real but not fitted
//! (outside a sub-mask), if it is excluded (another bone), or if it is
//! outside the image. Its weight is the 1D residual weight
//! at its depth ([`crate::model::residual_weight`]), so both models weight
//! their samples alike. The voxels are stored grouped by face (ragged), with
//! 24 bytes each, so a Gauss–Newton pass over them runs face by face.
//!
//! The voxels in the outer half of the outside band (`d ≤ −outside/2`), well
//! clear of the blurred periosteal edge, are also listed by grid index. They
//! sample soft tissue only, and their residuals measure the noise level and
//! its spatial correlation (the reliability outputs).

use std::collections::{HashMap, HashSet};

use ndarray::ArrayView3;
use rayon::prelude::*;

use crate::model::residual_weight;
use crate::surface::closest::MeshPoints;
use crate::surface::mesh::Mesh;
use crate::surface::voxelize::index_box;
use crate::util::{cross, norm, scale, sub, V3};
use crate::volume::{Geometry, Voxel};

/// What the band is built from, besides the image and the surface.
pub(crate) struct BandSpec<'a> {
    pub geom: &'a Geometry,
    pub outside: f64,
    pub inside: f64,
    /// Residual weight at the surface relative to the band's ends (`γ0`).
    pub boost: f64,
    /// The scanner axis (array axis) of the PSF.
    pub axis: usize,
    /// Voxels to drop (other bones).
    pub exclude: Option<ArrayView3<'a, bool>>,
    /// Use every `stride`-th voxel along each axis (1 = all).
    pub stride: usize,
}

/// The band's voxels, grouped by fitted face.
pub(crate) struct Band {
    /// Voxels of fitted face `f`: `offsets[f]..offsets[f + 1]`.
    pub offsets: Vec<usize>,
    pub value: Vec<f32>,
    /// Signed depth below the surface (positive inside).
    pub depth: Vec<f32>,
    /// Barycentric coordinates on the first two vertices of the face (the
    /// third is `1 − β₀ − β₁`).
    pub bary: Vec<[f32; 2]>,
    /// Squared component of the inward normal along the scanner axis.
    pub axial2: Vec<f32>,
    pub weight: Vec<f32>,
    /// Grid index of each voxel.
    pub grid: Vec<[u32; 3]>,
    /// Soft-tissue voxels (`d ≤ −outside/2`): (grid index, band position).
    pub soft: Vec<([u32; 3], u32)>,
}

/// One band voxel before grouping.
struct Sample {
    face: u32,
    grid: [u32; 3],
    value: f32,
    depth: f32,
    bary: [f32; 2],
    axial2: f32,
    weight: f32,
}

impl Band {
    /// The band of `mesh` in `image`. `fitted[f]` is the fitted-face number
    /// of mesh face `f` (`None` if the face is not fitted); there are
    /// `nfaces` fitted faces.
    pub fn build<T: Voxel>(
        image: &ArrayView3<T>,
        mesh: &Mesh,
        fitted: &[Option<u32>],
        nfaces: usize,
        spec: &BandSpec,
    ) -> Band {
        let mut samples = samples(image, mesh, fitted, spec);
        samples.par_sort_unstable_by_key(|s| (s.face, s.grid));
        group(samples, nfaces, spec.outside)
    }

    /// Number of voxels.
    pub fn len(&self) -> usize {
        self.value.len()
    }

    /// Whether the band has no voxels.
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    /// The voxels of fitted face `f`.
    #[inline]
    pub fn of(&self, f: usize) -> std::ops::Range<usize> {
        self.offsets[f]..self.offsets[f + 1]
    }
}

/// Every band voxel, in no particular order.
fn samples<T: Voxel>(image: &ArrayView3<T>, mesh: &Mesh, fitted: &[Option<u32>], spec: &BandSpec) -> Vec<Sample> {
    let (d0, d1, d2) = image.dim();
    let reach = spec.outside.max(spec.inside);
    let (lo, hi) = index_box(mesh, [d0, d1, d2], spec.geom, reach);
    let shape = (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1);
    let box_geom = Geometry {
        spacing: spec.geom.spacing,
        origin: spec.geom.to_physical(lo.map(|v| v as f64)),
    };
    let inside = mesh.voxelize(shape, &box_geom);
    let real = RealSurface::new(mesh, [d0, d1, d2], spec.geom);
    let points = MeshPoints::new(&real.mesh);
    let h = spec.geom.spacing.iter().cloned().fold(0.0, f64::max);
    let step = spec.stride.max(1);
    (lo[0]..=hi[0])
        .step_by(step)
        .collect::<Vec<_>>()
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut out = Vec::new();
            for j in (lo[1]..=hi[1]).step_by(step) {
                for k in (lo[2]..=hi[2]).step_by(step) {
                    if spec.exclude.as_ref().is_some_and(|e| e[[i, j, k]]) {
                        continue;
                    }
                    let q = spec.geom.to_physical([i as f64, j as f64, k as f64]);
                    let Some(c) = points.closest(q, reach + 2.0 * h) else {
                        continue;
                    };
                    if real.on_rim(&c) {
                        continue;
                    }
                    let Some(face) = fitted[real.to_face[c.face as usize] as usize] else {
                        continue;
                    };
                    let is_in = inside[[i - lo[0], j - lo[1], k - lo[2]]];
                    let depth = if is_in { c.dist } else { -c.dist };
                    if depth < -spec.outside || depth > spec.inside {
                        continue;
                    }
                    let n = inward_normal(q, &c, depth, &real.mesh);
                    out.push(Sample {
                        face,
                        grid: [i as u32, j as u32, k as u32],
                        value: image[[i, j, k]].to_f64() as f32,
                        depth: depth as f32,
                        bary: [c.bary[0] as f32, c.bary[1] as f32],
                        axial2: (n[spec.axis] * n[spec.axis]) as f32,
                        weight: residual_weight(depth, spec.boost, reach) as f32,
                    });
                }
            }
            out
        })
        .collect()
}

/// The bone's real surface: the mesh without its caps (the faces closing a
/// mask cut by the image edge, which are not bone surface), with the edges
/// and vertices of the rims they leave.
struct RealSurface {
    mesh: Mesh,
    /// Original face of every face of `mesh`.
    to_face: Vec<u32>,
    rim_edges: HashSet<(u32, u32)>,
    rim_vertices: Vec<bool>,
}

impl RealSurface {
    fn new(mesh: &Mesh, dims: [usize; 3], geom: &Geometry) -> Self {
        let outside_image = |v: u32| {
            let q = geom.to_index(mesh.vertices[v as usize]);
            (0..3).any(|a| q[a] < 0.0 || q[a] > dims[a] as f64 - 1.0)
        };
        let to_face: Vec<u32> = (0..mesh.faces.len() as u32)
            .filter(|&f| !mesh.faces[f as usize].iter().any(|&v| outside_image(v)))
            .collect();
        // only the vertices of real faces: the nearest-vertex search of the
        // closest point must not find cap vertices, which have no real faces
        let mut index = vec![u32::MAX; mesh.vertices.len()];
        let mut vertices = Vec::new();
        let faces = to_face
            .iter()
            .map(|&f| {
                mesh.faces[f as usize].map(|v| {
                    if index[v as usize] == u32::MAX {
                        index[v as usize] = vertices.len() as u32;
                        vertices.push(mesh.vertices[v as usize]);
                    }
                    index[v as usize]
                })
            })
            .collect();
        let real = Mesh { vertices, faces };
        let mut count: HashMap<(u32, u32), u32> = HashMap::new();
        for f in &real.faces {
            for k in 0..3 {
                let (a, b) = (f[k], f[(k + 1) % 3]);
                *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        let rim_edges: HashSet<(u32, u32)> = count.into_iter().filter(|&(_, c)| c == 1).map(|(e, _)| e).collect();
        let mut rim_vertices = vec![false; real.vertices.len()];
        for &(a, b) in &rim_edges {
            rim_vertices[a as usize] = true;
            rim_vertices[b as usize] = true;
        }
        RealSurface {
            mesh: real,
            to_face,
            rim_edges,
            rim_vertices,
        }
    }

    /// The closest point lies on a rim: the voxel is beyond the edge of the
    /// real surface, and its depth is not a depth below it.
    fn on_rim(&self, c: &crate::surface::closest::Closest) -> bool {
        let f = self.mesh.faces[c.face as usize];
        let zero: Vec<usize> = (0..3).filter(|&s| c.bary[s] == 0.0).collect();
        match zero.len() {
            2 => {
                let s = (0..3).find(|s| !zero.contains(s)).unwrap();
                self.rim_vertices[f[s] as usize]
            }
            1 => {
                let (a, b) = (f[(zero[0] + 1) % 3], f[(zero[0] + 2) % 3]);
                self.rim_edges.contains(&(a.min(b), a.max(b)))
            }
            _ => false,
        }
    }
}

/// The inward unit normal at voxel `q` with closest point `c` at signed
/// depth `depth`: the direction from the closest point, or, on the surface,
/// the inward face normal.
fn inward_normal(q: V3, c: &crate::surface::closest::Closest, depth: f64, mesh: &Mesh) -> V3 {
    if c.dist > 1e-9 {
        return scale(sub(q, c.point), 1.0 / depth);
    }
    let [a, b, cc] = mesh.faces[c.face as usize].map(|v| mesh.vertices[v as usize]);
    let n = cross(sub(b, a), sub(cc, a)); // outward
    let l = norm(n);
    if l > 0.0 {
        scale(n, -1.0 / l)
    } else {
        [0.0; 3]
    }
}

/// The band from samples sorted by face.
fn group(samples: Vec<Sample>, nfaces: usize, outside: f64) -> Band {
    let n = samples.len();
    let mut band = Band {
        offsets: vec![0; nfaces + 1],
        value: Vec::with_capacity(n),
        depth: Vec::with_capacity(n),
        bary: Vec::with_capacity(n),
        axial2: Vec::with_capacity(n),
        weight: Vec::with_capacity(n),
        grid: Vec::with_capacity(n),
        soft: Vec::new(),
    };
    for s in &samples {
        band.offsets[s.face as usize + 1] += 1;
    }
    for f in 0..nfaces {
        band.offsets[f + 1] += band.offsets[f];
    }
    for (pos, s) in samples.into_iter().enumerate() {
        if (s.depth as f64) <= -0.5 * outside {
            band.soft.push((s.grid, pos as u32));
        }
        band.value.push(s.value);
        band.depth.push(s.depth);
        band.bary.push(s.bary);
        band.axial2.push(s.axial2);
        band.weight.push(s.weight);
        band.grid.push(s.grid);
    }
    band.soft.sort_unstable_by_key(|s| s.0);
    band
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::mesh::Mesh;
    use ndarray::Array3;

    #[test]
    fn sphere_band_has_the_right_depths_and_normals() {
        let r = 6.0;
        let mut mesh = Mesh::uv_sphere(r, 96, 64);
        for v in &mut mesh.vertices {
            *v = crate::util::add(*v, [10.0; 3]);
        }
        let img = Array3::<f32>::zeros((21, 21, 21));
        let geom = Geometry::new([1.0; 3]);
        let fitted: Vec<Option<u32>> = (0..mesh.faces.len() as u32).map(Some).collect();
        let spec = BandSpec {
            geom: &geom,
            outside: 2.0,
            inside: 3.0,
            boost: 1.0,
            axis: 0,
            exclude: None,
            stride: 1,
        };
        let band = Band::build(&img.view(), &mesh, &fitted, mesh.faces.len(), &spec);
        assert!(band.len() > 1000);
        // every voxel's depth agrees with the sphere's (to the faceting error)
        // and lies in the band; the normal is radial
        for f in 0..mesh.faces.len() {
            for e in band.of(f) {
                let d = band.depth[e] as f64;
                assert!((-2.0..=3.0).contains(&d));
                assert!((band.bary[e][0] + band.bary[e][1]) <= 1.0 + 1e-6);
            }
        }
        let shell: usize = (0..band.len()).filter(|&e| band.depth[e] <= -1.0).count();
        assert_eq!(shell, band.soft.len());
        // against the exact sphere, at the soft-tissue voxels (grid known)
        let mut worst: f64 = 0.0;
        for &(g, e) in &band.soft {
            let x = g.map(|v| v as f64 - 10.0);
            let rx = norm(x);
            let d_true = r - rx;
            assert!(
                (band.depth[e as usize] as f64 - d_true).abs() < 0.02,
                "{} vs {d_true}",
                band.depth[e as usize]
            );
            let a2 = (x[0] / rx).powi(2);
            worst = worst.max((band.axial2[e as usize] as f64 - a2).abs());
        }
        // a voxel over a facet gets the facet's normal, which is off the radial
        // direction by up to half the facet's angle (π/96), so n₀² can be off
        // by up to |sin 2θ| · π/96 ≈ 0.033
        assert!(worst < std::f64::consts::PI / 96.0, "worst n₀² error {worst}");
    }
}
