//! The band of voxels the volumetric model fits, from a fixed surface.
//!
//! Every voxel `x` within `−outside ≤ d(x) ≤ inside` of the surface, where
//! `d` is the signed distance (positive inside, from the even–odd
//! rasterisation of the mesh for the sign and the exact closest point for the
//! magnitude), is a sample of the model. It belongs to the face of its
//! closest point `c(x)`, with that point's barycentric coordinates `β` on the
//! face, and its inward normal is `n(x) = (x − c(x)) / d(x)`, the gradient of
//! the distance (the face normal for a voxel on the surface). These are fixed
//! for a fit.
//!
//! The caps closing a mask cut by the image edge are not bone surface, so
//! closest points are taken on the rest of the mesh: a voxel inside the bone
//! near the image edge belongs to the lateral surface it lies under, as a
//! profile there would sample it, instead of to the nearer cap. A voxel is
//! dropped if its closest point lies on the rim the caps leave (it is beyond
//! the edge of the surface), if its closest face is real but not fitted
//! (outside a sub-mask), if it is excluded (another bone), or if it is
//! outside the image. Its weight is the 1D residual weight
//! at its depth ([`crate::model::residual_weight`], computed where it is
//! used from the depth and its face's reach), so both models weight their
//! samples alike. The voxels are stored grouped by face (ragged), in grid
//! order within a face, with 9 bytes each (16-bit intensity, depth and
//! barycentric coordinates, an 8-bit axial normal component; [`super::quantise`]),
//! so a Gauss–Newton pass over them runs face by face. They are generated slice by slice (the rows of a slice
//! in parallel) into arrays in grid order, which are then reordered by face
//! one at a time (a stable counting sort), so the build holds at most 13
//! bytes per voxel more than the band itself.
//!
//! Only for the reliability outputs ([`BandSpec::grid`]): every voxel's grid
//! index, and the voxels in the outer half of the outside band
//! (`d ≤ −outside/2`), well clear of the blurred periosteal edge, listed by
//! grid index. Those sample soft tissue only, and their residuals measure
//! the noise level and its spatial correlation.

use std::collections::{HashMap, HashSet};

use ndarray::ArrayView3;
use rayon::prelude::*;

use super::quantise::{Linear, BARY, UNIT8};
use crate::model::residual_weight;
use crate::surface::closest::MeshPoints;
use crate::surface::mesh::Mesh;
use crate::surface::voxelize::{boundary_distance2, index_box};
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
    /// Per fitted face, its reach `[outside, inside]` (the largest of its
    /// nodes'; [`super::reach`]); `None`: `outside` and `inside` everywhere.
    pub face_reach: Option<Vec<[f64; 2]>>,
    /// Keep the grid indices and the soft-tissue voxels (the reliability
    /// outputs need them; nothing else does).
    pub grid: bool,
}

/// The band's voxels, grouped by fitted face.
pub(crate) struct Band {
    /// Voxels of fitted face `f`: `offsets[f]..offsets[f + 1]`.
    pub offsets: Vec<usize>,
    /// The image value, coded on `value_scale` (read with [`Band::value`]).
    value: Vec<u16>,
    value_scale: Linear,
    /// Signed depth below the surface (positive inside), coded on
    /// `depth_scale` ([`Band::depth`]).
    depth: Vec<u16>,
    depth_scale: Linear,
    /// Barycentric coordinates on the first two vertices of the face (the
    /// third is `1 − β₀ − β₁`; [`Band::bary`]).
    bary: Vec<[u16; 2]>,
    /// Squared component of the inward normal along the scanner axis
    /// ([`Band::axial2`]).
    axial2: Vec<u8>,
    /// Residual weight at the surface relative to the band's ends (`γ0`).
    pub boost: f64,
    /// Per fitted face, the depth its weights reach (the larger of its
    /// outside and inside reach).
    pub weight_reach: Vec<f32>,
    /// Grid index of each voxel ([`BandSpec::grid`]; empty otherwise).
    pub grid: Vec<[u32; 3]>,
    /// Soft-tissue voxels (`d ≤ −outside/2`): (grid index, band position)
    /// ([`BandSpec::grid`]; empty otherwise).
    pub soft: Vec<([u32; 3], u32)>,
}

/// One band voxel, as a row finds it.
struct Sample {
    face: u32,
    grid: [u32; 3],
    value: u16,
    depth: u16,
    bary: [u16; 2],
    axial2: u8,
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
        let found = samples(image, mesh, fitted, spec);
        let st = crate::stage!("band: grouping by face");
        let band = group(found, nfaces, spec);
        st.end();
        band
    }

    /// The image value of voxel `e`.
    #[inline]
    pub fn value(&self, e: usize) -> f64 {
        self.value_scale.decode(self.value[e] as u32)
    }

    /// The signed depth of voxel `e` below the surface (positive inside).
    #[inline]
    pub fn depth(&self, e: usize) -> f64 {
        self.depth_scale.decode(self.depth[e] as u32)
    }

    /// The barycentric coordinates of voxel `e`'s closest point on its face.
    #[inline]
    pub fn bary(&self, e: usize) -> [f64; 3] {
        let [a, b] = self.bary[e].map(|q| BARY.decode(q as u32));
        [a, b, 1.0 - a - b]
    }

    /// The squared component of voxel `e`'s inward normal along the scanner
    /// axis.
    #[inline]
    pub fn axial2(&self, e: usize) -> f64 {
        UNIT8.decode(self.axial2[e] as u32)
    }

    /// Replace the image values (tests: data synthesised from the model).
    #[cfg(test)]
    pub fn set_values(&mut self, values: &[f64]) {
        let (lo, hi) = values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |a, &v| (a.0.min(v), a.1.max(v)));
        self.value_scale = Linear::u16(lo, hi);
        self.value = values.iter().map(|&v| self.value_scale.encode16(v)).collect();
    }

    /// The residual weight of voxel `e` of fitted face `f`.
    #[inline]
    pub fn weight(&self, f: usize, e: usize) -> f64 {
        residual_weight(self.depth(e), self.boost, self.weight_reach[f] as f64)
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

/// Slices whose band voxels are found together (in parallel over their
/// rows) before they are appended: enough for every thread, few enough that
/// their samples are a small part of the band.
const SLICES_PER_GROUP: usize = 8;

/// The band's voxels in grid order, before grouping by face (the grid
/// indices only with [`BandSpec::grid`]).
#[derive(Default)]
struct Found {
    face: Vec<u32>,
    value: Vec<u16>,
    value_scale: Option<Linear>,
    depth: Vec<u16>,
    depth_scale: Option<Linear>,
    bary: Vec<[u16; 2]>,
    axial2: Vec<u8>,
    grid: Vec<[u32; 3]>,
}

impl Found {
    fn push(&mut self, s: Sample, grid: bool) {
        self.face.push(s.face);
        self.value.push(s.value);
        self.depth.push(s.depth);
        self.bary.push(s.bary);
        self.axial2.push(s.axial2);
        if grid {
            self.grid.push(s.grid);
        }
    }
}

/// Every band voxel, in grid order.
fn samples<T: Voxel>(image: &ArrayView3<T>, mesh: &Mesh, fitted: &[Option<u32>], spec: &BandSpec) -> Found {
    let (d0, d1, d2) = image.dim();
    let reach = match &spec.face_reach {
        Some(r) => r
            .iter()
            .map(|r| r[0].max(r[1]))
            .fold(spec.outside.max(spec.inside), f64::max),
        None => spec.outside.max(spec.inside),
    };
    let (lo, hi) = index_box(mesh, [d0, d1, d2], spec.geom, reach);
    let shape = (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1);
    let box_geom = Geometry {
        spacing: spec.geom.spacing,
        origin: spec.geom.to_physical(lo.map(|v| v as f64)),
    };
    let st = crate::stage!("band: rasterise the surface");
    let inside = mesh.voxelize(shape, &box_geom);
    crate::done!(st, "box {shape:?}");
    let real = RealSurface::new(mesh, [d0, d1, d2], spec.geom);
    let points = MeshPoints::new(&real.mesh);
    let h = spec.geom.spacing.iter().cloned().fold(0.0, f64::max);
    // the surface passes within a voxel diagonal of the voxelisation's
    // boundary, so a voxel more than `reach + 2h` from that boundary is out
    // of the band: skip it without the closest-point search
    let st = crate::stage!("band: distance to the rasterised boundary");
    // the box's edges count as boundary: the mask may be cut there (caps)
    let near = boundary_distance2(&inside, spec.geom.spacing, true);
    st.end();
    let far2 = ((reach + 2.0 * h) * (reach + 2.0 * h)) as f32;
    let step = spec.stride.max(1);
    // the scales of the stored values: the image's range over the box, and
    // the band's depth range
    let (vlo, vhi) = image
        .slice(ndarray::s![lo[0]..=hi[0], lo[1]..=hi[1], lo[2]..=hi[2]])
        .outer_iter()
        .into_par_iter()
        .map(|sl| {
            sl.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, v| {
                (a.0.min(v.to_f64()), a.1.max(v.to_f64()))
            })
        })
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |a, b| (a.0.min(b.0), a.1.max(b.1)),
        );
    let value_scale = Linear::u16(vlo, vhi);
    let [max_out, max_in] = match &spec.face_reach {
        Some(r) => r
            .iter()
            .fold([spec.outside, spec.inside], |a, r| [a[0].max(r[0]), a[1].max(r[1])]),
        None => [spec.outside, spec.inside],
    };
    let depth_scale = Linear::u16(-max_out, max_in);
    let st = crate::stage!("band: closest points");
    let slices: Vec<usize> = (lo[0]..=hi[0]).step_by(step).collect();
    let rows: Vec<usize> = (lo[1]..=hi[1]).step_by(step).collect();
    let mut out = Found {
        value_scale: Some(value_scale),
        depth_scale: Some(depth_scale),
        ..Found::default()
    };
    // a group of slices at a time, its rows in parallel, appended in grid
    // order: only one group's samples are ever held twice, and the group is
    // large enough to keep every thread busy
    for group in slices.chunks(SLICES_PER_GROUP) {
        let pairs: Vec<(usize, usize)> = group.iter().flat_map(|&i| rows.iter().map(move |&j| (i, j))).collect();
        let slice: Vec<Vec<Sample>> = pairs
            .par_iter()
            .map(|&(i, j)| {
                let mut out = Vec::new();
                // along the row, the previous query bounds the next one's
                // nearest vertices (exactly: by its distance plus the step)
                let (mut bound, mut prev): (Option<f64>, V3) = (None, [0.0; 3]);
                for k in (lo[2]..=hi[2]).step_by(step) {
                    if near[[i - lo[0], j - lo[1], k - lo[2]]] > far2
                        || spec.exclude.as_ref().is_some_and(|e| e[[i, j, k]])
                    {
                        continue;
                    }
                    let q = spec.geom.to_physical([i as f64, j as f64, k as f64]);
                    let (found, far) = points.closest_bounded(
                        q,
                        reach + 2.0 * h,
                        bound.map(|b| b + crate::util::dist2(q, prev).sqrt()),
                    );
                    (bound, prev) = (far.is_finite().then_some(far), q);
                    let Some(c) = found else {
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
                    let [reach_out, reach_in] = spec
                        .face_reach
                        .as_ref()
                        .map_or([spec.outside, spec.inside], |r| r[face as usize]);
                    if depth < -reach_out || depth > reach_in {
                        continue;
                    }
                    let n = inward_normal(q, &c, depth, &real.mesh);
                    out.push(Sample {
                        face,
                        grid: [i as u32, j as u32, k as u32],
                        value: value_scale.encode16(image[[i, j, k]].to_f64()),
                        depth: depth_scale.encode16(depth),
                        bary: [BARY.encode16(c.bary[0]), BARY.encode16(c.bary[1])],
                        axial2: UNIT8.encode8(n[spec.axis] * n[spec.axis]),
                    });
                }
                out
            })
            .collect();
        for s in slice.into_iter().flatten() {
            out.push(s, spec.grid);
        }
    }
    crate::done!(st, "{} voxels kept, reach {reach:.3}", out.face.len());
    out
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

/// The band from its voxels in grid order: a stable counting sort by face
/// (so grid order within a face), each array reordered on its own.
fn group(found: Found, nfaces: usize, spec: &BandSpec) -> Band {
    let Found {
        face,
        value,
        value_scale,
        depth,
        depth_scale,
        bary,
        axial2,
        grid,
    } = found;
    let n = face.len();
    assert!(n <= u32::MAX as usize, "more than 2³² band voxels");
    let mut offsets = vec![0usize; nfaces + 1];
    for &f in &face {
        offsets[f as usize + 1] += 1;
    }
    for f in 0..nfaces {
        offsets[f + 1] += offsets[f];
    }
    // where each voxel goes
    let mut next = offsets[..nfaces].to_vec();
    let dest: Vec<u32> = face
        .iter()
        .map(|&f| {
            let pos = next[f as usize];
            next[f as usize] += 1;
            pos as u32
        })
        .collect();
    drop(face);
    let weight_reach: Vec<f32> = (0..nfaces)
        .map(|f| {
            let [o, i] = spec.face_reach.as_ref().map_or([spec.outside, spec.inside], |r| r[f]);
            o.max(i) as f32
        })
        .collect();
    let mut band = Band {
        offsets,
        value: reorder(value, &dest),
        value_scale: value_scale.unwrap_or(Linear { lo: 0.0, step: 1.0 }),
        depth: reorder(depth, &dest),
        depth_scale: depth_scale.unwrap_or(Linear { lo: 0.0, step: 1.0 }),
        bary: reorder(bary, &dest),
        axial2: reorder(axial2, &dest),
        boost: spec.boost,
        weight_reach,
        grid: if spec.grid { reorder(grid, &dest) } else { Vec::new() },
        soft: Vec::new(),
    };
    if spec.grid {
        band.soft = (0..n)
            .filter(|&e| band.depth(e) <= -0.5 * spec.outside)
            .map(|e| (band.grid[e], e as u32))
            .collect();
        band.soft.sort_unstable_by_key(|s| s.0);
    }
    band
}

/// `out[dest[e]] = src[e]`, freeing `src`.
fn reorder<T: Copy + Default + Send + Sync>(src: Vec<T>, dest: &[u32]) -> Vec<T> {
    let mut out = vec![T::default(); src.len()];
    for (e, &d) in dest.iter().enumerate() {
        out[d as usize] = src[e];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::mesh::Mesh;
    use ndarray::Array3;

    /// The far-voxel test (`boundary_distance2`) never drops a voxel that
    /// the closest-point search would have put in the band.
    #[test]
    fn pruning_keeps_every_band_voxel() {
        let mut mesh = Mesh::uv_sphere(5.5, 48, 32);
        for v in &mut mesh.vertices {
            *v = crate::util::add(*v, [9.0, 9.3, 8.7]);
        }
        let spacing = [0.9, 1.1, 1.0];
        let geom = Geometry::new(spacing);
        let dims = [20usize, 18, 20];
        let (outside, inside): (f64, f64) = (2.0, 2.5);
        let inside_mask = mesh.voxelize((dims[0], dims[1], dims[2]), &geom);
        let near = boundary_distance2(&inside_mask, spacing, true);
        let h: f64 = 1.1;
        let far2 = ((inside.max(outside) + 2.0 * h).powi(2)) as f32;
        let points = MeshPoints::new(&mesh);
        let mut in_band = 0;
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    let q = geom.to_physical([i as f64, j as f64, k as f64]);
                    let Some(c) = points.closest(q, 100.0) else { continue };
                    let depth = if inside_mask[[i, j, k]] { c.dist } else { -c.dist };
                    if (-outside..=inside).contains(&depth) {
                        in_band += 1;
                        assert!(near[[i, j, k]] <= far2, "voxel {:?} at depth {depth} pruned", [i, j, k]);
                    }
                }
            }
        }
        assert!(in_band > 1000);
    }

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
            face_reach: None,
            grid: true,
        };
        let band = Band::build(&img.view(), &mesh, &fitted, mesh.faces.len(), &spec);
        assert!(band.len() > 1000);
        // every voxel's depth agrees with the sphere's (to the faceting error)
        // and lies in the band; the normal is radial
        for f in 0..mesh.faces.len() {
            for e in band.of(f) {
                let d = band.depth(e);
                assert!((-2.0..=3.0).contains(&d));
                let b = band.bary(e);
                assert!(b[2] >= -1e-4 && (b[0] + b[1] + b[2] - 1.0).abs() < 1e-12);
            }
        }
        let shell: usize = (0..band.len()).filter(|&e| band.depth(e) <= -1.0).count();
        assert_eq!(shell, band.soft.len());
        // against the exact sphere, at the soft-tissue voxels (grid known)
        let mut worst: f64 = 0.0;
        for &(g, e) in &band.soft {
            let x = g.map(|v| v as f64 - 10.0);
            let rx = norm(x);
            let d_true = r - rx;
            assert!(
                (band.depth(e as usize) - d_true).abs() < 0.02,
                "{} vs {d_true}",
                band.depth(e as usize)
            );
            let a2 = (x[0] / rx).powi(2);
            worst = worst.max((band.axial2(e as usize) - a2).abs());
        }
        // a voxel over a facet gets the facet's normal, which is off the radial
        // direction by up to half the facet's angle (π/96), so n₀² can be off
        // by up to |sin 2θ| · π/96 ≈ 0.033
        assert!(worst < std::f64::consts::PI / 96.0, "worst n₀² error {worst}");
    }
}
