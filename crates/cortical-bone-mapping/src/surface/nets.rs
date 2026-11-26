//! Surface extraction from a binary mask or a level set with (naive)
//! surface nets.
//!
//! Surface nets (Gibson 1998) are a dual method: the iso-surface of a sampled
//! field `f` at level `ISO` gets one vertex per grid cell (a cube whose 8
//! corners are voxel centres) that straddles the level, and one quad per grid
//! edge whose endpoints straddle it, joining the four cells around that edge.
//! Gibson then relaxes the vertices iteratively within their cells; the
//! "naive" variant used here places each vertex once, at the mean of the
//! level crossings on its cell's edges, found by linear interpolation along
//! each edge,
//!
//! ```text
//! v = cell corner + mean over crossing edges (p, q) of  c_p + t (c_q − c_p),
//! t = (ISO − f_p) / (f_q − f_p)
//! ```
//!
//! as in dual contouring without the quadric fit (Ju et al. 2002). For a
//! binary mask `t = ½` on every edge; blurring the mask first
//! (`smoothing_sigma`) gives sub-voxel positions. Quads are oriented so that
//! the normal points out of the foreground and split into two triangles along
//! their shorter diagonal, which avoids the slivers of the longer one.
//!
//! Compared with marching cubes (Lorensen & Cline 1987) this needs no lookup
//! tables and gives about half as many, much better-shaped triangles, which
//! matters because every vertex becomes a CBM node. The cell→vertex index is kept for two
//! cell layers at a time, so memory is `O(slice)` + output. Voxels outside the
//! image count as background, so the surface is always closed; vertices whose
//! cell reaches outside the image are flagged as `border` (the caps where the
//! mask is cut by the image edge, which carry no periosteum).
//!
//! # References
//!
//! * S. F. F. Gibson, *Constrained elastic surface nets: generating smooth
//!   surfaces from binary segmented data*, Medical Image Computing and
//!   Computer-Assisted Intervention (MICCAI 1998), Lecture Notes in Computer
//!   Science 1496, Springer, 1998, pp. 888–898.
//! * T. Ju, F. Losasso, S. Schaefer, J. Warren, *Dual contouring of Hermite
//!   data*, ACM Transactions on Graphics 21(3):339–346, 2002.
//! * W. E. Lorensen, H. E. Cline, *Marching cubes: a high resolution 3D surface
//!   construction algorithm*, Computer Graphics (SIGGRAPH 87) 21(4):163–169,
//!   1987.

use ndarray::{s, Array3, ArrayView3, Zip};

use crate::segmentation::morphology::scaled_sq_distance;
use crate::surface::mesh::Mesh;
use crate::volume::{bounding_box, gaussian_filter, Geometry, Voxel};

/// Output of [`extract_surface`].
pub struct Extracted {
    pub mesh: Mesh,
    /// `true` for vertices whose cell touches the outside of the image
    /// domain, i.e. that lie on the cap closing a mask cut by the image edge.
    pub border: Vec<bool>,
}

/// Extract the boundary of `mask` (non-zero = inside) as a closed,
/// outward-oriented triangle mesh in physical coordinates.
///
/// `smoothing_sigma` (voxels) optionally blurs the mask before extraction,
/// which places vertices with sub-voxel interpolation instead of at edge
/// midpoints. It needs a float copy of the mask's bounding box.
pub fn extract_surface<T: Voxel>(mask: &ArrayView3<T>, geom: &Geometry, smoothing_sigma: f64) -> Extracted {
    let Some((lo, hi)) = bounding_box(mask) else {
        return Extracted::empty();
    };
    let d = mask.dim();
    let dims = [d.0 as isize, d.1 as isize, d.2 as isize];
    if smoothing_sigma > 0.0 {
        let pad = (3.0 * smoothing_sigma).ceil() as usize + 1;
        let (blurred, origin) = blurred_box(mask, lo, hi, pad, smoothing_sigma);
        let value = window(blurred.view(), origin);
        let field = |i: isize, j: isize, k: isize| value([i, j, k]).unwrap_or(0.0);
        // Blurring may grow the surface beyond the bounding box by ~3σ.
        let lo2 = lo.map(|v| v.saturating_sub(pad));
        let hi2 = [0, 1, 2].map(|a| (hi[a] + pad).min(dims[a] as usize - 1));
        nets(field, lo2, hi2, dims, geom)
    } else {
        let field = |i: isize, j: isize, k: isize| -> f64 {
            if i < 0 || j < 0 || k < 0 || i >= dims[0] || j >= dims[1] || k >= dims[2] {
                0.0
            } else {
                // SAFETY: bounds checked above.
                (unsafe { mask.uget((i as usize, j as usize, k as usize)) }.to_f64() != 0.0) as u8 as f64
            }
        };
        nets(field, lo, hi, dims, geom)
    }
}

/// The mask's bounding box `lo..=hi`, padded by `pad` voxels (outside the
/// image as background), as 0/1 blurred by `sigma`; and its origin.
fn blurred_box<T: Voxel>(
    mask: &ArrayView3<T>,
    lo: [usize; 3],
    hi: [usize; 3],
    pad: usize,
    sigma: f64,
) -> (Array3<f64>, [isize; 3]) {
    let d = mask.dim();
    let dims = [d.0 as isize, d.1 as isize, d.2 as isize];
    let origin = lo.map(|v| v as isize - pad as isize);
    let shape = (
        hi[0] - lo[0] + 1 + 2 * pad,
        hi[1] - lo[1] + 1 + 2 * pad,
        hi[2] - lo[2] + 1 + 2 * pad,
    );
    let sub = Array3::from_shape_fn(shape, |(i, j, k)| {
        let g = [i as isize + origin[0], j as isize + origin[1], k as isize + origin[2]];
        if (0..3).all(|a| g[a] >= 0 && g[a] < dims[a]) {
            (mask[[g[0] as usize, g[1] as usize, g[2] as usize]].to_f64() != 0.0) as u8 as f64
        } else {
            0.0
        }
    });
    (gaussian_filter(&sub, [sigma; 3]), origin)
}

/// Look-up of image voxel `g` in `arr`, which holds the voxels from `origin`
/// on; `None` outside it.
fn window<A: Copy>(arr: ArrayView3<'_, A>, origin: [isize; 3]) -> impl Fn([isize; 3]) -> Option<A> + '_ {
    let s = arr.dim();
    let shape = [s.0 as isize, s.1 as isize, s.2 as isize];
    move |g| {
        let l = [g[0] - origin[0], g[1] - origin[1], g[2] - origin[2]];
        (0..3)
            .all(|a| l[a] >= 0 && l[a] < shape[a])
            .then(|| arr[[l[0] as usize, l[1] as usize, l[2] as usize]])
    }
}

const ISO: f64 = 0.5;

/// [`extract_level_set`] on every `step`-th voxel of the image grid: the
/// voxels whose indices are multiples of `step` (the same lattice for every
/// call, so the image's edges stay where they are), as a grid with the
/// image's origin and `step` times its spacing.
pub fn extract_level_set_coarse(
    phi: &ArrayView3<f32>,
    lo: [usize; 3],
    dims: [usize; 3],
    geom: &Geometry,
    step: usize,
) -> Extracted {
    if step <= 1 {
        return extract_level_set(phi, lo, dims, geom);
    }
    let first = lo.map(|l| l.div_ceil(step) * step);
    let (s0, s1, s2) = phi.dim();
    if (0..3).any(|a| first[a] >= lo[a] + [s0, s1, s2][a]) {
        return Extracted::empty();
    }
    let lattice = phi.slice(s![
        (first[0] - lo[0])..;step,
        (first[1] - lo[1])..;step,
        (first[2] - lo[2])..;step
    ]);
    let coarse = Geometry {
        spacing: geom.spacing.map(|s| s * step as f64),
        origin: geom.origin,
    };
    extract_level_set(
        &lattice,
        first.map(|f| f / step),
        dims.map(|d| (d - 1) / step + 1),
        &coarse,
    )
}

/// The surface of `mask` on every `step`-th voxel (`step` > 1): the zero
/// level of the mask's exact signed distance (physical units, between voxel
/// centres, as the binary extraction places its vertices midway), computed
/// on the mask's bounding box and extracted by [`extract_level_set_coarse`].
pub fn extract_surface_coarse<T: Voxel>(mask: &ArrayView3<T>, geom: &Geometry, step: usize) -> Extracted {
    let Some((lo, hi)) = bounding_box(mask) else {
        return Extracted::empty();
    };
    let (d0, d1, d2) = mask.dim();
    let dims = [d0, d1, d2];
    let pad = step + 1;
    let lo = lo.map(|v| v.saturating_sub(pad));
    let hi: [usize; 3] = std::array::from_fn(|a| (hi[a] + pad).min(dims[a] - 1));
    let inside = mask
        .slice(s![lo[0]..=hi[0], lo[1]..=hi[1], lo[2]..=hi[2]])
        .mapv(|v| v.to_f64() != 0.0);
    let inv = geom.spacing.map(|s| 1.0 / s);
    let to_inside = scaled_sq_distance(&inside, inv);
    let to_outside = scaled_sq_distance(&inside.mapv(|b| !b), inv);
    let mut phi = Array3::<f32>::zeros(inside.dim());
    Zip::from(&mut phi)
        .and(&inside)
        .and(&to_inside)
        .and(&to_outside)
        .par_for_each(|p, &i, &di, &dout| *p = if i { -dout.sqrt() } else { di.sqrt() });
    extract_level_set_coarse(&phi.view(), lo, dims, geom, step)
}

/// Extract the zero level of a signed distance `phi` (negative inside),
/// stored for the image voxels `lo + index`, as a closed outward-oriented
/// mesh. Voxels outside `phi` or the image (`dims`) count as outside;
/// `border` flags vertices whose cell touches the outside of the image.
pub fn extract_level_set(phi: &ArrayView3<f32>, lo: [usize; 3], dims: [usize; 3], geom: &Geometry) -> Extracted {
    let (s0, s1, s2) = phi.dim();
    if s0 == 0 || s1 == 0 || s2 == 0 {
        return Extracted::empty();
    }
    let value = window(phi.view(), lo.map(|v| v as isize));
    let field = |i: isize, j: isize, k: isize| value([i, j, k]).map_or(0.0, |p| ISO - p as f64);
    let hi = [lo[0] + s0 - 1, lo[1] + s1 - 1, lo[2] + s2 - 1];
    nets(field, lo, hi, dims.map(|v| v as isize), geom)
}

impl Extracted {
    /// No surface.
    fn empty() -> Self {
        Extracted {
            mesh: Mesh::default(),
            border: Vec::new(),
        }
    }
}

/// Cube corner offsets, and the 12 cube edges as corner index pairs.
const CORNERS: [[isize; 3]; 8] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
];
const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];
const NONE: u32 = u32::MAX;

/// A field value inside the surface.
fn inside(v: f64) -> bool {
    v > ISO
}

/// Surface nets over the voxels `lo..=hi` of the field `f` (inside above
/// `ISO`), one cell layer at a time.
fn nets<F: Fn(isize, isize, isize) -> f64>(
    f: F,
    lo: [usize; 3],
    hi: [usize; 3],
    dims: [isize; 3],
    geom: &Geometry,
) -> Extracted {
    let mut nets = Nets::new(lo, hi, dims);
    for a in nets.c0[0]..=hi[0] as isize {
        nets.vertices(&f, a, geom);
        nets.quads(&f, a);
        std::mem::swap(&mut nets.prev, &mut nets.cur);
    }
    Extracted {
        mesh: Mesh {
            vertices: nets.verts,
            faces: nets.faces,
        },
        border: nets.border,
    }
}

/// Surface nets in progress. Cells `(a, b, c)` have the voxels
/// `(a, b, c) + {0,1}³` as corners and cover `c0 ..= hi`; the vertex index
/// is kept for the current and the previous cell layer.
struct Nets {
    c0: [isize; 3],
    hi: [isize; 3],
    n2: isize,
    dims: [isize; 3],
    prev: Vec<u32>,
    cur: Vec<u32>,
    verts: Vec<[f64; 3]>,
    border: Vec<bool>,
    faces: Vec<[u32; 3]>,
}

impl Nets {
    /// Cells over the voxels `lo..=hi`, one cell layer beyond them on the low side.
    fn new(lo: [usize; 3], hi: [usize; 3], dims: [isize; 3]) -> Self {
        let n1 = (hi[1] - lo[1] + 2) as isize;
        let n2 = (hi[2] - lo[2] + 2) as isize;
        Nets {
            c0: lo.map(|v| v as isize - 1),
            hi: hi.map(|v| v as isize),
            n2,
            dims,
            prev: vec![NONE; (n1 * n2) as usize],
            cur: vec![NONE; (n1 * n2) as usize],
            verts: Vec::new(),
            border: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Index of cell `(b, c)` within a layer.
    fn lidx(&self, b: isize, c: isize) -> usize {
        ((b - self.c0[1]) * self.n2 + (c - self.c0[2])) as usize
    }

    /// The vertex of cell `(b, c)` in a layer, `NONE` outside the cells.
    fn cell(&self, layer: &[u32], b: isize, c: isize) -> u32 {
        if b < self.c0[1] || c < self.c0[2] || b > self.hi[1] || c > self.hi[2] {
            NONE
        } else {
            layer[self.lidx(b, c)]
        }
    }

    /// One vertex per cell of layer `a` that straddles the boundary.
    fn vertices<F: Fn(isize, isize, isize) -> f64>(&mut self, f: &F, a: isize, geom: &Geometry) {
        self.cur.iter_mut().for_each(|v| *v = NONE);
        for b in self.c0[1]..=self.hi[1] {
            for c in self.c0[2]..=self.hi[2] {
                let vals: [f64; 8] = std::array::from_fn(|q| {
                    let o = CORNERS[q];
                    f(a + o[0], b + o[1], c + o[2])
                });
                let Some(off) = crossing_mean(&vals) else {
                    continue;
                };
                let idx = [a as f64 + off[0], b as f64 + off[1], c as f64 + off[2]];
                let k = self.lidx(b, c);
                self.cur[k] = self.verts.len() as u32;
                self.verts.push(geom.to_physical(idx));
                let d = self.dims;
                let on_border = a < 0 || b < 0 || c < 0 || a + 1 >= d[0] || b + 1 >= d[1] || c + 1 >= d[2];
                self.border.push(on_border);
            }
        }
    }

    /// Quads for the voxel edges whose four cells are all in layers `a − 1`
    /// and `a`.
    fn quads<F: Fn(isize, isize, isize) -> f64>(&mut self, f: &F, a: isize) {
        for b in self.c0[1]..=self.hi[1] + 1 {
            for c in self.c0[2]..=self.hi[2] + 1 {
                let here = inside(f(a, b, c));
                // axis-0 edge (a,b,c)–(a+1,b,c): cells (a, b-1..b, c-1..c), layer a
                if here != inside(f(a + 1, b, c)) {
                    let cur = &self.cur;
                    let q = [
                        self.cell(cur, b - 1, c - 1),
                        self.cell(cur, b, c - 1),
                        self.cell(cur, b, c),
                        self.cell(cur, b - 1, c),
                    ];
                    self.emit(q, !here);
                }
                if a == self.c0[0] {
                    continue;
                }
                let (prev, cur) = (&self.prev, &self.cur);
                // axis-1 edge (a,b,c)–(a,b+1,c): cells (a-1..a, b, c-1..c)
                if here != inside(f(a, b + 1, c)) {
                    let q = [
                        self.cell(prev, b, c - 1),
                        self.cell(prev, b, c),
                        self.cell(cur, b, c),
                        self.cell(cur, b, c - 1),
                    ];
                    self.emit(q, !here);
                }
                let (prev, cur) = (&self.prev, &self.cur);
                // axis-2 edge (a,b,c)–(a,b,c+1): cells (a-1..a, b-1..b, c)
                if here != inside(f(a, b, c + 1)) {
                    let q = [
                        self.cell(prev, b - 1, c),
                        self.cell(cur, b - 1, c),
                        self.cell(cur, b, c),
                        self.cell(prev, b, c),
                    ];
                    self.emit(q, !here);
                }
            }
        }
    }

    /// Two triangles for the quad `q` (reversed with `flip`), split along
    /// its shorter diagonal; nothing if a cell has no vertex.
    fn emit(&mut self, q: [u32; 4], flip: bool) {
        if q.contains(&NONE) {
            return;
        }
        let q = if flip { [q[0], q[3], q[2], q[1]] } else { q };
        let v = &self.verts;
        let d02 = crate::util::dist2(v[q[0] as usize], v[q[2] as usize]);
        let d13 = crate::util::dist2(v[q[1] as usize], v[q[3] as usize]);
        if d02 <= d13 {
            self.faces.push([q[0], q[1], q[2]]);
            self.faces.push([q[0], q[2], q[3]]);
        } else {
            self.faces.push([q[0], q[1], q[3]]);
            self.faces.push([q[1], q[2], q[3]]);
        }
    }
}

/// The mean of the boundary crossings on a cell's edges (corner values
/// `vals`), relative to its first corner; `None` if the cell is all inside
/// or all outside.
fn crossing_mean(vals: &[f64; 8]) -> Option<[f64; 3]> {
    let mask = (0..8).filter(|&q| inside(vals[q])).fold(0u8, |m, q| m | 1 << q);
    if mask == 0 || mask == 255 {
        return None;
    }
    let mut acc = [0.0; 3];
    let mut cnt = 0.0;
    for &(p, q) in &EDGES {
        if inside(vals[p]) != inside(vals[q]) {
            let t = ((ISO - vals[p]) / (vals[q] - vals[p])).clamp(0.0, 1.0);
            for d in 0..3 {
                acc[d] += CORNERS[p][d] as f64 + t * (CORNERS[q][d] - CORNERS[p][d]) as f64;
            }
            cnt += 1.0;
        }
    }
    Some([acc[0] / cnt, acc[1] / cnt, acc[2] / cnt])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::V3;

    fn ball(n: usize, r: f64) -> Array3<u8> {
        let c = (n as f64 - 1.0) / 2.0;
        Array3::from_shape_fn((n, n, n), |(i, j, k)| {
            let d = ((i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2)).sqrt();
            (d <= r) as u8
        })
    }

    #[test]
    fn sphere_is_closed_oriented_and_round() {
        for sigma in [0.0, 1.0] {
            let m = ball(30, 9.5);
            let e = extract_surface(&m.view(), &Geometry::new([1.0; 3]), sigma);
            let mesh = &e.mesh;
            assert!(mesh.is_closed_manifold(), "sigma={sigma}");
            assert_eq!(mesh.euler_characteristic(), 2);
            // outward orientation ⇒ positive enclosed volume close to the ball
            let vol = mesh.signed_volume();
            let want = 4.0 / 3.0 * std::f64::consts::PI * 9.5f64.powi(3);
            assert!((vol / want - 1.0).abs() < 0.08, "vol {vol} want {want}");
            let c = 14.5;
            for v in &mesh.vertices {
                let r = ((v[0] - c).powi(2) + (v[1] - c).powi(2) + (v[2] - c).powi(2)).sqrt();
                assert!((r - 9.5).abs() < 1.0, "r={r}");
            }
            assert!(e.border.iter().all(|&b| !b));
        }
    }

    #[test]
    fn border_vertices_flagged_and_anisotropic_spacing() {
        let m = Array3::from_shape_fn((6, 10, 10), |(_, j, k)| {
            (((j as f64 - 4.5).powi(2) + (k as f64 - 4.5).powi(2)) <= 9.0) as u8
        });
        let e = extract_surface(&m.view(), &Geometry::new([2.0, 0.5, 0.5]), 0.0);
        assert!(e.mesh.is_closed_manifold());
        assert!(e.border.iter().any(|&b| b));
        assert!(e.mesh.signed_volume() > 0.0);
    }

    #[test]
    fn coarse_grid_sphere_is_closed_round_and_sparser() {
        let m = ball(64, 25.0);
        let g = Geometry::new([1.0; 3]);
        let fine = extract_surface(&m.view(), &g, 0.0).mesh.vertices.len();
        for step in [2usize, 4] {
            let e = extract_surface_coarse(&m.view(), &g, step);
            let mesh = &e.mesh;
            assert!(mesh.is_closed_manifold(), "step {step}");
            assert_eq!(mesh.euler_characteristic(), 2);
            let n = mesh.vertices.len() as f64;
            let ratio = fine as f64 / n;
            assert!(
                ratio > 0.6 * (step * step) as f64,
                "step {step}: {fine} vs {n} vertices"
            );
            // the vertices lie on the binary surface (midway between voxel
            // centres): compare their radii with the fine extraction's
            let c = 31.5;
            let radii = |vs: &[V3]| -> (f64, f64, f64) {
                let r: Vec<f64> = vs
                    .iter()
                    .map(|v| ((v[0] - c).powi(2) + (v[1] - c).powi(2) + (v[2] - c).powi(2)).sqrt())
                    .collect();
                let mean = r.iter().sum::<f64>() / r.len() as f64;
                let (lo, hi) = r.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                (mean, lo, hi)
            };
            let f = radii(&extract_surface(&m.view(), &g, 0.0).mesh.vertices);
            let k = radii(&mesh.vertices);
            assert!(
                (k.0 - f.0).abs() < 0.25 && k.1 > f.1 - 0.75 && k.2 < f.2 + 0.75,
                "step {step}: fine (mean, min, max) {f:?}, coarse {k:?}"
            );
            assert!(e.border.iter().all(|&b| !b));
        }
    }

    #[test]
    fn coarse_grid_keeps_the_image_edge_as_border() {
        // a cylinder along axis 0, cut by both ends of the image
        let m = Array3::from_shape_fn((23, 40, 40), |(_, j, k)| {
            (((j as f64 - 19.5).powi(2) + (k as f64 - 19.5).powi(2)) <= 144.0) as u8
        });
        let g = Geometry::new([0.5, 0.25, 0.25]);
        let e = extract_surface_coarse(&m.view(), &g, 4);
        assert!(e.mesh.is_closed_manifold());
        assert!(e.border.iter().any(|&b| b));
        // the caps lie at the lattice's ends, within one coarse step of the image's
        for (v, &b) in e.mesh.vertices.iter().zip(&e.border) {
            if b {
                let i = v[0] / 0.5;
                assert!(i < 4.0 + 1e-9 || i > 22.0 - 4.0 - 1e-9, "cap vertex at slice {i}");
            }
        }
        assert!(e.mesh.signed_volume() > 0.0);
    }

    #[test]
    fn empty_and_single_voxel() {
        let m = Array3::<u8>::zeros((3, 3, 3));
        assert!(extract_surface(&m.view(), &Geometry::default(), 0.0)
            .mesh
            .vertices
            .is_empty());
        let mut m = m;
        m[[1, 1, 1]] = 1;
        let e = extract_surface(&m.view(), &Geometry::default(), 0.0);
        assert_eq!(e.mesh.vertices.len(), 8);
        assert_eq!(e.mesh.faces.len(), 12);
        assert!(e.mesh.is_closed_manifold());
        // vertices sit at the mean edge crossing: a cube of side 1/3
        assert!((e.mesh.signed_volume() - 1.0 / 27.0).abs() < 1e-9);
    }
}
