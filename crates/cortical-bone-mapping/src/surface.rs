//! Surface extraction from a binary mask with (naive) surface nets.
//!
//! Each grid cell (a cube whose 8 corners are voxel centres) that straddles
//! the boundary gets one vertex, at the mean of the boundary crossings on its
//! edges. Each voxel edge whose endpoints differ emits a quad joining the four
//! cells around it, oriented so the normal points out of the foreground. The
//! quads are split into triangles along their shorter diagonal.
//!
//! Compared with marching cubes this produces about half as many, much
//! better-shaped triangles and needs no lookup tables. The cell→vertex index
//! is kept for only two cell layers at a time, so memory is
//! `O(slice)` + output, regardless of the image size. Voxels outside the
//! image count as background, so the surface is always closed.

use ndarray::{Array3, ArrayView3};

use crate::mesh::Mesh;
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
        return Extracted {
            mesh: Mesh::default(),
            border: Vec::new(),
        };
    };
    let dims = {
        let d = mask.dim();
        [d.0 as isize, d.1 as isize, d.2 as isize]
    };
    if smoothing_sigma > 0.0 {
        // Blur a padded copy of the bounding box.
        let pad = (3.0 * smoothing_sigma).ceil() as usize + 1;
        let lo_p: Vec<isize> = (0..3).map(|a| lo[a] as isize - pad as isize).collect();
        let shape = (
            hi[0] - lo[0] + 1 + 2 * pad,
            hi[1] - lo[1] + 1 + 2 * pad,
            hi[2] - lo[2] + 1 + 2 * pad,
        );
        let sub = Array3::from_shape_fn(shape, |(i, j, k)| {
            let g = [i as isize + lo_p[0], j as isize + lo_p[1], k as isize + lo_p[2]];
            if (0..3).all(|a| g[a] >= 0 && g[a] < dims[a]) {
                (mask[[g[0] as usize, g[1] as usize, g[2] as usize]].to_f64() != 0.0) as u8 as f64
            } else {
                0.0
            }
        });
        let blurred = gaussian_filter(&sub, [smoothing_sigma; 3]);
        let field = |i: isize, j: isize, k: isize| -> f64 {
            let l = [i - lo_p[0], j - lo_p[1], k - lo_p[2]];
            let s = blurred.dim();
            let s = [s.0 as isize, s.1 as isize, s.2 as isize];
            if (0..3).all(|a| l[a] >= 0 && l[a] < s[a]) {
                blurred[[l[0] as usize, l[1] as usize, l[2] as usize]]
            } else {
                0.0
            }
        };
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

const ISO: f64 = 0.5;

/// Extract the zero level of a signed distance `phi` (negative inside),
/// stored for the image voxels `lo + index`, as a closed outward-oriented
/// mesh. Voxels outside `phi` or the image (`dims`) count as outside;
/// `border` flags vertices whose cell touches the outside of the image.
pub fn extract_level_set(phi: &ArrayView3<f32>, lo: [usize; 3], dims: [usize; 3], geom: &Geometry) -> Extracted {
    let (s0, s1, s2) = phi.dim();
    if s0 == 0 || s1 == 0 || s2 == 0 {
        return Extracted {
            mesh: Mesh::default(),
            border: Vec::new(),
        };
    }
    let shape = [s0 as isize, s1 as isize, s2 as isize];
    let lo_i = lo.map(|v| v as isize);
    let field = |i: isize, j: isize, k: isize| -> f64 {
        let l = [i - lo_i[0], j - lo_i[1], k - lo_i[2]];
        if (0..3).all(|a| l[a] >= 0 && l[a] < shape[a]) {
            ISO - phi[[l[0] as usize, l[1] as usize, l[2] as usize]] as f64
        } else {
            0.0
        }
    };
    let hi = [lo[0] + s0 - 1, lo[1] + s1 - 1, lo[2] + s2 - 1];
    nets(field, lo, hi, dims.map(|v| v as isize), geom)
}

fn nets<F: Fn(isize, isize, isize) -> f64>(
    f: F,
    lo: [usize; 3],
    hi: [usize; 3],
    dims: [isize; 3],
    geom: &Geometry,
) -> Extracted {
    // Cells c = (a, b, c) with corners voxels c + {0,1}^3, covering
    // a ∈ [lo0 − 1, hi0] etc.
    let c0 = [lo[0] as isize - 1, lo[1] as isize - 1, lo[2] as isize - 1];
    let n1 = (hi[1] - lo[1] + 2) as isize;
    let n2 = (hi[2] - lo[2] + 2) as isize;
    let none = u32::MAX;
    let mut prev = vec![none; (n1 * n2) as usize];
    let mut cur = vec![none; (n1 * n2) as usize];
    let mut verts: Vec<[f64; 3]> = Vec::new();
    let mut border: Vec<bool> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();

    let lidx = |b: isize, c: isize| -> usize { ((b - c0[1]) * n2 + (c - c0[2])) as usize };
    let inside = |v: f64| v > ISO;

    // Cube corner offsets and the 12 edges as corner index pairs.
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

    for a in c0[0]..=hi[0] as isize {
        // 1. vertices of cell layer a
        cur.iter_mut().for_each(|v| *v = none);
        for b in c0[1]..=hi[1] as isize {
            for c in c0[2]..=hi[2] as isize {
                let mut vals = [0.0; 8];
                let mut mask = 0u8;
                for (q, o) in CORNERS.iter().enumerate() {
                    vals[q] = f(a + o[0], b + o[1], c + o[2]);
                    if inside(vals[q]) {
                        mask |= 1 << q;
                    }
                }
                if mask == 0 || mask == 255 {
                    continue;
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
                let idx = [
                    a as f64 + acc[0] / cnt,
                    b as f64 + acc[1] / cnt,
                    c as f64 + acc[2] / cnt,
                ];
                cur[lidx(b, c)] = verts.len() as u32;
                verts.push(geom.to_physical(idx));
                let on_border = a < 0 || b < 0 || c < 0 || a + 1 >= dims[0] || b + 1 >= dims[1] || c + 1 >= dims[2];
                border.push(on_border);
            }
        }
        // 2. quads for edges whose cells are all in layers a−1 and a.
        let emit = |q: [u32; 4], flip: bool, verts: &Vec<[f64; 3]>, faces: &mut Vec<[u32; 3]>| {
            if q.contains(&none) {
                return;
            }
            let q = if flip { [q[0], q[3], q[2], q[1]] } else { q };
            let d02 = crate::util::dist2(verts[q[0] as usize], verts[q[2] as usize]);
            let d13 = crate::util::dist2(verts[q[1] as usize], verts[q[3] as usize]);
            if d02 <= d13 {
                faces.push([q[0], q[1], q[2]]);
                faces.push([q[0], q[2], q[3]]);
            } else {
                faces.push([q[0], q[1], q[3]]);
                faces.push([q[1], q[2], q[3]]);
            }
        };
        let cell = |layer: &Vec<u32>, b: isize, c: isize| -> u32 {
            if b < c0[1] || c < c0[2] || b > hi[1] as isize || c > hi[2] as isize {
                none
            } else {
                layer[lidx(b, c)]
            }
        };
        for b in c0[1]..=hi[1] as isize + 1 {
            for c in c0[2]..=hi[2] as isize + 1 {
                let here = inside(f(a, b, c));
                // axis-0 edge (a,b,c)–(a+1,b,c): cells (a, b-1..b, c-1..c), layer a
                if here != inside(f(a + 1, b, c)) {
                    let q = [
                        cell(&cur, b - 1, c - 1),
                        cell(&cur, b, c - 1),
                        cell(&cur, b, c),
                        cell(&cur, b - 1, c),
                    ];
                    emit(q, !here, &verts, &mut faces);
                }
                if a > c0[0] {
                    // axis-1 edge (a,b,c)–(a,b+1,c): cells (a-1..a, b, c-1..c)
                    if here != inside(f(a, b + 1, c)) {
                        let q = [
                            cell(&prev, b, c - 1),
                            cell(&prev, b, c),
                            cell(&cur, b, c),
                            cell(&cur, b, c - 1),
                        ];
                        emit(q, !here, &verts, &mut faces);
                    }
                    // axis-2 edge (a,b,c)–(a,b,c+1): cells (a-1..a, b-1..b, c)
                    if here != inside(f(a, b, c + 1)) {
                        let q = [
                            cell(&prev, b - 1, c),
                            cell(&cur, b - 1, c),
                            cell(&cur, b, c),
                            cell(&prev, b, c),
                        ];
                        emit(q, !here, &verts, &mut faces);
                    }
                }
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    Extracted {
        mesh: Mesh { vertices: verts, faces },
        border,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
