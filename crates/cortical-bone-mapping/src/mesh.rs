//! Triangle meshes: normals, Taubin smoothing, the cotangent Laplacian,
//! sub-meshes, rasterisation back to a mask, and VTK output.

use std::collections::HashMap;
use std::io::Write;

use ndarray::Array3;
use rayon::prelude::*;

use crate::util::{add, cross, dot, norm, scale, sub, V3};
use crate::volume::Geometry;

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

    /// Rasterise the region enclosed by the mesh into a mask of `shape`
    /// voxels: a voxel is inside if a ray from its centre along axis 0 crosses
    /// the surface an odd number of times (even–odd rule). Works for any
    /// closed mesh, including self-intersecting ones (where the parity rule
    /// decides). Memory is the output plus 8 bytes per ray crossing.
    pub fn voxelize(&self, shape: (usize, usize, usize), geom: &Geometry) -> Array3<bool> {
        let (d0, d1, d2) = shape;
        // Crossings (column id, axis-0 index coordinate).
        let tri_idx: Vec<[V3; 3]> = self
            .faces
            .iter()
            .map(|f| f.map(|i| geom.to_index(self.vertices[i as usize])))
            .collect();
        // Tiny irrational offset so rays never pass exactly through vertices
        // or edges lying on the integer lattice.
        const EPS1: f64 = 1.234_567e-7;
        const EPS2: f64 = 7.654_321e-8;
        let mut crossings: Vec<(u32, f32)> = tri_idx
            .par_iter()
            .flat_map_iter(|t| {
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
                    return out.into_iter();
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
                out.into_iter()
            })
            .collect();
        crossings.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        let mut out = Array3::from_elem(shape, false);
        // Fill column by column.
        let mut s = 0;
        while s < crossings.len() {
            let col = crossings[s].0;
            let mut e = s;
            while e < crossings.len() && crossings[e].0 == col {
                e += 1;
            }
            let (j, k) = (col as usize / d2, col as usize % d2);
            let xs = &crossings[s..e];
            for pair in xs.chunks(2) {
                if pair.len() < 2 {
                    break;
                }
                let a = (pair[0].1 as f64).ceil().max(0.0) as usize;
                let b = (pair[1].1 as f64).floor();
                if b < 0.0 {
                    continue;
                }
                let b = (b as usize).min(d0 - 1);
                for i in a..=b {
                    out[[i, j, k]] = true;
                }
            }
            s = e;
        }
        out
    }

    /// Write the mesh as a legacy binary VTK PolyData file, with optional
    /// per-point scalar and 3-vector arrays.
    pub fn write_vtk(
        &self,
        path: &std::path::Path,
        scalars: &[(&str, &[f64])],
        vectors: &[(&str, &[V3])],
    ) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let n = self.vertices.len();
        writeln!(w, "# vtk DataFile Version 3.0")?;
        writeln!(w, "cortical-bone-mapping")?;
        writeln!(w, "BINARY")?;
        writeln!(w, "DATASET POLYDATA")?;
        writeln!(w, "POINTS {n} double")?;
        for v in &self.vertices {
            for c in v {
                w.write_all(&c.to_be_bytes())?;
            }
        }
        writeln!(w)?;
        writeln!(w, "POLYGONS {} {}", self.faces.len(), self.faces.len() * 4)?;
        for f in &self.faces {
            w.write_all(&3i32.to_be_bytes())?;
            for &i in f {
                w.write_all(&(i as i32).to_be_bytes())?;
            }
        }
        writeln!(w)?;
        if !scalars.is_empty() || !vectors.is_empty() {
            writeln!(w, "POINT_DATA {n}")?;
        }
        for (name, data) in scalars {
            assert_eq!(data.len(), n, "scalar array {name} has the wrong length");
            writeln!(w, "SCALARS {name} double 1")?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for x in *data {
                w.write_all(&x.to_be_bytes())?;
            }
            writeln!(w)?;
        }
        for (name, data) in vectors {
            assert_eq!(data.len(), n, "vector array {name} has the wrong length");
            writeln!(w, "VECTORS {name} double")?;
            for v in *data {
                for c in v {
                    w.write_all(&c.to_be_bytes())?;
                }
            }
            writeln!(w)?;
        }
        w.flush()
    }
}

#[inline]
fn minmax(a: f64, b: f64, c: f64) -> (f64, f64) {
    (a.min(b).min(c), a.max(b).max(c))
}

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

    #[test]
    fn vtk_writes() {
        let m = Mesh::uv_sphere(1.0, 6, 4);
        let dir = std::env::temp_dir().join("cbm_vtk_test.vtk");
        let s: Vec<f64> = (0..m.vertices.len()).map(|i| i as f64).collect();
        m.write_vtk(&dir, &[("idx", &s)], &[("n", &m.vertex_normals())])
            .unwrap();
        let bytes = std::fs::read(&dir).unwrap();
        assert!(bytes.starts_with(b"# vtk DataFile"));
        let _ = std::fs::remove_file(dir);
    }
}
