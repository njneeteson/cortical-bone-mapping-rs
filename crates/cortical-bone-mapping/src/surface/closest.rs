//! Closest points on a triangle mesh, for the level set's distances and the
//! volumetric model's band.
//!
//! The closest point to `p` on a triangle is found by Ericson's (2005, §5.1.5)
//! method: classify `p` into the Voronoi region of a vertex, an edge or the
//! face from signed dot products, and project it onto that feature. The
//! barycentric coordinates `(β_a, β_b, β_c)` of the closest point come out of
//! the same classification (a vertex region has `(1, 0, 0)`, an edge region
//! `(1 − s, s, 0)`, the face region the normalised areas `(va, vb, vc)`).
//!
//! On a mesh, the closest point lies on a face around one of the nearest
//! vertices. [`MeshPoints`] searches the faces around the 4 nearest vertices
//! (k-d tree): the nearest vertex alone can miss the closest face of a long
//! triangle, and 4 make that very unlikely; on a smoothed surface-nets mesh it
//! agreed with brute force at 399,999 of 400,000 points (the one other point
//! off by 8·10⁻⁵ mm).
//!
//! # References
//!
//! * C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2005,
//!   §5.1.5.

use crate::kdtree::KdTree;
use crate::surface::mesh::Mesh;
use crate::util::{add, dot, norm, scale, sub, V3};

/// Vertices searched around a query point.
const NEAREST: usize = 4;

/// Faces incident to each vertex.
pub(crate) struct VertexFaces {
    offsets: Vec<usize>,
    faces: Vec<u32>,
}

impl VertexFaces {
    /// The faces of every vertex, in compressed rows.
    pub fn new(mesh: &Mesh) -> Self {
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

    /// The faces incident to vertex `v`.
    pub fn of(&self, v: usize) -> &[u32] {
        &self.faces[self.offsets[v]..self.offsets[v + 1]]
    }
}

/// The closest point of a mesh to a query point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Closest {
    /// The face it lies on.
    pub face: u32,
    pub point: V3,
    /// Its barycentric coordinates on the face's vertices, in face order.
    pub bary: [f64; 3],
    /// Its distance from the query point.
    pub dist: f64,
}

/// Closest-point queries on a mesh.
pub struct MeshPoints<'m> {
    mesh: &'m Mesh,
    tree: KdTree,
    faces: VertexFaces,
}

impl<'m> MeshPoints<'m> {
    pub fn new(mesh: &'m Mesh) -> Self {
        MeshPoints {
            mesh,
            tree: KdTree::new(&mesh.vertices),
            faces: VertexFaces::new(mesh),
        }
    }

    /// The closest point to `q` on the faces around its nearest vertices;
    /// `None` if the nearest vertex is farther than `reach` (or there is no
    /// vertex).
    pub fn closest(&self, q: V3, reach: f64) -> Option<Closest> {
        self.closest_bounded(q, reach, None).0
    }

    /// [`Self::closest`], with the search for the nearest vertices confined
    /// to `bound` of `q` when given (exact if they lie within it: a
    /// neighbouring query's [`NEAREST`]-th vertex distance plus the distance
    /// between the two queries), and the [`NEAREST`]-th vertex distance of
    /// `q`, the bound for the next query (∞ without vertices).
    pub fn closest_bounded(&self, q: V3, reach: f64, bound: Option<f64>) -> (Option<Closest>, f64) {
        let nn = match bound {
            Some(b) => self.tree.knn_bounded(q, NEAREST, b),
            None => self.nearest_vertices(q),
        };
        let far = nn.last().map_or(f64::INFINITY, |v| v.1.sqrt());
        if nn.is_empty() || nn[0].1.sqrt() > reach {
            return (None, far);
        }
        let mut best: Option<Closest> = None;
        for &(v, _) in &nn {
            for &f in self.faces.of(v as usize) {
                let t = self.mesh.faces[f as usize].map(|x| self.mesh.vertices[x as usize]);
                let (point, bary) = closest_on_triangle_bary(q, t[0], t[1], t[2]);
                let dist = norm(sub(q, point));
                if best.is_none_or(|b| dist < b.dist) {
                    best = Some(Closest {
                        face: f,
                        point,
                        bary,
                        dist,
                    });
                }
            }
        }
        (best, far)
    }

    /// The `NEAREST` vertices nearest to `q`, as `(index, squared distance)`.
    pub fn nearest_vertices(&self, q: V3) -> Vec<(u32, f64)> {
        self.tree.knn(q, NEAREST, None)
    }
}

/// Closest point to `p` on triangle `abc` (Ericson 2005, §5.1.5).
pub fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> V3 {
    closest_on_triangle_bary(p, a, b, c).0
}

/// Closest point to `p` on triangle `abc` and its barycentric coordinates
/// `(β_a, β_b, β_c)`.
pub fn closest_on_triangle_bary(p: V3, a: V3, b: V3, c: V3) -> (V3, [f64; 3]) {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (add(a, scale(ab, v)), [1.0 - v, v, 0.0]);
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (add(a, scale(ac, w)), [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (add(b, scale(sub(c, b), w)), [0.0, 1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (add(a, add(scale(ab, v), scale(ac, w))), [1.0 - v - w, v, w])
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn barycentric_coordinates_reproduce_the_point() {
        let (a, b, c) = ([0.3, -0.2, 0.1], [1.4, 0.1, -0.3], [0.2, 1.1, 0.4]);
        let mut rng = crate::util::Rng::new(5);
        for _ in 0..2000 {
            let p = [rng.normal(), rng.normal(), rng.normal()];
            let (q, w) = closest_on_triangle_bary(p, a, b, c);
            assert!(w.iter().all(|&x| x >= -1e-12) && (w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let r = add(add(scale(a, w[0]), scale(b, w[1])), scale(c, w[2]));
            assert!(norm(sub(q, r)) < 1e-12, "{q:?} vs {r:?}");
        }
    }
}
