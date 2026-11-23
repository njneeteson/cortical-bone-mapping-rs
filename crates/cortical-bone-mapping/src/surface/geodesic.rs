//! Distances along a surface mesh: truncated Dijkstra over its edge graph.
//!
//! Two points on opposite walls of a notch, or on the two cortices of a thin
//! plate, are close in space but far apart along the surface. Neighbourhoods
//! built from surface distance therefore do not couple them, where Euclidean
//! (k-d tree) neighbourhoods do. Separate components of a mesh (two bones)
//! are at infinite distance and never couple.
//!
//! The distance between two vertices is approximated by the shortest path
//! along mesh edges, with Euclidean edge lengths, found by Dijkstra's
//! algorithm (Dijkstra 1959) from the source and stopped when the frontier
//! passes a radius or enough vertices have been settled. A search touches
//! only the vertices it settles and their one-rings, so building a
//! neighbourhood of `k` vertices costs `O(k log k)`, and all of them run in
//! parallel. Edge paths zigzag, so they overestimate the geodesic distance.
//! The worst case depends on the edge directions: with edges in 2m evenly
//! spaced directions it is `1/cos(π/2m)`, i.e. 8% with 8 directions (a grid
//! split along both diagonals), 15% with 6 (equilateral triangles) and 41%
//! with 4. Surface-nets meshes split each quad along its shorter diagonal,
//! so both diagonals occur. For a Gaussian weight the overestimate is a
//! slightly narrower kernel. Local fast marching on the triangles (Kimmel &
//! Sethian 1998) would be exact to first order.
//!
//! Results are canonical: vertices are settled in order of (distance,
//! index), so ties are resolved by index.
//!
//! # References
//!
//! * E. W. Dijkstra, *A note on two problems in connexion with graphs*,
//!   Numerische Mathematik 1:269–271, 1959.
//! * R. Kimmel, J. A. Sethian, *Computing geodesic paths on manifolds*,
//!   Proceedings of the National Academy of Sciences 95(15):8431–8435, 1998.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::mesh::{Adjacency, Mesh};
use crate::util::{dist2, V3};

/// The edge graph of a mesh, for surface-distance queries.
pub struct SurfaceGraph {
    points: Vec<V3>,
    adj: Adjacency,
}

/// Reusable per-thread storage for searches: tentative distances (infinite
/// except where touched) and the vertices touched.
pub struct Scratch {
    dist: Vec<f64>,
    touched: Vec<u32>,
    heap: BinaryHeap<Entry>,
}

/// A heap entry, ordered so that the smallest (distance, index) pops first.
#[derive(PartialEq)]
struct Entry(f64, u32);

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl SurfaceGraph {
    /// The graph of the mesh's vertices and one-ring edges.
    pub fn new(mesh: &Mesh) -> Self {
        SurfaceGraph {
            points: mesh.vertices.clone(),
            adj: mesh.adjacency(),
        }
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the graph has no vertices.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Vertex positions.
    pub fn points(&self) -> &[V3] {
        &self.points
    }

    /// Storage for searches on this graph.
    pub fn scratch(&self) -> Scratch {
        Scratch {
            dist: vec![f64::INFINITY; self.points.len()],
            touched: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// The vertices within surface distance `radius` of `src`, nearest
    /// first, at most `max` of them (not counting `src` itself, which is
    /// left out), as `(vertex, squared distance)`; ties by index.
    pub fn nearest(&self, src: usize, max: usize, radius: f64, s: &mut Scratch) -> Vec<(u32, f64)> {
        let mut out = Vec::new();
        if max == 0 {
            return out;
        }
        self.search(src, radius, s, |v, d| {
            if v != src as u32 {
                out.push((v, d * d));
            }
            out.len() < max
        });
        out
    }

    /// Dijkstra from `src` up to `radius`: calls `visit(vertex, distance)`
    /// for each settled vertex in order of (distance, index), `src` first,
    /// until it returns `false` or the frontier passes `radius`.
    fn search(&self, src: usize, radius: f64, s: &mut Scratch, mut visit: impl FnMut(u32, f64) -> bool) {
        s.heap.clear();
        s.dist[src] = 0.0;
        s.touched.push(src as u32);
        s.heap.push(Entry(0.0, src as u32));
        while let Some(Entry(d, v)) = s.heap.pop() {
            if d > s.dist[v as usize] {
                continue; // a stale entry
            }
            if d > radius || !visit(v, d) {
                break;
            }
            for &u in self.adj.of(v as usize) {
                let du = d + dist2(self.points[v as usize], self.points[u as usize]).sqrt();
                if du < s.dist[u as usize] {
                    if s.dist[u as usize].is_infinite() {
                        s.touched.push(u);
                    }
                    s.dist[u as usize] = du;
                    s.heap.push(Entry(du, u));
                }
            }
        }
        for &v in &s.touched {
            s.dist[v as usize] = f64::INFINITY;
        }
        s.touched.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{norm, sub};

    /// A flat square grid of `n × n` vertices at spacing `h`, two triangles per cell.
    fn grid(n: usize, h: f64) -> Mesh {
        let vertices = (0..n * n)
            .map(|i| [(i / n) as f64 * h, (i % n) as f64 * h, 0.0])
            .collect();
        let id = |i: usize, j: usize| (i * n + j) as u32;
        let mut faces = Vec::new();
        for i in 0..n - 1 {
            for j in 0..n - 1 {
                faces.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
                faces.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
            }
        }
        Mesh { vertices, faces }
    }

    #[test]
    fn plane_distances_are_close_to_euclidean_and_never_shorter() {
        let m = grid(41, 0.1);
        let g = SurfaceGraph::new(&m);
        let mut s = g.scratch();
        let src = 20 * 41 + 20;
        let near = g.nearest(src, usize::MAX, 1.5, &mut s);
        assert!(!near.is_empty());
        for &(v, d2) in &near {
            let e = norm(sub(m.vertices[v as usize], m.vertices[src]));
            let d = d2.sqrt();
            // one diagonal per cell: paths in 6 directions here, but against
            // the diagonal only the axes help (√2 at worst)
            assert!(
                d >= e - 1e-12 && d <= std::f64::consts::SQRT_2 * e + 1e-12,
                "{v}: {d} vs {e}"
            );
            if (m.vertices[v as usize][0] - m.vertices[src][0]).abs() < 1e-9 {
                assert!((d - e).abs() < 1e-12, "along an axis the path is straight");
            }
            assert!(d <= 1.5 + 1e-12);
        }
        // sorted by (distance, index), source excluded
        assert!(near.windows(2).all(|w| (w[0].1, w[0].0) <= (w[1].1, w[1].0)));
        assert!(near.iter().all(|&(v, _)| v as usize != src));
        assert_eq!(g.nearest(src, 5, f64::INFINITY, &mut s).len(), 5);
    }

    #[test]
    fn separate_components_never_meet() {
        let a = grid(5, 0.1);
        let mut b = grid(5, 0.1);
        for v in &mut b.vertices {
            v[2] = 0.05; // 0.05 above, much closer than any surface path
        }
        let off = a.vertices.len() as u32;
        let mesh = Mesh {
            vertices: a.vertices.iter().chain(&b.vertices).cloned().collect(),
            faces: a
                .faces
                .iter()
                .cloned()
                .chain(b.faces.iter().map(|f| f.map(|v| v + off)))
                .collect(),
        };
        let g = SurfaceGraph::new(&mesh);
        let mut s = g.scratch();
        let near = g.nearest(0, usize::MAX, f64::INFINITY, &mut s);
        assert_eq!(near.len(), 24);
        assert!(near.iter().all(|&(v, _)| v < off));
    }
}
