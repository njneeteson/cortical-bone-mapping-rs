//! Neighbourhoods of the surface nodes, by Euclidean distance (k-d tree) or
//! by distance along the surface (truncated Dijkstra on the mesh,
//! [`crate::surface::geodesic`]).
//!
//! The regularisation operator `ω`, the Gaussian curvature Laplacian and the
//! multigrid basis all weight neighbours by a Gaussian of their distance;
//! only the metric differs. Euclidean neighbourhoods are those of our 2025
//! paper (Neeteson & Boyd 2025) and the default of the 1D methods; surface
//! neighbourhoods do not couple the two walls of a notch or the two cortices
//! of a plate, which are close in space but far apart along the surface.
//!
//! # References
//!
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.

use crate::kdtree::KdTree;
use crate::surface::geodesic::{Scratch, SurfaceGraph};
use crate::surface::mesh::Mesh;
use crate::util::V3;

/// How the distance between surface nodes is measured for neighbourhoods.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Distance {
    /// Straight-line distance (k-d tree).
    #[default]
    Euclidean,
    /// Distance along the surface (shortest path on the mesh edges).
    Surface,
}

/// Neighbour queries over the vertices of a mesh.
pub enum Neighbours {
    Euclidean { tree: KdTree, points: Vec<V3> },
    Surface(SurfaceGraph),
}

/// Per-thread storage for surface queries (none for Euclidean ones).
pub struct QueryScratch(Option<Scratch>);

impl Neighbours {
    /// Neighbour queries over the vertices of `mesh` by `distance`.
    pub fn new(mesh: &Mesh, distance: Distance) -> Self {
        log::debug!("neighbours: {} vertices, {distance:?} distance", mesh.vertices.len());
        match distance {
            Distance::Euclidean => Self::euclidean(&mesh.vertices),
            Distance::Surface => Neighbours::Surface(SurfaceGraph::new(mesh)),
        }
    }

    /// Euclidean queries over a point set (no mesh needed).
    pub fn euclidean(points: &[V3]) -> Self {
        Neighbours::Euclidean {
            tree: KdTree::new(points),
            points: points.to_vec(),
        }
    }

    /// The vertex positions.
    pub fn points(&self) -> &[V3] {
        match self {
            Neighbours::Euclidean { points, .. } => points,
            Neighbours::Surface(g) => g.points(),
        }
    }

    /// Storage for one thread's queries.
    pub fn scratch(&self) -> QueryScratch {
        QueryScratch(match self {
            Neighbours::Euclidean { .. } => None,
            Neighbours::Surface(g) => Some(g.scratch()),
        })
    }

    /// The `k` nearest vertices to vertex `i`, excluding `i`, as
    /// `(vertex, squared distance)`, nearest first (ties by index).
    pub fn nearest(&self, i: usize, k: usize, s: &mut QueryScratch) -> Vec<(u32, f64)> {
        match self {
            Neighbours::Euclidean { tree, points } => tree.knn(points[i], k, Some(i as u32)),
            Neighbours::Surface(g) => g.nearest(i, k, f64::INFINITY, s.0.as_mut().unwrap()),
        }
    }

    /// All vertices within distance `r` of vertex `i`, including `i`, as
    /// `(vertex, squared distance)`, sorted by index.
    pub fn within(&self, i: usize, r: f64, out: &mut Vec<(u32, f64)>, s: &mut QueryScratch) {
        match self {
            Neighbours::Euclidean { tree, points } => tree.within(points[i], r, out),
            Neighbours::Surface(g) => {
                out.clear();
                out.push((i as u32, 0.0));
                out.extend(g.nearest(i, usize::MAX, r, s.0.as_mut().unwrap()));
                out.sort_unstable_by_key(|c| c.0);
            }
        }
    }
}
