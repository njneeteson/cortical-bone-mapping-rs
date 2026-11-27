//! Moving the surface: the displacement each fit implies, its smooth
//! extension to unfitted vertices, and the move itself, either of the same
//! vertices or of a level set from which a new mesh is extracted (ours).
//!
//! * **Displacement** of a trusted node: `d_i = α (m_i − t_i/2) n_i`, capped at
//!   `max_step`. Its warm start for the next fit has `m` shifted by the move
//!   (`m ← m − α (m − t/2)`), since the profile origin moves with the node.
//! * **Extension** to untrusted, unfitted and excluded vertices: the
//!   harmonic extension of the known displacements ([`super::fill`]).
//! * **Caps** (vertices outside the image, where the mask is cut by the image
//!   edge) may only slide within their cap: the displacement is zeroed along
//!   the axes on which they lie outside.
//! * **Exclusion as a hard constraint**: a same-vertex move into an excluded
//!   voxel is halved up to six times (to 1/64 of the step) until it stays out,
//!   else dropped; with a level set the excluded voxels are kept outside
//!   (`φ ≥ h/2`) after every move.
//! * **State transfer** to a new mesh: every new vertex takes the state of the
//!   nearest *moved* old vertex, with its warm `m` shifted by how much farther
//!   inside it lies along that vertex's normal.

use ndarray::{Array3, ArrayView3};
use rayon::prelude::*;

use super::state::NodeState;
use crate::kdtree::KdTree;
use crate::model::{M, T};
use crate::result::CbmResult;
use crate::surface::levelset::LevelSet;
use crate::surface::mesh::Mesh;
use crate::surface::nets::Extracted;
use crate::util::{add, dot, scale, sub, V3};
use crate::volume::{is_set_at, Geometry};

/// What one fit implies for every surface vertex.
pub(super) struct Step {
    /// Displacement toward the fitted periosteal edge.
    pub disp: Vec<V3>,
    /// The displacement is set (fitted and trusted, held, or fixed at 0).
    pub known: Vec<bool>,
    /// The vertex was fitted.
    pub fitted: Vec<bool>,
    /// Its fit could not be trusted to locate the edge.
    pub failed: Vec<bool>,
    /// Change of thickness since the previous fit.
    pub t_change: Vec<Option<f64>>,
}

/// The displacement `α (m − t/2) n` of every trusted node (capped at
/// `max_step`), updating each node's warm start (`m` shifted by the move)
/// and previous thickness. Held nodes do not move; untrusted ones are filled
/// in later from their neighbours.
pub(super) fn displacements(
    result: &CbmResult,
    trusted: &[bool],
    alpha: f64,
    max_step: Option<f64>,
    st: &mut NodeState,
) -> Step {
    let nv = st.warm.len();
    let mut step = Step {
        disp: vec![[0.0; 3]; nv],
        known: vec![false; nv],
        fitted: vec![false; nv],
        failed: vec![false; nv],
        t_change: vec![None; nv],
    };
    for (k, &v) in result.vertex_ids.iter().enumerate() {
        let v = v as usize;
        step.fitted[v] = true;
        if st.held[v] {
            step.known[v] = true;
            continue;
        }
        let p = result.params[k];
        step.t_change[v] = st.prev_t[v].map(|t| (p[T] - t).abs());
        st.prev_t[v] = Some(p[T]);
        if !trusted[k] {
            st.warm[v] = None;
            step.failed[v] = true;
            continue;
        }
        let mut d = alpha * (p[M] - 0.5 * p[T]);
        if let Some(cap) = max_step {
            d = d.clamp(-cap, cap);
        }
        step.disp[v] = scale(result.normals[k], d);
        step.known[v] = true;
        let mut w = p;
        w[M] -= d;
        st.warm[v] = Some(w);
    }
    step
}

/// Axes along which each vertex lies outside the image (cap vertices, where
/// the mask is cut by the image edge, may only slide within their cap).
pub(super) fn locked_axes(mesh: &Mesh, geom: &Geometry, dims: [f64; 3]) -> Vec<[bool; 3]> {
    mesh.vertices
        .iter()
        .map(|&p| {
            let q = geom.to_index(p);
            [0, 1, 2].map(|a| q[a] < 0.0 || q[a] > dims[a] - 1.0)
        })
        .collect()
}

/// Move the same vertices by `disp`, except along their locked axes and
/// never into an `exclude` voxel (a move into one is halved until it stays
/// out, else dropped).
pub(super) fn move_vertices(
    mesh: &mut Mesh,
    disp: &[V3],
    locked: &[[bool; 3]],
    exclude: Option<&ArrayView3<bool>>,
    geom: &Geometry,
) {
    let blocked = |p: V3| exclude.is_some_and(|e| is_set_at(e, geom, p));
    for v in 0..mesh.vertices.len() {
        let mut d = disp[v];
        for a in 0..3 {
            if locked[v][a] {
                d[a] = 0.0;
            }
        }
        let mut step = d;
        for _ in 0..6 {
            if !blocked(add(mesh.vertices[v], step)) {
                break;
            }
            step = scale(step, 0.5);
        }
        if blocked(add(mesh.vertices[v], step)) {
            step = [0.0; 3];
        }
        mesh.vertices[v] = add(mesh.vertices[v], step);
    }
}

/// Move the level set by `s` (per vertex, along the inward normal), keep
/// `exclude` voxels outside, extract the new mesh and re-distance. Returns
/// the new surface, the time spent per stage, and for every new vertex the
/// nearest moved old vertex with how much further inside the new one lies.
pub(super) fn move_level_set(
    ls: &mut LevelSet,
    mesh: &Mesh,
    outward: &[V3],
    s: &[f64],
    exclude: Option<&ArrayView3<bool>>,
    timings: &mut Vec<(&'static str, f64)>,
) -> (Extracted, Vec<(usize, f64)>) {
    let clock = std::time::Instant::now;
    let ta = clock();
    let width = mesh.mean_edge_length().max(1e-9);
    let mut changed: Array3<bool> = ls.advance(&mesh.vertices, s, width, 8);
    if let Some(e) = exclude {
        ls.forbid(e, &mut changed);
    }
    timings.push(("level set: advance", ta.elapsed().as_secs_f64()));
    let te = clock();
    let e = ls.extract();
    timings.push(("level set: extraction", te.elapsed().as_secs_f64()));
    let tr = clock();
    ls.redistance(&e.mesh, Some(&changed));
    timings.push(("level set: redistancing", tr.elapsed().as_secs_f64()));
    let tc = clock();
    let moved: Vec<V3> = (0..mesh.vertices.len())
        .map(|v| sub(mesh.vertices[v], scale(outward[v], s[v])))
        .collect();
    let tree = KdTree::new(&moved);
    let map = e
        .mesh
        .vertices
        .par_iter()
        .map(|&p| {
            let j = tree.knn(p, 1, None)[0].0 as usize;
            (j, -dot(sub(p, moved[j]), outward[j]))
        })
        .collect();
    timings.push(("level set: state transfer", tc.elapsed().as_secs_f64()));
    log::debug!(
        "level set move: {} -> {} vertices; {}",
        mesh.vertices.len(),
        e.mesh.vertices.len(),
        timings
            .iter()
            .map(|(n, t)| format!("{n} {t:.3} s"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    (e, map)
}
