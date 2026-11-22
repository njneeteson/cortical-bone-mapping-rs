//! The per-node state refinement carries from fit to fit, and regional
//! convergence: which regions are frozen, when they thaw, and which frozen
//! nodes are refitted as boundary conditions.

use rayon::prelude::*;

use super::motion::Step;
use super::options::FreezeOptions;
use crate::kdtree::KdTree;
use crate::model::{Params, M};
use crate::options::{Method, Options};
use crate::surface::mesh::Mesh;
use crate::util::{norm, V3};

/// Per surface vertex.
pub(super) struct NodeState {
    /// Warm start of the next fit (`m` already shifted by the move).
    pub warm: Vec<Option<Params>>,
    /// Thickness of the previous fit.
    pub prev_t: Vec<Option<f64>>,
    /// Consecutive fits the node has been still.
    pub still: Vec<u32>,
    pub frozen: Vec<bool>,
    /// Movement next to a frozen node since it froze.
    pub near_motion: Vec<f64>,
    /// Held fixed in the current fit (a frozen node next to moving ones).
    pub held: Vec<bool>,
}

impl NodeState {
    pub fn new(n: usize) -> Self {
        NodeState {
            warm: vec![None; n],
            prev_t: vec![None; n],
            still: vec![0; n],
            frozen: vec![false; n],
            near_motion: vec![0.0; n],
            held: vec![false; n],
        }
    }

    /// The state on a new mesh: vertex `i` takes the state of old vertex
    /// `map[i].0`, with its warm `m` shifted by `map[i].1` (how far further
    /// inside the new vertex lies).
    pub fn remap(&self, map: &[(usize, f64)]) -> Self {
        NodeState {
            warm: map
                .iter()
                .map(|&(j, off)| {
                    self.warm[j].map(|mut w| {
                        w[M] -= off;
                        w
                    })
                })
                .collect(),
            prev_t: map.iter().map(|&(j, _)| self.prev_t[j]).collect(),
            still: map.iter().map(|&(j, _)| self.still[j]).collect(),
            frozen: map.iter().map(|&(j, _)| self.frozen[j]).collect(),
            near_motion: map.iter().map(|&(j, _)| self.near_motion[j]).collect(),
            held: vec![false; map.len()],
        }
    }
}

/// Regional convergence, resolved from [`FreezeOptions`].
pub(super) struct Freeze {
    patience: u32,
    /// A node is still if it moved less than this …
    pub dtol: f64,
    /// … and its thickness changed by less than this.
    ttol: f64,
    radius: f64,
    halo: f64,
    thaw: f64,
}

impl Freeze {
    pub fn new(f: &FreezeOptions, options: &Options, tol: f64, max_spacing: f64) -> Self {
        let radius = f.radius.unwrap_or(3.0 * max_spacing);
        let halo = f.halo.unwrap_or(match &options.method {
            Method::Regularized {
                sigma, cutoff, lambda, ..
            } if *lambda > 0.0 => sigma * if cutoff.is_finite() { *cutoff } else { 3.0 },
            _ => radius,
        });
        Freeze {
            patience: f.patience.max(1) as u32,
            dtol: f.displacement.unwrap_or(tol),
            ttol: f.thickness.unwrap_or(tol),
            radius,
            halo,
            thaw: f.thaw.unwrap_or(max_spacing),
        }
    }

    /// After a full fit (which covers frozen regions too): thaw the frozen
    /// nodes it moves.
    pub fn thaw_moved(&self, st: &mut NodeState, step: &Step) {
        for v in 0..st.frozen.len() {
            if st.frozen[v] && step.known[v] && norm(step.disp[v]) >= self.dtol {
                st.frozen[v] = false;
                st.still[v] = 0;
            }
        }
    }

    /// Update after a fit's displacements: thaw frozen nodes whose
    /// surroundings have moved, count still nodes (a node whose fit failed is
    /// never still), and freeze nodes whose whole neighbourhood is still.
    pub fn update(&self, st: &mut NodeState, mesh: &Mesh, step: &Step) {
        if st.frozen.iter().any(|&f| f) {
            self.thaw_near_motion(st, mesh, step);
        }
        self.count_still(st, step);
        self.freeze_still_regions(st, mesh);
    }

    /// Add the largest displacement of the moving nodes within `radius` of
    /// each frozen node to its accumulated nearby motion; thaw it beyond `thaw`.
    fn thaw_near_motion(&self, st: &mut NodeState, mesh: &Mesh, step: &Step) {
        let nv = st.frozen.len();
        let movers: Vec<usize> = (0..nv).filter(|&v| !st.frozen[v]).collect();
        let pts: Vec<V3> = movers.iter().map(|&v| mesh.vertices[v]).collect();
        let tree = KdTree::new(&pts);
        let nearby_max: Vec<f64> = (0..nv)
            .into_par_iter()
            .map(|v| {
                if !st.frozen[v] || pts.is_empty() {
                    return 0.0;
                }
                let mut near = Vec::new();
                tree.within(mesh.vertices[v], self.radius, &mut near);
                near.iter()
                    .map(|&(u, _)| norm(step.disp[movers[u as usize]]))
                    .fold(0.0, f64::max)
            })
            .collect();
        for v in 0..nv {
            if st.frozen[v] {
                st.near_motion[v] += nearby_max[v];
                if st.near_motion[v] > self.thaw {
                    st.frozen[v] = false;
                    st.still[v] = 0;
                    st.near_motion[v] = 0.0;
                }
            }
        }
    }

    /// Count the fits in a row each moving node stayed still: moved less than
    /// `dtol` and (if fitted and free) changed its thickness by less than `ttol`.
    fn count_still(&self, st: &mut NodeState, step: &Step) {
        for v in 0..st.frozen.len() {
            if st.frozen[v] {
                continue;
            }
            if step.failed[v] {
                st.still[v] = 0;
                continue;
            }
            let thick_ok = if step.fitted[v] && !st.held[v] {
                step.t_change[v].is_some_and(|d| d < self.ttol)
            } else {
                true
            };
            st.still[v] = if thick_ok && norm(step.disp[v]) < self.dtol {
                st.still[v] + 1
            } else {
                0
            };
        }
    }

    /// Freeze the nodes still for `patience` fits whose whole neighbourhood
    /// (within `radius`) is still or frozen.
    fn freeze_still_regions(&self, st: &mut NodeState, mesh: &Mesh) {
        let nv = st.frozen.len();
        let eligible: Vec<bool> = (0..nv).map(|v| st.frozen[v] || st.still[v] >= self.patience).collect();
        let tree = KdTree::new(&mesh.vertices);
        let newly: Vec<bool> = (0..nv)
            .into_par_iter()
            .map(|v| {
                if st.frozen[v] || !eligible[v] {
                    return false;
                }
                let mut near = Vec::new();
                tree.within(mesh.vertices[v], self.radius, &mut near);
                near.iter().all(|&(u, _)| eligible[u as usize])
            })
            .collect();
        for v in 0..nv {
            st.frozen[v] |= newly[v];
        }
    }

    /// The frozen nodes refitted as fixed values: those (with a warm start)
    /// within the halo of a moving node.
    pub fn held_nodes(&self, st: &NodeState, mesh: &Mesh, keep: &[bool]) -> Vec<bool> {
        let nv = keep.len();
        let active: Vec<V3> = (0..nv)
            .filter(|&v| keep[v] && !st.frozen[v])
            .map(|v| mesh.vertices[v])
            .collect();
        let tree = KdTree::new(&active);
        (0..nv)
            .into_par_iter()
            .map(|v| {
                keep[v]
                    && st.frozen[v]
                    && st.warm[v].is_some()
                    && tree.knn(mesh.vertices[v], 1, None)[0].1.sqrt() <= self.halo
            })
            .collect()
    }
}
