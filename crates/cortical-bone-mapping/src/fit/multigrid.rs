//! The multigrid method of our paper (Neeteson & Boyd 2025).
//!
//! The field parameters (`m`, `t`, `σ` by default) live on `Q ≪ N` control
//! points and are interpolated to the `N` nodes by a sparse matrix `A`
//! (`N × Q`), `v_nodes = A v_ctrl`; the other parameters are global. For each
//! control-point spacing `s`, coarsest first:
//!
//! 1. select control points with minimum separation `s`
//!    ([`crate::operators::select_control_points`]);
//! 2. build `A`: each node averages its `k` nearest control points with
//!    normalised Gaussian weights of standard deviation `s`
//!    ([`crate::operators::interpolation_matrix`]);
//! 3. solve the global least-squares problem ([`super::global`]) for the
//!    control-point values, starting from the node values of the previous level
//!    (or the global fit) at the control points, and the globals of the
//!    previous level.
//!
//! The interpolation is the regulariser: the fitted fields can only vary on
//! the scale of `s`. The coarse levels, with few unknowns and many profiles
//! per unknown, give the finer ones a start close to their solution, so each
//! level's non-linear solve needs only a few iterations.
//!
//! # References
//!
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.

use super::global::{self, Problem, Role, State};
use super::{base_roles, Data, Nodes, Start};
use crate::model::{Params, NPARAM, SIGMA};
use crate::operators::{interpolation_matrix, select_control_points, Neighbours};
use crate::options::Method;
use crate::result::FitInfo;

/// Fit with [`Method::Multigrid`]; returns the parameters per node.
pub(crate) fn fit(data: &Data, nodes: &Nodes, start: &Start, info: &mut FitInfo) -> Vec<Params> {
    let Method::Multigrid {
        spacings,
        neighbours,
        seed,
    } = &data.options.method
    else {
        unreachable!("multigrid::fit with another method")
    };
    let o = data.options;
    let mut node_params = start.nodes.clone();
    let mut roles = base_roles(o);
    let mut fixed: [Option<&[f64]>; NPARAM] = [None; NPARAM];
    if let Some(sf) = start.sigma_fixed {
        roles[SIGMA] = Role::Fixed;
        fixed[SIGMA] = Some(sf);
    }
    let fields: Vec<usize> = (0..NPARAM).filter(|&c| roles[c] == Role::Field).collect();
    let mut scalars = start.global;
    let near = Neighbours::new(nodes.mesh, o.distance());
    for (lvl, &s) in spacings.iter().enumerate() {
        let level = crate::stage!("multigrid level");
        let ctrl = select_control_points(&near, s, seed.wrapping_add(lvl as u64));
        let a = interpolation_matrix(&near, &ctrl, *neighbours, s);
        log::debug!(
            "multigrid level {}/{}: spacing {s:.3}, {} control points for {} nodes",
            lvl + 1,
            spacings.len(),
            ctrl.len(),
            node_params.len()
        );
        let pr = Problem {
            profiles: data.profiles,
            x: data.x,
            weights: data.weights,
            rho_c: data.rho_c,
            roles,
            fixed_per_node: fixed,
            interp: Some(&a),
            frozen: None,
            smoothing: None,
            curvature: None,
            lower: o.model.lower,
            upper: o.model.upper,
        };
        let mut st = State {
            field: fields
                .iter()
                .map(|&c| ctrl.iter().map(|&i| node_params[i as usize][c]).collect())
                .collect(),
            scalars,
        };
        let si = global::solve(&pr, &mut st, &data.solver);
        info.add_solve(&si);
        node_params = pr.node_params(&st);
        scalars = st.scalars;
        crate::done!(level, "level {} of {}", lvl + 1, spacings.len());
    }
    node_params
}
