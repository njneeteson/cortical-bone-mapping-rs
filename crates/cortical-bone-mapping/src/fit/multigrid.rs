//! The multigrid method (Neeteson & Boyd 2025): `m`, `t`, `σ` (and any
//! other field parameters) on sparse control points interpolated to the
//! nodes, solved from the coarsest control-point spacing to the finest.

use super::global::{self, Problem, Role, State};
use super::{base_roles, Data, Nodes, Start};
use crate::model::{Params, NPARAM, SIGMA};
use crate::operators::{interpolation_matrix, select_control_points};
use crate::options::Method;
use crate::result::FitInfo;
use crate::util::V3;

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
    let points = &nodes.mesh.vertices;
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
    for (lvl, &s) in spacings.iter().enumerate() {
        let ctrl = select_control_points(points, s, seed.wrapping_add(lvl as u64));
        let cpts: Vec<V3> = ctrl.iter().map(|&i| points[i as usize]).collect();
        let a = interpolation_matrix(points, &cpts, *neighbours, s);
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
    }
    node_params
}
