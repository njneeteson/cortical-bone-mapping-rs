//! The regularised method (Neeteson & Boyd 2025): field parameters per node
//! with the spatial smoothing penalty `½ κ_c ‖ω v_c‖²`, optionally plus the
//! curvature penalty on the fitted surfaces (extension).

use std::time::Instant;

use super::global::{self, Problem, Role, Smoothing, State};
use super::{base_roles, local, local_settings, Data, FitConstraints, Nodes, Start};
use crate::model::{Params, M, NPARAM, RHO_B, RHO_S, SIGMA, T};
use crate::operators::{CurvatureOperator, SmoothingOperator, SurfaceLaplacian};
use crate::options::Method;
use crate::result::FitInfo;

/// Fit with [`Method::Regularized`]; returns the parameters per node.
pub(crate) fn fit(
    data: &Data,
    nodes: &Nodes,
    start: &Start,
    constraints: &FitConstraints,
    info: &mut FitInfo,
    timings: &mut Vec<(String, f64)>,
) -> Vec<Params> {
    let Method::Regularized {
        lambda,
        sigma,
        neighbours,
        cutoff,
        init_from_local,
        curvature,
    } = &data.options.method
    else {
        unreachable!("regularized::fit with another method")
    };
    let o = data.options;
    let (rho_c, points) = (data.rho_c, &nodes.mesh.vertices);
    let n = points.len();
    let free_sigma = start.sigma_fixed.is_none();
    let mut first: Vec<Params> = if *init_from_local {
        local::fit_all(
            data.profiles,
            data.x,
            data.weights,
            rho_c,
            &start.nodes,
            &local_settings(o, free_sigma),
        )
        .iter()
        .map(|f| f.params)
        .collect()
    } else {
        start.nodes.clone()
    };
    if let Some(held) = start.held {
        for (k, &h) in held.iter().enumerate() {
            if h {
                first[k] = start.nodes[k];
            }
        }
    }

    // roles: σ fixed by a PSF, or held densities / σ during partial refits
    let mut roles = base_roles(o);
    if constraints.densities.is_some() {
        for c in [RHO_S, RHO_B] {
            if roles[c] == Role::Global {
                roles[c] = Role::Fixed;
            }
        }
    }
    if constraints.sigma.is_some() && roles[SIGMA] == Role::Global {
        roles[SIGMA] = Role::Fixed;
    }
    let mut fixed: [Option<&[f64]>; NPARAM] = [None; NPARAM];
    if let Some(sf) = start.sigma_fixed {
        roles[SIGMA] = Role::Fixed;
        fixed[SIGMA] = Some(sf);
    }

    // penalty weights, scaled by the global model
    let t0g = start.global[T].max(1e-6);
    let s0g = start.global[SIGMA].max(1e-6);
    let base = lambda * rho_c * rho_c / n as f64;
    let t_op = Instant::now();
    let op = (*lambda > 0.0).then(|| SmoothingOperator::new(points, *neighbours, *sigma, *cutoff));
    timings.push(("smoothing operator".to_string(), t_op.elapsed().as_secs_f64()));
    let mut coef = [0.0; NPARAM];
    coef[M] = base / (t0g * t0g);
    coef[T] = base / (t0g * t0g);
    if free_sigma && roles[SIGMA] == Role::Field {
        coef[SIGMA] = base / (s0g * s0g);
    }
    // density variation relative to the cortical density
    let cd = o.model.density_smoothing * base / (rho_c * rho_c);
    coef[RHO_S] = cd;
    coef[RHO_B] = cd;
    let curv = curvature.as_ref().map(|c| {
        let ell = c.length.unwrap_or(t0g);
        let cc = (c.weight / n as f64).sqrt() * rho_c * ell;
        let lap = match c.scale {
            Some(sc) => SurfaceLaplacian::gaussian(points, c.neighbours, sc, 3.0),
            None => SurfaceLaplacian::cotangent(nodes.mesh),
        };
        CurvatureOperator::new(lap, points, nodes.normals, c.mode, cc, [c.periosteal, c.endosteal])
    });

    let fields: Vec<usize> = (0..NPARAM).filter(|&c| roles[c] == Role::Field).collect();
    let pr = Problem {
        profiles: data.profiles,
        x: data.x,
        weights: data.weights,
        rho_c,
        roles,
        fixed_per_node: fixed,
        interp: None,
        frozen: start.held,
        smoothing: op.as_ref().map(|op| Smoothing { op, coef }),
        curvature: curv.as_ref(),
        lower: o.model.lower,
        upper: o.model.upper,
    };
    let mut scalars = start.global;
    if let Some([rs, rb]) = constraints.density_start {
        if constraints.densities.is_none() {
            scalars[RHO_S] = rs;
            scalars[RHO_B] = rb;
        }
    }
    if let Some(sg) = constraints.sigma.or(constraints.sigma_start) {
        if start.sigma_fixed.is_none() {
            scalars[SIGMA] = sg;
        }
    }
    if *init_from_local && constraints.densities.is_none() && constraints.density_start.is_none() {
        scalars[RHO_S] = first.iter().map(|p| p[RHO_S]).sum::<f64>() / n as f64;
        scalars[RHO_B] = first.iter().map(|p| p[RHO_B]).sum::<f64>() / n as f64;
    }
    let mut st = State {
        field: fields.iter().map(|&c| first.iter().map(|p| p[c]).collect()).collect(),
        scalars,
    };
    let si = global::solve(&pr, &mut st, &data.solver);
    info.add_solve(&si);
    pr.node_params(&st)
}
