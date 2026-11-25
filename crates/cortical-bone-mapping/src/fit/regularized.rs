//! The regularised method of our paper (Neeteson & Boyd 2025), optionally
//! with the curvature penalty on the fitted surfaces (an extension in this
//! crate).
//!
//! Every field parameter has one value per node, and neighbouring values are
//! tied by the spatial smoothing penalty
//!
//! ```text
//! ½ Σ_c κ_c ‖ω v_c‖²,    κ_m = κ_t = λ ρc² / (N t0²),   κ_σ = λ ρc² / (N σ0²)
//! ```
//!
//! added to the data term `1/(2NM) Σ_ij γ_j r_ij²`, with `ω` the smoothing
//! operator ([`crate::operators::SmoothingOperator`]: each value minus the
//! Gaussian-weighted mean of its neighbours) and `t0`, `σ0` the thickness and
//! blur of the global fit. The scaling makes the penalty commensurate with the
//! data term: a residual is a density (scale `ρc`), and `ω m / t0`, `ω σ / σ0`
//! are dimensionless, so `λ` is a pure number whatever the units and the
//! number of nodes. The original Python implementation in Bonelab applied this scaling
//! inconsistently between loss and gradient; here it is the same in both.
//!
//! Two additions of ours:
//!
//! * **density fields**: when `ρs`, `ρb` are fields instead of globals
//!   ([`crate::options::ModelOptions::global`]) they are smoothed with
//!   `κ_ρ = density_smoothing · λ ρc² / (N ρc²)`, i.e. their variation is
//!   measured relative to the cortical density;
//! * **curvature penalty** ([`crate::operators::CurvatureOperator`]): residual
//!   scale `c = √(μ/N) ρc ℓ` (`ℓ` the curvature length, default `t0`), which
//!   makes curvature dimensionless so that `μ` plays the same role as `λ`.
//!
//! The fields start from local fits (`init_from_local`) or from the node
//! starts; the globals from the global fit, or the mean of the local fits.
//! Held nodes (refinement) stay at their starts and act as boundary conditions
//! of the penalties.
//!
//! # References
//!
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.

use std::time::Instant;

use super::global::{self, Problem, Role, Smoothing, State};
use super::{base_roles, local, local_settings, Data, FitConstraints, Nodes, Start};
use crate::model::{Params, M, NPARAM, RHO_B, RHO_S, SIGMA, T};
use crate::operators::{CurvatureMode, CurvatureOperator, Neighbours, SmoothingOperator, SurfaceLaplacian};
use crate::options::{CurvatureOptions, Method};
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
    let (o, rho_c, points) = (data.options, data.rho_c, &nodes.mesh.vertices);
    let first = first_estimate(data, start, *init_from_local);
    let (roles, fixed) = roles(data, start, constraints);

    // penalties, weighted relative to the global model
    let base = lambda * rho_c * rho_c / points.len() as f64;
    let t_op = Instant::now();
    let near = Neighbours::new(nodes.mesh, o.distance());
    let op = (*lambda > 0.0).then(|| SmoothingOperator::new(&near, *neighbours, *sigma, *cutoff));
    timings.push(("smoothing operator".to_string(), t_op.elapsed().as_secs_f64()));
    let coef = smoothing_weights(data, start, &roles, base);
    let curv = curvature
        .as_ref()
        .map(|c| curvature_operator(c, nodes, &near, rho_c, start.global[T].max(1e-6)));

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
    let fields: Vec<usize> = (0..NPARAM).filter(|&c| roles[c] == Role::Field).collect();
    let mut st = State {
        field: fields.iter().map(|&c| first.iter().map(|p| p[c]).collect()).collect(),
        scalars: initial_scalars(start, constraints, &first, *init_from_local),
    };
    let si = global::solve(&pr, &mut st, &data.solver);
    info.add_solve(&si);
    pr.node_params(&st)
}

/// Where the fields start: local fits from the node starts, or the starts
/// themselves; held nodes always at their starts.
fn first_estimate(data: &Data, start: &Start, init_from_local: bool) -> Vec<Params> {
    let mut first: Vec<Params> = if init_from_local {
        let settings = local_settings(data.options, start.sigma_fixed.is_none());
        local::fit_all(data.profiles, data.x, data.weights, data.rho_c, &start.nodes, &settings)
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
    first
}

/// Roles of the parameters, and per-node values of the fixed ones: σ fixed
/// by a PSF, or held densities / σ during partial refits.
fn roles<'a>(
    data: &Data,
    start: &Start<'a>,
    constraints: &FitConstraints,
) -> ([Role; NPARAM], [Option<&'a [f64]>; NPARAM]) {
    let mut roles = base_roles(data.options);
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
    (roles, fixed)
}

/// Smoothing weights `κ_c`: `base` over the squared scale of each parameter
/// (global thickness and blur; densities relative to the cortical density).
fn smoothing_weights(data: &Data, start: &Start, roles: &[Role; NPARAM], base: f64) -> Params {
    let t0g = start.global[T].max(1e-6);
    let s0g = start.global[SIGMA].max(1e-6);
    let mut coef = [0.0; NPARAM];
    coef[M] = base / (t0g * t0g);
    coef[T] = base / (t0g * t0g);
    if start.sigma_fixed.is_none() && roles[SIGMA] == Role::Field {
        coef[SIGMA] = base / (s0g * s0g);
    }
    let cd = data.options.model.density_smoothing * base / (data.rho_c * data.rho_c);
    coef[RHO_S] = cd;
    coef[RHO_B] = cd;
    coef
}

/// The curvature penalty on the fitted surfaces; lengths default to the
/// global thickness `t0`.
pub(crate) fn curvature_operator(
    c: &CurvatureOptions,
    nodes: &Nodes,
    near: &Neighbours,
    rho_c: f64,
    t0: f64,
) -> CurvatureOperator {
    let points = &nodes.mesh.vertices;
    let ell = c.length.unwrap_or(t0);
    // the residual is dimensionless: curvature times ℓ, its gradient times ℓ²
    let lengths = if c.mode == CurvatureMode::Variation {
        ell * ell
    } else {
        ell
    };
    let scale = |w: f64, on: bool| {
        if on {
            (w / points.len() as f64).sqrt() * rho_c * lengths
        } else {
            0.0
        }
    };
    let coef = [
        scale(c.weight, c.periosteal),
        scale(c.endosteal_weight.unwrap_or(c.weight), c.endosteal),
    ];
    let lap = match c.scale.sigma(nodes.mesh.mean_edge_length()) {
        Some(sc) => SurfaceLaplacian::gaussian(near, c.neighbours, sc, 3.0),
        None => SurfaceLaplacian::cotangent(nodes.mesh),
    };
    CurvatureOperator::new(lap, points, nodes.normals, c.mode, c.measure, coef)
}

/// Starting values of the global and fixed parameters: the global model,
/// with warm-started (or held) densities and σ; densities from the mean of
/// the local fits when those start the fields.
fn initial_scalars(start: &Start, constraints: &FitConstraints, first: &[Params], init_from_local: bool) -> Params {
    let n = first.len() as f64;
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
    if init_from_local && constraints.densities.is_none() && constraints.density_start.is_none() {
        scalars[RHO_S] = first.iter().map(|p| p[RHO_S]).sum::<f64>() / n;
        scalars[RHO_B] = first.iter().map(|p| p[RHO_B]).sum::<f64>() / n;
    }
    scalars
}
