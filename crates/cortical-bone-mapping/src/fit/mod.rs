//! Fitting the profile model to the profiles of a surface.
//!
//! [`fit_surface`] runs the stages in order:
//!
//! 1. sample a profile along each node's inward normal ([`profiles`]);
//! 2. set the cortical density ([`density`]);
//! 3. set the blur: free, fixed, from a PSF or a PSF estimated from the
//!    profiles ([`psf`]);
//! 4. fit one model to all profiles (the global initialisation);
//! 5. fit with the chosen method: [`local`] (one fit per node, Treece et
//!    al.), [`multigrid`] or [`regularized`] (Neeteson & Boyd), the latter two
//!    on the [`global`] solver;
//! 6. compute the residuals.

pub mod density;
pub mod global;
pub mod local;
pub mod multigrid;
pub mod profiles;
pub mod psf;
pub mod regularized;

use std::time::Instant;

use ndarray::ArrayView3;

use crate::error::Error;
use crate::model::{eval, residual_weights, Params, NPARAM, RHO_B, RHO_S, SIGMA};
use crate::options::{Method, Options, SigmaMode};
use crate::result::{CbmResult, FitInfo};
use crate::surface::mesh::Mesh;
use crate::util::V3;
use crate::volume::Voxel;
use local::FitSettings;
use profiles::{sampling_normals, Profiles, SampleLine};

/// Constraints for a partial refit of a surface (used by refinement).
#[derive(Clone, Copy, Debug, Default)]
pub struct FitConstraints<'a> {
    /// Voxels whose profile samples are ignored (weight zero), e.g. the
    /// regions of other bones.
    pub exclude: Option<ArrayView3<'a, bool>>,
    /// Per surface vertex: hold the node at its warm start (which must be
    /// given). With the regularised method such nodes are boundary
    /// conditions for their free neighbours; other methods simply keep the
    /// warm-start values.
    pub fixed: Option<&'a [bool]>,
    /// Cortical density to use instead of estimating it (modes `Fixed` and
    /// `Estimate`).
    pub rho_c: Option<f64>,
    /// Hold `(ρs, ρb)` at these values (regularised method).
    pub densities: Option<[f64; 2]>,
    /// Start `(ρs, ρb)` from these values instead of the global fit
    /// (regularised method; a warm start).
    pub density_start: Option<[f64; 2]>,
    /// Stop the global solve once its steps change the fitted periosteal
    /// edges by less than this (RMS, physical units; 0 = solve fully).
    pub edge_tol: f64,
    /// Hold a global `σ` at this value.
    pub sigma: Option<f64>,
    /// Start a global `σ` from this value (a warm start).
    pub sigma_start: Option<f64>,
}

/// The sampled data every method fits.
#[derive(Clone, Copy)]
pub(crate) struct Data<'a> {
    pub profiles: &'a Profiles,
    /// Sample positions along the profiles.
    pub x: &'a [f64],
    /// Residual weights per sample position.
    pub weights: &'a [f64],
    pub rho_c: f64,
    pub options: &'a Options,
    /// Settings of the method's global solves.
    pub solver: global::Settings,
}

/// The fitted nodes: the sub-mesh of the surface, and inward sampling
/// directions.
pub(crate) struct Nodes<'a> {
    pub mesh: &'a Mesh,
    pub normals: &'a [V3],
}

/// Where a method starts.
pub(crate) struct Start<'a> {
    /// The single model fitted to all profiles.
    pub global: Params,
    /// Per node: its warm start, else `global`, with a fixed σ inserted.
    pub nodes: Vec<Params>,
    /// Per-node σ when the blur is fixed.
    pub sigma_fixed: Option<&'a [f64]>,
    /// Nodes held at their start (boundary conditions).
    pub held: Option<&'a [bool]>,
}

/// Roles of `[m, t, σ, ρs, ρb]` in the global methods: a single global value
/// or a field ([`ModelOptions::global`](crate::options::ModelOptions::global)).
pub(crate) fn base_roles(o: &Options) -> [global::Role; NPARAM] {
    o.model
        .global
        .map(|g| if g { global::Role::Global } else { global::Role::Field })
}

/// Settings of the per-node Levenberg–Marquardt fits.
pub(crate) fn local_settings(o: &Options, free_sigma: bool) -> FitSettings {
    let mut free = [true; NPARAM];
    free[SIGMA] = free_sigma;
    FitSettings {
        lower: o.model.lower,
        upper: o.model.upper,
        free,
        max_iterations: o.solver.max_iterations,
        ftol: o.solver.ftol,
        gtol: o.solver.gtol,
        xtol: o.solver.xtol,
    }
}

/// Settings of the global solves.
pub(crate) fn global_settings(o: &Options) -> global::Settings {
    global::Settings {
        max_iterations: o.solver.max_iterations,
        ftol: o.solver.ftol,
        gtol: o.solver.gtol,
        xtol: o.solver.xtol,
        cg_max_iterations: o.solver.cg_max_iterations,
        cg_tolerance: o.solver.cg_tolerance,
        cg_max_total: o.solver.cg_max_total,
        progress_every: o.solver.progress_every,
        edge_tol: 0.0,
    }
}

/// Fit the model on the nodes `keep` of an (already smoothed) `surface`.
/// `warm_start`, if given, holds initial parameters per surface vertex
/// (`None` entries fall back to the global fit).
pub fn fit_surface<T: Voxel>(
    image: &ArrayView3<T>,
    surface: Mesh,
    keep: &[bool],
    options: &Options,
    warm_start: Option<&[Option<Params>]>,
) -> Result<CbmResult, Error> {
    fit_surface_constrained(image, surface, keep, options, warm_start, &FitConstraints::default())
}

/// [`fit_surface`] with [`FitConstraints`].
pub fn fit_surface_constrained<T: Voxel>(
    image: &ArrayView3<T>,
    surface: Mesh,
    keep: &[bool],
    options: &Options,
    warm_start: Option<&[Option<Params>]>,
    constraints: &FitConstraints,
) -> Result<CbmResult, Error> {
    let o = options;
    let mut timings = Vec::new();
    let t0 = Instant::now();

    // ---- the nodes and their profiles
    let outward = surface.vertex_normals();
    let dirs = sampling_normals(&outward, o.sampling.normal_constraint);
    let keep: Vec<bool> = keep.iter().zip(&dirs).map(|(&k, d)| k && d.is_some()).collect();
    let (sub, vertex_ids) = surface.submesh(&keep);
    if vertex_ids.is_empty() {
        return Err(Error::NoNodes);
    }
    let normals: Vec<V3> = vertex_ids.iter().map(|&v| dirs[v as usize].unwrap()).collect();
    let points = sub.vertices.clone();
    let n = points.len();
    let resolution = o.sampling.resolution.unwrap_or_else(|| o.geometry.min_spacing());
    let line = SampleLine::new(o.sampling.outside, o.sampling.inside, resolution);
    let mut profiles = Profiles::sample(image, &o.geometry, &points, &normals, &line, o.sampling.smoothing);
    if let Some(ex) = constraints.exclude {
        profiles.exclude(&ex, &o.geometry, &points, &normals, &line);
    }
    if o.sampling.half_width_inside {
        let region = surface.voxelize(image.dim(), &o.geometry);
        let min_inside = o.geometry.min_spacing();
        profiles.limit_inside(&region.view(), &o.geometry, &points, &normals, &line, min_inside);
    }
    let x = line.x.clone();
    let weights = residual_weights(&x, o.model.residual_boost);

    // ---- cortical density
    let (rho_c, rho_c_node) = density::cortical_density(&mut profiles, &x, &weights, constraints.rho_c, o)?;
    timings.push(("sampling".to_string(), t0.elapsed().as_secs_f64()));
    let data = Data {
        profiles: &profiles,
        x: &x,
        weights: &weights,
        rho_c,
        options: o,
        solver: global::Settings {
            edge_tol: constraints.edge_tol,
            ..global_settings(o)
        },
    };
    let mut info = FitInfo {
        method: o.method.name().to_string(),
        ..Default::default()
    };

    // ---- the blur
    let mut psf: Option<[f64; 3]> = None;
    let mut sigma_fixed: Option<Vec<f64>> = match &o.model.sigma {
        SigmaMode::Free | SigmaMode::EstimatePsf { .. } => None,
        SigmaMode::Fixed(s) => Some(vec![*s; n]),
        SigmaMode::Psf(p) => {
            psf = Some(*p);
            Some(normals.iter().map(|&nr| psf::sigma_along(*p, nr)).collect())
        }
    };
    let mut init = o.model.init;
    if let Some(sf) = &sigma_fixed {
        init[SIGMA] = sf.iter().sum::<f64>() / n as f64;
    }

    // ---- global initialisation: one model for all profiles
    let t1 = Instant::now();
    let mut global_params = if o.init_from_global {
        density::fit_one_model(&data, init, sigma_fixed.as_deref())
    } else {
        init
    };
    if let Some([rs, rb]) = constraints.densities {
        global_params[RHO_S] = rs;
        global_params[RHO_B] = rb;
    }
    // fixed nodes (subset indices) must have a warm start to be held at
    let fixed_sub: Option<Vec<bool>> = constraints
        .fixed
        .map(|f| vertex_ids.iter().map(|&v| f[v as usize]).collect());
    if let Some(fs) = &fixed_sub {
        let has_warm = |k: usize| warm_start.is_some_and(|w| w[vertex_ids[k] as usize].is_some());
        if fs.iter().enumerate().any(|(k, &f)| f && !has_warm(k)) {
            return Err(Error::InvalidOption("fixed nodes need a warm start".into()));
        }
    }
    timings.push(("global fit".to_string(), t1.elapsed().as_secs_f64()));
    let starts =
        |global: &Params, sigma_fixed: Option<&[f64]>| node_starts(global, warm_start, &vertex_ids, sigma_fixed);

    // ---- PSF estimation (Treece-style fixed blur)
    if let SigmaMode::EstimatePsf { model, min_ratio } = &o.model.sigma {
        let t2 = Instant::now();
        let est = psf::estimate_from_fits(&data, &normals, &starts(&global_params, None), *model, *min_ratio)?;
        psf = Some(est);
        sigma_fixed = Some(normals.iter().map(|&nr| psf::sigma_along(est, nr)).collect());
        if o.init_from_global {
            global_params = density::fit_one_model(&data, global_params, sigma_fixed.as_deref());
        }
        timings.push(("psf estimation".to_string(), t2.elapsed().as_secs_f64()));
    }

    // ---- the chosen method
    let t3 = Instant::now();
    let start = Start {
        global: global_params,
        nodes: starts(&global_params, sigma_fixed.as_deref()),
        sigma_fixed: sigma_fixed.as_deref(),
        held: fixed_sub.as_deref(),
    };
    let nodes = Nodes {
        mesh: &sub,
        normals: &normals,
    };
    let mut params: Vec<Params> = match &o.method {
        Method::Local => local::fit_nodes(&data, &start, &mut info),
        Method::Multigrid { .. } => multigrid::fit(&data, &nodes, &start, &mut info),
        Method::Regularized { .. } => regularized::fit(&data, &nodes, &start, constraints, &mut info, &mut timings),
    };
    if let Some(held) = start.held {
        for (k, &h) in held.iter().enumerate() {
            if h {
                params[k] = start.nodes[k];
            }
        }
    }
    timings.push((format!("{} fit", o.method.name()), t3.elapsed().as_secs_f64()));

    // ---- residuals, and per-node densities scaled back
    let mut residual = residuals(&data, &params);
    if let Some(node) = &rho_c_node {
        let m = profiles.m;
        for (i, &rn) in node.iter().enumerate() {
            let s = rn / rho_c;
            params[i][RHO_S] *= s;
            params[i][RHO_B] *= s;
            residual[i] *= s;
            if o.keep_profiles {
                profiles.values[i * m..(i + 1) * m]
                    .iter_mut()
                    .for_each(|v| *v *= s as f32);
            }
        }
    }
    timings.append(&mut info.timings);
    info.timings = timings;
    Ok(CbmResult {
        surface,
        vertex_ids,
        points,
        faces: sub.faces,
        normals,
        params,
        residual,
        rho_c,
        rho_c_node,
        global_params,
        psf,
        info,
        profiles: o.keep_profiles.then_some((x, profiles)),
    })
}

/// Per fitted node: its warm start (by surface vertex), else `global`, with
/// a fixed σ inserted.
fn node_starts(
    global: &Params,
    warm_start: Option<&[Option<Params>]>,
    vertex_ids: &[u32],
    sigma_fixed: Option<&[f64]>,
) -> Vec<Params> {
    (0..vertex_ids.len())
        .map(|k| {
            let mut p = warm_start.and_then(|w| w[vertex_ids[k] as usize]).unwrap_or(*global);
            if let Some(sf) = sigma_fixed {
                p[SIGMA] = sf[k];
            }
            p
        })
        .collect()
}

/// Weighted RMS residual per node (over the samples used).
fn residuals(data: &Data, params: &[Params]) -> Vec<f64> {
    let (x, profiles) = (data.x, data.profiles);
    let wsum: f64 = data.weights.iter().sum();
    params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let y = profiles.row(i);
            let wr = profiles.row_weights(i, data.weights);
            let s: f64 = x
                .iter()
                .zip(y)
                .zip(wr.iter())
                .map(|((&xj, &yj), &w)| w * (eval(xj, p, data.rho_c) - yj as f64).powi(2))
                .sum();
            let ws: f64 = wr.iter().sum();
            (s / if ws > 0.0 { ws } else { wsum }).sqrt()
        })
        .collect()
}
