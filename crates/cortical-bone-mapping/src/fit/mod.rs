//! Fitting the profile model to the profiles of a surface.
//!
//! [`fit_surface`] runs the stages in order:
//!
//! 1. sample a profile along each node's inward normal ([`profiles`]),
//!    following Treece et al. (2010);
//! 2. set the cortical density ([`density`]);
//! 3. set the blur: free, fixed, from a PSF or a PSF estimated from the
//!    profiles ([`psf`]);
//! 4. fit one model to all profiles (the global initialisation of our paper,
//!    Neeteson & Boyd 2025): every parameter takes one value shared by all
//!    nodes, which gives each method a robust start where individual profiles
//!    are noisy;
//! 5. fit with the chosen method: [`local`] (one fit per node, Treece et
//!    al. 2010), [`multigrid`] or [`regularized`] (Neeteson & Boyd 2025), the
//!    latter two on the [`global`] solver;
//! 6. compute the residuals, `r_i = √(Σ_j w_ij (f(x_j) − f_ij)² / Σ_j w_ij)`
//!    (the weighted RMS over the samples used), and scale per-node densities
//!    back ([`density`]).
//!
//! All methods minimise the same weighted least-squares data term (with the
//! residual weights of [`crate::model::residual_weights`]), so results differ
//! only by what links the nodes.
//!
//! # References
//!
//! * G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
//!   cortical bone thickness measurement from clinical CT data*, Medical Image
//!   Analysis 14(3):276–290, 2010, doi:10.1016/j.media.2010.01.003.
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.

pub mod density;
pub mod global;
pub mod local;
pub mod multigrid;
pub mod profiles;
pub mod psf;
pub mod regularized;
pub mod volumetric;

use std::time::Instant;

use ndarray::ArrayView3;

use crate::error::Error;
use crate::model::{eval, residual_weights, Params, NPARAM, RHO_B, RHO_S, SIGMA};
use crate::options::{Method, Options, RhoCMode, SigmaMode};
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
    /// Hold the volumetric model's PSF `(σ_xy, σ_z)` at these values.
    pub psf: Option<[f64; 2]>,
    /// Start the volumetric model's PSF from these values (a warm start).
    pub psf_start: Option<[f64; 2]>,
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

/// The estimated cortical density ([`RhoCMode::Estimate`]) from the profiles
/// of the nodes `keep`, for the volumetric model: the 1D estimate looks for
/// cortex thick enough to show its density along every profile, where the
/// volumetric fit's own voxels were dominated by thin cortex on clinical CT
/// and its estimates were low.
fn profile_rho_c<T: Voxel>(
    image: &ArrayView3<T>,
    surface: &Mesh,
    keep: &[bool],
    o: &Options,
    exclude: Option<ArrayView3<bool>>,
) -> Result<f64, Error> {
    let (sub, _, normals) = select_nodes(surface, keep, o)?;
    let (mut profiles, x) = sample_profiles(image, surface, &sub.vertices, &normals, exclude, o);
    let weights = residual_weights(&x, o.model.residual_boost);
    Ok(density::cortical_density(&mut profiles, &x, &weights, None, o)?.0)
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
    if let Method::Volumetric(v) = &options.method {
        let mut c = *constraints;
        if c.rho_c.is_none() && options.model.rho_c_mode == RhoCMode::Estimate {
            c.rho_c = Some(profile_rho_c(image, &surface, keep, options, constraints.exclude)?);
        }
        return volumetric::fit_surface(image, surface, keep, options, v, warm_start, &c);
    }
    let o = options;
    let mut timings = Vec::new();
    let t0 = Instant::now();

    // ---- the nodes, their profiles and the cortical density
    let (sub, vertex_ids, normals) = select_nodes(&surface, keep, o)?;
    let points = sub.vertices.clone();
    let (mut profiles, x) = sample_profiles(image, &surface, &points, &normals, constraints.exclude, o);
    let weights = residual_weights(&x, o.model.residual_boost);
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

    // ---- the blur, and one model for all profiles
    let (mut psf, mut sigma_fixed) = given_blur(o, &normals);
    let t1 = Instant::now();
    let mut global_params = initial_model(&data, sigma_fixed.as_deref(), constraints.densities);
    let held = held_subset(constraints.fixed, warm_start, &vertex_ids)?;
    timings.push(("global fit".to_string(), t1.elapsed().as_secs_f64()));
    let starts =
        |global: &Params, sigma_fixed: Option<&[f64]>| node_starts(global, warm_start, &vertex_ids, sigma_fixed);
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
        held: held.as_deref(),
    };
    let nodes = Nodes {
        mesh: &sub,
        normals: &normals,
    };
    let mut params = fit_method(&data, &nodes, &start, constraints, &mut info, &mut timings);
    timings.push((format!("{} fit", o.method.name()), t3.elapsed().as_secs_f64()));

    // ---- residuals, and per-node densities scaled back
    let mut residual = residuals(&data, &params);
    if let Some(node) = &rho_c_node {
        let kept = o.keep_profiles.then_some(&mut profiles);
        scale_to_node_density(node, rho_c, &mut params, &mut residual, kept);
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
        volumetric: None,
    })
}

/// The nodes to fit: those of `keep` with a valid sampling direction. Their
/// sub-mesh, surface vertex ids, and inward sampling directions.
fn select_nodes(surface: &Mesh, keep: &[bool], o: &Options) -> Result<(Mesh, Vec<u32>, Vec<V3>), Error> {
    let outward = surface.vertex_normals();
    let dirs = sampling_normals(&outward, o.sampling.normal_constraint);
    let keep: Vec<bool> = keep.iter().zip(&dirs).map(|(&k, d)| k && d.is_some()).collect();
    let (sub, vertex_ids) = surface.submesh(&keep);
    if vertex_ids.is_empty() {
        return Err(Error::NoNodes);
    }
    let normals = vertex_ids.iter().map(|&v| dirs[v as usize].unwrap()).collect();
    Ok((sub, vertex_ids, normals))
}

/// The profiles of the nodes, with samples in `exclude` (and, with
/// `half_width_inside`, beyond half the object's width) unweighted; and
/// their sample positions.
fn sample_profiles<T: Voxel>(
    image: &ArrayView3<T>,
    surface: &Mesh,
    points: &[V3],
    normals: &[V3],
    exclude: Option<ArrayView3<bool>>,
    o: &Options,
) -> (Profiles, Vec<f64>) {
    let geom = &o.geometry;
    let resolution = o.sampling.resolution.unwrap_or_else(|| geom.min_spacing());
    let line = SampleLine::new(o.sampling.outside, o.sampling.inside, resolution);
    let mut profiles = Profiles::sample(image, geom, points, normals, &line, o.sampling.smoothing);
    if let Some(ex) = exclude {
        profiles.exclude(&ex, geom, points, normals, &line);
    }
    if o.sampling.half_width_inside {
        let region = surface.voxelize(image.dim(), geom);
        profiles.limit_inside(&region.view(), geom, points, normals, &line, geom.min_spacing());
    }
    (profiles, line.x)
}

/// A blur given by the options: the PSF, and σ per node (fixed or from the PSF).
fn given_blur(o: &Options, normals: &[V3]) -> (Option<[f64; 3]>, Option<Vec<f64>>) {
    match &o.model.sigma {
        SigmaMode::Free | SigmaMode::EstimatePsf { .. } => (None, None),
        SigmaMode::Fixed(s) => (None, Some(vec![*s; normals.len()])),
        SigmaMode::Psf(p) => (
            Some(*p),
            Some(normals.iter().map(|&nr| psf::sigma_along(*p, nr)).collect()),
        ),
    }
}

/// One model for all profiles (the global initialisation), or the initial
/// values; densities held at `densities` if given.
fn initial_model(data: &Data, sigma_fixed: Option<&[f64]>, densities: Option<[f64; 2]>) -> Params {
    let o = data.options;
    let mut init = o.model.init;
    if let Some(sf) = sigma_fixed {
        init[SIGMA] = sf.iter().sum::<f64>() / sf.len() as f64;
    }
    let mut global = if o.init_from_global {
        density::fit_one_model(data, init, sigma_fixed)
    } else {
        init
    };
    if let Some([rs, rb]) = densities {
        global[RHO_S] = rs;
        global[RHO_B] = rb;
    }
    global
}

/// The fixed surface vertices as fitted-node flags; each needs a warm start
/// to be held at.
pub(crate) fn held_subset(
    fixed: Option<&[bool]>,
    warm_start: Option<&[Option<Params>]>,
    vertex_ids: &[u32],
) -> Result<Option<Vec<bool>>, Error> {
    let Some(f) = fixed else {
        return Ok(None);
    };
    let held: Vec<bool> = vertex_ids.iter().map(|&v| f[v as usize]).collect();
    let has_warm = |k: usize| warm_start.is_some_and(|w| w[vertex_ids[k] as usize].is_some());
    if held.iter().enumerate().any(|(k, &h)| h && !has_warm(k)) {
        return Err(Error::InvalidOption("fixed nodes need a warm start".into()));
    }
    Ok(Some(held))
}

/// Fit with the chosen method; held nodes keep their starts.
fn fit_method(
    data: &Data,
    nodes: &Nodes,
    start: &Start,
    constraints: &FitConstraints,
    info: &mut FitInfo,
    timings: &mut Vec<(String, f64)>,
) -> Vec<Params> {
    let mut params = match &data.options.method {
        Method::Local => local::fit_nodes(data, start, info),
        Method::Multigrid { .. } => multigrid::fit(data, nodes, start, info),
        Method::Regularized { .. } => regularized::fit(data, nodes, start, constraints, info, timings),
        Method::Volumetric(_) => unreachable!("the volumetric model does not fit profiles"),
    };
    if let Some(held) = start.held {
        for (k, &h) in held.iter().enumerate() {
            if h {
                params[k] = start.nodes[k];
            }
        }
    }
    params
}

/// Densities, residuals (and kept profiles) fitted at the common `ρc`,
/// scaled back to each node's own cortical density `node`.
fn scale_to_node_density(
    node: &[f64],
    rho_c: f64,
    params: &mut [Params],
    residual: &mut [f64],
    mut profiles: Option<&mut Profiles>,
) {
    for (i, &rn) in node.iter().enumerate() {
        let s = rn / rho_c;
        params[i][RHO_S] *= s;
        params[i][RHO_B] *= s;
        residual[i] *= s;
        if let Some(p) = profiles.as_deref_mut() {
            let m = p.m;
            p.values[i * m..(i + 1) * m].iter_mut().for_each(|v| *v *= s as f32);
        }
    }
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
