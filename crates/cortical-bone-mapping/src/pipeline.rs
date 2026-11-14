//! The end-to-end cortical bone mapping pipeline.
//!
//! 1. Extract the outer bone surface from the binary mask (surface nets) and
//!    smooth it (Taubin λ|μ).
//! 2. Select the nodes to fit: optionally only near a sub-mask, never on the
//!    caps where the mask is cut by the image edge, and only where the
//!    (optionally constrained) sampling direction is valid.
//! 3. Sample a density profile along the inward normal at every node.
//! 4. Fit one model to all profiles (global initialisation).
//! 5. Fit with the chosen [`Method`]: local, multigrid or regularised (with an
//!    optional curvature penalty).

use std::time::Instant;

use ndarray::ArrayView3;

use crate::global::{self, Problem, Role, Smoothing, State};
use crate::local::{self, FitSettings, Status};
use crate::mesh::Mesh;
use crate::model::{eval, residual_weights, Params, M, NPARAM, RHO_B, RHO_S, SIGMA, T};
use crate::operators::{
    interpolation_matrix, select_control_points, CurvatureMode, CurvatureOperator, SmoothingOperator, SurfaceLaplacian,
};
use crate::profiles::{sampling_normals, NormalConstraint, Profiles, SampleLine};
use crate::psf::{estimate_psf, sigma_along, PsfModel};
use crate::surface::extract_surface;
use crate::util::{add, scale, V3};
use crate::volume::{Geometry, Voxel};

/// Errors from the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    EmptyMask,
    ShapeMismatch(String),
    NoNodes,
    InvalidOption(String),
    ThreadPool(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::EmptyMask => write!(f, "the mask is empty"),
            Error::ShapeMismatch(s) => write!(f, "shape mismatch: {s}"),
            Error::NoNodes => write!(f, "no surface nodes left to fit (check sub-mask and normal constraint)"),
            Error::InvalidOption(s) => write!(f, "invalid option: {s}"),
            Error::ThreadPool(s) => write!(f, "could not build thread pool: {s}"),
        }
    }
}
impl std::error::Error for Error {}

/// Surface extraction and smoothing.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceOptions {
    /// Gaussian blur of the mask before extraction (voxels; 0 = none).
    pub mask_smoothing: f64,
    pub taubin_iterations: usize,
    /// Taubin pass-band `k_pb` (`1/λ + 1/μ = k_pb`).
    pub taubin_passband: f64,
    pub taubin_lambda: f64,
    /// Skip nodes on caps where the mask touches the image edge.
    pub exclude_border: bool,
}

impl Default for SurfaceOptions {
    fn default() -> Self {
        SurfaceOptions {
            mask_smoothing: 0.0,
            taubin_iterations: 100,
            taubin_passband: 0.1,
            taubin_lambda: 0.5,
            exclude_border: true,
        }
    }
}

/// Profile sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingOptions {
    /// Distance sampled outside the surface (physical units).
    pub outside: f64,
    /// Distance sampled inside the surface.
    pub inside: f64,
    /// Sample spacing; `None` = the smallest voxel spacing.
    pub resolution: Option<f64>,
    /// 1D Gaussian smoothing of each profile (standard deviation in samples).
    pub smoothing: f64,
    pub normal_constraint: NormalConstraint,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        SamplingOptions {
            outside: 2.0,
            inside: 4.0,
            resolution: None,
            smoothing: 0.0,
            normal_constraint: NormalConstraint::None,
        }
    }
}

/// How the blur `σ` is treated.
#[derive(Clone, Debug, PartialEq)]
pub enum SigmaMode {
    /// Fitted per node (Neeteson & Boyd).
    Free,
    /// Fixed to one value everywhere.
    Fixed(f64),
    /// Fixed per node from a known Gaussian PSF (standard deviations per axis).
    Psf([f64; 3]),
    /// Estimate the PSF from a first local fit with `σ` free (nodes with
    /// `t ≥ min_ratio · σ`), then fit with `σ` fixed from it (Treece-style).
    EstimatePsf { model: PsfModel, min_ratio: f64 },
}

/// The profile model: fixed cortical density, residual weighting, initial
/// guesses and bounds for `[m, t, σ, ρs, ρb]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOptions {
    pub rho_c: f64,
    /// `γ0`: weight of residuals at the surface node relative to the inner end.
    pub residual_boost: f64,
    pub init: Params,
    pub lower: Params,
    pub upper: Params,
    pub sigma: SigmaMode,
}

impl Default for ModelOptions {
    fn default() -> Self {
        // Initial guesses and bounds follow Table 6.1 of Neeteson & Boyd.
        ModelOptions {
            rho_c: 1000.0,
            residual_boost: 10.0,
            init: [0.0, 1.0, 0.1, 0.0, 0.0],
            lower: [-3.0, 0.0607, 0.01, -200.0, -200.0],
            upper: [6.0, 12.0, 2.0, 400.0, 400.0],
            sigma: SigmaMode::Free,
        }
    }
}

/// Stopping criteria shared by the local and global solvers.
#[derive(Clone, Debug, PartialEq)]
pub struct SolverOptions {
    pub max_iterations: usize,
    pub ftol: f64,
    pub gtol: f64,
    pub xtol: f64,
    pub cg_max_iterations: usize,
    pub cg_tolerance: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        SolverOptions {
            max_iterations: 100,
            ftol: 1e-8,
            gtol: 1e-8,
            xtol: 1e-10,
            cg_max_iterations: 200,
            cg_tolerance: 1e-3,
        }
    }
}

/// Curvature penalty on the fitted surfaces (extension).
#[derive(Clone, Debug, PartialEq)]
pub struct CurvatureOptions {
    pub mode: CurvatureMode,
    /// Weight `μ` (dimensionless, like `λ`).
    pub weight: f64,
    /// Length scale `ℓ` that makes curvature dimensionless; `None` = the
    /// globally fitted thickness `t0`.
    pub length: Option<f64>,
    /// Penalise the periosteal and/or endosteal surface.
    pub periosteal: bool,
    pub endosteal: bool,
    /// Scale of the surface Laplacian: `Some(σ_c)` uses a Gaussian
    /// point-cloud Laplacian at that physical scale (well conditioned, fast);
    /// `None` uses the cotangent mesh Laplacian (curvature at mesh
    /// resolution; the Gauss–Newton systems are stiffer and need many more
    /// CG iterations).
    pub scale: Option<f64>,
    /// Neighbours for the Gaussian Laplacian.
    pub neighbours: usize,
}

impl Default for CurvatureOptions {
    fn default() -> Self {
        CurvatureOptions {
            mode: CurvatureMode::Change,
            weight: 1.0,
            length: None,
            periosteal: true,
            endosteal: true,
            scale: Some(0.25),
            neighbours: 64,
        }
    }
}

/// The model-fitting method.
#[derive(Clone, Debug, PartialEq)]
pub enum Method {
    /// Independent fit at every node (Treece et al.; "local" method).
    Local,
    /// `m, t, σ` on control points at decreasing spacings, interpolated to
    /// the nodes (Neeteson & Boyd, §6.1.2.1).
    Multigrid {
        /// Control point spacings (physical units), coarse to fine.
        spacings: Vec<f64>,
        neighbours: usize,
        seed: u64,
    },
    /// `m, t, σ` per node with spatial regularisation (Neeteson & Boyd,
    /// §6.1.2.2), optionally with a curvature penalty.
    Regularized {
        /// Regularisation coefficient `λ` (0 disables the smoothing term).
        lambda: f64,
        /// Regularisation Gaussian `σ_reg` (physical units).
        sigma: f64,
        neighbours: usize,
        /// Neighbours farther than `cutoff · σ_reg` are dropped (∞ = plain kNN).
        cutoff: f64,
        /// Start from the local fit instead of the global fit.
        init_from_local: bool,
        curvature: Option<CurvatureOptions>,
    },
}

impl Method {
    pub fn multigrid() -> Self {
        // Table 6.1 (radius & tibia): 128, 64, 32, 16 voxels of 60.7 µm.
        Method::Multigrid {
            spacings: vec![7.77, 3.885, 1.942, 0.971],
            neighbours: 5,
            seed: 0,
        }
    }
    pub fn regularized() -> Self {
        Method::Regularized {
            lambda: 1.0,
            sigma: 1.924,
            neighbours: 500,
            cutoff: 3.0,
            init_from_local: false,
            curvature: None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Method::Local => "local",
            Method::Multigrid { .. } => "multigrid",
            Method::Regularized { .. } => "regularized",
        }
    }
}

/// All pipeline options.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub geometry: Geometry,
    pub surface: SurfaceOptions,
    pub sampling: SamplingOptions,
    pub model: ModelOptions,
    pub solver: SolverOptions,
    pub method: Method,
    /// Initialise the fit from one model fitted to all profiles.
    pub init_from_global: bool,
    /// Keep the sampled profiles in the result.
    pub keep_profiles: bool,
    /// Worker threads (`None` = all cores).
    pub threads: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            geometry: Geometry::default(),
            surface: SurfaceOptions::default(),
            sampling: SamplingOptions::default(),
            model: ModelOptions::default(),
            solver: SolverOptions::default(),
            method: Method::Local,
            init_from_global: true,
            keep_profiles: false,
            threads: None,
        }
    }
}

/// Diagnostics of a fit.
#[derive(Clone, Debug, Default)]
pub struct FitInfo {
    pub method: String,
    /// Fraction of local fits that converged (local method), or 0/1.
    pub converged: f64,
    pub iterations: usize,
    pub cg_iterations: usize,
    pub cost: f64,
    pub data_cost: f64,
    pub smoothing_cost: f64,
    pub curvature_cost: f64,
    /// Wall time per stage (seconds).
    pub timings: Vec<(String, f64)>,
}

/// The result of cortical bone mapping.
#[derive(Clone, Debug)]
pub struct CbmResult {
    /// The full smoothed outer surface.
    pub surface: Mesh,
    /// Indices (into `surface.vertices`) of the fitted nodes.
    pub vertex_ids: Vec<u32>,
    /// The fitted nodes and the faces between them (indices into `points`).
    pub points: Vec<V3>,
    pub faces: Vec<[u32; 3]>,
    /// Inward unit sampling directions.
    pub normals: Vec<V3>,
    /// `[m, t, σ, ρs, ρb]` per node.
    pub params: Vec<Params>,
    /// Weighted RMS residual per node.
    pub residual: Vec<f64>,
    pub rho_c: f64,
    /// The single model fitted to all profiles.
    pub global_params: Params,
    /// The PSF used or estimated, if any.
    pub psf: Option<[f64; 3]>,
    pub info: FitInfo,
    /// Sample positions and profiles (if `keep_profiles`).
    pub profiles: Option<(Vec<f64>, Profiles)>,
}

impl CbmResult {
    pub fn n(&self) -> usize {
        self.points.len()
    }
    pub fn param(&self, c: usize) -> Vec<f64> {
        self.params.iter().map(|p| p[c]).collect()
    }
    pub fn thickness(&self) -> Vec<f64> {
        self.param(T)
    }
    /// Fitted surface `p + (m + side·t/2) n` (`side` −1 periosteal, +1 endosteal).
    fn offset_points(&self, side: f64) -> Vec<V3> {
        self.points
            .iter()
            .zip(&self.normals)
            .zip(&self.params)
            .map(|((&p, &n), q)| add(p, scale(n, q[M] + side * 0.5 * q[T])))
            .collect()
    }
    pub fn periosteal_points(&self) -> Vec<V3> {
        self.offset_points(-1.0)
    }
    pub fn endosteal_points(&self) -> Vec<V3> {
        self.offset_points(1.0)
    }
    /// The full surface with fitted vertices moved to the fitted periosteal
    /// (`side = -1`) or endosteal (`side = +1`) position; other vertices stay.
    pub fn displaced_surface(&self, side: f64) -> Mesh {
        let mut m = self.surface.clone();
        for (k, &v) in self.vertex_ids.iter().enumerate() {
            let q = &self.params[k];
            m.vertices[v as usize] = add(self.points[k], scale(self.normals[k], q[M] + side * 0.5 * q[T]));
        }
        m
    }
    /// Write the fitted nodes as VTK PolyData with all parameters attached.
    pub fn write_vtk(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mesh = Mesh {
            vertices: self.points.clone(),
            faces: self.faces.clone(),
        };
        let cols: Vec<Vec<f64>> = (0..NPARAM).map(|c| self.param(c)).collect();
        let names = ["center", "thickness", "sigma", "rho_s", "rho_b"];
        let mut scalars: Vec<(&str, &[f64])> = names.iter().zip(&cols).map(|(n, c)| (*n, c.as_slice())).collect();
        scalars.push(("residual", &self.residual));
        mesh.write_vtk(path, &scalars, &[("normals", &self.normals)])
    }
}

/// Run `f` on a pool with `threads` workers (or the global pool).
pub fn with_threads<R: Send>(threads: Option<usize>, f: impl FnOnce() -> R + Send) -> Result<R, Error> {
    match threads {
        None => Ok(f()),
        Some(0) => Err(Error::InvalidOption("threads must be positive".into())),
        Some(t) => rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build()
            .map(|p| p.install(f))
            .map_err(|e| Error::ThreadPool(e.to_string())),
    }
}

fn check_options(o: &Options) -> Result<(), Error> {
    let bad = |s: &str| Err(Error::InvalidOption(s.to_string()));
    if o.geometry.spacing.iter().any(|&s| !(s > 0.0)) {
        return bad("spacing must be positive");
    }
    if !(o.sampling.inside > 0.0) || !(o.sampling.outside >= 0.0) {
        return bad("sampling distances must be positive");
    }
    if o.sampling.resolution.is_some_and(|r| !(r > 0.0)) {
        return bad("sampling resolution must be positive");
    }
    for c in 0..NPARAM {
        if !(o.model.lower[c] <= o.model.upper[c]) {
            return bad("lower bounds must not exceed upper bounds");
        }
    }
    if !(o.model.lower[SIGMA] > 0.0) || !(o.model.lower[T] >= 0.0) {
        return bad("the lower bound of sigma must be positive and of t non-negative");
    }
    match &o.method {
        Method::Multigrid {
            spacings, neighbours, ..
        } => {
            if spacings.is_empty() || spacings.iter().any(|&s| !(s > 0.0)) || *neighbours == 0 {
                return bad("multigrid needs positive spacings and neighbours");
            }
        }
        Method::Regularized {
            lambda,
            sigma,
            neighbours,
            curvature,
            ..
        } => {
            if *lambda < 0.0 || !(*sigma > 0.0) || *neighbours == 0 {
                return bad("regularized needs lambda >= 0, sigma > 0, neighbours > 0");
            }
            if let Some(c) = curvature {
                if c.weight < 0.0 || c.length.is_some_and(|l| !(l > 0.0)) || c.scale.is_some_and(|l| !(l > 0.0)) {
                    return bad("curvature weight must be >= 0, length and scale > 0");
                }
                if c.scale.is_some() && c.neighbours < 3 {
                    return bad("curvature neighbours must be >= 3");
                }
                if let CurvatureMode::Excess { kappa } = c.mode {
                    if kappa < 0.0 {
                        return bad("curvature threshold must be >= 0");
                    }
                }
            }
        }
        Method::Local => {}
    }
    Ok(())
}

/// Run cortical bone mapping on `image` with outer-surface segmentation
/// `mask` (non-zero = bone) and an optional `sub_mask` restricting where the
/// surface is measured.
pub fn run<T: Voxel, U: Voxel, V: Voxel>(
    image: ArrayView3<T>,
    mask: ArrayView3<U>,
    sub_mask: Option<ArrayView3<V>>,
    options: &Options,
) -> Result<CbmResult, Error> {
    check_options(options)?;
    if image.dim() != mask.dim() {
        return Err(Error::ShapeMismatch(format!(
            "image {:?} vs mask {:?}",
            image.dim(),
            mask.dim()
        )));
    }
    if let Some(s) = &sub_mask {
        if s.dim() != mask.dim() {
            return Err(Error::ShapeMismatch(format!(
                "sub_mask {:?} vs mask {:?}",
                s.dim(),
                mask.dim()
            )));
        }
    }
    with_threads(options.threads, || {
        let t0 = Instant::now();
        let surf = prepare_surface(&mask, options);
        if surf.0.vertices.is_empty() {
            return Err(Error::EmptyMask);
        }
        let t_surface = t0.elapsed().as_secs_f64();
        let candidates = candidate_nodes(&surf.0, &surf.1, sub_mask.as_ref(), options);
        let mut res = fit_surface(&image, surf.0, &candidates, options, None)?;
        res.info.timings.insert(0, ("surface".into(), t_surface));
        Ok(res)
    })?
}

/// Extract and smooth the outer surface; returns the mesh and border flags.
pub fn prepare_surface<U: Voxel>(mask: &ArrayView3<U>, options: &Options) -> (Mesh, Vec<bool>) {
    let s = &options.surface;
    let e = extract_surface(mask, &options.geometry, s.mask_smoothing);
    let mut mesh = e.mesh;
    mesh.taubin_smooth(s.taubin_iterations, s.taubin_lambda, s.taubin_passband, None);
    (mesh, e.border)
}

/// Nodes that may be fitted: not on the border caps (if excluded) and, with
/// a sub-mask, adjacent to a sub-mask voxel (any of the 8 voxels around it).
pub fn candidate_nodes<V: Voxel>(
    mesh: &Mesh,
    border: &[bool],
    sub_mask: Option<&ArrayView3<V>>,
    options: &Options,
) -> Vec<bool> {
    let g = &options.geometry;
    mesh.vertices
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            if options.surface.exclude_border && border[i] {
                return false;
            }
            match sub_mask {
                None => true,
                Some(sm) => {
                    let (d0, d1, d2) = sm.dim();
                    let q = g.to_index(p);
                    let base = q.map(|v| v.floor() as isize);
                    for a in 0..2 {
                        for b in 0..2 {
                            for c in 0..2 {
                                let (i0, i1, i2) = (base[0] + a, base[1] + b, base[2] + c);
                                if i0 >= 0
                                    && i1 >= 0
                                    && i2 >= 0
                                    && (i0 as usize) < d0
                                    && (i1 as usize) < d1
                                    && (i2 as usize) < d2
                                    && sm[[i0 as usize, i1 as usize, i2 as usize]].to_f64() != 0.0
                                {
                                    return true;
                                }
                            }
                        }
                    }
                    false
                }
            }
        })
        .collect()
}

fn fit_settings(o: &Options, free_sigma: bool) -> FitSettings {
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

fn global_settings(o: &Options) -> global::Settings {
    global::Settings {
        max_iterations: o.solver.max_iterations,
        ftol: o.solver.ftol,
        gtol: o.solver.gtol,
        xtol: o.solver.xtol,
        cg_max_iterations: o.solver.cg_max_iterations,
        cg_tolerance: o.solver.cg_tolerance,
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
    let o = options;
    let mut timings = Vec::new();
    let t0 = Instant::now();
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
    let profiles = Profiles::sample(image, &o.geometry, &points, &normals, &line, o.sampling.smoothing);
    let x = line.x.clone();
    let weights = residual_weights(&x, o.model.residual_boost);
    timings.push(("sampling".to_string(), t0.elapsed().as_secs_f64()));

    let rho_c = o.model.rho_c;
    let mut info = FitInfo {
        method: o.method.name().to_string(),
        ..Default::default()
    };

    // ---- σ handling
    let mut psf: Option<[f64; 3]> = None;
    let mut sigma_fixed: Option<Vec<f64>> = match &o.model.sigma {
        SigmaMode::Free | SigmaMode::EstimatePsf { .. } => None,
        SigmaMode::Fixed(s) => Some(vec![*s; n]),
        SigmaMode::Psf(p) => {
            psf = Some(*p);
            Some(normals.iter().map(|&nr| sigma_along(*p, nr)).collect())
        }
    };
    let mut init = o.model.init;
    if let Some(sf) = &sigma_fixed {
        init[SIGMA] = sf.iter().sum::<f64>() / n as f64;
    }

    // ---- global initialisation: one model for all profiles
    let t1 = Instant::now();
    let global_fit = |init: Params, sigma_fixed: &Option<Vec<f64>>| -> Params {
        let mut roles = [Role::Global; NPARAM];
        let mut fixed: [Option<&[f64]>; NPARAM] = [None; NPARAM];
        if let Some(sf) = sigma_fixed {
            roles[SIGMA] = Role::Fixed;
            fixed[SIGMA] = Some(sf.as_slice());
        }
        let pr = Problem {
            profiles: &profiles,
            x: &x,
            weights: &weights,
            rho_c,
            roles,
            fixed_per_node: fixed,
            interp: None,
            smoothing: None,
            curvature: None,
            lower: o.model.lower,
            upper: o.model.upper,
        };
        let mut st = State {
            field: vec![],
            scalars: init,
        };
        global::solve(&pr, &mut st, &global_settings(o));
        st.scalars
    };
    let mut global_params = if o.init_from_global {
        global_fit(init, &sigma_fixed)
    } else {
        init
    };
    timings.push(("global fit".to_string(), t1.elapsed().as_secs_f64()));

    let per_node_init = |gp: &Params, sigma_fixed: &Option<Vec<f64>>| -> Vec<Params> {
        (0..n)
            .map(|k| {
                let mut p = warm_start.and_then(|w| w[vertex_ids[k] as usize]).unwrap_or(*gp);
                if let Some(sf) = sigma_fixed {
                    p[SIGMA] = sf[k];
                }
                p
            })
            .collect()
    };

    // ---- PSF estimation (Treece-style fixed blur)
    if let SigmaMode::EstimatePsf { model, min_ratio } = &o.model.sigma {
        let t2 = Instant::now();
        let fits = local::fit_all(
            &profiles,
            &x,
            &weights,
            rho_c,
            &per_node_init(&global_params, &None),
            &fit_settings(o, true),
        );
        let ps: Vec<Params> = fits.iter().map(|f| f.params).collect();
        let est = estimate_psf(
            &normals,
            &ps,
            *model,
            *min_ratio,
            o.model.lower[SIGMA],
            o.model.upper[SIGMA],
        )
        .ok_or_else(|| Error::InvalidOption("too few nodes with a thick cortex to estimate the PSF".into()))?;
        let est = est.map(|v| v.clamp(o.model.lower[SIGMA], o.model.upper[SIGMA]));
        psf = Some(est);
        let sf: Vec<f64> = normals.iter().map(|&nr| sigma_along(est, nr)).collect();
        sigma_fixed = Some(sf);
        if o.init_from_global {
            global_params = global_fit(global_params, &sigma_fixed);
        }
        timings.push(("psf estimation".to_string(), t2.elapsed().as_secs_f64()));
    }
    let free_sigma = sigma_fixed.is_none();

    // ---- the chosen method
    let t3 = Instant::now();
    let params: Vec<Params> = match &o.method {
        Method::Local => {
            let fits = local::fit_all(
                &profiles,
                &x,
                &weights,
                rho_c,
                &per_node_init(&global_params, &sigma_fixed),
                &fit_settings(o, free_sigma),
            );
            info.converged = fits.iter().filter(|f| f.status.converged()).count() as f64 / n as f64;
            info.iterations = fits.iter().map(|f| f.iterations as usize).max().unwrap_or(0);
            fits.iter().map(|f| f.params).collect()
        }
        Method::Multigrid {
            spacings,
            neighbours,
            seed,
        } => {
            let mut node_params = per_node_init(&global_params, &sigma_fixed);
            let mut roles = [Role::Field, Role::Field, Role::Field, Role::Global, Role::Global];
            let mut fixed: [Option<&[f64]>; NPARAM] = [None; NPARAM];
            if let Some(sf) = &sigma_fixed {
                roles[SIGMA] = Role::Fixed;
                fixed[SIGMA] = Some(sf.as_slice());
            }
            let fields: Vec<usize> = (0..NPARAM).filter(|&c| roles[c] == Role::Field).collect();
            let mut scalars = global_params;
            for (lvl, &s) in spacings.iter().enumerate() {
                let ctrl = select_control_points(&points, s, seed.wrapping_add(lvl as u64));
                let cpts: Vec<V3> = ctrl.iter().map(|&i| points[i as usize]).collect();
                let a = interpolation_matrix(&points, &cpts, *neighbours, s);
                let pr = Problem {
                    profiles: &profiles,
                    x: &x,
                    weights: &weights,
                    rho_c,
                    roles,
                    fixed_per_node: fixed,
                    interp: Some(&a),
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
                let si = global::solve(&pr, &mut st, &global_settings(o));
                accumulate(&mut info, &si);
                node_params = pr.node_params(&st);
                scalars = st.scalars;
            }
            node_params
        }
        Method::Regularized {
            lambda,
            sigma,
            neighbours,
            cutoff,
            init_from_local,
            curvature,
        } => {
            let start: Vec<Params> = if *init_from_local {
                local::fit_all(
                    &profiles,
                    &x,
                    &weights,
                    rho_c,
                    &per_node_init(&global_params, &sigma_fixed),
                    &fit_settings(o, free_sigma),
                )
                .iter()
                .map(|f| f.params)
                .collect()
            } else {
                per_node_init(&global_params, &sigma_fixed)
            };
            let mut roles = [Role::Field, Role::Field, Role::Field, Role::Global, Role::Global];
            let mut fixed: [Option<&[f64]>; NPARAM] = [None; NPARAM];
            if let Some(sf) = &sigma_fixed {
                roles[SIGMA] = Role::Fixed;
                fixed[SIGMA] = Some(sf.as_slice());
            }
            let t0g = global_params[T].max(1e-6);
            let s0g = global_params[SIGMA].max(1e-6);
            let base = lambda * rho_c * rho_c / n as f64;
            let op = (*lambda > 0.0).then(|| SmoothingOperator::new(&points, *neighbours, *sigma, *cutoff));
            let mut coef = [0.0; NPARAM];
            coef[M] = base / (t0g * t0g);
            coef[T] = base / (t0g * t0g);
            if free_sigma {
                coef[SIGMA] = base / (s0g * s0g);
            }
            let curv = curvature.as_ref().map(|c| {
                let ell = c.length.unwrap_or(t0g);
                let cc = (c.weight / n as f64).sqrt() * rho_c * ell;
                let lap = match c.scale {
                    Some(sc) => SurfaceLaplacian::gaussian(&points, c.neighbours, sc, 3.0),
                    None => SurfaceLaplacian::cotangent(&sub),
                };
                CurvatureOperator::new(lap, &points, &normals, c.mode, cc, [c.periosteal, c.endosteal])
            });
            let fields: Vec<usize> = (0..NPARAM).filter(|&c| roles[c] == Role::Field).collect();
            let pr = Problem {
                profiles: &profiles,
                x: &x,
                weights: &weights,
                rho_c,
                roles,
                fixed_per_node: fixed,
                interp: None,
                smoothing: op.as_ref().map(|op| Smoothing { op, coef }),
                curvature: curv.as_ref(),
                lower: o.model.lower,
                upper: o.model.upper,
            };
            let mut scalars = global_params;
            if *init_from_local {
                scalars[RHO_S] = start.iter().map(|p| p[RHO_S]).sum::<f64>() / n as f64;
                scalars[RHO_B] = start.iter().map(|p| p[RHO_B]).sum::<f64>() / n as f64;
            }
            let mut st = State {
                field: fields.iter().map(|&c| start.iter().map(|p| p[c]).collect()).collect(),
                scalars,
            };
            let si = global::solve(&pr, &mut st, &global_settings(o));
            accumulate(&mut info, &si);
            pr.node_params(&st)
        }
    };
    timings.push((format!("{} fit", o.method.name()), t3.elapsed().as_secs_f64()));

    // ---- residuals
    let wsum: f64 = weights.iter().sum();
    let residual: Vec<f64> = params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let y = profiles.row(i);
            let s: f64 = x
                .iter()
                .zip(y)
                .zip(&weights)
                .map(|((&xj, &yj), &w)| w * (eval(xj, p, rho_c) - yj as f64).powi(2))
                .sum();
            (s / wsum).sqrt()
        })
        .collect();
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
        global_params,
        psf,
        info,
        profiles: o.keep_profiles.then_some((x, profiles)),
    })
}

fn accumulate(info: &mut FitInfo, si: &global::SolveInfo) {
    info.iterations += si.iterations;
    info.cg_iterations += si.cg_iterations;
    info.cost = si.cost;
    info.data_cost = si.data_cost;
    info.smoothing_cost = si.smoothing_cost;
    info.curvature_cost = si.curvature_cost;
    info.converged = if si.status == Status::MaxIterations { 0.0 } else { 1.0 };
}
