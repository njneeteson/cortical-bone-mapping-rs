//! All options of the pipeline, the methods, and the published parameter
//! presets (Table 1 of Neeteson & Boyd 2025).

use crate::error::Error;
use crate::fit::profiles::NormalConstraint;
use crate::fit::psf::PsfModel;
use crate::model::{Params, NPARAM, SIGMA, T};
use crate::operators::{CurvatureMeasure, CurvatureMode, Distance};
use crate::volume::Geometry;

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
    /// Limit each profile's inside part to half the object's width along
    /// its normal (the distance to where the line leaves the region enclosed
    /// by the surface), but not below one voxel. Samples beyond get zero
    /// weight. At thin tips and plates the profile then stops at the
    /// object's middle instead of running out through the far side, where
    /// the model's trabecular plateau does not exist (extension).
    pub half_width_inside: bool,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        SamplingOptions {
            outside: 2.0,
            inside: 4.0,
            resolution: None,
            smoothing: 0.0,
            normal_constraint: NormalConstraint::None,
            half_width_inside: false,
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

/// How the (fixed, not fitted) cortical density `ρc` is chosen.
#[derive(Clone, Debug, PartialEq)]
pub enum RhoCMode {
    /// [`ModelOptions::rho_c`] as given (e.g. the mineralised tissue density
    /// of a density-calibrated scanner).
    Fixed,
    /// One value for all nodes, estimated from the data as the cortical
    /// plateau level where the cortex is thick enough to have one: local fits
    /// (σ free), then for nodes whose fitted cortex has samples farther than
    /// 2σ from both edges (at least 3), the median of those samples; `ρc` is
    /// the median over such nodes. Starts from the 90th percentile of the
    /// profile maxima and is iterated twice. Profile maxima alone are biased
    /// up by noise, and down by blur and profile smoothing on a thin cortex.
    Estimate,
    /// One value per node: the maximum of its (smoothed) profile, as the
    /// original Python implementation does when no density is given. A thin
    /// or blurred cortex never reaches the tissue density, so this trades
    /// the model's partial-volume correction for robustness to density
    /// variation.
    ProfileMax,
}

/// The profile model: fixed cortical density, residual weighting, initial
/// guesses and bounds for `[m, t, σ, ρs, ρb]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOptions {
    pub rho_c: f64,
    pub rho_c_mode: RhoCMode,
    /// `γ0`: weight of residuals at the surface node relative to the inner end.
    pub residual_boost: f64,
    pub init: Params,
    pub lower: Params,
    pub upper: Params,
    pub sigma: SigmaMode,
    /// Global methods: which of `[m, t, σ, ρs, ρb]` take one value for the
    /// whole surface; the others are fields (per node, regularised in the
    /// regularised method). Neeteson & Boyd: `ρs`, `ρb` global. Per-node
    /// densities (all `false`) are needed where the trabecular density
    /// varies over the bone, e.g. dense epiphyses and fatty diaphyseal
    /// marrow in a clinical CT (extension).
    pub global: [bool; NPARAM],
    /// Weight of the smoothing of density fields relative to that of `m`
    /// and `t` (regularised method).
    pub density_smoothing: f64,
}

impl Default for ModelOptions {
    fn default() -> Self {
        // Initial guesses and bounds follow Table 1 of Neeteson & Boyd (2025).
        ModelOptions {
            rho_c: 1000.0,
            rho_c_mode: RhoCMode::Fixed,
            residual_boost: 10.0,
            init: [0.0, 1.0, 0.1, 0.0, 0.0],
            lower: [-3.0, 0.0607, 0.01, -200.0, -200.0],
            upper: [6.0, 12.0, 2.0, 400.0, 400.0],
            sigma: SigmaMode::Free,
            global: [false, false, false, true, true],
            density_smoothing: 1.0,
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
    /// CG iterations allowed over one global solve (bounds the work on a
    /// badly posed problem; the fit is then flagged in
    /// [`FitInfo::budget_exhausted`](crate::result::FitInfo::budget_exhausted)).
    pub cg_max_total: usize,
    /// Print the state of the global (multigrid / regularised) solver to
    /// stderr every this many iterations; 0 = never.
    pub progress_every: usize,
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
            cg_max_total: 2000,
            progress_every: 1,
        }
    }
}

/// Curvature penalty on the fitted surfaces (extension).
#[derive(Clone, Debug, PartialEq)]
pub struct CurvatureOptions {
    pub mode: CurvatureMode,
    /// The curvature the absolute mode measures (the change mode always
    /// measures mean curvature).
    pub measure: CurvatureMeasure,
    /// Weight `μ` (dimensionless, like `λ`).
    pub weight: f64,
    /// Weight of the endosteal surface, if different (`None` = `weight`):
    /// the endosteal edge, a transition into trabecular bone, is noisier.
    pub endosteal_weight: Option<f64>,
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
            measure: CurvatureMeasure::Mean,
            weight: 1.0,
            endosteal_weight: None,
            length: None,
            periosteal: true,
            endosteal: true,
            scale: Some(0.25),
            neighbours: 64,
        }
    }
}

/// How the volumetric model treats the blur along the scanner axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AxialBlur {
    /// Fitted, as a second global parameter.
    Fit,
    /// Equal to the in-plane blur (an isotropic PSF; HR-pQCT, or a short
    /// shaft whose normals are all nearly in-plane, where `σ_z` is not
    /// determined).
    Tied,
    /// Fixed to this value (physical units).
    Fixed(f64),
}

/// Options of the volumetric model ([`Method::Volumetric`]; extension, see
/// `docs/volumetric_model.md`).
#[derive(Clone, Debug, PartialEq)]
pub struct VolumetricOptions {
    /// The scanner axis (array axis) along which the PSF has `σ_z`.
    pub scanner_axis: usize,
    /// Fix the in-plane blur `σ_xy` (physical units); `None` = fitted.
    pub sigma_xy: Option<f64>,
    pub sigma_z: AxialBlur,
    /// Smoothing of the density fields: weight `λ` (`κ = λ/N`; 0 = none),
    /// Gaussian `σ_reg`, neighbours, and neighbour cut-off (× `σ_reg`).
    pub lambda: f64,
    pub sigma: f64,
    pub neighbours: usize,
    pub cutoff: f64,
    /// One global `ρs` and `ρb` instead of fields.
    pub global_densities: bool,
    /// Curvature penalty on the fitted surfaces (on `m`, `t`), the only
    /// regulariser of the edge positions: without it each node's support is
    /// roughly one profile's worth of voxels and the precision is that of the
    /// local fit. The default weight 0.3 is the best compromise over the
    /// phantoms (`docs/volumetric_validation.md`): thickness RMS error
    /// against weights 0.1 and 1.0 of 0.0077 (0.0069, 0.0106) and 0.0087
    /// (0.0099, 0.0104) mm on the shell at noise 30 and 120, 0.029 (0.041,
    /// 0.021) mm on the distal radius, 0.063 (0.063, 0.065) mm on the
    /// proximal femur and 0.162 (0.158, 0.166) mm on the knee. Heavier
    /// weights suit HR-pQCT; at clinical resolution the choice matters little.
    pub curvature: Option<CurvatureOptions>,
    /// Use every `voxel_stride`-th voxel along each axis (1 = all).
    pub voxel_stride: usize,
}

impl Default for VolumetricOptions {
    fn default() -> Self {
        VolumetricOptions {
            scanner_axis: 0,
            sigma_xy: None,
            sigma_z: AxialBlur::Fit,
            lambda: 1.0,
            sigma: 1.9424,
            neighbours: 500,
            cutoff: 3.0,
            global_densities: false,
            curvature: Some(CurvatureOptions {
                weight: 0.3,
                ..CurvatureOptions::default()
            }),
            voxel_stride: 1,
        }
    }
}

/// The model-fitting method.
#[derive(Clone, Debug, PartialEq)]
pub enum Method {
    /// Independent fit at every node (Treece et al.; "local" method).
    Local,
    /// `m, t, σ` on control points at decreasing spacings, interpolated to
    /// the nodes (Neeteson & Boyd 2025).
    Multigrid {
        /// Control point spacings (physical units), coarse to fine.
        spacings: Vec<f64>,
        neighbours: usize,
        seed: u64,
    },
    /// `m, t, σ` per node with spatial regularisation (Neeteson & Boyd
    /// 2025), optionally with a curvature penalty.
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
    /// The volumetric model: the profile model evaluated at every voxel of a
    /// band around the surface, with a global PSF, a curvature penalty on
    /// `m`, `t` and spatial smoothing of the density fields (extension).
    Volumetric(VolumetricOptions),
}

impl Method {
    /// The multigrid preset of Table 1 of Neeteson & Boyd (2025) for the radius
    /// and tibia: control-point spacings of 128, 64, 32 and 16 voxels of 60.7 µm,
    /// each node interpolating its 5 nearest control points.
    pub fn multigrid() -> Self {
        // Table 1 of Neeteson & Boyd (2025), radius & tibia: 128, 64, 32, 16
        // voxels of 60.7 µm.
        Method::Multigrid {
            spacings: vec![7.7696, 3.8848, 1.9424, 0.9712],
            neighbours: 5,
            seed: 0,
        }
    }
    /// The regularised preset of Table 1 of Neeteson & Boyd (2025): `λ = 1`,
    /// `σ_reg = 1.9424 mm` (32 voxels of 60.7 µm), 500 neighbours; plus our
    /// neighbour cut-off at `3σ_reg` (which leaves the paper's operator unchanged
    /// at HR-pQCT node spacing, where 500 neighbours lie well within it).
    pub fn regularized() -> Self {
        // Table 1: λ = 1, σ_reg = 1.942 mm (32 voxels), 500 neighbours.
        Method::Regularized {
            lambda: 1.0,
            sigma: 1.9424,
            neighbours: 500,
            cutoff: 3.0,
            init_from_local: false,
            curvature: None,
        }
    }
    /// Multigrid with the control-point spacings of Table 1 for `site`, in
    /// voxels of size `voxel` (radius & tibia 128…16, knee 256…16).
    pub fn multigrid_for(site: Site, voxel: f64) -> Self {
        let coarsest = match site {
            Site::RadiusTibia => 128.0,
            Site::Knee => 256.0,
        };
        let mut spacings = vec![];
        let mut s: f64 = coarsest;
        while s >= 16.0 {
            spacings.push(s * voxel);
            s /= 2.0;
        }
        Method::Multigrid {
            spacings,
            neighbours: 5,
            seed: 0,
        }
    }
    /// Regularised fit with the Table 1 parameters for `site` (500
    /// neighbours for radius & tibia, 1000 for the knee).
    pub fn regularized_for(site: Site) -> Self {
        let mut m = Method::regularized();
        if let (Site::Knee, Method::Regularized { neighbours, .. }) = (site, &mut m) {
            *neighbours = 1000;
        }
        m
    }
    /// The method's name (as in the Python options).
    pub fn name(&self) -> &'static str {
        match self {
            Method::Local => "local",
            Method::Multigrid { .. } => "multigrid",
            Method::Regularized { .. } => "regularized",
            Method::Volumetric(_) => "volumetric",
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
    /// How node neighbourhoods are measured for the regularisation operator
    /// `ω`, the Gaussian curvature Laplacian and the multigrid basis
    /// ([`Distance`]); `None` = the method's default (Euclidean for the 1D
    /// methods, as in Neeteson & Boyd 2025).
    pub neighbour_distance: Option<Distance>,
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
            neighbour_distance: None,
        }
    }
}

/// Checks of [`CurvatureOptions`].
fn validate_curvature(c: &CurvatureOptions) -> Result<(), Error> {
    let bad = |s: &str| Err(Error::InvalidOption(s.to_string()));
    let negative = c.weight < 0.0 || c.endosteal_weight.is_some_and(|w| !(w >= 0.0));
    if negative || c.length.is_some_and(|l| !(l > 0.0)) || c.scale.is_some_and(|l| !(l > 0.0)) {
        return bad("curvature weights must be >= 0, length and scale > 0");
    }
    if c.scale.is_some() && c.neighbours < 3 {
        return bad("curvature neighbours must be >= 3");
    }
    match c.mode {
        CurvatureMode::Absolute { kappa } if !(kappa >= 0.0) => bad("curvature threshold must be >= 0"),
        CurvatureMode::Change | CurvatureMode::Variation if c.measure == CurvatureMeasure::Total => {
            bad("the change and variation modes measure mean curvature only")
        }
        _ => Ok(()),
    }
}

/// Checks of [`VolumetricOptions`].
fn validate_volumetric(v: &VolumetricOptions, o: &Options) -> Result<(), Error> {
    let bad = |s: &str| Err(Error::InvalidOption(s.to_string()));
    if v.scanner_axis > 2 {
        return bad("the scanner axis must be 0, 1 or 2");
    }
    if v.sigma_xy.is_some_and(|s| !(s > 0.0)) || matches!(v.sigma_z, AxialBlur::Fixed(s) if !(s > 0.0)) {
        return bad("fixed PSF widths must be positive");
    }
    if v.lambda < 0.0 || !(v.sigma > 0.0) || v.neighbours == 0 || v.voxel_stride == 0 {
        return bad("volumetric needs lambda >= 0, sigma > 0, neighbours > 0, voxel_stride > 0");
    }
    if let Some(c) = &v.curvature {
        validate_curvature(c)?;
    }
    if o.model.rho_c_mode == RhoCMode::ProfileMax {
        return bad("the volumetric model takes a fixed or estimated rho_c, not profile maxima");
    }
    Ok(())
}

/// Anatomical sites with published parameter sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// Distal radius and tibia (HR-pQCT): normals in the transverse plane.
    RadiusTibia,
    /// Proximal tibia and distal femur at the knee: normals along the axis
    /// (subchondral plate of the articular surfaces).
    Knee,
}

impl Options {
    /// The parameters of Table 1 of Neeteson & Boyd (2025) for `site`, with
    /// isotropic voxels of size `voxel` and the regularised method. Array
    /// axis 0 is taken as the scan (axial) direction.
    ///
    /// Taubin smoothing 100 iterations (pass-band 0.1, knee 0.01); sampling
    /// 5 mm outside and 15 mm inside at the voxel size; profile smoothing σ = 3
    /// samples; model guesses and bounds as [`ModelOptions::default`]. The
    /// optimiser tolerances of Table 1 belong to L-BFGS-B and are not used.
    pub fn preset(site: Site, voxel: f64) -> Options {
        let (passband, constraint) = match site {
            Site::RadiusTibia => (0.1, NormalConstraint::InPlane(0)),
            Site::Knee => (0.01, NormalConstraint::AlongAxis(0)),
        };
        Options {
            geometry: Geometry::new([voxel; 3]),
            surface: SurfaceOptions {
                taubin_iterations: 100,
                taubin_passband: passband,
                ..Default::default()
            },
            sampling: SamplingOptions {
                outside: 5.0,
                inside: 15.0,
                resolution: Some(voxel),
                smoothing: 3.0,
                normal_constraint: constraint,
                half_width_inside: false,
            },
            method: Method::regularized_for(site),
            ..Default::default()
        }
    }
}

impl Options {
    /// Check the options for values the fit cannot use.
    /// The distance used for node neighbourhoods (see
    /// [`Options::neighbour_distance`]).
    pub fn distance(&self) -> Distance {
        let default = match self.method {
            Method::Volumetric(_) => Distance::Surface,
            _ => Distance::Euclidean,
        };
        self.neighbour_distance.unwrap_or(default)
    }

    pub fn validate(&self) -> Result<(), Error> {
        let o = self;
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
                if let Some(c) = curvature.as_ref() {
                    validate_curvature(c)?;
                }
            }
            Method::Local => {}
            Method::Volumetric(v) => validate_volumetric(v, o)?,
        }
        Ok(())
    }
}
