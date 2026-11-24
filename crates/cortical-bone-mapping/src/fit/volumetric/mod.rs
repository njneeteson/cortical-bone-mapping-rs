//! The volumetric model (extension): the profile model evaluated at every
//! voxel of a band around the surface (`docs/volumetric_model.md`).
//!
//! The 1D methods sample one interpolated profile per node along a normal
//! taken from the mesh. Here every voxel within `[−outside, inside]` of the
//! fixed surface is a sample, placed by its signed distance and owning the
//! parameters interpolated from the vertices of its closest face (`band`).
//! So the samples are real voxels, each used once, the normal of a sample is
//! the gradient of the distance at the sample (different on either side of a
//! corner), and the blur is one scanner PSF, `σ_n² = σ_xy² (1 − n_a²) +
//! σ_z² n_a²` for a normal with component `n_a` along the scanner axis,
//! instead of a fitted `σ` per node. That is the in-plane / out-of-plane
//! blur of Treece & Gee (2015, appendix A), which they regress on the blurs
//! of per-location 1D fits; here `σ_xy` and `σ_z` are fitted with the
//! model.
//!
//! The fit (`fit_surface`) runs:
//!
//! 1. the band, from the fixed surface;
//! 2. one model for all band voxels, every parameter global (as the 1D
//!    methods' initialisation; skipped when refinement passes a PSF to start
//!    from);
//! 3. the fields released: `m`, `t` per node with the curvature penalty on
//!    the fitted surfaces (they define the periosteal and endosteal
//!    surfaces), `ρs`, `ρb` per node with the spatial smoothing `ω`, and the
//!    PSF global (`problem`, on the shared Gauss–Newton/PCG solver);
//! 4. with `ρc` estimated, the cortical plateau from the voxels well inside
//!    the fitted cortex, and one refit;
//! 5. the reliability outputs (`reliability`).
//!
//! The surface is fixed during a fit; refinement moves it by the fitted
//! `m − t/2`, as for the 1D methods. Neighbourhoods default to distance along
//! the surface ([`crate::operators::Distance::Surface`]).
//!
//! # References
//!
//! * G. M. Treece, A. H. Gee, *Independent measurement of femoral cortical
//!   thickness and cortical bone density using clinical CT*, Medical Image
//!   Analysis 20(1):249–264, 2015.

pub(crate) mod band;
pub(crate) mod problem;
mod reach;
mod reliability;
#[cfg(test)]
mod tests;

use std::time::Instant;

use ndarray::ArrayView3;

use self::band::{Band, BandSpec};
use self::problem::{Prepared, VolumetricProblem, M, NV, RHO_B, RHO_S, SXY, SZ, T};
use self::reach::{node_reach, uniform_reach, Reach, ReachSpec};
use super::global::{minimise, Role};
use super::profiles::{sampling_normals, NormalConstraint};
use super::regularized::curvature_operator;
use super::{global_settings, held_subset, FitConstraints, Nodes};
use crate::error::Error;
use crate::model::{Params, M as PM, RHO_B as PRB, RHO_S as PRS, SIGMA as PS, T as PT};
use crate::operators::{CurvatureOperator, Neighbours, SmoothingOperator};
use crate::options::{AxialBlur, Options, RhoCMode, VolumetricOptions};
use crate::result::{CbmResult, FitInfo, VolumetricInfo};
use crate::surface::mesh::Mesh;
use crate::util::{dot, quantile, scale, V3};
use crate::volume::{Geometry, Voxel};

/// Fit the volumetric model on the nodes `keep` of an (already smoothed)
/// `surface`; see the module documentation. `warm_start` holds initial
/// parameters per surface vertex.
pub(crate) fn fit_surface<T: Voxel>(
    image: &ArrayView3<T>,
    surface: Mesh,
    keep: &[bool],
    options: &Options,
    v: &VolumetricOptions,
    warm_start: Option<&[Option<Params>]>,
    c: &FitConstraints,
) -> Result<CbmResult, Error> {
    let t0 = Instant::now();
    let mut timings = Vec::new();
    let (d0, d1, d2) = image.dim();
    let cut = ImageEdge::new([d0, d1, d2], &options.geometry);
    let nodes = FitNodes::new(&surface, keep, options.sampling.normal_constraint, &cut)?;
    let reach = band_reach(image, &nodes, options, v);
    let mut spec = band_spec(options, v, c.exclude);
    if v.adaptive_band.is_some() {
        spec.face_reach = Some(face_reach(&nodes.mesh, &reach));
    }
    let band = Band::build(image, &surface, &nodes.fitted, nodes.mesh.faces.len(), &spec);
    if band.is_empty() {
        return Err(Error::NoNodes);
    }
    timings.push(("band".to_string(), t0.elapsed().as_secs_f64()));
    let held = held_subset(c.fixed, warm_start, &nodes.ids)?;
    let fit = Fit::new(&band, &nodes, options, v, *c, held.as_deref());
    let mut info = FitInfo {
        method: "volumetric".to_string(),
        ..Default::default()
    };
    let mut rho_c = fit.initial_rho_c();
    let tg = Instant::now();
    let global = fit.global_stage(rho_c, &mut info);
    timings.push(("global fit".to_string(), tg.elapsed().as_secs_f64()));
    let tf = Instant::now();
    let starts = fit.node_starts(&global, warm_start);
    let mut x = fit.field_stage(rho_c, &global, &starts, &mut info);
    if c.rho_c.is_none() && options.model.rho_c_mode == RhoCMode::Estimate {
        if let Some(rc) = fit.plateau(rho_c, &global, &x) {
            let refit = fit.field_starts_from(rho_c, &global, &x);
            rho_c = rc;
            x = fit.field_stage(rho_c, &global, &refit, &mut info);
        }
    }
    timings.push(("volumetric fit".to_string(), tf.elapsed().as_secs_f64()));
    let tr = Instant::now();
    let out = fit.outputs(rho_c, &global, &x);
    timings.push(("reliability".to_string(), tr.elapsed().as_secs_f64()));
    timings.append(&mut info.timings);
    info.timings = timings;
    Ok(CbmResult {
        vertex_ids: nodes.ids.clone(),
        points: nodes.mesh.vertices.clone(),
        faces: nodes.mesh.faces.clone(),
        normals: nodes.directions.clone(),
        params: out.params,
        residual: out.residual,
        rho_c,
        rho_c_node: None,
        global_params: global_params(&global),
        psf: Some(psf_per_axis(global_psf(&out.scalars), v.scanner_axis)),
        info,
        profiles: None,
        volumetric: Some(VolumetricInfo {
            reach: reach.outside.iter().zip(&reach.inside).map(|(&o, &i)| [o, i]).collect(),
            ..out.info
        }),
        surface,
    })
}

/// The fitted nodes: the sub-mesh of `keep`, the surface vertex of each
/// node, its inward normal and reporting direction, and the fitted-face
/// number of every surface face (`None` unless all its vertices are
/// fitted).
///
/// The model measures `m` and `t` along the surface normal (each voxel's
/// direction to its closest point); the curvature penalty and the axial
/// part of the PSF use the node's inward normal `n`. The results report
/// them along the direction `d` of the sampling's normal constraint, as the
/// 1D methods' profiles are: for a planar cortex, an edge at distance `x`
/// along `n` lies at `x / (n · d)` along `d`, so reported values are the
/// model's divided by `n · d` (and warm starts, which come in reported
/// values, multiplied by it). Without a constraint `d = n`. On a shaft the
/// in-plane constraint also keeps the reported edges from folding where the
/// surface is rounded off at the image's cut ends, whose normals tilt
/// towards the axis. That tilt is the rounding, not anatomy, so within
/// [`ImageEdge::MARGIN`] voxels of the image's edge no conversion is made
/// (`n · d` taken as 1); elsewhere it is clamped at [`MIN_COS`].
struct FitNodes {
    mesh: Mesh,
    ids: Vec<u32>,
    normals: Vec<V3>,
    directions: Vec<V3>,
    /// `n · d`, at least [`MIN_COS`].
    cos: Vec<f64>,
    fitted: Vec<Option<u32>>,
}

/// The smallest `n · d` used to convert between the model's and the
/// reported distances (60°).
const MIN_COS: f64 = 0.5;

/// The edge of the image, where a bone that leaves it is cut.
struct ImageEdge<'a> {
    dims: [usize; 3],
    geom: &'a Geometry,
}

impl<'a> ImageEdge<'a> {
    /// Voxels from the edge within which the surface is the cut's rounding.
    const MARGIN: f64 = 2.0;

    fn new(dims: [usize; 3], geom: &'a Geometry) -> Self {
        ImageEdge { dims, geom }
    }

    /// Whether physical point `p` is within [`Self::MARGIN`] voxels of the
    /// image's edge.
    fn near(&self, p: V3) -> bool {
        let q = self.geom.to_index(p);
        (0..3).any(|a| q[a] < Self::MARGIN || q[a] > self.dims[a] as f64 - 1.0 - Self::MARGIN)
    }
}

impl FitNodes {
    fn new(surface: &Mesh, keep: &[bool], constraint: NormalConstraint, cut: &ImageEdge) -> Result<Self, Error> {
        let outward = surface.vertex_normals();
        let dirs = sampling_normals(&outward, constraint);
        let keep: Vec<bool> = keep.iter().zip(&dirs).map(|(&k, d)| k && d.is_some()).collect();
        let (mesh, ids) = surface.submesh(&keep);
        if ids.is_empty() || mesh.faces.is_empty() {
            return Err(Error::NoNodes);
        }
        let mut next = 0u32;
        let fitted = surface
            .faces
            .iter()
            .map(|f| {
                f.iter().all(|&v| keep[v as usize]).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        let normals: Vec<V3> = ids.iter().map(|&v| scale(outward[v as usize], -1.0)).collect();
        let directions: Vec<V3> = ids.iter().map(|&v| dirs[v as usize].unwrap()).collect();
        let cos = ids
            .iter()
            .zip(normals.iter().zip(&directions))
            .map(|(&v, (n, d))| {
                if cut.near(surface.vertices[v as usize]) {
                    1.0
                } else {
                    dot(*n, *d).max(MIN_COS)
                }
            })
            .collect();
        Ok(FitNodes {
            mesh,
            ids,
            normals,
            directions,
            cos,
            fitted,
        })
    }
}

/// Each node's reach: adaptive ([`reach`]) with
/// [`VolumetricOptions::adaptive_band`], else the sampling range.
fn band_reach<T: Voxel>(image: &ArrayView3<T>, nodes: &FitNodes, o: &Options, v: &VolumetricOptions) -> Reach {
    let s = &o.sampling;
    match v.adaptive_band {
        Some(cap) => {
            let points: Vec<V3> = nodes.mesh.vertices.clone();
            let spec = ReachSpec {
                geom: &o.geometry,
                outside: s.outside,
                inside: s.inside,
                cap,
            };
            node_reach(image, &points, &nodes.normals, &spec)
        }
        None => uniform_reach(nodes.ids.len(), s.outside, s.inside),
    }
}

/// The reach of every fitted face: the largest of its nodes'.
fn face_reach(mesh: &Mesh, reach: &Reach) -> Vec<[f64; 2]> {
    mesh.faces
        .iter()
        .map(|f| {
            let pick = |r: &[f64]| f.iter().map(|&v| r[v as usize]).fold(0.0, f64::max);
            [pick(&reach.outside), pick(&reach.inside)]
        })
        .collect()
}

/// The band's settings from the options.
fn band_spec<'a>(o: &'a Options, v: &VolumetricOptions, exclude: Option<ArrayView3<'a, bool>>) -> BandSpec<'a> {
    BandSpec {
        geom: &o.geometry,
        outside: o.sampling.outside,
        inside: o.sampling.inside,
        boost: o.model.residual_boost,
        axis: v.scanner_axis,
        exclude,
        stride: v.voxel_stride,
        face_reach: None,
    }
}

/// What the stages of one fit share.
struct Fit<'a, 'c> {
    band: &'a Band,
    nodes: &'a FitNodes,
    o: &'a Options,
    v: &'a VolumetricOptions,
    c: FitConstraints<'c>,
    held: Option<&'a [bool]>,
    neighbours: Neighbours,
    /// The smoothing operator of the density fields (if they are fields) and
    /// of the edge fields (with `edge_lambda`).
    smoothing: Option<SmoothingOperator>,
    lower: [f64; NV],
    upper: [f64; NV],
}

/// The fitted values at the nodes, and the reliability outputs.
struct Outputs {
    params: Vec<Params>,
    residual: Vec<f64>,
    scalars: [f64; NV],
    info: VolumetricInfo,
}

impl<'a, 'c> Fit<'a, 'c> {
    fn new(
        band: &'a Band,
        nodes: &'a FitNodes,
        o: &'a Options,
        v: &'a VolumetricOptions,
        c: FitConstraints<'c>,
        held: Option<&'a [bool]>,
    ) -> Self {
        let neighbours = Neighbours::new(&nodes.mesh, o.distance());
        let density_fields = !v.global_densities && c.densities.is_none();
        let smoothing = (density_fields && v.lambda > 0.0 || v.edge_lambda > 0.0)
            .then(|| SmoothingOperator::new(&neighbours, v.neighbours, v.sigma, v.cutoff));
        let (lo, hi) = (o.model.lower, o.model.upper);
        Fit {
            band,
            nodes,
            o,
            v,
            c,
            held,
            neighbours,
            smoothing,
            lower: [lo[PM], lo[PT], lo[PRS], lo[PRB], lo[PS], lo[PS]],
            upper: [hi[PM], hi[PT], hi[PRS], hi[PRB], hi[PS], hi[PS]],
        }
    }

    /// The cortical density to start from: given, fixed, or (to be
    /// estimated) the 90th percentile of the band voxels inside the surface.
    fn initial_rho_c(&self) -> f64 {
        if let Some(rc) = self.c.rho_c {
            return rc;
        }
        match self.o.model.rho_c_mode {
            RhoCMode::Estimate => {
                let inside: Vec<f64> = (0..self.band.len())
                    .filter(|&e| self.band.depth[e] >= 0.0)
                    .map(|e| self.band.value[e] as f64)
                    .collect();
                quantile(inside, 0.9)
            }
            _ => self.o.model.rho_c,
        }
    }

    /// Roles of the parameters: `fields` for the field stage, all global
    /// for the initialisation; held densities and PSF are fixed. Also
    /// whether `σ_z` is tied to `σ_xy`.
    fn roles(&self, fields: bool) -> ([Role; NV], bool) {
        let node = if fields { Role::Field } else { Role::Global };
        let density = |global: bool| {
            if self.c.densities.is_some() {
                Role::Fixed
            } else if global {
                Role::Global
            } else {
                node
            }
        };
        let rho_s = density(self.v.global_densities || self.v.global_soft_tissue);
        let rho_b = density(self.v.global_densities);
        let held_psf = self.c.psf.is_some();
        let sxy = if held_psf || self.v.sigma_xy.is_some() {
            Role::Fixed
        } else {
            Role::Global
        };
        let (sz, tied) = match (held_psf, self.v.sigma_z) {
            (true, _) | (false, AxialBlur::Fixed(_)) => (Role::Fixed, false),
            (false, AxialBlur::Tied) => (Role::Global, true),
            (false, AxialBlur::Fit) => (Role::Global, false),
        };
        ([node, node, rho_s, rho_b, sxy, sz], tied)
    }

    /// `scalars` with the fixed parameters (held densities and PSF, given
    /// widths) set.
    fn fixed_values(&self, scalars: &[f64; NV]) -> [f64; NV] {
        let mut f = *scalars;
        if let Some([rs, rb]) = self.c.densities {
            f[RHO_S] = rs;
            f[RHO_B] = rb;
        }
        if let Some(s) = self.v.sigma_xy {
            f[SXY] = s;
        }
        if let AxialBlur::Fixed(s) = self.v.sigma_z {
            f[SZ] = s;
        }
        if let Some([sxy, sz]) = self.c.psf {
            f[SXY] = sxy;
            f[SZ] = sz;
        }
        f
    }

    /// The problem of a stage (`fields` or the all-global initialisation).
    fn problem<'p>(
        &'p self,
        fields: bool,
        rho_c: f64,
        scalars: &[f64; NV],
        curvature: Option<&'p CurvatureOperator>,
    ) -> VolumetricProblem<'p> {
        let (roles, tied) = self.roles(fields);
        // the regularised method's weights: λ/N on the densities (relative to
        // ρc), λ ρc² / (N t0²) on m and t (relative to the global thickness)
        let nn = self.nodes.ids.len() as f64;
        let kappa = self.v.lambda / nn;
        let t0 = scalars[T].max(1e-6);
        let kappa_edges = self.v.edge_lambda * rho_c * rho_c / (nn * t0 * t0);
        VolumetricProblem {
            band: self.band,
            faces: &self.nodes.mesh.faces,
            nodes: self.nodes.ids.len(),
            rho_c,
            roles,
            tied,
            fixed: self.fixed_values(scalars),
            frozen: if fields { self.held } else { None },
            smoothing: if fields {
                self.smoothing
                    .as_ref()
                    .map(|op| (op, [kappa_edges, kappa_edges, kappa, kappa]))
            } else {
                None
            },
            curvature: if fields { curvature } else { None },
            lower: self.lower,
            upper: self.upper,
        }
    }

    /// The initial scalars: the options' initial model, with warm-started
    /// densities and PSF.
    fn initial_scalars(&self) -> [f64; NV] {
        let i = self.o.model.init;
        let mut s = [i[PM], i[PT], i[PRS], i[PRB], i[PS], i[PS]];
        if let Some([rs, rb]) = self.c.density_start {
            s[RHO_S] = rs;
            s[RHO_B] = rb;
        }
        if let Some([sxy, sz]) = self.c.psf_start {
            s[SXY] = sxy;
            s[SZ] = sz;
        }
        self.fixed_values(&s)
    }

    /// Stage 2: one model for all band voxels (unless not wanted, or the PSF
    /// comes from a previous fit).
    fn global_stage(&self, rho_c: f64, info: &mut FitInfo) -> [f64; NV] {
        let start = self.initial_scalars();
        if !self.o.init_from_global || self.c.psf_start.is_some() {
            return start;
        }
        let pr = self.problem(false, rho_c, &start, None);
        let prep = Prepared::new(&pr);
        let mut x: Vec<f64> = prep.lay.globals.iter().map(|&c| start[c]).collect();
        let si = minimise(&prep, &mut x, &global_settings(self.o));
        info.add_solve(&si);
        let mut out = pr.fixed;
        for (gi, &c) in prep.lay.globals.iter().enumerate() {
            out[c] = x[gi];
        }
        if pr.tied {
            out[SZ] = out[SXY];
        }
        out
    }

    /// The starting node parameters `[m, t, ρs, ρb]`: warm starts, else the
    /// global model.
    fn node_starts(&self, global: &[f64; NV], warm: Option<&[Option<Params>]>) -> Vec<[f64; 4]> {
        self.nodes
            .ids
            .iter()
            .zip(&self.nodes.cos)
            .map(|(&v, &c)| match warm.and_then(|w| w[v as usize]) {
                // warm starts are in reported distances (along `d`)
                Some(p) => [p[PM] * c, p[PT] * c, p[PRS], p[PRB]],
                None => [global[M], global[T], global[RHO_S], global[RHO_B]],
            })
            .collect()
    }

    /// Node starts from a previous field fit `x`.
    fn field_starts_from(&self, rho_c: f64, global: &[f64; NV], x: &[f64]) -> Vec<[f64; 4]> {
        let pr = self.problem(true, rho_c, global, None);
        let values = Prepared::new(&pr).node_values(x);
        (0..self.nodes.ids.len())
            .map(|v| [values[M][v], values[T][v], values[RHO_S][v], values[RHO_B][v]])
            .collect()
    }

    /// The curvature penalty on the fitted surfaces (lengths default to the
    /// global thickness).
    fn curvature(&self, rho_c: f64, global: &[f64; NV]) -> Option<CurvatureOperator> {
        let nodes = Nodes {
            mesh: &self.nodes.mesh,
            normals: &self.nodes.normals,
        };
        self.v
            .curvature
            .as_ref()
            .map(|cv| curvature_operator(cv, &nodes, &self.neighbours, rho_c, global[T].max(1e-6)))
    }

    /// Stage 3: the field fit from `starts`; returns the variables.
    fn field_stage(&self, rho_c: f64, global: &[f64; NV], starts: &[[f64; 4]], info: &mut FitInfo) -> Vec<f64> {
        let curv = self.curvature(rho_c, global);
        let pr = self.problem(true, rho_c, global, curv.as_ref());
        let prep = Prepared::new(&pr);
        let mut x = Vec::with_capacity(prep.lay.fields.len() * starts.len() + prep.lay.globals.len());
        for &c in &prep.lay.fields {
            x.extend(starts.iter().map(|s| s[c]));
        }
        x.extend(prep.lay.globals.iter().map(|&c| global[c]));
        let settings = super::global::Settings {
            edge_tol: self.c.edge_tol,
            ..global_settings(self.o)
        };
        let si = minimise(&prep, &mut x, &settings);
        info.add_solve(&si);
        x
    }

    /// Stage 4: the cortical plateau (`ρc` estimate) from the field fit `x`:
    /// per node the median of its voxels well inside the fitted cortex (at
    /// least 3), and the median over nodes (at least 10).
    fn plateau(&self, rho_c: f64, global: &[f64; NV], x: &[f64]) -> Option<f64> {
        let pr = self.problem(true, rho_c, global, None);
        let mut samples = Prepared::new(&pr).cortical_samples(x);
        samples.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let medians: Vec<f64> = samples
            .chunk_by(|a, b| a.0 == b.0)
            .filter(|g| g.len() >= 3)
            .map(|g| g[g.len() / 2].1 as f64)
            .collect();
        (medians.len() >= 10).then(|| quantile(medians, 0.5))
    }

    /// Stage 5: node parameters, residuals and reliability at `x` (only the
    /// residuals for an inexact inner fit, `edge_tol > 0`).
    fn outputs(&self, rho_c: f64, global: &[f64; NV], x: &[f64]) -> Outputs {
        let curv = self.curvature(rho_c, global);
        let pr = self.problem(true, rho_c, global, curv.as_ref());
        let prep = Prepared::new(&pr);
        let values = prep.node_values(x);
        let scalars: [f64; NV] = std::array::from_fn(|c| values[c].first().copied().unwrap_or(global[c]));
        let axis = self.v.scanner_axis;
        let params = (0..self.nodes.ids.len())
            .map(|v| {
                let a2 = self.nodes.normals[v][axis].powi(2);
                let sn = (scalars[SXY].powi(2) * (1.0 - a2) + scalars[SZ].powi(2) * a2).sqrt();
                // reported along `d` (FitNodes)
                let c = self.nodes.cos[v];
                [
                    values[M][v] / c,
                    values[T][v] / c,
                    sn,
                    values[RHO_S][v],
                    values[RHO_B][v],
                ]
            })
            .collect();
        // inexact inner fits of a refinement only need the residuals
        let rel = if self.c.edge_tol > 0.0 {
            reliability::residuals_only(&prep, x, global_psf(&scalars), axis)
        } else {
            reliability::assess(&prep, x, global_psf(&scalars), axis)
        };
        Outputs {
            params,
            residual: rel.residual,
            scalars,
            info: rel.info,
        }
    }
}

/// The PSF `(σ_xy, σ_z)` of a set of scalars.
fn global_psf(s: &[f64; NV]) -> [f64; 2] {
    [s[SXY], s[SZ]]
}

/// The PSF as standard deviations per array axis.
fn psf_per_axis([sxy, sz]: [f64; 2], axis: usize) -> [f64; 3] {
    let mut p = [sxy; 3];
    p[axis] = sz;
    p
}

/// The global model in the 1D parameter order (`σ` = the in-plane blur).
fn global_params(s: &[f64; NV]) -> Params {
    [s[M], s[T], s[SXY], s[RHO_S], s[RHO_B]]
}
