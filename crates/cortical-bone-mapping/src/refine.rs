//! Model-driven segmentation refinement (extension).
//!
//! A cortical bone mapping fit locates the periosteal edge of every profile
//! at `x = m − t/2` from its surface node, independently of where the input
//! segmentation put the node. Refinement repeatedly moves the surface toward
//! that fitted edge with under-relaxation:
//!
//! ```text
//! v_i ← v_i + α (m_i − t_i/2) n_i,     0 < α ≤ 1
//! ```
//!
//! then recomputes normals, resamples the profiles and refits (warm-started,
//! with `m` shifted by the move), until the RMS displacement falls below a
//! tolerance. Under-relaxation damps the oscillation that a full step can
//! cause when the normals change between iterations. The output is the
//! refined periosteal surface and masks of the periosteal, endosteal and
//! cortical regions.
//!
//! **Inexact inner fits.** While the surface still moves, a fit only needs
//! the edges to a fraction of the next move: the global solve stops once its
//! steps change the edges by less than `edge_ratio` × the last displacement
//! (RMS). The initial fit and the final one are full solves: refinement has
//! converged only when a full solve, itself converged, moves the surface by
//! less than the tolerance. If the displacement stops decreasing (and no
//! region converges) for `stall_iterations` iterations, refinement stops as
//! *stalled*, with a diagnosis of the profiles that miss the cortex.
//!
//! Two options change how the surface is carried between iterations:
//!
//! * **Level set** ([`RefineOptions::level_set`]): the surface is the zero
//!   level of a signed distance ([`crate::levelset`]); every iteration moves
//!   the level set and extracts a new mesh, so the node spacing stays at the
//!   grid spacing and the surface cannot fold, however far it travels.
//! * **Regional convergence** ([`RefineOptions::freeze`]): regions whose nodes
//!   have stopped moving (in position and thickness) are frozen. They are no
//!   longer refitted, except near moving regions, where they are refitted as
//!   fixed values: boundary conditions for the regularisation. The work of
//!   each iteration then goes where the surface still moves. The verifying
//!   full solve covers the whole surface and unfreezes what still moves.

use std::time::Instant;

use ndarray::{Array3, ArrayView3, Zip};
use rayon::prelude::*;

use crate::kdtree::KdTree;
use crate::levelset::LevelSet;
use crate::mesh::Adjacency;
use crate::mesh::Mesh;
use crate::model::{Params, M, RHO_B, RHO_S, SIGMA, T};
use crate::pipeline::{
    candidate_nodes, fit_surface_constrained, prepare_surface, with_threads, CbmResult, Error, FitConstraints, Method,
    Options, RhoCMode,
};
use crate::util::{add, dot, norm, scale, sub, V3};
use crate::volume::{Geometry, Voxel};

#[derive(Clone, Debug, PartialEq)]
pub struct RefineOptions {
    /// Maximum number of moves.
    pub max_iterations: usize,
    /// Under-relaxation factor `α` in (0, 1].
    pub relaxation: f64,
    /// Converged when a full fit moves the surface by less than this (RMS,
    /// physical units); `None` = a tenth of the smallest voxel spacing. The
    /// displacement cannot fall much below `α` × the noise of the fitted edge
    /// position, so very small tolerances are never reached on noisy data.
    pub tolerance: Option<f64>,
    /// Taubin iterations applied after each move (keeps the mesh regular;
    /// not used with `level_set`, where the mesh is extracted anew).
    pub smoothing_iterations: usize,
    /// Cap on the displacement of a vertex per iteration (physical units).
    pub max_step: Option<f64>,
    /// Keep the surface of every iteration (for visualisation).
    pub keep_history: bool,
    /// Print one line to stderr after every fit (displacement, thickness,
    /// fit iterations, time), and the time per stage at the end.
    pub verbose: bool,
    /// Prefix of the progress lines (e.g. the bone's name).
    pub label: String,
    /// Inexact inner fits: stop the global solve once its steps change the
    /// edges by less than this × the last displacement (0 = always solve
    /// fully).
    pub edge_ratio: f64,
    /// Stop as stalled when, for this many fits, the displacement has not
    /// fallen below 95% of its best and no further region has converged.
    pub stall_iterations: usize,
    /// Regional convergence test: the displacement RMS over the nodes
    /// within this distance of each node (`None` = 5 voxels) …
    pub region_radius: Option<f64>,
    /// … must be below this everywhere (`None` = 2 × the tolerance;
    /// `f64::INFINITY` disables the test). A narrow band of nodes that still
    /// moves (e.g. a surface zipping into a concavity) barely changes the RMS
    /// over the whole surface, but stands out regionally.
    pub region_tolerance: Option<f64>,
    /// Carry the surface as a level set and extract a new mesh every
    /// iteration (remeshing) instead of moving the same vertices.
    pub level_set: bool,
    /// Regional convergence: freeze regions that have stopped moving.
    pub freeze: Option<FreezeOptions>,
}

impl Default for RefineOptions {
    fn default() -> Self {
        RefineOptions {
            max_iterations: 20,
            relaxation: 0.5,
            tolerance: None,
            smoothing_iterations: 2,
            max_step: None,
            keep_history: false,
            verbose: true,
            label: String::new(),
            edge_ratio: 0.1,
            stall_iterations: 5,
            region_radius: None,
            region_tolerance: None,
            level_set: false,
            freeze: None,
        }
    }
}

/// When a region counts as converged; see [`RefineOptions::freeze`].
#[derive(Clone, Debug, PartialEq)]
pub struct FreezeOptions {
    /// Number of consecutive iterations a node must be still.
    pub patience: usize,
    /// A node is still if it moved less than this (`None` = the refinement
    /// tolerance) …
    pub displacement: Option<f64>,
    /// … and its thickness changed by less than this (`None` = the
    /// refinement tolerance).
    pub thickness: Option<f64>,
    /// A node freezes only when every node within this distance is still
    /// (`None` = 3 voxels).
    pub radius: Option<f64>,
    /// Frozen nodes within this distance of a moving node are refitted as
    /// fixed values (`None` = the reach of the regularisation, `cutoff ·
    /// σ_reg`, or `radius` without it).
    pub halo: Option<f64>,
    /// A frozen node thaws once the moving nodes within `radius` of it have
    /// moved this far in total (the largest move nearby, summed over
    /// iterations; `None` = 1 voxel): the surface next to it is changing.
    pub thaw: Option<f64>,
}

impl Default for FreezeOptions {
    fn default() -> Self {
        FreezeOptions {
            patience: 3,
            displacement: None,
            thickness: None,
            radius: None,
            halo: None,
            thaw: None,
        }
    }
}

/// Statistics of one fit of the refinement and the move it implies.
#[derive(Clone, Copy, Debug)]
pub struct IterationStats {
    /// RMS and maximum displacement implied by the fit (over the fitted,
    /// not fixed, nodes).
    pub rms_displacement: f64,
    pub max_displacement: f64,
    pub mean_thickness: f64,
    pub mean_residual: f64,
    /// Surface nodes that may be fitted.
    pub nodes: usize,
    /// Of those, not frozen.
    pub moving_nodes: usize,
    /// A full solve on the whole surface (initial or verifying fit).
    pub full_fit: bool,
    /// The global solve converged (did not exhaust its budget).
    pub fit_converged: bool,
    /// Fitted nodes whose fit could not be trusted to locate the edge.
    pub untrusted: usize,
    /// Largest regional displacement RMS (see
    /// [`RefineOptions::region_radius`]).
    pub max_regional_rms: f64,
}

pub struct RefineResult {
    /// The final fit, on the refined surface.
    pub result: CbmResult,
    /// The refined periosteal surface (= `result.surface`).
    pub periosteal: Mesh,
    /// The fitted endosteal surface.
    pub endosteal: Mesh,
    pub periosteal_mask: Array3<bool>,
    pub endosteal_mask: Array3<bool>,
    /// `periosteal_mask & !endosteal_mask`.
    pub cortical_mask: Array3<bool>,
    /// One entry per fit: `history[k]` is the fit on `surfaces[k]`.
    pub history: Vec<IterationStats>,
    /// The input segmentation's surface and the surface after every move (if
    /// `keep_history`). With `level_set` every surface has its own faces.
    pub surfaces: Vec<Mesh>,
    /// A full solve moved the surface by less than the tolerance.
    pub converged: bool,
    /// Stopped because the surface stopped making progress (see `message`).
    pub stalled: bool,
    /// Why refinement stopped, with a diagnosis if it stalled.
    pub message: String,
    /// Wall time per stage, summed over all fits (seconds).
    pub timings: Vec<(String, f64)>,
}

struct Freeze {
    patience: u32,
    dtol: f64,
    ttol: f64,
    radius: f64,
    halo: f64,
    thaw: f64,
}

#[derive(Default)]
struct Timings(Vec<(String, f64)>);

impl Timings {
    fn add(&mut self, name: &str, t: f64) {
        match self.0.iter_mut().find(|(n, _)| n == name) {
            Some(e) => e.1 += t,
            None => self.0.push((name.to_string(), t)),
        }
    }
    fn add_fit(&mut self, r: &CbmResult) {
        for (n, t) in &r.info.timings {
            self.add(&format!("fit: {n}"), *t);
        }
    }
    fn since(&mut self, name: &str, t: Instant) {
        self.add(name, t.elapsed().as_secs_f64());
    }
}

/// Axes along which each vertex lies outside the image (cap vertices, where
/// the mask is cut by the image edge, may only slide within their cap).
fn locked_axes(mesh: &Mesh, geom: &Geometry, dims: [f64; 3]) -> Vec<[bool; 3]> {
    mesh.vertices
        .iter()
        .map(|&p| {
            let q = geom.to_index(p);
            [0, 1, 2].map(|a| q[a] < 0.0 || q[a] > dims[a] - 1.0)
        })
        .collect()
}

fn mean_densities(r: &CbmResult) -> [f64; 2] {
    let n = r.n().max(1) as f64;
    [
        r.params.iter().map(|p| p[RHO_S]).sum::<f64>() / n,
        r.params.iter().map(|p| p[RHO_B]).sum::<f64>() / n,
    ]
}

/// Refine the outer-surface segmentation `mask` of `image`.
pub fn refine_segmentation<T: Voxel, U: Voxel, V: Voxel>(
    image: ArrayView3<T>,
    mask: ArrayView3<U>,
    sub_mask: Option<ArrayView3<V>>,
    options: &Options,
    refine: &RefineOptions,
) -> Result<RefineResult, Error> {
    refine_segmentation_excluding(image, mask, sub_mask, None::<ArrayView3<u8>>, options, refine)
}

/// [`refine_segmentation`] next to other structures: profile samples in the
/// `exclude` voxels (e.g. other bones) get zero weight, and the surface is
/// never moved into them (a hard constraint: with `level_set` those voxels
/// are kept outside the surface, also at the start; otherwise a move into
/// one is shortened until it stays out).
pub fn refine_segmentation_excluding<T: Voxel, U: Voxel, V: Voxel, W: Voxel>(
    image: ArrayView3<T>,
    mask: ArrayView3<U>,
    sub_mask: Option<ArrayView3<V>>,
    exclude: Option<ArrayView3<W>>,
    options: &Options,
    refine: &RefineOptions,
) -> Result<RefineResult, Error> {
    let exclude: Option<Array3<bool>> = match exclude {
        Some(e) if e.dim() != image.dim() => {
            return Err(Error::ShapeMismatch(format!(
                "image {:?} vs exclude {:?}",
                image.dim(),
                e.dim()
            )))
        }
        Some(e) => Some(e.mapv(|v| v.to_f64() != 0.0)),
        None => None,
    };
    let exclude = exclude.as_ref().map(|e| e.view());
    let inside_excluded = |p: V3| -> bool {
        match &exclude {
            None => false,
            Some(e) => {
                let q = options.geometry.to_index(p).map(|v| v.round());
                let (d0, d1, d2) = e.dim();
                q[0] >= 0.0
                    && q[1] >= 0.0
                    && q[2] >= 0.0
                    && (q[0] as usize) < d0
                    && (q[1] as usize) < d1
                    && (q[2] as usize) < d2
                    && e[[q[0] as usize, q[1] as usize, q[2] as usize]]
            }
        }
    };
    if image.dim() != mask.dim() {
        return Err(Error::ShapeMismatch(format!(
            "image {:?} vs mask {:?}",
            image.dim(),
            mask.dim()
        )));
    }
    if !(refine.relaxation > 0.0 && refine.relaxation <= 1.0) {
        return Err(Error::InvalidOption("relaxation must be in (0, 1]".into()));
    }
    with_threads(options.threads, || {
        let t0 = Instant::now();
        let mut clock = Timings::default();
        let (mut mesh, mut border) = prepare_surface(&mask, options);
        if mesh.vertices.is_empty() {
            return Err(Error::EmptyMask);
        }
        let mut keep = candidate_nodes(&mesh, &border, sub_mask.as_ref(), options);
        clock.since("surface extraction", t0);
        let geom = &options.geometry;
        let shape = image.dim();
        let dims_u = [shape.0, shape.1, shape.2];
        let dims = dims_u.map(|v| v as f64);
        let max_spacing = geom.spacing.iter().cloned().fold(0.0, f64::max);
        let region_radius = refine.region_radius.unwrap_or(5.0 * max_spacing);
        let alpha = refine.relaxation;
        let tol = refine.tolerance.unwrap_or(0.1 * geom.min_spacing());
        let region_tol = refine.region_tolerance.unwrap_or(2.0 * tol);
        let max_move = refine
            .max_step
            .unwrap_or(f64::INFINITY)
            .min(alpha * options.sampling.outside.max(options.sampling.inside));

        // Lagrangian mode: cap vertices keep their cap plane
        let locked = locked_axes(&mesh, geom, dims);
        let pinned0: Vec<bool> = locked.iter().map(|l| l.iter().any(|&b| b)).collect();
        let tl = Instant::now();
        let mut level_set = refine.level_set.then(|| {
            let band = max_move + 3.0 * max_spacing;
            let margin = options.sampling.outside + options.sampling.inside + band;
            let mut ls = LevelSet::from_mesh(&mesh, dims_u, geom, margin, band);
            if let Some(e) = &exclude {
                let mut ch = Array3::from_elem(ls.phi.dim(), false);
                ls.forbid(e, &mut ch);
            }
            ls
        });
        if level_set.is_some() {
            clock.since("level set: initialisation", tl);
        }

        let freeze = refine.freeze.as_ref().map(|f| {
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
        });

        let mut adj = mesh.adjacency();
        let mut nv = mesh.vertices.len();
        let mut warm: Vec<Option<Params>> = vec![None; nv];
        let mut prev_t: Vec<Option<f64>> = vec![None; nv];
        let mut still = vec![0u32; nv];
        let mut frozen = vec![false; nv];
        let mut near_motion = vec![0.0f64; nv]; // movement next to a frozen node
        let mut held = vec![false; nv]; // fixed in the current fit
        let mut history: Vec<IterationStats> = Vec::new();
        let mut converged = false;
        let mut stalled = false;
        let message;
        let tag = if refine.label.is_empty() {
            "refine".to_string()
        } else {
            format!("refine {}", refine.label)
        };
        let fit_note = |r: &CbmResult| {
            let i = &r.info;
            format!(
                "{} nodes, fit {} it / {} CG{}",
                r.n(),
                i.iterations,
                i.cg_iterations,
                if i.budget_exhausted {
                    " (budget exhausted)"
                } else if i.edge_stop {
                    " (edges settled)"
                } else {
                    ""
                }
            )
        };
        let tf = Instant::now();
        let mut result = fit_surface_constrained(
            &image,
            mesh.clone(),
            &keep,
            options,
            None,
            &FitConstraints {
                exclude,
                ..FitConstraints::default()
            },
        )?;
        clock.since("fits", tf);
        clock.add_fit(&result);
        let mut last_full = true;
        let mut fit_kind = "initial fit".to_string();
        // With regional convergence, partial fits must share the densities
        // of the whole surface.
        let fixed_rho = freeze.is_some() && options.model.rho_c_mode != RhoCMode::ProfileMax;
        let rho_c = fixed_rho.then_some(result.rho_c);
        let global_method = matches!(options.method, Method::Regularized { .. });
        let densities = (fixed_rho && global_method && (options.model.global[RHO_S] || options.model.global[RHO_B]))
            .then(|| mean_densities(&result));
        let mut density_start = mean_densities(&result);
        // a global blur is a property of the scanner: held during partial
        // fits, carried over otherwise
        let mean_sigma = |r: &CbmResult| r.params.iter().map(|p| p[SIGMA]).sum::<f64>() / r.n().max(1) as f64;
        let global_sigma = options.model.global[SIGMA] && global_method;
        let sigma_held = (global_sigma && freeze.is_some()).then(|| mean_sigma(&result));
        let mut sigma_start = mean_sigma(&result);
        let mut surfaces = Vec::new();
        if refine.keep_history {
            surfaces.push(mesh.clone());
        }
        let (mut best_rms, mut best_moving, mut since_best) = (f64::INFINITY, usize::MAX, 0usize);
        let mut moves = 0;
        loop {
            // ---- displacement toward the fitted periosteal edge
            let ts = Instant::now();
            let mut disp = vec![[0.0f64; 3]; nv];
            let mut known = vec![false; nv];
            let mut fitted = vec![false; nv];
            let mut t_change: Vec<Option<f64>> = vec![None; nv];
            let mut failed = vec![false; nv];
            let trusted = trusted_fits(&result, options);
            for (k, &v) in result.vertex_ids.iter().enumerate() {
                let v = v as usize;
                fitted[v] = true;
                if held[v] {
                    known[v] = true;
                    continue;
                }
                let p = result.params[k];
                t_change[v] = prev_t[v].map(|t| (p[T] - t).abs());
                prev_t[v] = Some(p[T]);
                if !trusted[k] {
                    // filled in from trusted neighbours below; a node whose
                    // fit failed is never still
                    warm[v] = None;
                    failed[v] = true;
                    continue;
                }
                let mut d = alpha * (p[M] - 0.5 * p[T]);
                if let Some(cap) = refine.max_step {
                    d = d.clamp(-cap, cap);
                }
                disp[v] = scale(result.normals[k], d);
                known[v] = true;
                let mut w = p;
                w[M] -= d;
                warm[v] = Some(w);
            }
            if let Some(fz) = &freeze {
                if last_full {
                    // the full fit covered frozen regions too: thaw what moves
                    for v in 0..nv {
                        if frozen[v] && known[v] && norm(disp[v]) >= fz.dtol {
                            frozen[v] = false;
                            still[v] = 0;
                        }
                    }
                }
            }
            for v in 0..nv {
                if frozen[v] || (level_set.is_some() && border[v] && !fitted[v]) {
                    disp[v] = [0.0; 3];
                    known[v] = true;
                }
            }
            // extend it smoothly to unfitted vertices (harmonic fill)
            harmonic_fill(&adj, &known, &mut disp, 50);

            // ---- regional convergence
            if let Some(fz) = &freeze {
                // thaw frozen nodes whose surroundings have moved
                if frozen.iter().any(|&f| f) {
                    let movers: Vec<usize> = (0..nv).filter(|&v| !frozen[v]).collect();
                    let pts: Vec<V3> = movers.iter().map(|&v| mesh.vertices[v]).collect();
                    let tree = KdTree::new(&pts);
                    let nearby_max: Vec<f64> = (0..nv)
                        .into_par_iter()
                        .map(|v| {
                            if !frozen[v] || pts.is_empty() {
                                return 0.0;
                            }
                            let mut near = Vec::new();
                            tree.within(mesh.vertices[v], fz.radius, &mut near);
                            near.iter()
                                .map(|&(u, _)| norm(disp[movers[u as usize]]))
                                .fold(0.0, f64::max)
                        })
                        .collect();
                    for v in 0..nv {
                        if frozen[v] {
                            near_motion[v] += nearby_max[v];
                            if near_motion[v] > fz.thaw {
                                frozen[v] = false;
                                still[v] = 0;
                                near_motion[v] = 0.0;
                            }
                        }
                    }
                }
                for v in 0..nv {
                    if frozen[v] {
                        continue;
                    }
                    if failed[v] {
                        still[v] = 0;
                        continue;
                    }
                    let thick_ok = if fitted[v] && !held[v] {
                        t_change[v].is_some_and(|d| d < fz.ttol)
                    } else {
                        true
                    };
                    still[v] = if thick_ok && norm(disp[v]) < fz.dtol {
                        still[v] + 1
                    } else {
                        0
                    };
                }
                let eligible: Vec<bool> = (0..nv).map(|v| frozen[v] || still[v] >= fz.patience).collect();
                let tree = KdTree::new(&mesh.vertices);
                let newly: Vec<bool> = (0..nv)
                    .into_par_iter()
                    .map(|v| {
                        if frozen[v] || !eligible[v] {
                            return false;
                        }
                        let mut near = Vec::new();
                        tree.within(mesh.vertices[v], fz.radius, &mut near);
                        near.iter().all(|&(u, _)| eligible[u as usize])
                    })
                    .collect();
                for v in 0..nv {
                    frozen[v] |= newly[v];
                }
            }

            // ---- statistics of this fit
            let moving_fitted: Vec<usize> = (0..nv).filter(|&v| fitted[v] && !held[v]).collect();
            let outward = mesh.vertex_normals();
            let s: Vec<f64> = (0..nv)
                .map(|v| (-dot(disp[v], outward[v])).clamp(-max_move, max_move))
                .collect();
            let mag = |v: usize| if level_set.is_some() { s[v].abs() } else { norm(disp[v]) };
            let rms =
                (moving_fitted.iter().map(|&v| mag(v).powi(2)).sum::<f64>() / moving_fitted.len().max(1) as f64).sqrt();
            let maxd = moving_fitted.iter().map(|&v| mag(v)).fold(0.0, f64::max);
            let regional = {
                let pts: Vec<V3> = moving_fitted.iter().map(|&v| mesh.vertices[v]).collect();
                let m2: Vec<f64> = moving_fitted.iter().map(|&v| mag(v).powi(2)).collect();
                let tree = KdTree::new(&pts);
                (0..pts.len())
                    .into_par_iter()
                    .map(|i| {
                        let mut near = Vec::new();
                        tree.within(pts[i], region_radius, &mut near);
                        let sum: f64 = near.iter().map(|&(u, _)| m2[u as usize]).sum();
                        (sum / near.len().max(1) as f64).sqrt()
                    })
                    .reduce(|| 0.0, f64::max)
            };
            let n_keep = keep.iter().filter(|&&k| k).count();
            let n_moving = (0..nv).filter(|&v| keep[v] && !frozen[v]).count();
            let fit_moving: Vec<usize> = (0..result.n())
                .filter(|&k| !held[result.vertex_ids[k] as usize])
                .collect();
            let nm = fit_moving.len().max(1) as f64;
            let stats = IterationStats {
                rms_displacement: rms,
                max_displacement: maxd,
                mean_thickness: fit_moving.iter().map(|&k| result.params[k][T]).sum::<f64>() / nm,
                mean_residual: fit_moving.iter().map(|&k| result.residual[k]).sum::<f64>() / nm,
                nodes: n_keep,
                moving_nodes: n_moving,
                full_fit: last_full,
                fit_converged: !result.info.budget_exhausted,
                untrusted: fit_moving.iter().filter(|&&k| !trusted[k]).count(),
                max_regional_rms: regional,
            };
            clock.since("displacements", ts);
            if refine.verbose {
                let moving_note = if freeze.is_some() {
                    format!(", {n_moving} of {n_keep} nodes moving")
                } else {
                    String::new()
                };
                eprintln!(
                    "{tag}: {fit_kind}: displacement RMS {rms:.3} max {maxd:.3} regional {regional:.3} \
                     (tolerance {tol:.3} / {region_tol:.3}), mean thickness {:.3}, {} untrusted{moving_note}, {}, {:.1} s",
                    stats.mean_thickness,
                    stats.untrusted,
                    fit_note(&result),
                    t0.elapsed().as_secs_f64()
                );
            }
            history.push(stats);

            // ---- stop?
            let settled = rms < tol && regional < region_tol;
            if last_full && stats.fit_converged && settled {
                converged = true;
                message = format!(
                    "converged: a full fit moves the surface by {rms:.3} RMS (< {tol:.3}), \
                     at most {regional:.3} RMS in any region (< {region_tol:.3})"
                );
                break;
            }
            if moves >= refine.max_iterations {
                message = format!(
                    "not converged after {moves} moves: the last fit moves the surface by {rms:.3} RMS (tolerance {tol:.3})"
                );
                break;
            }
            if rms < 0.95 * best_rms || n_moving < best_moving {
                best_rms = best_rms.min(rms);
                best_moving = best_moving.min(n_moving);
                since_best = 0;
            } else {
                since_best += 1;
            }
            // a settled surface goes to its verifying full fit, not to a stall
            if refine.stall_iterations > 0 && !settled && since_best >= refine.stall_iterations {
                stalled = true;
                message = stall_diagnosis(&result, &trusted, &held, options, rms, tol, refine.stall_iterations);
                break;
            }

            // ---- move
            let tm = Instant::now();
            if let Some(ls) = level_set.as_mut() {
                let moved: Vec<V3> = (0..nv)
                    .map(|v| sub(mesh.vertices[v], scale(outward[v], s[v])))
                    .collect();
                let width = mesh.mean_edge_length().max(1e-9);
                let ta = Instant::now();
                let mut changed = ls.advance(&mesh.vertices, &s, width, 8);
                if let Some(e) = &exclude {
                    ls.forbid(e, &mut changed);
                }
                clock.since("level set: advance", ta);
                let te = Instant::now();
                let e = ls.extract();
                clock.since("level set: extraction", te);
                let tr = Instant::now();
                ls.redistance(&e.mesh, Some(&changed));
                clock.since("level set: redistancing", tr);
                if e.mesh.vertices.is_empty() {
                    return Err(Error::EmptyMask);
                }
                // carry the per-node state to the new mesh (nearest moved node)
                let tc = Instant::now();
                let tree = KdTree::new(&moved);
                let map: Vec<(usize, f64)> = e
                    .mesh
                    .vertices
                    .par_iter()
                    .map(|&p| {
                        let j = tree.knn(p, 1, None)[0].0 as usize;
                        (j, -dot(sub(p, moved[j]), outward[j]))
                    })
                    .collect();
                warm = map
                    .iter()
                    .map(|&(j, off)| {
                        warm[j].map(|mut w| {
                            w[M] -= off;
                            w
                        })
                    })
                    .collect();
                prev_t = map.iter().map(|&(j, _)| prev_t[j]).collect();
                still = map.iter().map(|&(j, _)| still[j]).collect();
                frozen = map.iter().map(|&(j, _)| frozen[j]).collect();
                near_motion = map.iter().map(|&(j, _)| near_motion[j]).collect();
                mesh = e.mesh;
                border = e.border;
                nv = mesh.vertices.len();
                adj = mesh.adjacency();
                keep = candidate_nodes(&mesh, &border, sub_mask.as_ref(), options);
                clock.since("level set: state transfer", tc);
            } else {
                for v in 0..nv {
                    let mut d = disp[v];
                    for a in 0..3 {
                        if locked[v][a] {
                            d[a] = 0.0;
                        }
                    }
                    // never into an excluded region: shorten the move
                    let mut step = d;
                    for _ in 0..6 {
                        if !inside_excluded(add(mesh.vertices[v], step)) {
                            break;
                        }
                        step = scale(step, 0.5);
                    }
                    if inside_excluded(add(mesh.vertices[v], step)) {
                        step = [0.0; 3];
                    }
                    mesh.vertices[v] = add(mesh.vertices[v], step);
                }
                if refine.smoothing_iterations > 0 {
                    let pinned: Vec<bool> = (0..nv).map(|v| pinned0[v] || frozen[v]).collect();
                    mesh.taubin_smooth(
                        refine.smoothing_iterations,
                        options.surface.taubin_lambda,
                        options.surface.taubin_passband,
                        Some(&pinned),
                    );
                }
            }
            clock.since("moves", tm);
            moves += 1;
            if refine.keep_history {
                surfaces.push(mesh.clone());
            }

            // ---- refit: a full solve of the whole surface to verify an
            // apparently converged surface, otherwise an inexact solve of
            // the moving nodes (and the frozen ones next to them)
            let n_keep = keep.iter().filter(|&&k| k).count();
            let any_moving = (0..nv).any(|v| keep[v] && !frozen[v]);
            let full = settled || !any_moving || refine.edge_ratio <= 0.0 && freeze.is_none();
            let fit_mask: Vec<bool>;
            if let (false, Some(fz)) = (full, freeze.as_ref()) {
                let active: Vec<V3> = (0..nv)
                    .filter(|&v| keep[v] && !frozen[v])
                    .map(|v| mesh.vertices[v])
                    .collect();
                let tree = KdTree::new(&active);
                held = (0..nv)
                    .into_par_iter()
                    .map(|v| {
                        keep[v]
                            && frozen[v]
                            && warm[v].is_some()
                            && tree.knn(mesh.vertices[v], 1, None)[0].1.sqrt() <= fz.halo
                    })
                    .collect();
                fit_mask = (0..nv).map(|v| keep[v] && (!frozen[v] || held[v])).collect();
            } else {
                fit_mask = keep.clone();
                held = vec![false; nv];
            }
            let constraints = FitConstraints {
                exclude,
                fixed: (!full && freeze.is_some()).then_some(held.as_slice()),
                rho_c,
                densities,
                density_start: Some(density_start),
                edge_tol: if full { 0.0 } else { refine.edge_ratio * rms },
                sigma: if full { None } else { sigma_held },
                sigma_start: global_sigma.then_some(sigma_start),
            };
            let tf = Instant::now();
            result = fit_surface_constrained(&image, mesh.clone(), &fit_mask, options, Some(&warm), &constraints)?;
            clock.since("fits", tf);
            clock.add_fit(&result);
            if densities.is_none() {
                density_start = mean_densities(&result);
            }
            if global_sigma && sigma_held.is_none() {
                sigma_start = mean_sigma(&result);
            }
            last_full = full && result.n() >= n_keep.min(fit_mask.iter().filter(|&&k| k).count());
            fit_kind = format!(
                "iteration {moves}/{}{}",
                refine.max_iterations,
                if full { " (full fit)" } else { "" }
            );
        }
        if !last_full && freeze.is_some() {
            // the last fit covered only part of the surface: fit all of it
            let tf = Instant::now();
            result = fit_surface_constrained(
                &image,
                mesh.clone(),
                &keep,
                options,
                Some(&warm),
                &FitConstraints {
                    exclude,
                    rho_c,
                    density_start: Some(density_start),
                    sigma_start: global_sigma.then_some(sigma_start),
                    ..FitConstraints::default()
                },
            )?;
            clock.since("fits", tf);
            clock.add_fit(&result);
            if refine.verbose {
                eprintln!(
                    "{tag}: output fit on the whole surface, {}, {:.1} s",
                    fit_note(&result),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        let tk = Instant::now();
        let periosteal = result.surface.clone();
        let endosteal = result.displaced_surface(1.0);
        let periosteal_mask = periosteal.voxelize(shape, &options.geometry);
        let endosteal_mask = endosteal.voxelize(shape, &options.geometry);
        let mut cortical_mask = periosteal_mask.clone();
        Zip::from(&mut cortical_mask)
            .and(&endosteal_mask)
            .par_for_each(|c, &e| *c = *c && !e);
        clock.since("masks", tk);
        clock.add("total", t0.elapsed().as_secs_f64());
        if refine.verbose {
            eprintln!("{tag}: {message}");
            let total = t0.elapsed().as_secs_f64().max(1e-9);
            let parts: Vec<String> = clock
                .0
                .iter()
                .filter(|(n, _)| n != "total")
                .map(|(n, t)| format!("{n} {t:.1} s ({:.0}%)", 100.0 * t / total))
                .collect();
            eprintln!("{tag}: time: {}", parts.join(", "));
        }
        Ok(RefineResult {
            result,
            periosteal,
            endosteal,
            periosteal_mask,
            endosteal_mask,
            cortical_mask,
            history,
            surfaces,
            converged,
            stalled,
            message,
            timings: clock.0,
        })
    })?
}

/// Why refinement stopped making progress: of the fitted nodes that are not
/// trusted (whose profiles do not show a cortex where the model can place
/// it), how many put the edge at the inner or outer end of the profile.
fn stall_diagnosis(
    result: &CbmResult,
    trusted: &[bool],
    held: &[bool],
    options: &Options,
    rms: f64,
    tol: f64,
    window: usize,
) -> String {
    let x0 = -options.sampling.outside;
    let x1 = options.sampling.inside;
    let (lo, hi) = (options.model.lower, options.model.upper);
    let med = crate::util::median(&result.residual);
    let eps = |a: f64, b: f64| 1e-6 * (b - a).abs().max(1e-12);
    // reasons (a fit may have several): inner end, outer end, m at a bound,
    // t at its upper bound, densities out of order, large residual
    let mut why = [0usize; 6];
    let (mut n, mut bad) = (0usize, 0usize);
    for k in 0..result.n() {
        if held[result.vertex_ids[k] as usize] {
            continue;
        }
        n += 1;
        if trusted[k] {
            continue;
        }
        bad += 1;
        let p = result.params[k];
        let edge = p[M] - 0.5 * p[T];
        let rc = result.rho_c_at(k);
        let flags = [
            edge >= x1 - p[SIGMA],
            edge <= x0 + p[SIGMA],
            p[M] <= lo[M] + eps(lo[M], hi[M]) || p[M] >= hi[M] - eps(lo[M], hi[M]),
            p[T] >= hi[T] - eps(lo[T], hi[T]),
            rc <= p[RHO_S] || rc <= p[RHO_B],
            med > 0.0 && result.residual[k] > 3.0 * med,
        ];
        for (w, f) in why.iter_mut().zip(flags) {
            *w += f as usize;
        }
    }
    let pc = |a: usize| 100.0 * a as f64 / n.max(1) as f64;
    let names = [
        "edge at the inner end",
        "edge at the outer end",
        "m at a bound",
        "t at its upper bound",
        "densities out of order",
        "residual > 3x median",
    ];
    let parts: Vec<String> = names
        .iter()
        .zip(why)
        .filter(|(_, c)| *c > 0)
        .map(|(nm, c)| format!("{nm} {c} ({:.1}%)", pc(c)))
        .collect();
    let mut m = format!(
        "stalled: no progress in {window} fits (displacement RMS {rms:.3}, tolerance {tol:.3}); \
         {bad} of {n} moving fitted profiles ({:.1}%) do not locate an edge: {}",
        pc(bad),
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join(", ")
        }
    );
    let (deep, shallow) = (why[0], why[1]);
    if deep > shallow && deep > 0 {
        m.push_str(&format!(
            "; the surface may lie farther inside than the profiles reach: try a longer inside sampling (now {x1})"
        ));
    } else if shallow > 0 {
        m.push_str(&format!(
            "; the surface may lie farther outside than the profiles reach: try a longer outside sampling (now {})",
            -x0
        ));
    }
    m
}

/// Whether each node's fit can be trusted to locate the periosteal edge:
/// the edge lies inside the sampled range (with a margin of the blur), `m`
/// and `t` are not on their upper bounds (a thickness at its lower bound is
/// a cortex thinner than resolvable: the profile is then a blurred step whose
/// position still locates the edge), the cortex is brighter than both
/// neighbouring tissues, and the residual is below 3× the median. Profiles
/// that miss the cortex (e.g. lines running along the flank of a gross
/// segmentation error) fail these tests; refinement then moves those vertices
/// with their trusted neighbours instead of by a meaningless fit.
pub fn trusted_fits(result: &CbmResult, options: &Options) -> Vec<bool> {
    let lo = options.model.lower;
    let hi = options.model.upper;
    let x0 = -options.sampling.outside;
    let x1 = options.sampling.inside;
    let med = crate::util::median(&result.residual);
    let eps = |a: f64, b: f64| 1e-6 * (b - a).abs().max(1e-12);
    result
        .params
        .iter()
        .zip(&result.residual)
        .enumerate()
        .map(|(i, (p, &r))| {
            let edge = p[M] - 0.5 * p[T];
            let s = p[SIGMA];
            edge > x0 + s
                && edge < x1 - s
                && p[M] > lo[M] + eps(lo[M], hi[M])
                && p[M] < hi[M] - eps(lo[M], hi[M])
                && p[T] < hi[T] - eps(lo[T], hi[T])
                && result.rho_c_at(i) > p[RHO_S]
                && result.rho_c_at(i) > p[RHO_B]
                && (!(med > 0.0) || r <= 3.0 * med)
        })
        .collect()
}

/// Jacobi iterations of neighbour averaging over the vertices without a
/// known value, so that unknown displacements interpolate the known ones.
fn harmonic_fill(adj: &Adjacency, known: &[bool], vals: &mut [V3], iterations: usize) {
    if known.iter().all(|&k| k) {
        return;
    }
    let mut next = vals.to_vec();
    for _ in 0..iterations {
        next.par_iter_mut().enumerate().for_each(|(i, o)| {
            if known[i] {
                *o = vals[i];
                return;
            }
            let ring = adj.of(i);
            if ring.is_empty() {
                return;
            }
            let mut acc = [0.0; 3];
            for &j in ring {
                acc = add(acc, vals[j as usize]);
            }
            *o = scale(acc, 1.0 / ring.len() as f64);
        });
        vals.copy_from_slice(&next);
    }
}

/// Masks `(periosteal, endosteal, cortical)` from a fit: the regions enclosed
/// by the fitted periosteal and endosteal surfaces.
pub fn cortical_masks(result: &CbmResult, shape: (usize, usize, usize), options: &Options) -> [Array3<bool>; 3] {
    let p = result.displaced_surface(-1.0).voxelize(shape, &options.geometry);
    let e = result.displaced_surface(1.0).voxelize(shape, &options.geometry);
    let mut c = p.clone();
    Zip::from(&mut c).and(&e).par_for_each(|c, &e| *c = *c && !e);
    [p, e, c]
}
