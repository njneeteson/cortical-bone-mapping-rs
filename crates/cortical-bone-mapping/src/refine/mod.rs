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
//! tolerance. This is a damped fixed-point iteration `v ← v + α (F(v) − v)`
//! whose fixed point is a surface on which every trusted fit puts the edge at
//! the node (`m = t/2`): with `α = ½` the remaining displacement roughly halves
//! each iteration, and damping prevents the oscillation that a full step can
//! cause when the normals, and hence the profiles, change between iterations.
//! The output is the refined periosteal surface and masks of the periosteal,
//! endosteal and cortical regions.
//!
//! **Inexact inner fits.** While the surface still moves, a fit only needs
//! the edges to a fraction of the next move: the global solve stops once its
//! steps change the edges by less than `edge_ratio` × the last displacement
//! (RMS), as an inexact Newton method stops its linear solves early. The
//! initial fit and the final one are full solves: refinement has converged
//! only when a full solve, itself converged, moves the surface by less than
//! the tolerance, both overall and in every region of radius `region_radius`
//! (an RMS over the whole surface can hide a small region that still moves).
//! If the displacement stops decreasing (and no region converges) for
//! `stall_iterations` iterations, refinement stops as *stalled*, with a
//! diagnosis of the profiles that miss the cortex (`diagnosis`).
//!
//! Two options change how the surface is carried between iterations:
//!
//! * **Level set** ([`RefineOptions::level_set`]): the surface is the zero
//!   level of a signed distance ([`crate::surface::levelset`]; Osher & Sethian
//!   1988); every iteration moves the level set and extracts a new mesh
//!   (`motion`), so the node spacing stays at the grid spacing and the
//!   surface cannot fold, however far it travels.
//! * **Regional convergence** ([`RefineOptions::freeze`]): regions whose nodes
//!   have stopped moving (in position and thickness) are frozen (`state`).
//!   They are no longer refitted, except near moving regions, where they are
//!   refitted as fixed values: boundary conditions for the regularisation.
//!   The verifying full solve covers the whole surface and unfreezes what
//!   still moves.
//!
//! Next to other structures ([`refine_segmentation_excluding`]), their
//! voxels are excluded: profile samples there get zero weight and the surface
//! is never moved into them.
//!
//! The whole procedure is ours; no published method is reproduced here.
//!
//! # References
//!
//! * S. Osher, J. A. Sethian, *Fronts propagating with curvature-dependent
//!   speed: algorithms based on Hamilton–Jacobi formulations*, Journal of
//!   Computational Physics 79(1):12–49, 1988.

mod diagnosis;
mod motion;
mod options;
mod state;

pub use crate::result::cortical_masks;
pub use diagnosis::trusted_fits;
pub use options::{FreezeOptions, IterationStats, RefineOptions, RefineResult};

use std::time::Instant;

use ndarray::{Array3, ArrayView3};

use crate::error::Error;
use crate::fit::{fit_surface_constrained, FitConstraints};
use crate::model::{RHO_B, RHO_S, SIGMA, T};
use crate::options::{Method, Options, RhoCMode};
use crate::pipeline::{candidate_nodes, prepare_surface, with_threads};
use crate::result::{add_timing, region_masks, CbmResult};
use crate::surface::levelset::LevelSet;
use crate::surface::mesh::{Adjacency, Mesh};
use crate::util::{dot, norm, V3};
use crate::volume::Voxel;
use diagnosis::{max_regional_rms, stall_diagnosis};
use motion::{displacements, harmonic_fill, locked_axes, move_level_set, move_vertices, Step};
use state::{Freeze, NodeState};

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
    let inputs = Inputs {
        image,
        sub_mask,
        exclude: exclude.as_ref().map(|e| e.view()),
        options,
        refine,
    };
    with_threads(options.threads, || {
        let mut r = Refinement::start(inputs, &mask)?;
        let end = loop {
            let (step, trusted) = r.displacements();
            let measured = r.measure(&step, &trusted);
            if let Some(end) = r.should_stop(&measured, &trusted) {
                break end;
            }
            r.move_surface(&step, &measured)?;
            r.refit(&measured)?;
        };
        r.finish(end)
    })?
}

/// What refinement works on.
#[derive(Clone, Copy)]
struct Inputs<'a, T: Voxel, V: Voxel> {
    image: ArrayView3<'a, T>,
    sub_mask: Option<ArrayView3<'a, V>>,
    exclude: Option<ArrayView3<'a, bool>>,
    options: &'a Options,
    refine: &'a RefineOptions,
}

/// Tolerances and step limits resolved from the options.
struct Limits {
    alpha: f64,
    tol: f64,
    region_tol: f64,
    region_radius: f64,
    max_move: f64,
}

impl Limits {
    /// Defaults: tolerance a tenth of the smallest voxel spacing, the regional
    /// tolerance twice that over a radius of 5 voxels, and a move capped at `α`
    /// times the longer sampling half-range (a fitted edge beyond the profile is
    /// not measured).
    fn new(options: &Options, refine: &RefineOptions) -> Self {
        let geom = &options.geometry;
        let max_spacing = geom.spacing.iter().cloned().fold(0.0, f64::max);
        let tol = refine.tolerance.unwrap_or(0.1 * geom.min_spacing());
        Limits {
            alpha: refine.relaxation,
            tol,
            region_tol: refine.region_tolerance.unwrap_or(2.0 * tol),
            region_radius: refine.region_radius.unwrap_or(5.0 * max_spacing),
            max_move: refine
                .max_step
                .unwrap_or(f64::INFINITY)
                .min(refine.relaxation * options.sampling.outside.max(options.sampling.inside)),
        }
    }
}

/// The densities and blur shared by the fits of one refinement. With
/// regional convergence, partial fits must use the whole surface's `ρc` and
/// global densities. A global blur is a property of the scanner: held during
/// partial fits, carried over as a warm start otherwise.
struct Shared {
    rho_c: Option<f64>,
    densities: Option<[f64; 2]>,
    density_start: [f64; 2],
    global_sigma: bool,
    sigma_held: Option<f64>,
    sigma_start: f64,
}

impl Shared {
    /// From the first, full fit: with freezing (and a cortical density that is
    /// one value), hold `ρc` and, for the regularised method, global densities and
    /// a global `σ` at the whole surface's values, so that partial fits measure on
    /// the same scale; otherwise only warm-start them.
    fn new(first: &CbmResult, options: &Options, freezing: bool) -> Self {
        let fixed_rho = freezing && options.model.rho_c_mode != RhoCMode::ProfileMax;
        let global_method = matches!(options.method, Method::Regularized { .. });
        let global_densities = options.model.global[RHO_S] || options.model.global[RHO_B];
        let global_sigma = options.model.global[SIGMA] && global_method;
        Shared {
            rho_c: fixed_rho.then_some(first.rho_c),
            densities: (fixed_rho && global_method && global_densities).then(|| first.mean_densities()),
            density_start: first.mean_densities(),
            global_sigma,
            sigma_held: (global_sigma && freezing).then(|| first.mean_sigma()),
            sigma_start: first.mean_sigma(),
        }
    }

    /// Carry the warm starts over from a fit.
    fn update(&mut self, r: &CbmResult) {
        if self.densities.is_none() {
            self.density_start = r.mean_densities();
        }
        if self.global_sigma && self.sigma_held.is_none() {
            self.sigma_start = r.mean_sigma();
        }
    }

    /// Constraints of a fit: `full` (verifying, whole surface) or partial,
    /// with nodes `held` fixed and inner solves stopped at `edge_tol`.
    fn constraints<'a>(
        &self,
        exclude: Option<ArrayView3<'a, bool>>,
        full: bool,
        held: Option<&'a [bool]>,
        edge_tol: f64,
    ) -> FitConstraints<'a> {
        FitConstraints {
            exclude,
            fixed: held,
            rho_c: self.rho_c,
            densities: self.densities,
            density_start: Some(self.density_start),
            edge_tol: if full { 0.0 } else { edge_tol },
            sigma: if full { None } else { self.sigma_held },
            sigma_start: self.global_sigma.then_some(self.sigma_start),
        }
    }
}

/// What a fit implies, measured before the surface moves.
struct Measured {
    stats: IterationStats,
    /// Inward displacement per vertex (capped), and the outward normals.
    s: Vec<f64>,
    outward: Vec<V3>,
    /// Both the whole-surface and the regional displacement are small.
    settled: bool,
}

/// Why refinement stopped.
struct End {
    converged: bool,
    stalled: bool,
    message: String,
}

/// Progress toward the best displacement, for stall detection.
struct Progress {
    best_rms: f64,
    best_moving: usize,
    since_best: usize,
}

/// A refinement in progress: the surface, the last fit, and the state
/// carried from fit to fit.
struct Refinement<'a, T: Voxel, V: Voxel> {
    inp: Inputs<'a, T, V>,
    lim: Limits,
    tag: String,
    t0: Instant,
    clock: Timings,
    // the surface
    mesh: Mesh,
    border: Vec<bool>,
    keep: Vec<bool>,
    adj: Adjacency,
    level_set: Option<LevelSet>,
    /// Same-vertex moves: axes locked at the caps, and pinned vertices.
    locked: Vec<[bool; 3]>,
    pinned: Vec<bool>,
    freeze: Option<Freeze>,
    state: NodeState,
    // the last fit
    result: CbmResult,
    last_full: bool,
    fit_kind: String,
    shared: Shared,
    // progress
    moves: usize,
    progress: Progress,
    history: Vec<IterationStats>,
    surfaces: Vec<Mesh>,
}

impl<'a, T: Voxel, V: Voxel> Refinement<'a, T, V> {
    /// Extract the surface, set up the level set and freezing, and fit once.
    fn start<U: Voxel>(inp: Inputs<'a, T, V>, mask: &ArrayView3<U>) -> Result<Self, Error> {
        let (options, refine) = (inp.options, inp.refine);
        let t0 = Instant::now();
        let mut clock = Timings::default();
        let (mesh, border) = prepare_surface(mask, options);
        if mesh.vertices.is_empty() {
            return Err(Error::EmptyMask);
        }
        let keep = candidate_nodes(&mesh, &border, inp.sub_mask.as_ref(), options);
        clock.since("surface extraction", t0);
        let lim = Limits::new(options, refine);
        let geom = &options.geometry;
        let shape = inp.image.dim();
        let dims = [shape.0, shape.1, shape.2];
        let max_spacing = geom.spacing.iter().cloned().fold(0.0, f64::max);
        let locked = locked_axes(&mesh, geom, dims.map(|v| v as f64));
        let pinned = locked.iter().map(|l| l.iter().any(|&b| b)).collect();
        let tl = Instant::now();
        let level_set = refine.level_set.then(|| {
            let band = lim.max_move + 3.0 * max_spacing;
            initial_level_set(&mesh, dims, band, inp.exclude.as_ref(), options)
        });
        if level_set.is_some() {
            clock.since("level set: initialisation", tl);
        }
        let freeze = refine
            .freeze
            .as_ref()
            .map(|f| Freeze::new(f, options, lim.tol, max_spacing));
        let tag = if refine.label.is_empty() {
            "refine".to_string()
        } else {
            format!("refine {}", refine.label)
        };
        let tf = Instant::now();
        let first = FitConstraints {
            exclude: inp.exclude,
            ..FitConstraints::default()
        };
        let result = fit_surface_constrained(&inp.image, mesh.clone(), &keep, options, None, &first)?;
        clock.since("fits", tf);
        clock.add_fit(&result);
        let shared = Shared::new(&result, options, freeze.is_some());
        let surfaces = if refine.keep_history {
            vec![mesh.clone()]
        } else {
            Vec::new()
        };
        Ok(Refinement {
            inp,
            lim,
            tag,
            t0,
            clock,
            adj: mesh.adjacency(),
            state: NodeState::new(mesh.vertices.len()),
            mesh,
            border,
            keep,
            level_set,
            locked,
            pinned,
            freeze,
            result,
            last_full: true,
            fit_kind: "initial fit".to_string(),
            shared,
            moves: 0,
            progress: Progress {
                best_rms: f64::INFINITY,
                best_moving: usize::MAX,
                since_best: 0,
            },
            history: Vec::new(),
            surfaces,
        })
    }

    /// The displacements the last fit implies for every vertex: trusted
    /// nodes toward their fitted edge, frozen and unfitted cap vertices not
    /// at all, the rest filled in from their neighbours. Updates freezing.
    fn displacements(&mut self) -> (Step, Vec<bool>) {
        let ts = Instant::now();
        let trusted = trusted_fits(&self.result, self.inp.options);
        let mut step = displacements(
            &self.result,
            &trusted,
            self.lim.alpha,
            self.inp.refine.max_step,
            &mut self.state,
        );
        if let (true, Some(fz)) = (self.last_full, &self.freeze) {
            fz.thaw_moved(&mut self.state, &step);
        }
        for v in 0..self.mesh.vertices.len() {
            let fixed_cap = self.level_set.is_some() && self.border[v] && !step.fitted[v];
            if self.state.frozen[v] || fixed_cap {
                step.disp[v] = [0.0; 3];
                step.known[v] = true;
            }
        }
        harmonic_fill(&self.adj, &step.known, &mut step.disp, 50);
        if let Some(fz) = &self.freeze {
            fz.update(&mut self.state, &self.mesh, &step);
        }
        self.clock.since("displacements", ts);
        (step, trusted)
    }

    /// How far the move would take the surface (RMS, maximum and regional,
    /// over the fitted nodes that are not held), and the fit's statistics.
    fn measure(&mut self, step: &Step, trusted: &[bool]) -> Measured {
        let ts = Instant::now();
        let nv = self.mesh.vertices.len();
        let (state, result) = (&self.state, &self.result);
        let moving_fitted: Vec<usize> = (0..nv).filter(|&v| step.fitted[v] && !state.held[v]).collect();
        let outward = self.mesh.vertex_normals();
        let max_move = self.lim.max_move;
        let s: Vec<f64> = (0..nv)
            .map(|v| (-dot(step.disp[v], outward[v])).clamp(-max_move, max_move))
            .collect();
        let level_set = self.level_set.is_some();
        let mag = |v: usize| if level_set { s[v].abs() } else { norm(step.disp[v]) };
        let m2: Vec<f64> = moving_fitted.iter().map(|&v| mag(v).powi(2)).collect();
        let rms = (m2.iter().sum::<f64>() / moving_fitted.len().max(1) as f64).sqrt();
        let maxd = moving_fitted.iter().map(|&v| mag(v)).fold(0.0, f64::max);
        let pts: Vec<V3> = moving_fitted.iter().map(|&v| self.mesh.vertices[v]).collect();
        let regional = max_regional_rms(&pts, &m2, self.lim.region_radius);
        let fit_moving: Vec<usize> = (0..result.n())
            .filter(|&k| !state.held[result.vertex_ids[k] as usize])
            .collect();
        let nm = fit_moving.len().max(1) as f64;
        let stats = IterationStats {
            rms_displacement: rms,
            max_displacement: maxd,
            mean_thickness: fit_moving.iter().map(|&k| result.params[k][T]).sum::<f64>() / nm,
            mean_residual: fit_moving.iter().map(|&k| result.residual[k]).sum::<f64>() / nm,
            nodes: self.keep.iter().filter(|&&k| k).count(),
            moving_nodes: (0..nv).filter(|&v| self.keep[v] && !state.frozen[v]).count(),
            full_fit: self.last_full,
            fit_converged: !result.info.budget_exhausted,
            untrusted: fit_moving.iter().filter(|&&k| !trusted[k]).count(),
            max_regional_rms: regional,
        };
        self.clock.since("displacements", ts);
        self.report(&stats);
        self.history.push(stats);
        Measured {
            stats,
            s,
            outward,
            settled: rms < self.lim.tol && regional < self.lim.region_tol,
        }
    }

    /// One progress line about the last fit.
    fn report(&self, st: &IterationStats) {
        if !self.inp.refine.verbose {
            return;
        }
        let moving_note = if self.freeze.is_some() {
            format!(", {} of {} nodes moving", st.moving_nodes, st.nodes)
        } else {
            String::new()
        };
        eprintln!(
            "{}: {}: displacement RMS {:.3} max {:.3} regional {:.3} (tolerance {:.3} / {:.3}), \
             mean thickness {:.3}, {} untrusted{moving_note}, {}, {:.1} s",
            self.tag,
            self.fit_kind,
            st.rms_displacement,
            st.max_displacement,
            st.max_regional_rms,
            self.lim.tol,
            self.lim.region_tol,
            st.mean_thickness,
            st.untrusted,
            self.result.summary(),
            self.t0.elapsed().as_secs_f64()
        );
    }

    /// Converged (a full, converged fit that barely moves the surface), out
    /// of moves, or stalled (no new low of the displacement and no region
    /// converging for `stall_iterations` fits, unless the surface is already
    /// settled and only awaits its verifying fit).
    fn should_stop(&mut self, m: &Measured, trusted: &[bool]) -> Option<End> {
        let (st, lim, refine) = (&m.stats, &self.lim, self.inp.refine);
        let (rms, regional, tol) = (st.rms_displacement, st.max_regional_rms, lim.tol);
        if self.last_full && st.fit_converged && m.settled {
            let message = format!(
                "converged: a full fit moves the surface by {rms:.3} RMS (< {tol:.3}), \
                 at most {regional:.3} RMS in any region (< {:.3})",
                lim.region_tol
            );
            return Some(End {
                converged: true,
                stalled: false,
                message,
            });
        }
        if self.moves >= refine.max_iterations {
            let message = format!(
                "not converged after {} moves: the last fit moves the surface by {rms:.3} RMS (tolerance {tol:.3})",
                self.moves
            );
            return Some(End {
                converged: false,
                stalled: false,
                message,
            });
        }
        let p = &mut self.progress;
        if rms < 0.95 * p.best_rms || st.moving_nodes < p.best_moving {
            p.best_rms = p.best_rms.min(rms);
            p.best_moving = p.best_moving.min(st.moving_nodes);
            p.since_best = 0;
        } else {
            p.since_best += 1;
        }
        if refine.stall_iterations > 0 && !m.settled && p.since_best >= refine.stall_iterations {
            let (held, window) = (&self.state.held, refine.stall_iterations);
            let message = stall_diagnosis(&self.result, trusted, held, self.inp.options, rms, tol, window);
            return Some(End {
                converged: false,
                stalled: true,
                message,
            });
        }
        None
    }

    /// Move the surface: the level set (and a new mesh, with the per-node
    /// state carried over), or the same vertices (then lightly smoothed).
    fn move_surface(&mut self, step: &Step, m: &Measured) -> Result<(), Error> {
        let tm = Instant::now();
        let options = self.inp.options;
        if let Some(ls) = self.level_set.as_mut() {
            let mut stages = Vec::new();
            let (e, map) = move_level_set(ls, &self.mesh, &m.outward, &m.s, self.inp.exclude.as_ref(), &mut stages);
            for (name, t) in stages {
                add_timing(&mut self.clock.0, name, t);
            }
            if e.mesh.vertices.is_empty() {
                return Err(Error::EmptyMask);
            }
            self.state = self.state.remap(&map);
            self.mesh = e.mesh;
            self.border = e.border;
            self.adj = self.mesh.adjacency();
            self.keep = candidate_nodes(&self.mesh, &self.border, self.inp.sub_mask.as_ref(), options);
        } else {
            move_vertices(
                &mut self.mesh,
                &step.disp,
                &self.locked,
                self.inp.exclude.as_ref(),
                &options.geometry,
            );
            let iterations = self.inp.refine.smoothing_iterations;
            if iterations > 0 {
                let pinned: Vec<bool> = (0..self.pinned.len())
                    .map(|v| self.pinned[v] || self.state.frozen[v])
                    .collect();
                let s = &options.surface;
                self.mesh
                    .taubin_smooth(iterations, s.taubin_lambda, s.taubin_passband, Some(&pinned));
            }
        }
        self.clock.since("moves", tm);
        self.moves += 1;
        if self.inp.refine.keep_history {
            self.surfaces.push(self.mesh.clone());
        }
        Ok(())
    }

    /// Refit: a full solve of the whole surface to verify an apparently
    /// converged surface, otherwise an inexact solve of the moving nodes (and
    /// of the frozen ones next to them, held fixed).
    fn refit(&mut self, m: &Measured) -> Result<(), Error> {
        let refine = self.inp.refine;
        let nv = self.mesh.vertices.len();
        let keep = &self.keep;
        let n_keep = keep.iter().filter(|&&k| k).count();
        let any_moving = (0..nv).any(|v| keep[v] && !self.state.frozen[v]);
        let full = m.settled || !any_moving || refine.edge_ratio <= 0.0 && self.freeze.is_none();
        let fit_mask: Vec<bool> = match (full, &self.freeze) {
            (false, Some(fz)) => {
                self.state.held = fz.held_nodes(&self.state, &self.mesh, keep);
                (0..nv)
                    .map(|v| keep[v] && (!self.state.frozen[v] || self.state.held[v]))
                    .collect()
            }
            _ => {
                self.state.held = vec![false; nv];
                keep.clone()
            }
        };
        let held = (!full && self.freeze.is_some()).then_some(self.state.held.as_slice());
        let edge_tol = refine.edge_ratio * m.stats.rms_displacement;
        let constraints = self.shared.constraints(self.inp.exclude, full, held, edge_tol);
        let tf = Instant::now();
        let warm = Some(self.state.warm.as_slice());
        self.result = fit_surface_constrained(
            &self.inp.image,
            self.mesh.clone(),
            &fit_mask,
            self.inp.options,
            warm,
            &constraints,
        )?;
        self.clock.since("fits", tf);
        self.clock.add_fit(&self.result);
        self.shared.update(&self.result);
        self.last_full = full && self.result.n() >= n_keep.min(fit_mask.iter().filter(|&&k| k).count());
        self.fit_kind = format!(
            "iteration {}/{}{}",
            self.moves,
            refine.max_iterations,
            if full { " (full fit)" } else { "" }
        );
        Ok(())
    }

    /// A fit of the whole surface with nothing held, for the outputs (after
    /// a last fit that covered only part of it).
    fn output_fit(&mut self) -> Result<(), Error> {
        let tf = Instant::now();
        let shared = &self.shared;
        let constraints = FitConstraints {
            exclude: self.inp.exclude,
            rho_c: shared.rho_c,
            density_start: Some(shared.density_start),
            sigma_start: shared.global_sigma.then_some(shared.sigma_start),
            ..FitConstraints::default()
        };
        let warm = Some(self.state.warm.as_slice());
        let (image, options) = (&self.inp.image, self.inp.options);
        self.result = fit_surface_constrained(image, self.mesh.clone(), &self.keep, options, warm, &constraints)?;
        self.clock.since("fits", tf);
        self.clock.add_fit(&self.result);
        if self.inp.refine.verbose {
            let t = self.t0.elapsed().as_secs_f64();
            eprintln!(
                "{}: output fit on the whole surface, {}, {t:.1} s",
                self.tag,
                self.result.summary()
            );
        }
        Ok(())
    }

    /// The outputs: if the last fit covered only part of the surface, a fit
    /// of all of it; the fitted surfaces and their masks; and the timings.
    fn finish(mut self, end: End) -> Result<RefineResult, Error> {
        let (options, tag) = (self.inp.options, self.tag.clone());
        if !self.last_full && self.freeze.is_some() {
            self.output_fit()?;
        }
        let tk = Instant::now();
        let periosteal = self.result.surface.clone();
        let endosteal = self.result.displaced_surface(1.0);
        let shape = self.inp.image.dim();
        let [periosteal_mask, endosteal_mask, cortical_mask] =
            region_masks(&periosteal, &endosteal, shape, &options.geometry);
        self.clock.since("masks", tk);
        add_timing(&mut self.clock.0, "total", self.t0.elapsed().as_secs_f64());
        if self.inp.refine.verbose {
            eprintln!("{tag}: {}", end.message);
            eprintln!("{tag}: time: {}", self.clock.summary(self.t0));
        }
        Ok(RefineResult {
            result: self.result,
            periosteal,
            endosteal,
            periosteal_mask,
            endosteal_mask,
            cortical_mask,
            history: self.history,
            surfaces: self.surfaces,
            converged: end.converged,
            stalled: end.stalled,
            message: end.message,
            timings: self.clock.0,
        })
    }
}

/// The level set of the starting surface, over the image grid within reach
/// of the profiles and `band` of moves; `exclude` voxels kept outside.
fn initial_level_set(
    mesh: &Mesh,
    dims: [usize; 3],
    band: f64,
    exclude: Option<&ArrayView3<bool>>,
    options: &Options,
) -> LevelSet {
    let margin = options.sampling.outside + options.sampling.inside + band;
    let mut ls = LevelSet::from_mesh(mesh, dims, &options.geometry, margin, band);
    if let Some(e) = exclude {
        let mut changed = Array3::from_elem(ls.phi.dim(), false);
        ls.forbid(e, &mut changed);
    }
    ls
}

/// Wall time per stage of a refinement.
#[derive(Default)]
struct Timings(Vec<(String, f64)>);

impl Timings {
    fn since(&mut self, name: &str, t: Instant) {
        add_timing(&mut self.0, name, t.elapsed().as_secs_f64());
    }
    fn add_fit(&mut self, r: &CbmResult) {
        for (n, t) in &r.info.timings {
            add_timing(&mut self.0, &format!("fit: {n}"), *t);
        }
    }
    /// The stages with their share of the total since `t0`.
    fn summary(&self, t0: Instant) -> String {
        let total = t0.elapsed().as_secs_f64().max(1e-9);
        let parts: Vec<String> = self
            .0
            .iter()
            .filter(|(n, _)| n != "total")
            .map(|(n, t)| format!("{n} {t:.1} s ({:.0}%)", 100.0 * t / total))
            .collect();
        parts.join(", ")
    }
}
