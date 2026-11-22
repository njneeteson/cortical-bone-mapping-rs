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
//!   level of a signed distance ([`crate::surface::levelset`]); every
//!   iteration moves the level set and extracts a new mesh ([`motion`]), so
//!   the node spacing stays at the grid spacing and the surface cannot fold,
//!   however far it travels.
//! * **Regional convergence** ([`RefineOptions::freeze`]): regions whose nodes
//!   have stopped moving (in position and thickness) are frozen ([`state`]).
//!   They are no longer refitted, except near moving regions, where they are
//!   refitted as fixed values: boundary conditions for the regularisation.
//!   The verifying full solve covers the whole surface and unfreezes what
//!   still moves.
//!
//! Next to other structures ([`refine_segmentation_excluding`]), their
//! voxels are excluded: profile samples there get zero weight and the surface
//! is never moved into them.

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
use crate::util::{dot, norm, V3};
use crate::volume::Voxel;
use diagnosis::{max_regional_rms, stall_diagnosis};
use motion::{displacements, harmonic_fill, locked_axes, move_level_set, move_vertices};
use state::{Freeze, NodeState};

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
        let max_spacing = geom.spacing.iter().cloned().fold(0.0, f64::max);
        let region_radius = refine.region_radius.unwrap_or(5.0 * max_spacing);
        let alpha = refine.relaxation;
        let tol = refine.tolerance.unwrap_or(0.1 * geom.min_spacing());
        let region_tol = refine.region_tolerance.unwrap_or(2.0 * tol);
        let max_move = refine
            .max_step
            .unwrap_or(f64::INFINITY)
            .min(alpha * options.sampling.outside.max(options.sampling.inside));

        // the same vertices: cap vertices keep their cap plane
        let locked = locked_axes(&mesh, geom, dims_u.map(|v| v as f64));
        let pinned0: Vec<bool> = locked.iter().map(|l| l.iter().any(|&b| b)).collect();
        // or a level set
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
        let freeze = refine
            .freeze
            .as_ref()
            .map(|f| Freeze::new(f, options, tol, max_spacing));

        let mut adj = mesh.adjacency();
        let mut state = NodeState::new(mesh.vertices.len());
        let mut history: Vec<IterationStats> = Vec::new();
        let mut converged = false;
        let mut stalled = false;
        let message;
        let tag = if refine.label.is_empty() {
            "refine".to_string()
        } else {
            format!("refine {}", refine.label)
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
        // of the whole surface. A global blur is a property of the scanner:
        // held during partial fits, carried over otherwise.
        let fixed_rho = freeze.is_some() && options.model.rho_c_mode != RhoCMode::ProfileMax;
        let rho_c = fixed_rho.then_some(result.rho_c);
        let global_method = matches!(options.method, Method::Regularized { .. });
        let densities = (fixed_rho && global_method && (options.model.global[RHO_S] || options.model.global[RHO_B]))
            .then(|| result.mean_densities());
        let mut density_start = result.mean_densities();
        let global_sigma = options.model.global[SIGMA] && global_method;
        let sigma_held = (global_sigma && freeze.is_some()).then(|| result.mean_sigma());
        let mut sigma_start = result.mean_sigma();
        let mut surfaces = Vec::new();
        if refine.keep_history {
            surfaces.push(mesh.clone());
        }
        let (mut best_rms, mut best_moving, mut since_best) = (f64::INFINITY, usize::MAX, 0usize);
        let mut moves = 0;
        loop {
            // ---- the displacements this fit implies
            let ts = Instant::now();
            let nv = mesh.vertices.len();
            let trusted = trusted_fits(&result, options);
            let mut step = displacements(&result, &trusted, alpha, refine.max_step, &mut state);
            if let (true, Some(fz)) = (last_full, &freeze) {
                fz.thaw_moved(&mut state, &step);
            }
            for v in 0..nv {
                if state.frozen[v] || (level_set.is_some() && border[v] && !step.fitted[v]) {
                    step.disp[v] = [0.0; 3];
                    step.known[v] = true;
                }
            }
            harmonic_fill(&adj, &step.known, &mut step.disp, 50);
            if let Some(fz) = &freeze {
                fz.update(&mut state, &mesh, &step);
            }

            // ---- statistics of this fit
            let moving_fitted: Vec<usize> = (0..nv).filter(|&v| step.fitted[v] && !state.held[v]).collect();
            let outward = mesh.vertex_normals();
            let s: Vec<f64> = (0..nv)
                .map(|v| (-dot(step.disp[v], outward[v])).clamp(-max_move, max_move))
                .collect();
            let mag = |v: usize| {
                if level_set.is_some() {
                    s[v].abs()
                } else {
                    norm(step.disp[v])
                }
            };
            let rms =
                (moving_fitted.iter().map(|&v| mag(v).powi(2)).sum::<f64>() / moving_fitted.len().max(1) as f64).sqrt();
            let maxd = moving_fitted.iter().map(|&v| mag(v)).fold(0.0, f64::max);
            let regional = {
                let pts: Vec<V3> = moving_fitted.iter().map(|&v| mesh.vertices[v]).collect();
                let m2: Vec<f64> = moving_fitted.iter().map(|&v| mag(v).powi(2)).collect();
                max_regional_rms(&pts, &m2, region_radius)
            };
            let n_keep = keep.iter().filter(|&&k| k).count();
            let n_moving = (0..nv).filter(|&v| keep[v] && !state.frozen[v]).count();
            let fit_moving: Vec<usize> = (0..result.n())
                .filter(|&k| !state.held[result.vertex_ids[k] as usize])
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
                    result.summary(),
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
                message = stall_diagnosis(
                    &result,
                    &trusted,
                    &state.held,
                    options,
                    rms,
                    tol,
                    refine.stall_iterations,
                );
                break;
            }

            // ---- move
            let tm = Instant::now();
            if let Some(ls) = level_set.as_mut() {
                let mut stages = Vec::new();
                let (e, map) = move_level_set(ls, &mesh, &outward, &s, exclude.as_ref(), &mut stages);
                for (name, t) in stages {
                    add_timing(&mut clock.0, name, t);
                }
                if e.mesh.vertices.is_empty() {
                    return Err(Error::EmptyMask);
                }
                state = state.remap(&map);
                mesh = e.mesh;
                border = e.border;
                adj = mesh.adjacency();
                keep = candidate_nodes(&mesh, &border, sub_mask.as_ref(), options);
            } else {
                move_vertices(&mut mesh, &step.disp, &locked, exclude.as_ref(), geom);
                if refine.smoothing_iterations > 0 {
                    let pinned: Vec<bool> = (0..nv).map(|v| pinned0[v] || state.frozen[v]).collect();
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
            let nv = mesh.vertices.len();
            let n_keep = keep.iter().filter(|&&k| k).count();
            let any_moving = (0..nv).any(|v| keep[v] && !state.frozen[v]);
            let full = settled || !any_moving || refine.edge_ratio <= 0.0 && freeze.is_none();
            let fit_mask: Vec<bool>;
            if let (false, Some(fz)) = (full, freeze.as_ref()) {
                state.held = fz.held_nodes(&state, &mesh, &keep);
                fit_mask = (0..nv)
                    .map(|v| keep[v] && (!state.frozen[v] || state.held[v]))
                    .collect();
            } else {
                fit_mask = keep.clone();
                state.held = vec![false; nv];
            }
            let constraints = FitConstraints {
                exclude,
                fixed: (!full && freeze.is_some()).then_some(state.held.as_slice()),
                rho_c,
                densities,
                density_start: Some(density_start),
                edge_tol: if full { 0.0 } else { refine.edge_ratio * rms },
                sigma: if full { None } else { sigma_held },
                sigma_start: global_sigma.then_some(sigma_start),
            };
            let tf = Instant::now();
            result = fit_surface_constrained(
                &image,
                mesh.clone(),
                &fit_mask,
                options,
                Some(&state.warm),
                &constraints,
            )?;
            clock.since("fits", tf);
            clock.add_fit(&result);
            if densities.is_none() {
                density_start = result.mean_densities();
            }
            if global_sigma && sigma_held.is_none() {
                sigma_start = result.mean_sigma();
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
                Some(&state.warm),
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
                    result.summary(),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        let tk = Instant::now();
        let periosteal = result.surface.clone();
        let endosteal = result.displaced_surface(1.0);
        let [periosteal_mask, endosteal_mask, cortical_mask] = region_masks(&periosteal, &endosteal, shape, geom);
        clock.since("masks", tk);
        add_timing(&mut clock.0, "total", t0.elapsed().as_secs_f64());
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
