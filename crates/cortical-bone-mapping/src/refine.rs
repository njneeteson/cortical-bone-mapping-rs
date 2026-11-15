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
//! then lightly re-smooths the mesh, recomputes normals, resamples the
//! profiles and refits (warm-started, with `m` shifted by the move), until
//! the RMS displacement falls below a tolerance. Under-relaxation damps the
//! oscillation that a full step can cause when the normals change between
//! iterations. The output is the refined periosteal surface and masks of the
//! periosteal, endosteal and cortical regions.

use ndarray::{Array3, ArrayView3, Zip};

use crate::mesh::Adjacency;
use crate::mesh::Mesh;
use crate::model::{Params, M, T};
use crate::pipeline::{candidate_nodes, fit_surface, prepare_surface, with_threads, CbmResult, Error, Options};
use crate::util::{add, norm, scale, sub, V3};
use crate::volume::Voxel;

#[derive(Clone, Debug, PartialEq)]
pub struct RefineOptions {
    pub max_iterations: usize,
    /// Under-relaxation factor `α` in (0, 1].
    pub relaxation: f64,
    /// Stop when the RMS vertex displacement of an iteration is below this
    /// (physical units); `None` = a tenth of the smallest voxel spacing. The
    /// displacement cannot fall much below `α` × the noise of the fitted edge
    /// position, so very small tolerances are never reached on noisy data.
    pub tolerance: Option<f64>,
    /// Taubin iterations applied after each move (keeps the mesh regular).
    pub smoothing_iterations: usize,
    /// Cap on the displacement of a vertex per iteration (physical units).
    pub max_step: Option<f64>,
    /// Keep the surface of every iteration (for visualisation).
    pub keep_history: bool,
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
        }
    }
}

/// Statistics of one refinement iteration.
#[derive(Clone, Copy, Debug)]
pub struct IterationStats {
    pub rms_displacement: f64,
    pub max_displacement: f64,
    pub mean_thickness: f64,
    pub mean_residual: f64,
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
    pub history: Vec<IterationStats>,
    /// Surfaces before the first and after every iteration (if
    /// `keep_history`); index 0 is the input segmentation's surface.
    pub surfaces: Vec<Mesh>,
    pub converged: bool,
}

/// Refine the outer-surface segmentation `mask` of `image`.
pub fn refine_segmentation<T: Voxel, U: Voxel, V: Voxel>(
    image: ArrayView3<T>,
    mask: ArrayView3<U>,
    sub_mask: Option<ArrayView3<V>>,
    options: &Options,
    refine: &RefineOptions,
) -> Result<RefineResult, Error> {
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
        let (mut mesh, border) = prepare_surface(&mask, options);
        if mesh.vertices.is_empty() {
            return Err(Error::EmptyMask);
        }
        let keep = candidate_nodes(&mesh, &border, sub_mask.as_ref(), options);
        let geom = &options.geometry;
        let dims = image.dim();
        let dims = [dims.0 as f64, dims.1 as f64, dims.2 as f64];
        // Cap vertices (where the mask is cut by the image edge) may only
        // slide within their cap plane: lock the axes along which they lie
        // outside the image.
        let locked: Vec<[bool; 3]> = mesh
            .vertices
            .iter()
            .map(|&p| {
                let q = geom.to_index(p);
                [0, 1, 2].map(|a| q[a] < 0.0 || q[a] > dims[a] - 1.0)
            })
            .collect();
        let pinned: Vec<bool> = locked.iter().map(|l| l.iter().any(|&b| b)).collect();
        let adj = mesh.adjacency();
        let nv = mesh.vertices.len();
        let mut warm: Vec<Option<Params>> = vec![None; nv];
        let mut history = Vec::new();
        let mut converged = false;
        let alpha = refine.relaxation;
        let tol = refine.tolerance.unwrap_or(0.1 * geom.min_spacing());
        let mut result = fit_surface(&image, mesh.clone(), &keep, options, None)?;
        let mut surfaces = Vec::new();
        if refine.keep_history {
            surfaces.push(mesh.clone());
        }
        for _ in 0..refine.max_iterations {
            // displacement toward the fitted periosteal edge at fitted nodes
            let mut disp = vec![[0.0f64; 3]; nv];
            let mut known = vec![false; nv];
            let trusted = trusted_fits(&result, options);
            for (k, &v) in result.vertex_ids.iter().enumerate() {
                if !trusted[k] {
                    // filled in from trusted neighbours below
                    warm[v as usize] = None;
                    continue;
                }
                let p = result.params[k];
                let mut d = alpha * (p[M] - 0.5 * p[T]);
                if let Some(cap) = refine.max_step {
                    d = d.clamp(-cap, cap);
                }
                disp[v as usize] = scale(result.normals[k], d);
                known[v as usize] = true;
                let mut w = p;
                w[M] -= d;
                warm[v as usize] = Some(w);
            }
            // extend it smoothly to unfitted vertices (harmonic fill)
            harmonic_fill(&adj, &known, &mut disp, 50);
            let before = mesh.vertices.clone();
            for v in 0..nv {
                let mut d = disp[v];
                for a in 0..3 {
                    if locked[v][a] {
                        d[a] = 0.0;
                    }
                }
                mesh.vertices[v] = add(mesh.vertices[v], d);
            }
            if refine.smoothing_iterations > 0 {
                mesh.taubin_smooth(
                    refine.smoothing_iterations,
                    options.surface.taubin_lambda,
                    options.surface.taubin_passband,
                    Some(&pinned),
                );
            }
            // actual displacement of fitted nodes (after smoothing)
            let mut sum2 = 0.0;
            let mut maxd: f64 = 0.0;
            for &v in &result.vertex_ids {
                let d = norm(sub(mesh.vertices[v as usize], before[v as usize]));
                sum2 += d * d;
                maxd = maxd.max(d);
            }
            let rms = (sum2 / result.vertex_ids.len().max(1) as f64).sqrt();
            result = fit_surface(&image, mesh.clone(), &keep, options, Some(&warm))?;
            if refine.keep_history {
                surfaces.push(mesh.clone());
            }
            history.push(IterationStats {
                rms_displacement: rms,
                max_displacement: maxd,
                mean_thickness: result.params.iter().map(|p| p[T]).sum::<f64>() / result.n() as f64,
                mean_residual: result.residual.iter().sum::<f64>() / result.n() as f64,
            });
            if rms < tol {
                converged = true;
                break;
            }
        }
        let shape = image.dim();
        let periosteal = result.surface.clone();
        let endosteal = result.displaced_surface(1.0);
        let periosteal_mask = periosteal.voxelize(shape, &options.geometry);
        let endosteal_mask = endosteal.voxelize(shape, &options.geometry);
        let mut cortical_mask = periosteal_mask.clone();
        Zip::from(&mut cortical_mask)
            .and(&endosteal_mask)
            .par_for_each(|c, &e| *c = *c && !e);
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
        })
    })?
}

/// Whether each node's fit can be trusted to locate the periosteal edge:
/// the edge lies inside the sampled range (with a margin of the blur), `m`
/// and `t` are not on their bounds, the cortex is brighter than both
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
            let s = p[crate::model::SIGMA];
            edge > x0 + s
                && edge < x1 - s
                && p[M] > lo[M] + eps(lo[M], hi[M])
                && p[M] < hi[M] - eps(lo[M], hi[M])
                && p[T] > lo[T] + eps(lo[T], hi[T])
                && p[T] < hi[T] - eps(lo[T], hi[T])
                && result.rho_c_at(i) > p[crate::model::RHO_S]
                && result.rho_c_at(i) > p[crate::model::RHO_B]
                && (!(med > 0.0) || r <= 3.0 * med)
        })
        .collect()
}

/// Jacobi iterations of neighbour averaging over the vertices without a
/// known value, so that unknown displacements interpolate the known ones.
fn harmonic_fill(adj: &Adjacency, known: &[bool], vals: &mut [V3], iterations: usize) {
    use rayon::prelude::*;
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
