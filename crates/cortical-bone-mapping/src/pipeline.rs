//! The end-to-end cortical bone mapping pipeline.
//!
//! 1. Extract the outer bone surface from the binary mask (surface nets) and
//!    smooth it (Taubin λ|μ).
//! 2. Select the nodes to fit: optionally only near a sub-mask, never on the
//!    caps where the mask is cut by the image edge, and only where the
//!    (optionally constrained) sampling direction is valid.
//! 3. Fit the profile model to the nodes' profiles ([`fit_surface`]).

use std::time::Instant;

use ndarray::ArrayView3;

use crate::error::Error;
use crate::fit::fit_surface;
use crate::options::Options;
use crate::result::CbmResult;
use crate::surface::mesh::Mesh;
use crate::surface::nets::extract_surface;
use crate::volume::Voxel;

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

/// Run cortical bone mapping on `image` with outer-surface segmentation
/// `mask` (non-zero = bone) and an optional `sub_mask` restricting where the
/// surface is measured.
pub fn run<T: Voxel, U: Voxel, V: Voxel>(
    image: ArrayView3<T>,
    mask: ArrayView3<U>,
    sub_mask: Option<ArrayView3<V>>,
    options: &Options,
) -> Result<CbmResult, Error> {
    options.validate()?;
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
