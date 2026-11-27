//! The end-to-end cortical bone mapping pipeline.
//!
//! 1. Extract the outer bone surface from the binary mask with surface nets
//!    ([`crate::surface::nets`]; Gibson 1998) and smooth it with Taubin λ|μ
//!    smoothing ([`Mesh::taubin_smooth`]; Taubin 1995), which removes the voxel
//!    staircase without shrinking the surface.
//! 2. Select the nodes to fit: optionally only near a sub-mask (vertices with
//!    a sub-mask voxel among the 8 voxels of their cell), never on the caps
//!    where the mask is cut by the image edge (they are not periosteum), and
//!    only where the (optionally constrained) sampling direction is valid.
//! 3. Fit the profile model to the nodes' profiles ([`fit_surface`]; Treece et
//!    al. 2010, Neeteson & Boyd 2025).
//!
//! # References
//!
//! * S. F. F. Gibson, *Constrained elastic surface nets: generating smooth
//!   surfaces from binary segmented data*, MICCAI 1998, Lecture Notes in
//!   Computer Science 1496, Springer, 1998, pp. 888–898.
//! * G. Taubin, *A signal processing approach to fair surface design*,
//!   Proceedings of SIGGRAPH 95, 1995, pp. 351–358.
//! * G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
//!   cortical bone thickness measurement from clinical CT data*, Medical Image
//!   Analysis 14(3):276–290, 2010, doi:10.1016/j.media.2010.01.003.
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.

use ndarray::ArrayView3;

use crate::error::Error;
use crate::fit::fit_surface;
use crate::options::Options;
use crate::result::CbmResult;
use crate::surface::mesh::Mesh;
use crate::surface::nets::{extract_surface, extract_surface_coarse};
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
    log::info!(
        "cortical bone mapping: image {:?}, method {}, sub-mask {}",
        image.dim(),
        options.method.name(),
        sub_mask.is_some()
    );
    log::debug!("options: {options:?}");
    with_threads(options.threads, || {
        let run = crate::stage!("cortical bone mapping");
        let s = crate::stage!("surface");
        let surf = prepare_surface(&mask, options);
        if surf.0.vertices.is_empty() {
            log::warn!("the mask is empty");
            return Err(Error::EmptyMask);
        }
        let t_surface = s.end();
        let candidates = candidate_nodes(&surf.0, &surf.1, sub_mask.as_ref(), options);
        log::debug!(
            "{} of {} surface vertices are candidate nodes",
            candidates.iter().filter(|&&c| c).count(),
            candidates.len()
        );
        let mut res = fit_surface(&image, surf.0, &candidates, options, None)?;
        res.info.timings.insert(0, ("surface".into(), t_surface));
        let t = crate::done!(run, "{} nodes", res.points.len());
        log::info!("cortical bone mapping: {} nodes in {t:.1} s", res.points.len());
        Ok(res)
    })?
}

/// Extract and smooth the outer surface; returns the mesh and border flags.
pub fn prepare_surface<U: Voxel>(mask: &ArrayView3<U>, options: &Options) -> (Mesh, Vec<bool>) {
    let s = &options.surface;
    let st = crate::stage!("surface nets");
    let e = if s.grid_step > 1 {
        extract_surface_coarse(mask, &options.geometry, s.grid_step)
    } else {
        extract_surface(mask, &options.geometry, s.mask_smoothing)
    };
    crate::done!(
        st,
        "grid step {}: {} vertices, {} faces, {} on the image's border",
        s.grid_step,
        e.mesh.vertices.len(),
        e.mesh.faces.len(),
        e.border.iter().filter(|&&b| b).count()
    );
    let mut mesh = e.mesh;
    let st = crate::stage!("Taubin smoothing");
    mesh.taubin_smooth(s.taubin_iterations, s.taubin_lambda, s.taubin_passband, None);
    crate::done!(
        st,
        "{} iterations, pass-band {}",
        s.taubin_iterations,
        s.taubin_passband
    );
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
