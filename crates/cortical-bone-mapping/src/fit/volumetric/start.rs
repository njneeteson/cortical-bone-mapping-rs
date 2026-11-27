//! What the volumetric fit takes from the nodes' 1D profiles: the estimated
//! cortical density and, with [`VolumetricOptions::init_from_local`], its
//! start.
//!
//! **Why not one global model.** The volumetric fit otherwise starts every
//! node from one model fitted to all band voxels. Where the cortex varies a
//! lot over the bone and the band reaches deep into trabecular bone (a whole
//! HR-pQCT proximal tibia), that model collapses the cortex to its lower
//! bound and widens the blur to absorb the rest, and from a start with no
//! cortex the field fit cannot recover.
//!
//! **The start.** Every node's profile is fitted on its own (the local 1D
//! method, `σ` free, from the 1D global model); a fit is trusted if it
//! converged with thickness and blur off their bounds and its periosteal
//! edge inside the sampled range. Every node starts from the median of the
//! trusted fits within `σ_reg` (the regularisation's scale) of it: single
//! fits are noisy from node to node, and a rough start makes the fitted
//! surfaces rough, which the curvature penalty (relative to the input
//! surface) punishes by many orders of magnitude more than the data. Nodes
//! with no trusted fit nearby start from the 1D global model. The PSF starts from the local fits' blur on thick-cortex nodes
//! ([`psf::estimate_psf`], after Treece & Gee 2015), less the widening of the
//! profile smoothing (subtracted in quadrature). The densities start from the
//! global model's.
//!
//! The profiles are sampled once, for the density estimate and the start.

use ndarray::ArrayView3;

use crate::error::Error;
use crate::fit::psf::{self, PsfModel};
use crate::fit::{
    density, global_settings, local, local_settings, sample_profiles, select_nodes, Data, FitConstraints,
};
use crate::kdtree::KdTree;
use crate::model::residual_weights;
use crate::model::{Params, M, RHO_B, RHO_S, SIGMA, T};
use crate::options::{Options, RhoCMode, VolumetricOptions};
use crate::surface::mesh::Mesh;
use crate::util::{median, V3};
use crate::volume::Voxel;
use rayon::prelude::*;

/// The ratio `t / σ` above which a local fit's blur measures the PSF (the
/// cortex is wide enough for its plateau to show).
const THICK_CORTEX: f64 = 3.0;

/// What the profiles give the volumetric fit.
#[derive(Default)]
pub(crate) struct ProfileStart {
    /// The estimated cortical density (with `rho_c="estimate"` and none given).
    pub rho_c: Option<f64>,
    /// Per surface vertex: the start of a node with a trusted local fit.
    pub warm: Option<Vec<Option<Params>>>,
    /// The PSF `(σ_xy, σ_z)` to start from.
    pub psf: Option<[f64; 2]>,
    /// The densities `(ρs, ρb)` to start from.
    pub densities: Option<[f64; 2]>,
}

/// The profile-based inputs of a volumetric fit on the nodes `keep` of
/// `surface` (with the constraints' cortical density, if given, and
/// excluded voxels).
pub(crate) fn profile_start<T: Voxel>(
    image: &ArrayView3<T>,
    surface: &Mesh,
    keep: &[bool],
    o: &Options,
    v: &VolumetricOptions,
    constraints: &FitConstraints,
    warm_given: bool,
) -> Result<ProfileStart, Error> {
    let (rho_c, exclude) = (constraints.rho_c, constraints.exclude);
    let estimate = rho_c.is_none() && o.model.rho_c_mode == RhoCMode::Estimate;
    let start = v.init_from_local && !warm_given;
    if !estimate && !start {
        return Ok(ProfileStart::default());
    }
    let s = crate::stage!("profiles for the volumetric fit");
    let (sub, ids, normals) = select_nodes(surface, keep, o)?;
    let (mut profiles, x) = sample_profiles(image, surface, &sub.vertices, &normals, exclude, o);
    let weights = residual_weights(&x, o.model.residual_boost);
    let mut out = ProfileStart::default();
    if estimate {
        out.rho_c = Some(density::cortical_density(&mut profiles, &x, &weights, None, o)?.0);
    }
    if start {
        let rho_c = out.rho_c.or(rho_c).unwrap_or(o.model.rho_c);
        let data = Data {
            profiles: &profiles,
            x: &x,
            weights: &weights,
            rho_c,
            options: o,
            solver: global_settings(o),
        };
        let global = density::fit_one_model(&data, o.model.init, None);
        let fits = local::fit_all(&profiles, &x, &weights, rho_c, &[global], &local_settings(o, true));
        let params: Vec<Params> = fits.iter().map(|f| f.params).collect();
        let trusted: Vec<bool> = fits
            .iter()
            .map(|f| f.status.converged() && trusted(&f.params, o))
            .collect();
        let smoothed = neighbourhood_medians(&sub.vertices, &params, &trusted, v.sigma);
        let mut warm = vec![None; surface.vertices.len()];
        for (k, &vtx) in ids.iter().enumerate() {
            warm[vtx as usize] = smoothed[k];
        }
        out.psf = start_psf(&normals, &params, &trusted, o, v);
        out.densities = Some([global[RHO_S], global[RHO_B]]);
        out.warm = Some(warm);
        crate::done!(
            s,
            "{} nodes, {} trusted local fits; start psf {:?}, densities ({:.1}, {:.1})",
            ids.len(),
            trusted.iter().filter(|&&t| t).count(),
            out.psf,
            global[RHO_S],
            global[RHO_B]
        );
    } else {
        crate::done!(s, "rho_c {:?} from {} profiles", out.rho_c, ids.len());
    }
    Ok(out)
}

/// Per node: the parameter-wise median of the trusted fits within `radius`
/// (`None` if there is none).
fn neighbourhood_medians(points: &[V3], params: &[Params], trusted: &[bool], radius: f64) -> Vec<Option<Params>> {
    let ids: Vec<usize> = (0..points.len()).filter(|&k| trusted[k]).collect();
    let pts: Vec<V3> = ids.iter().map(|&k| points[k]).collect();
    let tree = KdTree::new(&pts);
    points
        .par_iter()
        .map_init(Vec::new, |near, &p| {
            near.clear();
            tree.within(p, radius, near);
            if near.is_empty() {
                return None;
            }
            Some(std::array::from_fn(|c| {
                median(
                    &near
                        .iter()
                        .map(|&(j, _)| params[ids[j as usize]][c])
                        .collect::<Vec<_>>(),
                )
            }))
        })
        .collect()
}

/// A local fit locates the cortex: thickness and blur off their bounds, the
/// periosteal edge inside the sampled range.
fn trusted(p: &Params, o: &Options) -> bool {
    let (lo, hi) = (o.model.lower, o.model.upper);
    let off = |k: usize| {
        let margin = 1e-3 * (hi[k] - lo[k]);
        p[k] > lo[k] + margin && p[k] < hi[k] - margin
    };
    let edge = p[M] - 0.5 * p[T];
    off(T) && off(SIGMA) && edge > -o.sampling.outside && edge < o.sampling.inside
}

/// The PSF `(σ_xy, σ_z)` from the trusted local fits' blur, less the profile
/// smoothing's; `None` with too few thick-cortex nodes.
fn start_psf(
    normals: &[V3],
    params: &[Params],
    trusted: &[bool],
    o: &Options,
    v: &VolumetricOptions,
) -> Option<[f64; 2]> {
    let (n, p): (Vec<_>, Vec<_>) = normals
        .iter()
        .zip(params)
        .zip(trusted)
        .filter(|(_, &t)| t)
        .map(|((n, p), _)| (*n, *p))
        .unzip();
    let (lo, hi) = (o.model.lower[SIGMA], o.model.upper[SIGMA]);
    let axis = v.scanner_axis;
    let est = psf::estimate_psf(&n, &p, PsfModel::Axial(axis), THICK_CORTEX, lo, hi)?;
    let resolution = o.sampling.resolution.unwrap_or_else(|| o.geometry.min_spacing());
    let widening = o.sampling.smoothing * resolution;
    let floor = 0.25 * o.geometry.min_spacing();
    let unsmoothed = |s: f64| (s * s - widening * widening).max(floor * floor).sqrt();
    Some([unsmoothed(est[(axis + 1) % 3]), unsmoothed(est[axis])])
}
