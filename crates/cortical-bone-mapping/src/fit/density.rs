//! The cortical density `ρc` ([`RhoCMode`]) and the global initialisation:
//! one model fitted to all profiles.
//!
//! `ρc` is never fitted: thickness and density trade off (a thinner, denser
//! cortex blurs into nearly the same profile), so with `ρc` free `t` would be
//! poorly determined wherever the cortex is thin (Treece et al. 2012). Because
//! the fit conserves the cortical mass under the blurred profile, a wrong
//! `ρc` biases thickness roughly as `t ∝ 1/ρc`. Three ways to choose it:
//!
//! * **fixed**: as given, e.g. the tissue density of a density-calibrated
//!   scanner (the default, 1000, is that of Table 1 of Neeteson & Boyd 2025);
//! * **estimate** (ours): one value, the cortical *plateau* where the cortex
//!   is thick enough to show one. Profile maxima alone are biased up by noise
//!   (the maximum of noisy samples) and down by blur and profile smoothing on
//!   a thin cortex, which never reaches its tissue density. So, starting from
//!   the 90th percentile of the maxima, every node is fitted locally (`σ`
//!   free); nodes whose fitted cortex has at least 3 samples more than `2σ`
//!   from both edges (where the blurred edges contribute < 2.3% of a step) give
//!   the median of those samples, and `ρc` is the median over nodes; this is
//!   repeated once with the new `ρc`. It measures the *apparent* cortical
//!   density, including sub-resolution porosity, which is what a
//!   uniform-cortex model sees. Treece & Gee (2015) measure cortical density
//!   from the data by a different procedure; this one is a simple, robust
//!   stand-in, not a reproduction of theirs;
//! * **profile maximum**: per node, the maximum of its (smoothed) profile, as
//!   the original Python implementation in Bonelab does when no density is
//!   given. It underestimates `ρc`, and so overestimates `t`, exactly where the
//!   cortex is thin.
//!
//! Per-node densities need no change to the solvers: the model is linear in
//! `(ρs, ρc, ρb)`, so scaling profile `i` by `s_i = ρ̄c / ρc,i` (`ρ̄c` the
//! median) makes it a profile with cortical density `ρ̄c`; `m, t, σ` are
//! unchanged and `ρs, ρb` and the residual are scaled back by `1/s_i` after
//! the fit. The only difference from a true per-node density is the relative
//! weight of each node's data term against the regularisation.
//!
//! # References
//!
//! * G. M. Treece, K. E. S. Poole, A. H. Gee, *Imaging the femoral cortex:
//!   thickness, density and mass from clinical CT*, Medical Image Analysis
//!   16(5):952–965, 2012.
//! * G. M. Treece, A. H. Gee, *Independent measurement of femoral cortical
//!   thickness and cortical bone density using clinical CT*, Medical Image
//!   Analysis 20(1):249–264, 2015.
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.

use rayon::prelude::*;

use super::global::{self, Problem, Role, State};
use super::profiles::Profiles;
use super::{global_settings, local, local_settings, Data};
use crate::error::Error;
use crate::model::{Params, M, NPARAM, SIGMA, T};
use crate::options::{Options, RhoCMode};
use crate::util::quantile;

/// The cortical density, and per-node densities for
/// [`RhoCMode::ProfileMax`]. Per-node densities are handled by scaling each
/// profile to the common `ρc`: the model is linear in `(ρs, ρc, ρb)`, so `m`,
/// `t`, `σ` are unchanged and `ρs`, `ρb` (and the residual) scale back after
/// the fit. `given` overrides the fixed or estimated value.
pub(crate) fn cortical_density(
    profiles: &mut Profiles,
    x: &[f64],
    weights: &[f64],
    given: Option<f64>,
    o: &Options,
) -> Result<(f64, Option<Vec<f64>>), Error> {
    Ok(match o.model.rho_c_mode {
        RhoCMode::Fixed => (given.unwrap_or(o.model.rho_c), None),
        RhoCMode::Estimate => match given {
            Some(rc) => (rc, None),
            None => {
                let data = Data {
                    profiles,
                    x,
                    weights,
                    rho_c: 0.0,
                    options: o,
                    solver: global_settings(o),
                };
                (estimate_rho_c(&data, profile_maxima(profiles)), None)
            }
        },
        RhoCMode::ProfileMax => {
            let mx = profile_maxima(profiles);
            let rc = quantile(mx.clone(), 0.5);
            if !(rc > 0.0) {
                return Err(Error::InvalidOption("profile maxima are not positive".into()));
            }
            let node: Vec<f64> = mx.iter().map(|&v| v.max(0.05 * rc)).collect();
            let m = profiles.m;
            profiles.values.chunks_mut(m).zip(&node).for_each(|(row, &rn)| {
                let s = (rc / rn) as f32;
                row.iter_mut().for_each(|v| *v *= s);
            });
            (rc, Some(node))
        }
    })
}

/// The maximum of each profile over the samples used (the per-node density of
/// the profile-maximum mode, and the start of the estimate).
fn profile_maxima(profiles: &Profiles) -> Vec<f64> {
    (0..profiles.n)
        .map(|i| {
            let row = profiles.row(i);
            match profiles.used_row(i) {
                None => row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64,
                Some(u) => row
                    .iter()
                    .zip(u)
                    .filter(|(_, &k)| k)
                    .fold(f32::NEG_INFINITY, |a, (&b, _)| a.max(b)) as f64,
            }
        })
        .collect()
}

/// The global initialisation (Neeteson & Boyd 2025): one model
/// `[m, t, σ, ρs, ρb]` fitted to all profiles at once, every parameter global
/// (with a per-node σ if `sigma_fixed`), starting from `init`.
pub(crate) fn fit_one_model(data: &Data, init: Params, sigma_fixed: Option<&[f64]>) -> Params {
    let mut roles = [Role::Global; NPARAM];
    let mut fixed: [Option<&[f64]>; NPARAM] = [None; NPARAM];
    if let Some(sf) = sigma_fixed {
        roles[SIGMA] = Role::Fixed;
        fixed[SIGMA] = Some(sf);
    }
    let o = data.options;
    let pr = Problem {
        profiles: data.profiles,
        x: data.x,
        weights: data.weights,
        rho_c: data.rho_c,
        roles,
        fixed_per_node: fixed,
        interp: None,
        frozen: None,
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
}

/// [`RhoCMode::Estimate`]: the cortical plateau level of thick-cortex nodes
/// (module documentation), from the 90th percentile of the profile `maxima`,
/// two rounds.
fn estimate_rho_c(data: &Data, maxima: Vec<f64>) -> f64 {
    let (profiles, x, o) = (data.profiles, data.x, data.options);
    let n = profiles.n;
    let mut rc = quantile(maxima, 0.9);
    for _ in 0..2 {
        let d = Data { rho_c: rc, ..*data };
        let g = fit_one_model(&d, o.model.init, None);
        let fits = local::fit_all(profiles, x, data.weights, rc, &vec![g; n], &local_settings(o, true));
        let plateaus: Vec<f64> = fits
            .par_iter()
            .enumerate()
            .filter_map(|(i, f)| {
                let p = f.params;
                let (lo, hi) = (p[M] - 0.5 * p[T] + 2.0 * p[SIGMA], p[M] + 0.5 * p[T] - 2.0 * p[SIGMA]);
                let mut v: Vec<f64> = x
                    .iter()
                    .zip(profiles.row(i))
                    .filter(|(&xj, _)| xj >= lo && xj <= hi)
                    .map(|(_, &y)| y as f64)
                    .collect();
                (v.len() >= 3).then(|| {
                    v.sort_by(f64::total_cmp);
                    v[v.len() / 2]
                })
            })
            .collect();
        if plateaus.len() < 10 {
            break;
        }
        rc = quantile(plateaus, 0.5);
    }
    rc
}
