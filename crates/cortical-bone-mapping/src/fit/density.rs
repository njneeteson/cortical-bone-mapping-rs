//! The cortical density `ρc` ([`RhoCMode`]) and the global initialisation:
//! one model fitted to all profiles.

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

/// The maximum of each profile (over the samples used).
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

/// One model `[m, t, σ, ρs, ρb]` fitted to all profiles (with a per-node σ
/// if `sigma_fixed`), starting from `init`.
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

/// [`RhoCMode::Estimate`]: the cortical plateau level of thick-cortex nodes.
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
