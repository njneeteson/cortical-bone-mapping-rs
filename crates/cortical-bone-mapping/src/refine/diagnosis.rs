//! Diagnosing fits during refinement: which fits locate an edge, why a
//! refinement stalled, and where the surface still moves (ours).
//!
//! A fit is **trusted** to locate the periosteal edge when it passes all six
//! tests listed in `REASONS`:
//!
//! * the edge `m − t/2` lies inside the sampled range by more than `σ` at the
//!   inner and at the outer end (an edge at the end of a profile is
//!   extrapolated, not measured);
//! * `m` is off its bounds (a bounded `m` means the optimum lies outside);
//! * `t` is below its upper bound (a thickness at its *lower* bound is a
//!   cortex thinner than resolvable, whose blurred step still locates the
//!   edge, so that is allowed);
//! * the cortex is denser than both neighbouring tissues (`ρc > ρs, ρb`),
//!   otherwise the profile does not show a cortex the model can place;
//! * the residual is at most 3× the median residual.
//!
//! Profiles that miss the cortex, e.g. lines running along the flank of a
//! gross segmentation error, fail these tests; without the gate, such fits
//! pushed bumps outward. Untrusted nodes move with their trusted neighbours
//! ([`super::motion`]).
//!
//! The **regional RMS** of the displacement is the RMS over the nodes within a
//! radius of each node, maximised over nodes: it catches a small region that
//! still moves while the surface-wide RMS is already small.

use rayon::prelude::*;

use crate::kdtree::KdTree;
use crate::model::{M, RHO_B, RHO_S, SIGMA, T};
use crate::options::Options;
use crate::result::CbmResult;
use crate::util::{median, V3};

/// The largest regional displacement RMS: for each point, the RMS of `m2`
/// (squared displacements) over the points within `radius` of it.
pub(super) fn max_regional_rms(points: &[V3], m2: &[f64], radius: f64) -> f64 {
    let tree = KdTree::new(points);
    (0..points.len())
        .into_par_iter()
        .map(|i| {
            let mut near = Vec::new();
            tree.within(points[i], radius, &mut near);
            let sum: f64 = near.iter().map(|&(u, _)| m2[u as usize]).sum();
            (sum / near.len().max(1) as f64).sqrt()
        })
        .reduce(|| 0.0, f64::max)
}

/// The tests a fit must pass to locate the periosteal edge, as the reason
/// given when it fails.
const REASONS: [&str; 6] = [
    "edge at the inner end",
    "edge at the outer end",
    "m at a bound",
    "t at its upper bound",
    "densities out of order",
    "residual > 3x median",
];

/// Which of the [`REASONS`] tests node `k`'s fit fails (`med`: the median
/// residual).
fn failed_tests(result: &CbmResult, k: usize, options: &Options, med: f64) -> [bool; 6] {
    let (lo, hi) = (options.model.lower, options.model.upper);
    // the band's reach at the node (volumetric model), else the sampling range
    let (x0, x1) = match result.volumetric.as_ref().and_then(|v| v.reach.get(k)) {
        Some(&[o, i]) => (-o, i),
        None => (-options.sampling.outside, options.sampling.inside),
    };
    let eps = |a: f64, b: f64| 1e-6 * (b - a).abs().max(1e-12);
    let p = result.params[k];
    let edge = p[M] - 0.5 * p[T];
    let s = p[SIGMA];
    let rc = result.rho_c_at(k);
    [
        !(edge < x1 - s),
        !(edge > x0 + s),
        !(p[M] > lo[M] + eps(lo[M], hi[M]) && p[M] < hi[M] - eps(lo[M], hi[M])),
        !(p[T] < hi[T] - eps(lo[T], hi[T])),
        !(rc > p[RHO_S] && rc > p[RHO_B]),
        !(!(med > 0.0) || result.residual[k] <= 3.0 * med),
    ]
}

/// Why refinement stopped making progress: of the fitted nodes that are not
/// trusted (whose profiles do not show a cortex where the model can place
/// it), how many fail each test (a fit may fail several), with advice when
/// the edge lies beyond an end of the profiles.
pub(super) fn stall_diagnosis(
    result: &CbmResult,
    trusted: &[bool],
    held: &[bool],
    options: &Options,
    rms: f64,
    tol: f64,
    window: usize,
) -> String {
    let med = median(&result.residual);
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
        for (w, f) in why.iter_mut().zip(failed_tests(result, k, options, med)) {
            *w += f as usize;
        }
    }
    let pc = |a: usize| 100.0 * a as f64 / n.max(1) as f64;
    let parts: Vec<String> = REASONS
        .iter()
        .zip(why)
        .filter(|(_, c)| *c > 0)
        .map(|(nm, c)| format!("{nm} {c} ({:.1}%)", pc(c)))
        .collect();
    let reasons = if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(", ")
    };
    format!(
        "stalled: no progress in {window} fits (displacement RMS {rms:.3}, tolerance {tol:.3}); \
         {bad} of {n} moving fitted profiles ({:.1}%) do not locate an edge: {reasons}{}",
        pc(bad),
        sampling_advice(why[0], why[1], options)
    )
}

/// Advice on the sampling range from the counts of edges at the inner
/// (`deep`) and outer (`shallow`) end of the profiles.
fn sampling_advice(deep: usize, shallow: usize, options: &Options) -> String {
    let s = &options.sampling;
    if deep > shallow && deep > 0 {
        format!(
            "; the surface may lie farther inside than the profiles reach: try a longer inside sampling (now {})",
            s.inside
        )
    } else if shallow > 0 {
        format!(
            "; the surface may lie farther outside than the profiles reach: try a longer outside sampling (now {})",
            s.outside
        )
    } else {
        String::new()
    }
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
    let med = median(&result.residual);
    let failed: Vec<[bool; 6]> = (0..result.n()).map(|k| failed_tests(result, k, options, med)).collect();
    if log::log_enabled!(log::Level::Debug) {
        let counts: Vec<String> = REASONS
            .iter()
            .enumerate()
            .map(|(t, r)| format!("{r} {}", failed.iter().filter(|f| f[t]).count()))
            .collect();
        log::debug!(
            "trusted fits: {} of {} (failed: {})",
            failed.iter().filter(|f| !f.iter().any(|&b| b)).count(),
            failed.len(),
            counts.join(", ")
        );
    }
    failed.iter().map(|f| !f.iter().any(|&b| b)).collect()
}
