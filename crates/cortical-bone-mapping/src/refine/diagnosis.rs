//! Diagnosing fits during refinement: which fits locate an edge, why a
//! refinement stalled, and where the surface still moves.

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

/// Why refinement stopped making progress: of the fitted nodes that are not
/// trusted (whose profiles do not show a cortex where the model can place
/// it), how many put the edge at the inner or outer end of the profile.
pub(super) fn stall_diagnosis(
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
    let med = median(&result.residual);
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
    let med = median(&result.residual);
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
