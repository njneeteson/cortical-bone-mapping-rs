//! How far the band reaches from each node (ours).
//!
//! With a fixed band `[−outside, inside]`, a surface that has sunk into the
//! bone sees only bone in its band, and one bridging a concavity sees only
//! soft tissue below it. The density fields then take the values of what the
//! band does contain (a "soft tissue" of trabecular bone, a "marrow" of soft
//! tissue), the fit sees no edge, yet passes every trust test, and the
//! surface drifts.
//!
//! The adaptive band therefore extends each node's reach along its normal
//! until the band has crossed a bone edge, as one would look along a profile
//! for the edge before fitting it:
//!
//! * **outward**, past bright bone (above the level `high`) into soft tissue
//!   (below `low`): where the surface lies inside the bone, the band reaches
//!   through the cortex into the soft tissue beyond it;
//! * **inward**, where the profile starts in soft tissue (below `low`), to
//!   the first bright bone: where the surface bridges a concavity, the band
//!   reaches its floor;
//!
//! plus the default reach beyond the crossing as a margin, and never beyond
//! `cap` times the default reach. The levels come from the profiles
//! themselves: soft tissue `s` is the median sample of the outer half of
//! the default outside range, bone `c` the 90th percentile of the samples
//! inside (as the initial cortical density), and `low = s + ¼ (c − s)`,
//! `high = s + ½ (c − s)`.

use ndarray::ArrayView3;
use rayon::prelude::*;

use crate::util::{add, quantile, scale, V3};
use crate::volume::{trilinear, Geometry, Voxel};

/// Each node's reach outside and inside its surface (physical units).
pub(crate) struct Reach {
    pub outside: Vec<f64>,
    pub inside: Vec<f64>,
}

/// The settings of [`node_reach`].
pub(crate) struct ReachSpec<'a> {
    pub geom: &'a Geometry,
    /// The default reach.
    pub outside: f64,
    pub inside: f64,
    /// The largest reach, as a multiple of the default.
    pub cap: f64,
}

/// The default reach at every node.
pub(crate) fn uniform_reach(n: usize, outside: f64, inside: f64) -> Reach {
    Reach {
        outside: vec![outside; n],
        inside: vec![inside; n],
    }
}

/// The adaptive reach of the nodes at `points` with inward unit `normals`
/// (module documentation).
pub(crate) fn node_reach<T: Voxel>(image: &ArrayView3<T>, points: &[V3], normals: &[V3], spec: &ReachSpec) -> Reach {
    let step = 0.5 * spec.geom.min_spacing();
    let (out_max, in_max) = (spec.cap * spec.outside, spec.cap * spec.inside);
    let n_out = (out_max / step).ceil() as usize;
    let n_in = (in_max / step).ceil() as usize;
    // profile k: samples at x = (j − n_out) · step, j = 0..=n_out + n_in
    let profiles: Vec<Vec<f64>> = points
        .par_iter()
        .zip(normals)
        .map(|(&p, &n)| {
            (0..=n_out + n_in)
                .map(|j| {
                    let x = (j as f64 - n_out as f64) * step;
                    trilinear(image, spec.geom.to_index(add(p, scale(n, x))))
                })
                .collect()
        })
        .collect();
    let (low, high) = levels(&profiles, n_out, spec.outside, step);
    log::debug!(
        "adaptive band: soft-tissue level {low:.1}, bone level {high:.1}, cap {}x",
        spec.cap
    );
    let (outside, inside): (Vec<f64>, Vec<f64>) = profiles
        .par_iter()
        .map(|prof| {
            let outward: Vec<f64> = prof[..=n_out].iter().rev().copied().collect();
            let inward = &prof[n_out..];
            let o = crossed_out(&outward, low, high).map_or(spec.outside, |j| j as f64 * step + spec.outside);
            let i = crossed_in(inward, low, high).map_or(spec.inside, |j| j as f64 * step + spec.inside);
            (o.clamp(spec.outside, out_max), i.clamp(spec.inside, in_max))
        })
        .unzip();
    log::debug!(
        "adaptive band: median reach outside {:.2}, inside {:.2} ({} nodes)",
        crate::util::median(&outside),
        crate::util::median(&inside),
        outside.len()
    );
    Reach { outside, inside }
}

/// The soft-tissue and bone levels `(low, high)` from the profiles.
fn levels(profiles: &[Vec<f64>], n_out: usize, outside: f64, step: f64) -> (f64, f64) {
    let half = ((0.5 * outside) / step).round() as usize;
    let lo = n_out.saturating_sub(2 * half);
    let soft: Vec<f64> = profiles
        .iter()
        .flat_map(|p| p[lo..=n_out - half].iter().copied())
        .collect();
    let bone: Vec<f64> = profiles.iter().flat_map(|p| p[n_out + 1..].iter().copied()).collect();
    let (s, c) = (quantile(soft, 0.5), quantile(bone, 0.9));
    (s + 0.25 * (c - s), s + 0.5 * (c - s))
}

/// Outward from the node (`outward[0]` at the node): the first sample in
/// soft tissue after one in bright bone, if the profile passes through bone
/// before soft tissue (otherwise the default reach holds).
fn crossed_out(outward: &[f64], low: f64, high: f64) -> Option<usize> {
    let bright = outward.iter().position(|&v| v > high)?;
    let soft_first = outward.iter().position(|&v| v < low).unwrap_or(usize::MAX);
    if soft_first < bright {
        return None; // soft tissue first: the edge is within the default reach
    }
    outward[bright..].iter().position(|&v| v < low).map(|j| bright + j)
}

/// Inward from the node: if the profile starts in soft tissue, the first
/// sample in bright bone.
fn crossed_in(inward: &[f64], low: f64, high: f64) -> Option<usize> {
    if inward.first().is_none_or(|&v| v >= low) {
        return None;
    }
    inward.iter().position(|&v| v > high)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_profile_starting_in_bone_reaches_past_the_cortex() {
        // trabecular (300) at the node, cortex (1200) 2-3 samples out, soft (0) beyond
        let outward = [300.0, 300.0, 1200.0, 1200.0, 0.0, 0.0];
        assert_eq!(crossed_out(&outward, 250.0, 600.0), Some(4));
        // soft tissue right outside: the default reach holds
        assert_eq!(crossed_out(&[0.0, 0.0, 1200.0], 250.0, 600.0), None);
    }

    #[test]
    fn a_profile_starting_in_soft_tissue_reaches_the_bone_below() {
        let inward = [0.0, 0.0, 0.0, 1200.0, 300.0];
        assert_eq!(crossed_in(&inward, 250.0, 600.0), Some(3));
        assert_eq!(crossed_in(&[300.0, 1200.0], 250.0, 600.0), None);
    }
}
