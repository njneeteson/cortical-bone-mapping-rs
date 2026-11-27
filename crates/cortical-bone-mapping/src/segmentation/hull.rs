//! Convex-hull starting segmentations: the per-slice convex hull of a mask,
//! and the single-bone hull segmentation built on it (ours).
//!
//! A starting segmentation for refinement has to be *topologically* right (one
//! closed surface around the bone) and close enough that the profiles cross
//! the cortex. A thresholded bone leaks into the marrow through cortical
//! breaks and porous cortex; its convex hull cannot: the hull of a slice
//! contains the slice's bone, so its only errors are outward ones over
//! concave parts of the periosteum, which refinement pulls in. Hulls are taken
//! per slice (along axis 0) because a long bone is convex in cross-section
//! far more often than in 3D.
//!
//! The hull of a slice needs only the first and last set pixel of every row
//! (any other set pixel lies on a segment between them). It is computed by
//! Andrew's monotone chain (Andrew 1979): sort the points, build the lower and
//! upper chains keeping only left turns (cross product > 0), O(n log n). The
//! polygon is filled row by row between its edge crossings, pixel centres on
//! the boundary included.
//!
//! # References
//!
//! * A. M. Andrew, *Another efficient algorithm for convex hulls in two
//!   dimensions*, Information Processing Letters 9(5):216–219, 1979.

use ndarray::{Array3, ArrayView3, Axis, Zip};
use rayon::prelude::*;

use super::morphology::{binary_median, largest_component};
use crate::volume::Voxel;

/// Options of [`hull_segmentation`].
#[derive(Clone, Debug, PartialEq)]
pub struct HullOptions {
    /// Bone threshold (image units; 320 mg HA/cm³ for calibrated HR-pQCT).
    pub threshold: f64,
    /// Binary median kernel (as Buie's step 3) to remove noise speckle.
    pub median_kernel: [usize; 3],
    /// Keep only the largest 3D component (removes a separate fibula/ulna).
    pub largest_component: bool,
}

impl Default for HullOptions {
    fn default() -> Self {
        HullOptions {
            threshold: 320.0,
            median_kernel: [1, 3, 3],
            largest_component: true,
        }
    }
}

/// A conservative periosteal segmentation: threshold, median, largest
/// component, then the convex hull of every slice (along axis 0).
///
/// Unlike a closing, a hull cannot leak through cortical breaks or porous
/// cortex into the marrow: it always contains the thresholded bone, so its
/// only errors are outward ones over concave parts of the periosteum, the
/// kind that model-driven refinement removes. It is a starting point for
/// [`refine_segmentation`](crate::refine::refine_segmentation), not a
/// segmentation in its own right.
pub fn hull_segmentation<T: Voxel>(image: ArrayView3<T>, o: &HullOptions) -> Array3<bool> {
    let s = crate::stage!("hull segmentation");
    log::debug!("hull segmentation: image {:?}, {o:?}", image.dim());
    let mut m = Array3::from_elem(image.dim(), false);
    Zip::from(&mut m)
        .and(&image)
        .par_for_each(|b, &v| *b = v.to_f64() >= o.threshold);
    if o.median_kernel.iter().any(|&k| k > 1) {
        m = binary_median(&m, o.median_kernel);
    }
    if o.largest_component {
        m = largest_component(&m, false);
    }
    let hull = slice_hulls(&m);
    crate::done!(
        s,
        "{} voxels above the threshold kept, {} in the hull",
        m.iter().filter(|&&v| v).count(),
        hull.iter().filter(|&&v| v).count()
    );
    hull
}

/// The convex hull of every slice (along axis 0) of a binary mask.
pub fn slice_hulls(m: &Array3<bool>) -> Array3<bool> {
    let mut out = Array3::from_elem(m.dim(), false);
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(m.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut o, sl)| {
            let hull = convex_hull(&row_extremes(&sl));
            fill_convex(&hull, &mut o);
        });
    out
}

/// The leftmost and rightmost set pixel of every row, as (row, column)
/// points: enough to determine the convex hull of a slice.
fn row_extremes(sl: &ndarray::ArrayView2<bool>) -> Vec<(f64, f64)> {
    let mut pts = Vec::new();
    for (j, row) in sl.outer_iter().enumerate() {
        let first = row.iter().position(|&b| b);
        let last = row.iter().rposition(|&b| b);
        if let (Some(a), Some(b)) = (first, last) {
            pts.push((j as f64, a as f64));
            if b != a {
                pts.push((j as f64, b as f64));
            }
        }
    }
    pts
}

/// The convex hull by Andrew's monotone chain (Andrew 1979): sort the points
/// lexicographically, then build the lower and the upper chain, popping the
/// last point while it does not make a strict left turn (cross product
/// `≤ 0`, which also drops collinear points). Counter-clockwise.
pub(super) fn convex_hull(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut p = pts.to_vec();
    p.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    p.dedup();
    if p.len() < 3 {
        return p;
    }
    let cross = |o: (f64, f64), a: (f64, f64), b: (f64, f64)| (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0);
    let mut h: Vec<(f64, f64)> = Vec::with_capacity(2 * p.len());
    for pass in 0..2 {
        let start = h.len();
        let iter: Box<dyn Iterator<Item = &(f64, f64)>> = if pass == 0 {
            Box::new(p.iter())
        } else {
            Box::new(p.iter().rev())
        };
        for &q in iter {
            while h.len() >= start + 2 && cross(h[h.len() - 2], h[h.len() - 1], q) <= 0.0 {
                h.pop();
            }
            h.push(q);
        }
        h.pop();
    }
    h
}

/// Set the pixels whose centres lie in the convex polygon `hull` (boundary
/// included, with a tolerance of 10⁻⁹): per row, the extent between the
/// polygon's crossings of that row.
pub(super) fn fill_convex(hull: &[(f64, f64)], out: &mut ndarray::ArrayViewMut2<bool>) {
    if hull.is_empty() {
        return;
    }
    let (_, nk) = out.dim();
    let eps = 1e-9;
    let jmin = hull.iter().map(|q| q.0).fold(f64::INFINITY, f64::min);
    let jmax = hull.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max);
    for j in (jmin.ceil() as usize)..=(jmax.floor() as usize) {
        let y = j as f64;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let n = hull.len();
        for e in 0..n {
            let (a, b) = (hull[e], hull[(e + 1) % n]);
            if (a.0 - y).abs() < eps {
                lo = lo.min(a.1);
                hi = hi.max(a.1);
            }
            if (a.0 - y) * (b.0 - y) < 0.0 {
                let x = a.1 + (y - a.0) / (b.0 - a.0) * (b.1 - a.1);
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        if lo <= hi {
            let k0 = (lo - eps).ceil().max(0.0) as usize;
            let k1 = ((hi + eps).floor() as usize).min(nk - 1);
            for k in k0..=k1 {
                out[[j, k]] = true;
            }
        }
    }
}
