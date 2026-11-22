//! Thresholds from the image histogram by (multi-level) Otsu's method.

use ndarray::ArrayView3;
use rayon::prelude::*;

use crate::volume::Voxel;

/// Options of [`otsu_thresholds`].
#[derive(Clone, Debug, PartialEq)]
pub struct OtsuOptions {
    /// Only voxels `≥ floor` enter the histogram (e.g. −500 HU to exclude the
    /// air around the body); `None` uses every voxel.
    pub floor: Option<f64>,
    /// Number of classes: 2 (one threshold) or 3 (two thresholds).
    pub classes: usize,
    /// Number of histogram bins between the smallest and largest value.
    pub bins: usize,
}

impl Default for OtsuOptions {
    fn default() -> Self {
        OtsuOptions {
            floor: None,
            classes: 2,
            bins: 1024,
        }
    }
}

/// Thresholds from the image histogram by (multi-level) Otsu's method: the
/// thresholds that maximise the between-class variance of the voxel values
/// (Otsu 1979; exhaustive search for 3 classes). Returned in ascending order
/// and placed at bin edges; empty if no voxel passes `floor`.
///
/// For a non-calibrated CT (Hounsfield units) the histogram has peaks for
/// fat and soft tissue but none for bone, whose values form a long tail;
/// with `floor` excluding the air, the two-class threshold separates soft
/// tissue from bone, and is a reasonable starting threshold for
/// [`label_bones`] and the hull. With three classes the upper threshold is
/// higher (fat | soft tissue | bone, with the bone class dominated by cortex).
pub fn otsu_thresholds<T: Voxel>(image: ArrayView3<T>, o: &OtsuOptions) -> Vec<f64> {
    assert!(o.classes == 2 || o.classes == 3, "classes must be 2 or 3");
    assert!(o.bins >= 4, "bins must be at least 4");
    let floor = o.floor.unwrap_or(f64::NEG_INFINITY);
    let (lo, hi) = image
        .as_slice_memory_order()
        .map(|s| {
            s.par_iter()
                .map(|v| v.to_f64())
                .filter(|v| *v >= floor && v.is_finite())
                .fold(|| (f64::INFINITY, f64::NEG_INFINITY), |a, v| (a.0.min(v), a.1.max(v)))
                .reduce(
                    || (f64::INFINITY, f64::NEG_INFINITY),
                    |a, b| (a.0.min(b.0), a.1.max(b.1)),
                )
        })
        .unwrap_or_else(|| {
            image
                .iter()
                .map(|v| v.to_f64())
                .filter(|v| *v >= floor && v.is_finite())
                .fold((f64::INFINITY, f64::NEG_INFINITY), |a, v| (a.0.min(v), a.1.max(v)))
        });
    if !(hi > lo) {
        return Vec::new();
    }
    let n = o.bins;
    let w = (hi - lo) / n as f64;
    let bin = |v: f64| (((v - lo) / w) as usize).min(n - 1);
    let add = |mut h: Vec<f64>, v: f64| {
        if v >= floor && v.is_finite() {
            h[bin(v)] += 1.0;
        }
        h
    };
    let merge = |a: Vec<f64>, b: Vec<f64>| a.iter().zip(&b).map(|(x, y)| x + y).collect::<Vec<f64>>();
    let h = match image.as_slice_memory_order() {
        Some(s) => s
            .par_iter()
            .map(|v| v.to_f64())
            .fold(|| vec![0.0; n], add)
            .reduce(|| vec![0.0; n], merge),
        None => image.iter().map(|v| v.to_f64()).fold(vec![0.0; n], add),
    };
    let total: f64 = h.iter().sum();
    // cumulative weight and first moment; classes are bin ranges [a, b)
    let mut pc = vec![0.0; n + 1];
    let mut mc = vec![0.0; n + 1];
    for i in 0..n {
        let c = lo + (i as f64 + 0.5) * w;
        pc[i + 1] = pc[i] + h[i] / total;
        mc[i + 1] = mc[i] + h[i] / total * c;
    }
    let mt = mc[n];
    let term = |a: usize, b: usize| {
        let wk = pc[b] - pc[a];
        if wk <= 0.0 {
            None
        } else {
            let mk = (mc[b] - mc[a]) / wk;
            Some(wk * (mk - mt) * (mk - mt))
        }
    };
    let edge = |i: f64| lo + i * w;
    // all candidate splits with their between-class variance
    let cands: Vec<(f64, Vec<usize>)> = if o.classes == 2 {
        (1..n)
            .filter_map(|i| Some((term(0, i)? + term(i, n)?, vec![i])))
            .collect()
    } else {
        (1..n - 1)
            .into_par_iter()
            .flat_map_iter(|i| {
                let t0 = term(0, i);
                (i + 1..n).filter_map(move |j| Some((t0? + term(i, j)? + term(j, n)?, vec![i, j])))
            })
            .collect()
    };
    let Some(best) = cands.iter().map(|c| c.0).max_by(f64::total_cmp) else {
        return Vec::new();
    };
    // a threshold anywhere in an empty stretch of the histogram scores the
    // same: take the middle of the tied candidates
    let tied: Vec<&Vec<usize>> = cands
        .iter()
        .filter(|c| c.0 >= best - 1e-12 * best.abs())
        .map(|c| &c.1)
        .collect();
    (0..o.classes - 1)
        .map(|a| {
            let lo_i = tied.iter().map(|t| t[a]).min().unwrap();
            let hi_i = tied.iter().map(|t| t[a]).max().unwrap();
            edge(0.5 * (lo_i + hi_i) as f64)
        })
        .collect()
}
