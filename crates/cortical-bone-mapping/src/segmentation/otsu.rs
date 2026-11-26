//! Thresholds from the image histogram by (multi-level) Otsu's method.
//!
//! Otsu (1979) chooses the thresholds that split the histogram into classes
//! with the largest between-class variance. With the normalised histogram `h`
//! over bins with centres `c_b`, a class `C = [a, b)` of bins has weight
//! `ω_C = Σ_{b∈C} h_b` and mean `μ_C = Σ_{b∈C} h_b c_b / ω_C`, and the
//! between-class variance of a split into classes `C_1 … C_K` is
//!
//! ```text
//! σ_B² = Σ_k ω_k (μ_k − μ_T)²,     μ_T = Σ_b h_b c_b
//! ```
//!
//! Maximising it is equivalent to minimising the within-class variance, since
//! the two add up to the total. With cumulative sums of `h` and `h c` every
//! class term is O(1) (Liao et al. 2001), so the exhaustive search costs O(B)
//! for two classes and O(B²) for three (`B` bins). Thresholds are placed at
//! bin edges; a threshold anywhere in an empty stretch of the histogram gives
//! the same `σ_B²`, so the middle of the tied candidates is returned.
//!
//! For a non-calibrated CT (Hounsfield units) the histogram has peaks for fat
//! and soft tissue but none for bone, whose values form a long tail: with
//! `floor` excluding the air, the two-class threshold separates soft tissue
//! from bone.
//!
//! # References
//!
//! * N. Otsu, *A threshold selection method from gray-level histograms*, IEEE
//!   Transactions on Systems, Man, and Cybernetics 9(1):62–66, 1979.
//! * P.-S. Liao, T.-S. Chen, P.-C. Chung, *A fast algorithm for multilevel
//!   thresholding*, Journal of Information Science and Engineering
//!   17(5):713–727, 2001.

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
/// [`label_bones`](super::label_bones) and the hull. With three classes the upper threshold is
/// higher (fat | soft tissue | bone, with the bone class dominated by cortex).
pub fn otsu_thresholds<T: Voxel>(image: ArrayView3<T>, o: &OtsuOptions) -> Vec<f64> {
    assert!(o.classes == 2 || o.classes == 3, "classes must be 2 or 3");
    assert!(o.bins >= 4, "bins must be at least 4");
    let floor = o.floor.unwrap_or(f64::NEG_INFINITY);
    let (lo, hi) = value_range(&image, floor);
    if !(hi > lo) {
        return Vec::new();
    }
    let w = (hi - lo) / o.bins as f64;
    let moments = Moments::new(&histogram(&image, floor, lo, w, o.bins), lo, w);
    let cands = moments.candidates(o.classes);
    let Some(best) = cands.iter().map(|c| c.0).max_by(f64::total_cmp) else {
        log::warn!("Otsu: no candidate thresholds");
        return Vec::new();
    };
    // a threshold anywhere in an empty stretch of the histogram scores the
    // same: take the middle of the tied candidates
    let tied: Vec<&Vec<usize>> = cands
        .iter()
        .filter(|c| c.0 >= best - 1e-12 * best.abs())
        .map(|c| &c.1)
        .collect();
    let out: Vec<f64> = (0..o.classes - 1)
        .map(|a| {
            let lo_i = tied.iter().map(|t| t[a]).min().unwrap();
            let hi_i = tied.iter().map(|t| t[a]).max().unwrap();
            lo + 0.5 * (lo_i + hi_i) as f64 * w
        })
        .collect();
    log::debug!(
        "Otsu ({} classes, floor {:?}, values {lo:.1} to {hi:.1}, {} bins): thresholds {out:.1?}",
        o.classes,
        o.floor,
        o.bins
    );
    out
}

/// Smallest and largest finite value `≥ floor`.
fn value_range<T: Voxel>(image: &ArrayView3<T>, floor: f64) -> (f64, f64) {
    let none = (f64::INFINITY, f64::NEG_INFINITY);
    let widen = |a: (f64, f64), v: f64| (a.0.min(v), a.1.max(v));
    let counted = |v: &f64| *v >= floor && v.is_finite();
    match image.as_slice_memory_order() {
        Some(s) => s
            .par_iter()
            .map(|v| v.to_f64())
            .filter(counted)
            .fold(|| none, widen)
            .reduce(|| none, |a, b| (a.0.min(b.0), a.1.max(b.1))),
        None => image.iter().map(|v| v.to_f64()).filter(counted).fold(none, widen),
    }
}

/// Histogram of the values `≥ floor` in `bins` bins of width `w` from `lo`.
fn histogram<T: Voxel>(image: &ArrayView3<T>, floor: f64, lo: f64, w: f64, bins: usize) -> Vec<f64> {
    let bin = |v: f64| (((v - lo) / w) as usize).min(bins - 1);
    let add = |mut h: Vec<f64>, v: f64| {
        if v >= floor && v.is_finite() {
            h[bin(v)] += 1.0;
        }
        h
    };
    let merge = |a: Vec<f64>, b: Vec<f64>| a.iter().zip(&b).map(|(x, y)| x + y).collect::<Vec<f64>>();
    match image.as_slice_memory_order() {
        Some(s) => s
            .par_iter()
            .map(|v| v.to_f64())
            .fold(|| vec![0.0; bins], add)
            .reduce(|| vec![0.0; bins], merge),
        None => image.iter().map(|v| v.to_f64()).fold(vec![0.0; bins], add),
    }
}

/// Cumulative weight and first moment of the normalised histogram; classes
/// are bin ranges `[a, b)`.
struct Moments {
    pc: Vec<f64>,
    mc: Vec<f64>,
}

impl Moments {
    /// Cumulative sums of the normalised histogram and of its first moment over
    /// the bin centres `lo + (i + ½) w`, so that any class `[a, b)` has weight
    /// `pc[b] − pc[a]` and moment `mc[b] − mc[a]` (Liao et al. 2001).
    fn new(h: &[f64], lo: f64, w: f64) -> Self {
        let n = h.len();
        let total: f64 = h.iter().sum();
        let mut pc = vec![0.0; n + 1];
        let mut mc = vec![0.0; n + 1];
        for i in 0..n {
            let c = lo + (i as f64 + 0.5) * w;
            pc[i + 1] = pc[i] + h[i] / total;
            mc[i + 1] = mc[i] + h[i] / total * c;
        }
        Moments { pc, mc }
    }

    /// The between-class variance term `ω_C (μ_C − μ_T)²` of class `[a, b)`;
    /// `None` if it is empty.
    fn term(&self, a: usize, b: usize) -> Option<f64> {
        let mt = self.mc[self.mc.len() - 1];
        let wk = self.pc[b] - self.pc[a];
        if wk <= 0.0 {
            None
        } else {
            let mk = (self.mc[b] - self.mc[a]) / wk;
            Some(wk * (mk - mt) * (mk - mt))
        }
    }

    /// Every split into `classes` non-empty classes (bin indices of the
    /// thresholds) with its between-class variance.
    fn candidates(&self, classes: usize) -> Vec<(f64, Vec<usize>)> {
        let n = self.pc.len() - 1;
        if classes == 2 {
            return (1..n)
                .filter_map(|i| Some((self.term(0, i)? + self.term(i, n)?, vec![i])))
                .collect();
        }
        (1..n - 1)
            .into_par_iter()
            .flat_map_iter(|i| {
                let t0 = self.term(0, i);
                (i + 1..n).filter_map(move |j| Some((t0? + self.term(i, j)? + self.term(j, n)?, vec![i, j])))
            })
            .collect()
    }
}
