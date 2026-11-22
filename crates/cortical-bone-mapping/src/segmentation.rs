//! Buie's dual-threshold cortical/trabecular segmentation and the binary
//! morphology it needs.
//!
//! Implements the algorithm of
//!
//! > H. R. Buie, G. M. Campbell, R. J. Klinck, J. A. MacNeil, S. K. Boyd,
//! > *Automatic segmentation of cortical and trabecular compartments based on
//! > a dual threshold technique for in vivo micro-CT bone analysis*, Bone
//! > 41(4):505–515, 2007, doi:10.1016/j.bone.2007.07.007
//!
//! following its Fig. 1 and Methods. Step numbers are those of Fig. 1.
//!
//! **Step 1, the periosteal surface.**
//! 2. threshold at user input 1 (bone ≥ T1);
//! 3. median, kernel (3,3,1);
//! 4. dilate, kernel (15,15,1), closing Volkmann's canals;
//! 5. connectivity: select the non-bone (black) region outside the bone;
//!    everything else is bone (the trabecular region is enclosed by cortex);
//! 6. erode, kernel (15,15,1) → periosteal mask.
//!
//! **Step 2, the endosteal surface.**
//! 7. threshold the original data at user input 2;
//! 8. mask with the region found in step 1, leaving only the marrow
//!    cavities (below T2 inside the periosteal mask);
//! 9. dilate the marrow cavities, kernel (10,10,1), reconnecting them and
//!    eliminating trabeculae;
//! 10. connectivity: select the outer white (non-marrow) region;
//! 11. erode the marrow, kernel (10,10,1);
//! 12. Gaussian smooth, σ = 3, support (3,3,1);
//! 13. threshold by lower = 100 (on the 0/127-valued mask);
//! 14. mask: cortex = periosteal ∧ non-marrow; trabecular = periosteal ∧ marrow.
//!
//! Implementation notes and interpretations (the paper's implementation was
//! VTK; where the paper is silent the VTK filter semantics are used):
//!
//! * Kernel sizes in the paper are VTK `(x, y, z)`; here they are per array
//!   axis `(axis 0, axis 1, axis 2)` = `(z, y, x)`, so `(3,3,1)` → `[1, 3, 3]`.
//!   As in the paper, kernels are in voxels and are not rescaled with
//!   resolution (the paper used the same kernels from 10 to 82 µm).
//! * Dilation/erosion use the ellipsoidal footprint of `vtkImageDilateErode3D`
//!   (semi-axes `(k − 1)/2`), computed exactly with a separable squared
//!   Euclidean distance transform (cost independent of kernel size).
//! * Gaussian "support" is VTK's radius factor: kernel radius
//!   `⌊σ · factor⌋` voxels per axis (9 in-plane, 3 along the scan axis for
//!   σ = 3), weights normalised, edges clamped.
//! * Step 13: masks take the values 0/127 (Scanco convention, assumed), so
//!   "threshold by lower = 100" means a smoothed fraction ≤ 100/127 is marrow.
//! * "Outer" regions (steps 5 and 10) are the connected components touching
//!   the in-plane faces of the (padded) image. 3D (6-connected) by default;
//!   2D per-slice connectivity is optional (the paper reports identical results
//!   in the typical case).
//! * Pre-processing as described in the paper: optionally keep only the largest
//!   bone component (removing the fibula/ulna), and pad the data in-plane by the
//!   dilation radius so that dilation cannot clip at the image edge.
//! * Thresholds are inputs in the image's units. The paper used native scanner
//!   units (human XtremeCT: 3000 and 5000; the endosteal threshold is usually
//!   the higher of the two). For density-calibrated HR-pQCT, 450 and
//!   320 mg HA/cm³ are commonly used (Burghardt et al. 2010).

use std::collections::VecDeque;

use ndarray::{s, Array3, ArrayView3, Axis, Zip};
use rayon::prelude::*;

use crate::volume::Voxel;

/// Parameters of [`buie`]. Kernels are in voxels per array axis `(z, y, x)`.
#[derive(Clone, Debug, PartialEq)]
pub struct BuieOptions {
    /// User input 1 (step 2): image ≥ T1 is bone.
    pub periosteal_threshold: f64,
    /// User input 2 (step 7): image < T2 inside the periosteal mask is marrow.
    pub endosteal_threshold: f64,
    /// Step 3 median kernel.
    pub median_kernel: [usize; 3],
    /// Steps 4 and 6 dilation/erosion kernel.
    pub periosteal_kernel: [usize; 3],
    /// Steps 9 and 11 dilation/erosion kernel.
    pub endosteal_kernel: [usize; 3],
    /// Step 12 Gaussian standard deviation (voxels).
    pub gaussian_sigma: f64,
    /// Step 12 support (radius factors) per axis.
    pub gaussian_support: [f64; 3],
    /// Step 13 threshold as a fraction of the mask value (100/127).
    pub smoothed_threshold: f64,
    /// Connectivity in 2D (per slice along axis 0) instead of 3D.
    pub connectivity_2d: bool,
    /// Pre-processing: keep only the largest bone component.
    pub largest_bone_only: bool,
}

impl BuieOptions {
    /// The paper's filter parameters with the given thresholds.
    pub fn new(periosteal_threshold: f64, endosteal_threshold: f64) -> Self {
        BuieOptions {
            periosteal_threshold,
            endosteal_threshold,
            median_kernel: [1, 3, 3],
            periosteal_kernel: [1, 15, 15],
            endosteal_kernel: [1, 10, 10],
            gaussian_sigma: 3.0,
            gaussian_support: [1.0, 3.0, 3.0],
            smoothed_threshold: 100.0 / 127.0,
            connectivity_2d: false,
            largest_bone_only: true,
        }
    }
}

impl Default for BuieOptions {
    /// The paper's filter parameters with 450 / 320 mg HA/cm³ thresholds,
    /// commonly used for density-calibrated HR-pQCT.
    fn default() -> Self {
        BuieOptions::new(450.0, 320.0)
    }
}

/// Masks from [`buie`].
pub struct BuieResult {
    /// Step 6: the region inside the periosteal surface.
    pub periosteal: Array3<bool>,
    /// Step 14: cortical compartment.
    pub cortical: Array3<bool>,
    /// Step 14: trabecular compartment.
    pub trabecular: Array3<bool>,
}

/// Buie's dual-threshold segmentation.
pub fn buie<T: Voxel>(image: ArrayView3<T>, o: &BuieOptions) -> BuieResult {
    let (d0, d1, d2) = image.dim();
    // pad in-plane by the largest dilation radius (+1 so that the outside is
    // connected around the bone)
    let pad = [0, 1, 2].map(|a| {
        if a == 0 {
            0
        } else {
            o.periosteal_kernel[a].max(o.endosteal_kernel[a]) / 2 + 2
        }
    });
    let shape = (d0, d1 + 2 * pad[1], d2 + 2 * pad[2]);
    let inner = s![.., pad[1]..pad[1] + d1, pad[2]..pad[2] + d2];
    let conn = |m: &Array3<bool>| outer_region(m, o.connectivity_2d);

    // ---- step 1: periosteal surface
    // 2. threshold
    let mut s2 = Array3::from_elem(shape, false);
    Zip::from(s2.slice_mut(inner))
        .and(&image)
        .par_for_each(|b, &v| *b = v.to_f64() >= o.periosteal_threshold);
    // 3. median
    let mut s3 = binary_median(&s2, o.median_kernel);
    drop(s2);
    // pre-processing: keep the largest bone before any dilation, so that a
    // nearby fibula / ulna cannot be merged with it by the closing
    if o.largest_bone_only {
        s3 = largest_component(&s3, o.connectivity_2d);
    }
    // 4. dilate
    let s4 = dilate(&s3, o.periosteal_kernel);
    drop(s3);
    // 5. select the outer non-bone region; bone is everything else
    let bg = s4.mapv(|b| !b);
    drop(s4);
    let s5 = conn(&bg).mapv(|b| !b);
    drop(bg);
    // 6. erode
    let periosteal = erode(&s5, o.periosteal_kernel);
    drop(s5);

    // ---- step 2: endosteal surface
    // 7–8. marrow cavities: below T2 inside the periosteal mask
    let mut marrow = Array3::from_elem(shape, false);
    Zip::from(marrow.slice_mut(inner))
        .and(&image)
        .and(periosteal.slice(inner))
        .par_for_each(|m, &v, &p| *m = p && v.to_f64() < o.endosteal_threshold);
    // 9. dilate the marrow
    let s9 = dilate(&marrow, o.endosteal_kernel);
    drop(marrow);
    // 10. select the outer white (non-marrow) region
    let white = s9.mapv(|b| !b);
    drop(s9);
    let s10_white = conn(&white);
    drop(white);
    // 11. erode the marrow = dilate the white region
    let s11_white = dilate(&s10_white, o.endosteal_kernel);
    drop(s10_white);
    // 12. Gaussian smooth, 13. threshold by lower (marrow where ≤ threshold)
    let r = o.gaussian_support.map(|f| (o.gaussian_sigma * f).floor() as usize);
    let sm = gaussian_truncated(&s11_white, o.gaussian_sigma, r);
    drop(s11_white);
    let non_marrow = sm.mapv(|v| v as f64 > o.smoothed_threshold);
    drop(sm);
    // 14. combine
    let mut cortical = Array3::from_elem((d0, d1, d2), false);
    let mut trabecular = Array3::from_elem((d0, d1, d2), false);
    Zip::from(&mut cortical)
        .and(&mut trabecular)
        .and(periosteal.slice(inner))
        .and(non_marrow.slice(inner))
        .par_for_each(|c, t, &p, &n| {
            *c = p && n;
            *t = p && !n;
        });
    BuieResult {
        periosteal: periosteal.slice(inner).to_owned(),
        cortical,
        trabecular,
    }
}

/// Binary median (majority vote) over a box kernel, edges clamped.
pub fn binary_median(m: &Array3<bool>, k: [usize; 3]) -> Array3<bool> {
    let (d0, d1, d2) = m.dim();
    let r = k.map(|v| (v / 2) as isize);
    let n: usize = k.iter().product();
    let need = n / 2 + 1;
    let mut out = Array3::from_elem(m.dim(), false);
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut sl)| {
            for j in 0..d1 {
                for l in 0..d2 {
                    let mut c = 0;
                    for a in -r[0]..=r[0] {
                        let ii = (i as isize + a).clamp(0, d0 as isize - 1) as usize;
                        for b in -r[1]..=r[1] {
                            let jj = (j as isize + b).clamp(0, d1 as isize - 1) as usize;
                            for e in -r[2]..=r[2] {
                                let ll = (l as isize + e).clamp(0, d2 as isize - 1) as usize;
                                c += m[[ii, jj, ll]] as usize;
                            }
                        }
                    }
                    sl[[j, l]] = c >= need;
                }
            }
        });
    out
}

/// Squared scaled distance to the nearest `true` voxel, with per-axis
/// semi-axes `radii` (voxels): `Σ_a (Δ_a / r_a)²`. Axes with radius 0 do not
/// propagate. Exact (Felzenszwalb–Huttenlocher lower envelope per axis).
pub fn scaled_sq_distance(m: &Array3<bool>, radii: [f64; 3]) -> Array3<f32> {
    let mut d = m.mapv(|b| if b { 0.0f32 } else { f32::INFINITY });
    for (axis, &r) in radii.iter().enumerate() {
        if !(r > 0.0) {
            continue;
        }
        let w = 1.0 / (r * r);
        let len = d.len_of(Axis(axis));
        Zip::from(d.lanes_mut(Axis(axis))).par_for_each(|mut lane| {
            let f: Vec<f64> = lane.iter().map(|&v| v as f64).collect();
            let mut out = vec![f64::INFINITY; len];
            edt_1d(&f, w, &mut out);
            lane.iter_mut().zip(out).for_each(|(o, v)| *o = v as f32);
        });
    }
    d
}

/// 1D lower envelope of parabolas `w (q − p)² + f(p)`.
fn edt_1d(f: &[f64], w: f64, out: &mut [f64]) {
    let n = f.len();
    let mut v: Vec<usize> = Vec::with_capacity(n);
    let mut z: Vec<f64> = Vec::with_capacity(n + 1);
    for q in 0..n {
        if !f[q].is_finite() {
            continue;
        }
        loop {
            match v.last() {
                None => {
                    v.push(q);
                    z.clear();
                    z.push(f64::NEG_INFINITY);
                    break;
                }
                Some(&p) => {
                    let s =
                        ((f[q] + w * (q * q) as f64) - (f[p] + w * (p * p) as f64)) / (2.0 * w * (q as f64 - p as f64));
                    if s <= *z.last().unwrap() {
                        v.pop();
                        z.pop();
                        if v.is_empty() {
                            continue;
                        }
                    } else {
                        v.push(q);
                        z.push(s);
                        break;
                    }
                }
            }
        }
    }
    if v.is_empty() {
        out.iter_mut().for_each(|o| *o = f64::INFINITY);
        return;
    }
    z.push(f64::INFINITY);
    let mut k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        *o = w * (q as f64 - p as f64).powi(2) + f[p];
    }
}

fn radii(k: [usize; 3]) -> [f64; 3] {
    k.map(|v| (v.max(1) - 1) as f64 / 2.0)
}

/// Binary dilation with an ellipsoidal footprint of `kernel` voxels per axis
/// (as `vtkImageDilateErode3D`).
pub fn dilate(m: &Array3<bool>, kernel: [usize; 3]) -> Array3<bool> {
    let d = scaled_sq_distance(m, radii(kernel));
    d.mapv(|v| v <= 1.0 + 1e-6)
}

/// Binary erosion with an ellipsoidal footprint (voxels outside the image do
/// not erode).
pub fn erode(m: &Array3<bool>, kernel: [usize; 3]) -> Array3<bool> {
    let inv = m.mapv(|b| !b);
    dilate(&inv, kernel).mapv(|b| !b)
}

/// Label 6-connected components (4-connected within slices if `two_d`).
/// Returns labels (0 = background) and component sizes (index = label).
fn label_components(m: &Array3<bool>, two_d: bool) -> (Vec<u32>, Vec<usize>) {
    let (d0, d1, d2) = m.dim();
    let n = d0 * d1 * d2;
    let s = m.as_standard_layout();
    let flat = s.as_slice().unwrap();
    let mut label = vec![0u32; n];
    let mut sizes = vec![0usize];
    let mut queue = VecDeque::new();
    for start in 0..n {
        if !flat[start] || label[start] != 0 {
            continue;
        }
        let id = sizes.len() as u32;
        let mut size = 0;
        label[start] = id;
        queue.push_back(start);
        while let Some(p) = queue.pop_front() {
            size += 1;
            let (i, j, k) = (p / (d1 * d2), (p / d2) % d1, p % d2);
            let mut visit = |q: usize| {
                if flat[q] && label[q] == 0 {
                    label[q] = id;
                    queue.push_back(q);
                }
            };
            if !two_d {
                if i > 0 {
                    visit(p - d1 * d2);
                }
                if i + 1 < d0 {
                    visit(p + d1 * d2);
                }
            }
            if j > 0 {
                visit(p - d2);
            }
            if j + 1 < d1 {
                visit(p + d2);
            }
            if k > 0 {
                visit(p - 1);
            }
            if k + 1 < d2 {
                visit(p + 1);
            }
        }
        sizes.push(size);
    }
    (label, sizes)
}

/// The largest connected component of `m` (per slice if `two_d`).
pub fn largest_component(m: &Array3<bool>, two_d: bool) -> Array3<bool> {
    if two_d {
        return per_slice(m, |s| largest_component(s, false));
    }
    let (label, sizes) = label_components(m, false);
    let best = (1..sizes.len()).max_by_key(|&i| sizes[i]).unwrap_or(0) as u32;
    let out: Vec<bool> = label.par_iter().map(|&l| l != 0 && l == best).collect();
    Array3::from_shape_vec(m.dim(), out).unwrap()
}

/// The connected components of `m` touching the in-plane faces of the volume
/// (axes 1 and 2), i.e. the "outer" region (per slice if `two_d`).
pub fn outer_region(m: &Array3<bool>, two_d: bool) -> Array3<bool> {
    if two_d {
        return per_slice(m, |s| outer_region(s, false));
    }
    let (d0, d1, d2) = m.dim();
    let (label, sizes) = label_components(m, false);
    let mut outer = vec![false; sizes.len()];
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                if j == 0 || k == 0 || j + 1 == d1 || k + 1 == d2 {
                    outer[label[(i * d1 + j) * d2 + k] as usize] = true;
                }
            }
        }
    }
    outer[0] = false;
    let out: Vec<bool> = label.par_iter().map(|&l| outer[l as usize]).collect();
    Array3::from_shape_vec(m.dim(), out).unwrap()
}

/// Fill the holes of every slice perpendicular to `axis`.
fn fill_slice_holes(m: &Array3<bool>, axis: usize) -> Array3<bool> {
    let order = match axis {
        0 => [0, 1, 2],
        1 => [1, 0, 2],
        _ => [2, 0, 1],
    };
    let p = m.view().permuted_axes(order).as_standard_layout().to_owned();
    let filled = outer_region(&p.mapv(|b| !b), true).mapv(|b| !b);
    // undo the permutation
    let mut inv = [0usize; 3];
    for (i, &o) in order.iter().enumerate() {
        inv[o] = i;
    }
    filled.permuted_axes(inv).as_standard_layout().to_owned()
}

fn per_slice(m: &Array3<bool>, f: impl Fn(&Array3<bool>) -> Array3<bool> + Sync) -> Array3<bool> {
    let mut out = Array3::from_elem(m.dim(), false);
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(m.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut o, sl)| {
            let one = sl.to_owned().insert_axis(Axis(0));
            o.assign(&f(&one).index_axis(Axis(0), 0));
        });
    out
}

/// Gaussian smoothing of a binary mask with standard deviation `sigma`
/// (voxels) and kernel radius `r` voxels per axis (0 = no smoothing along
/// that axis), edges clamped, as `vtkImageGaussianSmooth`.
pub fn gaussian_truncated(m: &Array3<bool>, sigma: f64, r: [usize; 3]) -> Array3<f32> {
    let mut f = m.mapv(|b| b as u8 as f32);
    for (axis, &ra) in r.iter().enumerate() {
        if ra == 0 || !(sigma > 0.0) {
            continue;
        }
        let ra = ra as isize;
        let k: Vec<f64> = (-ra..=ra).map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp()).collect();
        let ks: f64 = k.iter().sum();
        let k: Vec<f64> = k.iter().map(|v| v / ks).collect();
        let len = f.len_of(Axis(axis)) as isize;
        Zip::from(f.lanes_mut(Axis(axis))).par_for_each(|mut lane| {
            let src: Vec<f32> = lane.iter().copied().collect();
            for (q, o) in lane.iter_mut().enumerate() {
                let mut acc = 0.0;
                for (t, &w) in k.iter().enumerate() {
                    let j = (q as isize + t as isize - ra).clamp(0, len - 1) as usize;
                    acc += w * src[j] as f64;
                }
                *o = acc as f32;
            }
        });
    }
    f
}

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
    slice_hulls(&m)
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

/// Options of [`label_bones`].
#[derive(Clone, Debug, PartialEq)]
pub struct BoneLabelOptions {
    /// Bone threshold (image units), e.g. from [`otsu_thresholds`].
    pub threshold: f64,
    /// Binary median kernel against noise speckle.
    pub median_kernel: [usize; 3],
    /// Components smaller than this (voxels) are dropped.
    pub min_size: usize,
    /// Keep at most this many bones (the largest).
    pub max_bones: Option<usize>,
}

impl Default for BoneLabelOptions {
    fn default() -> Self {
        BoneLabelOptions {
            threshold: 320.0,
            median_kernel: [1, 3, 3],
            min_size: 1000,
            max_bones: None,
        }
    }
}

/// Bones labelled by [`label_bones`].
#[derive(Clone, Debug)]
pub struct BoneLabels {
    /// 0 = background, `k` = the `k`-th largest bone.
    pub labels: Array3<u16>,
    /// Size (voxels) of bone `k` at index `k − 1`.
    pub sizes: Vec<usize>,
}

/// Separate the bones of an image: threshold, binary median, 6-connected
/// components, keep those of at least `min_size` voxels (at most
/// `max_bones`), labelled 1, 2, … from the largest.
///
/// Bones that touch at the threshold (blur, noise or motion across a thin
/// soft-tissue gap) become one component; raise the threshold, or separate
/// them by other means, if that happens.
pub fn label_bones<T: Voxel>(image: ArrayView3<T>, o: &BoneLabelOptions) -> BoneLabels {
    let mut m = Array3::from_elem(image.dim(), false);
    Zip::from(&mut m)
        .and(&image)
        .par_for_each(|b, &v| *b = v.to_f64() >= o.threshold);
    if o.median_kernel.iter().any(|&k| k > 1) {
        m = binary_median(&m, o.median_kernel);
    }
    let (label, sizes) = label_components(&m, false);
    let mut order: Vec<usize> = (1..sizes.len()).filter(|&i| sizes[i] >= o.min_size.max(1)).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    order.truncate(o.max_bones.unwrap_or(usize::MAX).min(u16::MAX as usize));
    let mut map = vec![0u16; sizes.len()];
    for (k, &c) in order.iter().enumerate() {
        map[c] = (k + 1) as u16;
    }
    let out: Vec<u16> = label.par_iter().map(|&l| map[l as usize]).collect();
    BoneLabels {
        labels: Array3::from_shape_vec(m.dim(), out).unwrap(),
        sizes: order.iter().map(|&c| sizes[c]).collect(),
    }
}

/// Starting segmentations for several bones: the per-slice convex hull
/// ([`slice_hulls`]) of every labelled bone, kept apart.
///
/// Hulls of neighbouring bones overlap where a concave surface faces
/// another bone (the femoral trochlea and the patella, the intercondylar
/// notch and the tibial eminence). Each voxel claimed by several hulls goes
/// to the bone whose thresholded voxels are nearest (Euclidean distance in
/// physical units, `spacing` per array axis); then hull voxels that touch
/// (6-neighbourhood) another bone's region are cleared, unless they belong
/// to their own thresholded bone, so that every bone is a separate closed
/// surface. Each label still contains its thresholded bone.
///
/// With `max_distance` (physical units), each hull is first trimmed to the
/// voxels within that distance of its bone and the holes of every slice are
/// filled. The marrow cavity, and gaps narrower than twice the distance
/// (cortical breaks), stay closed because the cortex encloses them; a wide,
/// open concavity, such as the femoral intercondylar notch, opens up instead
/// of being bridged by the hull. Use it when a concavity is deeper than the
/// inward sampling range.
///
/// With `closing_radius` (physical units), each bone's start is its
/// morphological closing by a ball of that radius instead of its per-slice
/// convex hull (a "local convex hull"): concavities narrower than about twice
/// the radius are filled, wider ones (e.g. the intercondylar notch with a
/// radius of a few mm) are kept, and the start follows the curvature of the
/// bone at larger scales. Slice holes (the marrow cavity) are then filled.
pub fn bone_hulls(
    labels: ArrayView3<u16>,
    spacing: [f64; 3],
    max_distance: Option<f64>,
    closing_radius: Option<f64>,
) -> Array3<u16> {
    let dim = labels.dim();
    let n = labels.iter().copied().max().unwrap_or(0);
    let mut owner = Array3::<u16>::zeros(dim);
    let mut best = Array3::from_elem(dim, f32::INFINITY);
    let inv = spacing.map(|s| 1.0 / s);
    for b in 1..=n {
        let mb = labels.mapv(|l| l == b);
        if !mb.iter().any(|&v| v) {
            continue;
        }
        let d = scaled_sq_distance(&mb, inv);
        let mut hull = match closing_radius {
            Some(r) => closing(&mb, &d, inv, r),
            None => slice_hulls(&mb),
        };
        if let Some(r) = max_distance {
            let r2 = (r * r) as f32;
            Zip::from(&mut hull).and(&d).par_for_each(|h, &dv| *h &= dv <= r2);
            hull = outer_region(&hull.mapv(|b| !b), true).mapv(|b| !b);
        }
        Zip::from(&mut owner)
            .and(&mut best)
            .and(&hull)
            .and(&d)
            .par_for_each(|o, bst, &h, &dv| {
                if h && dv < *bst {
                    *bst = dv;
                    *o = b;
                }
            });
    }
    let (d0, d1, d2) = dim;
    let mut out = owner.clone();
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
        let b = owner[[i, j, k]];
        if b == 0 || labels[[i, j, k]] == b {
            return;
        }
        let other = |q: [usize; 3]| {
            let c = owner[q];
            c != 0 && c != b
        };
        let touches = (i > 0 && other([i - 1, j, k]))
            || (i + 1 < d0 && other([i + 1, j, k]))
            || (j > 0 && other([i, j - 1, k]))
            || (j + 1 < d1 && other([i, j + 1, k]))
            || (k > 0 && other([i, j, k - 1]))
            || (k + 1 < d2 && other([i, j, k + 1]));
        if touches {
            *o = 0;
        }
    });
    // keep, per bone, only the pieces that contain some of its thresholded
    // bone (the assignment and the clearing can cut off small fragments)
    for b in 1..=n {
        let mb = out.mapv(|v| v == b);
        let (comp, sizes) = label_components(&mb, false);
        if sizes.len() <= 2 {
            continue;
        }
        let mut has_bone = vec![false; sizes.len()];
        for (c, &l) in comp.iter().zip(labels.iter()) {
            if *c != 0 && l == b {
                has_bone[*c as usize] = true;
            }
        }
        for (o, c) in out.iter_mut().zip(comp.iter()) {
            if *c != 0 && !has_bone[*c as usize] {
                *o = 0;
            }
        }
    }
    out
}

/// Morphological closing of `m` by a ball of radius `r` (physical units)
/// with its holes filled: dilate by `r`, fill every background region not
/// connected to the in-plane faces of the volume (voids inside the bone,
/// e.g. sparse trabecular or fatty marrow regions below the threshold), then
/// erode by `r` (as in the periosteal mask of Buie et al. 2007). `d` is the
/// squared distance map of `m` (physical units²; `inv` = 1 / spacing).
/// Slice holes left after the erosion are filled too.
fn closing(m: &Array3<bool>, d: &Array3<f32>, inv: [f64; 3], r: f64) -> Array3<bool> {
    let r2 = (r * r) as f32;
    // dilation, then fill enclosed background (the outside is the background
    // connected to the in-plane faces, so a bone cut by the first or last
    // slice still gets its marrow filled)
    let dilated_bg = d.mapv(|v| v > r2);
    let mut filled = outer_region(&dilated_bg, false).mapv(|b| !b);
    // a void can reach the outside through a gap in a cortex the threshold
    // missed that is wider than the closing; it is still enclosed in most
    // slices through it, so fill slice holes in all three orientations
    for _ in 0..2 {
        for axis in 0..3 {
            filled = fill_slice_holes(&filled, axis);
        }
    }
    let outside = filled.mapv(|b| !b);
    // erosion: voxels farther than r from the outside (the image border
    // counts as inside)
    let e = scaled_sq_distance(&outside, inv);
    let mut c = e.mapv(|v| v > r2);
    Zip::from(&mut c).and(m).par_for_each(|c, &b| *c |= b);
    outer_region(&c.mapv(|b| !b), true).mapv(|b| !b)
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

/// Andrew's monotone chain; counter-clockwise, without collinear points.
fn convex_hull(pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
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

/// Set the pixels whose centres lie in the convex polygon `hull` (inclusive).
fn fill_convex(hull: &[(f64, f64)], out: &mut ndarray::ArrayViewMut2<bool>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn brute_dilate(m: &Array3<bool>, k: [usize; 3]) -> Array3<bool> {
        let r = radii(k);
        let (d0, d1, d2) = m.dim();
        let ri = r.map(|v| v.floor() as isize);
        Array3::from_shape_fn(m.dim(), |(i, j, l)| {
            for a in -ri[0]..=ri[0] {
                for b in -ri[1]..=ri[1] {
                    for c in -ri[2]..=ri[2] {
                        let e = [a, b, c];
                        let s: f64 = (0..3)
                            .map(|x| if r[x] > 0.0 { (e[x] as f64 / r[x]).powi(2) } else { 0.0 })
                            .sum();
                        if s > 1.0 + 1e-6 {
                            continue;
                        }
                        let (ii, jj, ll) = (i as isize + a, j as isize + b, l as isize + c);
                        if ii >= 0
                            && jj >= 0
                            && ll >= 0
                            && (ii as usize) < d0
                            && (jj as usize) < d1
                            && (ll as usize) < d2
                            && m[[ii as usize, jj as usize, ll as usize]]
                        {
                            return true;
                        }
                    }
                }
            }
            false
        })
    }

    fn random_mask(seed: u64, dims: (usize, usize, usize), p: f64) -> Array3<bool> {
        let mut r = crate::util::Rng::new(seed);
        Array3::from_shape_fn(dims, |_| r.uniform() < p)
    }

    #[test]
    fn dilation_matches_brute_force_ellipsoid() {
        for (seed, k) in [(1, [1, 5, 5]), (2, [3, 7, 4]), (3, [1, 15, 15]), (4, [5, 1, 2])] {
            let m = random_mask(seed, (9, 21, 23), 0.01);
            assert_eq!(dilate(&m, k), brute_dilate(&m, k), "kernel {k:?}");
        }
    }

    #[test]
    fn erosion_is_dual_and_median_majority() {
        let m = random_mask(5, (6, 15, 15), 0.7);
        let e = erode(&m, [1, 5, 5]);
        assert!(e.iter().zip(m.iter()).all(|(&a, &b)| !a || b));
        let mut one = Array3::from_elem((1, 5, 5), false);
        one[[0, 2, 2]] = true;
        assert!(!binary_median(&one, [1, 3, 3]).iter().any(|&b| b));
        let full = Array3::from_elem((1, 5, 5), true);
        assert!(binary_median(&full, [1, 3, 3]).iter().all(|&b| b));
    }

    #[test]
    fn components_outer_region_and_gaussian() {
        let mut m = Array3::from_elem((3, 10, 10), false);
        for j in 0..5 {
            m[[1, j, 1]] = true;
        }
        m[[1, 8, 8]] = true;
        let l = largest_component(&m, false);
        assert_eq!(l.iter().filter(|&&b| b).count(), 5);
        assert!(!l[[1, 8, 8]]);
        // a ring's outside is outer, its enclosed hole is not
        let ring = Array3::from_shape_fn((2, 12, 12), |(_, j, k)| {
            let r = ((j as f64 - 5.5).powi(2) + (k as f64 - 5.5).powi(2)).sqrt();
            r > 2.5 && r < 4.5
        });
        let bg = ring.mapv(|b| !b);
        let out = outer_region(&bg, false);
        assert!(out[[0, 0, 0]] && !out[[0, 6, 6]]);
        assert_eq!(out, outer_region(&bg, true));
        let g = gaussian_truncated(&Array3::from_elem((2, 6, 6), true), 3.0, [3, 9, 9]);
        assert!(g.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn buie_on_a_ring_with_trabeculae_and_a_second_bone() {
        // cortex ring 1000 (r 24..30), marrow 100 with bright trabecular rods,
        // a separate small "fibula" disc, outside 0
        let (d1, d2) = (110usize, 100usize);
        let img = Array3::from_shape_fn((4, d1, d2), |(_, j, k)| {
            let r = ((j as f64 - 39.5).powi(2) + (k as f64 - 49.5).powi(2)).sqrt();
            let rf = ((j as f64 - 95.0).powi(2) + (k as f64 - 49.5).powi(2)).sqrt();
            if rf < 6.0 {
                1000.0f32
            } else if r > 30.0 {
                0.0
            } else if r > 24.0 {
                1000.0
            } else if (j % 9 == 0) && r < 20.0 {
                800.0
            } else {
                100.0
            }
        });
        let res = buie(img.view(), &BuieOptions::default());
        let n = |m: &Array3<bool>| m.iter().filter(|&&b| b).count() as f64;
        let peri_area = std::f64::consts::PI * 30.0f64.powi(2) * 4.0;
        assert!(
            (n(&res.periosteal) / peri_area - 1.0).abs() < 0.05,
            "{}",
            n(&res.periosteal) / peri_area
        );
        // the second bone is removed
        assert!(!res.periosteal[[1, 95, 49]]);
        // Step 12–13 (σ = 3, re-threshold at 100/127) moves the endosteal
        // boundary into the cortex by ≈ Φ⁻¹(100/127)·σ ≈ 0.8σ = 2.4 voxels on
        // a straight edge (a little less on this concave one).
        let r_in = (30.0f64.powi(2) - n(&res.cortical) / 4.0 / std::f64::consts::PI).sqrt();
        assert!(r_in > 24.5 && r_in < 26.9, "endosteal radius {r_in} (true 24)");
        // trabeculae are trabecular, not cortex
        assert!(!res.cortical[[1, 36, 49]] && res.trabecular[[1, 36, 49]]);
        // compartments partition the periosteal mask
        assert!(res
            .periosteal
            .iter()
            .zip(res.cortical.iter().zip(res.trabecular.iter()))
            .all(|(&p, (&c, &t))| p == (c || t) && !(c && t)));
    }

    #[test]
    fn hull_contains_the_bone_and_closes_breaks() {
        // a ring with a wide gap, a speckle, a separate small disc
        let (n0, n1, n2) = (3, 80, 80);
        let img = Array3::from_shape_fn((n0, n1, n2), |(_, j, k)| {
            let (y, x) = (j as f64 - 35.0, k as f64 - 35.0);
            let r = (y * y + x * x).sqrt();
            let ring = (20.0..25.0).contains(&r) && !(x > 15.0 && y.abs() < 8.0);
            let disc = ((j as f64 - 72.0).powi(2) + (k as f64 - 72.0).powi(2)).sqrt() < 4.0;
            let speckle = j == 2 && k == 70;
            if ring || disc || speckle {
                1000.0f32
            } else {
                0.0
            }
        });
        let h = hull_segmentation(img.view(), &HullOptions::default());
        let thr = img.mapv(|v| v >= 320.0);
        for (t, (j, k)) in thr.indexed_iter().filter(|(_, &t)| t).map(|((_, j, k), t)| (t, (j, k))) {
            let in_ring = ((j as f64 - 35.0).powi(2) + (k as f64 - 35.0).powi(2)).sqrt() < 26.0;
            assert!(!*t || !in_ring || h[[1, j, k]], "ring pixel ({j},{k}) outside hull");
        }
        // the gap and the marrow are filled; the disc and the speckle are not included
        assert!(h[[1, 35, 35]] && h[[1, 35, 57]]);
        assert!(!h[[1, 72, 72]] && !h[[1, 2, 70]]);
        // the hull of a disc is about the disc
        let filled = h.index_axis(Axis(0), 1).iter().filter(|&&b| b).count() as f64;
        let area = std::f64::consts::PI * 24.5f64.powi(2);
        assert!((filled - area).abs() < 0.06 * area, "{filled} vs {area}");
    }

    #[test]
    fn convex_hull_of_a_square_with_interior_points() {
        let pts = vec![(0.0, 0.0), (0.0, 4.0), (4.0, 0.0), (4.0, 4.0), (2.0, 2.0), (0.0, 2.0)];
        let h = convex_hull(&pts);
        assert_eq!(h.len(), 4);
        let mut out = Array2::from_elem((6, 6), false);
        fill_convex(&h, &mut out.view_mut());
        assert_eq!(out.iter().filter(|&&b| b).count(), 25);
    }

    #[test]
    fn otsu_separates_a_trimodal_histogram() {
        // air (excluded by the floor), soft tissue ~0, bone ~1000 (fewer voxels)
        let mut r = crate::util::Rng::new(3);
        let img = Array3::from_shape_fn((4, 60, 60), |(_, j, k)| {
            let base = if j < 10 {
                -1000.0
            } else if k < 15 {
                1000.0
            } else if k < 35 {
                -100.0
            } else {
                40.0
            };
            (base + 20.0 * r.normal()) as f32
        });
        let o = OtsuOptions {
            floor: Some(-500.0),
            ..OtsuOptions::default()
        };
        let t2 = otsu_thresholds(img.view(), &o);
        assert_eq!(t2.len(), 1);
        assert!(t2[0] > 100.0 && t2[0] < 900.0, "{t2:?}");
        let t3 = otsu_thresholds(img.view(), &OtsuOptions { classes: 3, ..o });
        assert_eq!(t3.len(), 2);
        assert!(
            t3[0] > -80.0 && t3[0] < 20.0 && t3[1] > 100.0 && t3[1] < 900.0,
            "{t3:?}"
        );
        // nothing above the floor
        let none = otsu_thresholds(
            img.view(),
            &OtsuOptions {
                floor: Some(5000.0),
                ..OtsuOptions::default()
            },
        );
        assert!(none.is_empty());
    }

    /// Two "bones" facing each other: a C-shaped bone whose concavity holds
    /// a small disc (as the trochlea holds the patella), and a speckle.
    fn two_bones() -> Array3<f32> {
        Array3::from_shape_fn((4, 80, 80), |(_, j, k)| {
            let (y, x) = (j as f64 - 40.0, k as f64 - 35.0);
            let r = (y * y + x * x).sqrt();
            let c_shape = (15.0..22.0).contains(&r) && !(x > 5.0 && y.abs() < 12.0);
            let disc = ((j as f64 - 40.0).powi(2) + (k as f64 - 50.0).powi(2)).sqrt() < 5.0;
            let speckle = j == 3 && k == 3;
            if c_shape || disc || speckle {
                1000.0
            } else {
                0.0
            }
        })
    }

    #[test]
    fn bones_are_labelled_by_size() {
        let img = two_bones();
        let b = label_bones(
            img.view(),
            &BoneLabelOptions {
                threshold: 500.0,
                min_size: 20,
                ..BoneLabelOptions::default()
            },
        );
        assert_eq!(b.sizes.len(), 2);
        assert!(b.sizes[0] > b.sizes[1]);
        assert_eq!(b.labels[[1, 40, 50]], 2); // the disc
        assert_eq!(b.labels[[1, 40, 15]], 1); // the C
        assert_eq!(b.labels[[1, 3, 3]], 0); // speckle removed (median and size)
        let one = label_bones(
            img.view(),
            &BoneLabelOptions {
                threshold: 500.0,
                min_size: 20,
                max_bones: Some(1),
                ..BoneLabelOptions::default()
            },
        );
        assert_eq!(one.sizes.len(), 1);
        assert_eq!(one.labels[[1, 40, 50]], 0);
    }

    #[test]
    fn bone_hulls_contain_their_bones_and_stay_apart() {
        let img = two_bones();
        let b = label_bones(
            img.view(),
            &BoneLabelOptions {
                threshold: 500.0,
                min_size: 20,
                ..BoneLabelOptions::default()
            },
        );
        let h = bone_hulls(b.labels.view(), [1.0, 1.0, 1.0], None, None);
        // every bone voxel is in its own hull
        assert!(b.labels.iter().zip(h.iter()).all(|(&l, &hv)| l == 0 || hv == l));
        // the C's hull would cover the disc; the disc keeps it, and the C's
        // hull still closes the C's own marrow
        assert_eq!(h[[1, 40, 35]], 1);
        assert_eq!(h[[1, 40, 50]], 2);
        // trimmed to 3 voxels from the bone: the C's wide mouth (24 voxels) is
        // an open concavity, so its inside opens up; the bone stays in
        let t = bone_hulls(b.labels.view(), [1.0, 1.0, 1.0], Some(3.0), None);
        assert!(b.labels.iter().zip(t.iter()).all(|(&l, &hv)| l == 0 || hv == l));
        assert_eq!(t[[1, 40, 35]], 0);
        // a ring with a narrow (4 voxel) break keeps its marrow
        let ring = Array3::from_shape_fn((4, 60, 60), |(_, j, k)| {
            let (y, x) = (j as f64 - 30.0, k as f64 - 30.0);
            let r = (y * y + x * x).sqrt();
            ((15.0..20.0).contains(&r) && !(x > 5.0 && y.abs() < 2.0)) as u16
        });
        let tr = bone_hulls(ring.view(), [1.0, 1.0, 1.0], Some(3.0), None);
        assert_eq!(tr[[1, 30, 30]], 1);
        assert_eq!(tr[[1, 30, 47]], 1); // the break is closed
        assert_eq!(tr[[1, 2, 2]], 0);
        // closing by a 3-voxel ball: the C's 24-voxel mouth stays open, the
        // ring's 4-voxel break closes and its marrow is filled
        let c = bone_hulls(b.labels.view(), [1.0, 1.0, 1.0], None, Some(3.0));
        assert!(b.labels.iter().zip(c.iter()).all(|(&l, &cv)| l == 0 || cv == l));
        assert_eq!(c[[1, 40, 35]], 0);
        let cr = bone_hulls(ring.view(), [1.0, 1.0, 1.0], None, Some(3.0));
        assert_eq!(cr[[1, 30, 47]], 1);
        assert_eq!(cr[[1, 30, 30]], 1);
        assert_eq!(cr[[1, 2, 2]], 0);
        // no voxel of one hull (or closing; a 6-voxel closing of the C would
        // reach the disc) 6-touches another
        let wide = bone_hulls(b.labels.view(), [1.0, 1.0, 1.0], None, Some(6.0));
        assert!(b.labels.iter().zip(wide.iter()).all(|(&l, &cv)| l == 0 || cv == l));
        for h in [&h, &c, &wide] {
            let (d0, d1, d2) = h.dim();
            for ((i, j, k), &v) in h.indexed_iter() {
                if v == 0 {
                    continue;
                }
                for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                    let (a, bb, cc) = (i + di, j + dj, k + dk);
                    if a < d0 && bb < d1 && cc < d2 {
                        let w = h[[a, bb, cc]];
                        assert!(w == 0 || w == v, "labels {v} and {w} touch at ({i},{j},{k})");
                    }
                }
            }
        }
    }
}
