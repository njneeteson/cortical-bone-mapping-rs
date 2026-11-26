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
//!
//! # References
//!
//! * H. R. Buie, G. M. Campbell, R. J. Klinck, J. A. MacNeil, S. K. Boyd,
//!   *Automatic segmentation of cortical and trabecular compartments based on
//!   a dual threshold technique for in vivo micro-CT bone analysis*, Bone
//!   41(4):505–515, 2007, doi:10.1016/j.bone.2007.07.007.
//! * A. J. Burghardt, H. R. Buie, A. Laib, S. Majumdar, S. K. Boyd,
//!   *Reproducibility of direct quantitative measures
//!   of cortical bone microarchitecture of the distal radius and tibia by
//!   HR-pQCT*, Bone 47(3):519–528, 2010.

use ndarray::{s, Array3, ArrayView3, Zip};

use super::morphology::{binary_median, dilate, erode, gaussian_truncated, largest_component, outer_region};
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
    let s = crate::stage!("Buie segmentation");
    log::debug!("Buie segmentation: image {:?}, {o:?}", image.dim());
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
    let periosteal = {
        let mut s2 = Array3::from_elem(shape, false);
        Zip::from(s2.slice_mut(inner))
            .and(&image)
            .par_for_each(|b, &v| *b = v.to_f64() >= o.periosteal_threshold);
        periosteal_region(s2, o)
    };
    let mut marrow = Array3::from_elem(shape, false);
    Zip::from(marrow.slice_mut(inner))
        .and(&image)
        .and(periosteal.slice(inner))
        .par_for_each(|m, &v, &p| *m = p && v.to_f64() < o.endosteal_threshold);
    let non_marrow = non_marrow_region(marrow, o);
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
    crate::done!(
        s,
        "{} cortical and {} trabecular voxels",
        cortical.iter().filter(|&&v| v).count(),
        trabecular.iter().filter(|&&v| v).count()
    );
    BuieResult {
        periosteal: periosteal.slice(inner).to_owned(),
        cortical,
        trabecular,
    }
}

/// Steps 3–6, the periosteal surface: from the thresholded bone `s2`
/// (step 2), median, (largest bone,) dilate, fill everything but the outer
/// background, erode.
fn periosteal_region(s2: Array3<bool>, o: &BuieOptions) -> Array3<bool> {
    let mut s3 = binary_median(&s2, o.median_kernel);
    drop(s2);
    // pre-processing: keep the largest bone before any dilation, so that a
    // nearby fibula / ulna cannot be merged with it by the closing
    if o.largest_bone_only {
        s3 = largest_component(&s3, o.connectivity_2d);
    }
    let s4 = dilate(&s3, o.periosteal_kernel);
    drop(s3);
    let bg = s4.mapv(|b| !b);
    drop(s4);
    let s5 = outer_region(&bg, o.connectivity_2d).mapv(|b| !b);
    drop(bg);
    erode(&s5, o.periosteal_kernel)
}

/// Steps 9–13, the endosteal surface: from the marrow cavities (steps 7–8,
/// below the endosteal threshold inside the periosteal region), dilate, take
/// the outer non-marrow region, dilate it back, smooth and threshold.
fn non_marrow_region(marrow: Array3<bool>, o: &BuieOptions) -> Array3<bool> {
    let s9 = dilate(&marrow, o.endosteal_kernel);
    drop(marrow);
    let white = s9.mapv(|b| !b);
    drop(s9);
    let s10_white = outer_region(&white, o.connectivity_2d);
    drop(white);
    // 11. erode the marrow = dilate the white region
    let s11_white = dilate(&s10_white, o.endosteal_kernel);
    drop(s10_white);
    // 12. Gaussian smooth, 13. threshold by lower (marrow where ≤ threshold)
    let r = o.gaussian_support.map(|f| (o.gaussian_sigma * f).floor() as usize);
    let sm = gaussian_truncated(&s11_white, o.gaussian_sigma, r);
    drop(s11_white);
    sm.mapv(|v| v as f64 > o.smoothed_threshold)
}
