//! Segmentation from the image alone: thresholds, bones, starting
//! segmentations for refinement, and Buie's method for comparison.
//!
//! * [`otsu`]: (multi-level) Otsu thresholds from the histogram.
//! * [`bones`]: bones as labelled connected components, hysteresis growth,
//!   and per-bone starting segmentations (per-slice convex hulls or
//!   closings), kept apart.
//! * [`hull`]: the single-bone convex-hull start and the slice-hull geometry.
//! * [`buie`]: Buie et al.'s (2007) dual-threshold segmentation.
//! * [`morphology`]: the binary morphology they share (median, exact
//!   distance transforms, dilation/erosion, connected components, fills).

pub mod bones;
pub mod buie;
pub mod hull;
pub mod morphology;
pub mod otsu;
#[cfg(test)]
mod tests;

pub use bones::{bone_hulls, grow_bones, label_bones, BoneLabelOptions, BoneLabels};
pub use buie::{buie, BuieOptions, BuieResult};
pub use hull::{hull_segmentation, slice_hulls, HullOptions};
pub use morphology::{
    binary_median, dilate, erode, gaussian_truncated, largest_component, outer_region, scaled_sq_distance,
};
pub use otsu::{otsu_thresholds, OtsuOptions};
