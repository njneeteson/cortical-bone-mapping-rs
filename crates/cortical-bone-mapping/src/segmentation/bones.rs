//! Several bones in one image: labelled connected components, hysteresis
//! growth, and per-bone starting segmentations kept apart.

use ndarray::{Array3, ArrayView3, Zip};
use rayon::prelude::*;

use super::hull::slice_hulls;
use super::morphology::{binary_median, fill_slice_holes, label_components, outer_region, scaled_sq_distance};
use crate::volume::Voxel;

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

/// Hysteresis thresholding: grow labelled bones (e.g. from [`label_bones`]
/// at a high threshold) into the connected voxels at or above
/// `low_threshold`, after a binary median with `median_kernel`. Thin
/// cortices that fall below the high threshold through partial volume are
/// recovered, while the bones are still told apart at the high threshold: a
/// voxel reachable from several bones goes to the one that reaches it first
/// (fewest 6-connected steps, a geodesic nearest bone; ties to the lower
/// label). Labels and their order are unchanged.
pub fn grow_bones<T: Voxel>(
    image: ArrayView3<T>,
    labels: ArrayView3<u16>,
    low_threshold: f64,
    median_kernel: [usize; 3],
) -> Array3<u16> {
    let mut low = Array3::from_elem(image.dim(), false);
    Zip::from(&mut low)
        .and(&image)
        .par_for_each(|b, &v| *b = v.to_f64() >= low_threshold);
    if median_kernel.iter().any(|&k| k > 1) {
        low = binary_median(&low, median_kernel);
    }
    let (d0, d1, d2) = image.dim();
    let mut out = labels.to_owned();
    let mut queue: std::collections::VecDeque<(usize, usize, usize)> = std::collections::VecDeque::new();
    for ((i, j, k), &l) in out.indexed_iter() {
        if l != 0 {
            queue.push_back((i, j, k));
        }
    }
    while let Some((i, j, k)) = queue.pop_front() {
        let l = out[[i, j, k]];
        let mut visit = |a: usize, b: usize, c: usize| {
            if out[[a, b, c]] == 0 && low[[a, b, c]] {
                out[[a, b, c]] = l;
                queue.push_back((a, b, c));
            }
        };
        if i > 0 {
            visit(i - 1, j, k);
        }
        if i + 1 < d0 {
            visit(i + 1, j, k);
        }
        if j > 0 {
            visit(i, j - 1, k);
        }
        if j + 1 < d1 {
            visit(i, j + 1, k);
        }
        if k > 0 {
            visit(i, j, k - 1);
        }
        if k + 1 < d2 {
            visit(i, j, k + 1);
        }
    }
    out
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
