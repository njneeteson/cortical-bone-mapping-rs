//! Several bones in one image: labelled connected components, hysteresis
//! growth, and per-bone starting segmentations kept apart (ours, built on
//! standard morphology).
//!
//! * **Labelling** ([`label_bones`]): threshold, binary median, 6-connected
//!   components, kept from the largest. Bones that touch at the threshold
//!   become one component; [`grow_bones`] and the Python `separate_bones`
//!   handle the two failure modes of a single threshold.
//! * **Hysteresis** ([`grow_bones`]): as in Canny's edge detector (Canny 1986),
//!   a high threshold finds reliable seeds (the bones, told apart) and a low
//!   threshold decides how far they extend: voxels at or above the low
//!   threshold that are connected to a seed join it. Thin cortices that fall
//!   below the high threshold through partial volume are recovered without
//!   joining neighbouring bones at the low one. Growth is a multi-source
//!   breadth-first search, so a voxel reachable from several bones goes to the
//!   geodesically nearest (fewest 6-connected steps; ties to the lower label).
//! * **Starts** ([`bone_hulls`]): each bone's per-slice convex hull, or its
//!   morphological closing (Serra 1982) by a ball of radius `r`: dilate by
//!   `r`, fill the enclosed background, erode by `r`. A closing fills
//!   concavities narrower than about `2r` and keeps wider ones (a "local
//!   convex hull"), so it follows a concave bone where a hull would bridge it.
//!   The background enclosed after the dilation is filled before eroding,
//!   since a void that the erosion would reopen (sparse marrow below the
//!   threshold) is inside the bone; slice holes are filled in all three
//!   orientations because a void reaching the outside through a cortical gap
//!   wider than the closing is still enclosed in most slices through it.
//! * **Kept apart**: starts of neighbouring bones overlap where a concave
//!   surface faces another bone. Each overlapping voxel goes to the bone whose
//!   thresholded voxels are nearest (Euclidean, physical units, from the
//!   distance transforms), the voxels touching another bone's start are
//!   cleared (unless they are their own bone), and pieces cut off from their
//!   bone are dropped, so that every bone is a separate closed surface that
//!   still contains its thresholded bone.
//!
//! # References
//!
//! * J. Canny, *A computational approach to edge detection*, IEEE Transactions
//!   on Pattern Analysis and Machine Intelligence 8(6):679–698, 1986.
//! * J. Serra, *Image Analysis and Mathematical Morphology*, Academic Press,
//!   1982.

use std::collections::VecDeque;
use std::ops::Range;

use ndarray::{s, Array3, ArrayView3, ArrayViewMut3, Zip};
use rayon::prelude::*;

use super::hull::slice_hulls;
use super::morphology::{
    binary_median, fill_slice_holes, label_components, neighbours6, outer_region, scaled_sq_distance,
};
use crate::volume::Voxel;

/// Options of [`label_bones`].
#[derive(Clone, Debug, PartialEq)]
pub struct BoneLabelOptions {
    /// Bone threshold (image units), e.g. from [`otsu_thresholds`](super::otsu_thresholds).
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
    let s = crate::stage!("label bones");
    log::debug!("label bones: image {:?}, {o:?}", image.dim());
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
    crate::done!(
        s,
        "{} components, {} kept (at least {} voxels): sizes {:?}",
        sizes.len().saturating_sub(1),
        order.len(),
        o.min_size,
        order.iter().map(|&c| sizes[c]).collect::<Vec<_>>()
    );
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
    let s = crate::stage!("grow bones");
    let mut low = Array3::from_elem(image.dim(), false);
    Zip::from(&mut low)
        .and(&image)
        .par_for_each(|b, &v| *b = v.to_f64() >= low_threshold);
    if median_kernel.iter().any(|&k| k > 1) {
        low = binary_median(&low, median_kernel);
    }
    let mut out = labels.to_owned();
    let mut queue: VecDeque<[usize; 3]> = out
        .indexed_iter()
        .filter(|(_, &l)| l != 0)
        .map(|((i, j, k), _)| [i, j, k])
        .collect();
    while let Some(p) = queue.pop_front() {
        let l = out[p];
        for q in neighbours6(p, image.dim()) {
            if out[q] == 0 && low[q] {
                out[q] = l;
                queue.push_back(q);
            }
        }
    }
    let labelled = |a: &ArrayView3<u16>| a.iter().filter(|&&l| l != 0).count();
    crate::done!(
        s,
        "low threshold {low_threshold:.1}: {} -> {} labelled voxels",
        labelled(&labels),
        labelled(&out.view())
    );
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
    log::debug!(
        "bone hulls: {n} bones, image {dim:?}, {}",
        match (closing_radius, max_distance) {
            (Some(r), _) => format!("closing radius {r}"),
            (None, Some(d)) => format!("slice hulls trimmed to {d}"),
            (None, None) => "slice hulls".to_string(),
        }
    );
    let mut owner = Array3::<u16>::zeros(dim);
    let mut best = Array3::from_elem(dim, f32::INFINITY);
    let inv = spacing.map(|s| 1.0 / s);
    // each bone's start lies within its bounding box grown by the closing
    // radius (or trimming distance): work on that box plus a voxel, whose
    // outer ring is background like the rest of the image
    let reach = closing_radius.unwrap_or(0.0).max(max_distance.unwrap_or(0.0));
    let pad = spacing.map(|s| (reach / s).ceil() as usize + 1);
    let full = [dim.0, dim.1, dim.2];
    let boxes: Vec<Option<[Range<usize>; 3]>> = bounding_boxes(&labels, n)
        .into_iter()
        .map(|bx| {
            bx.map(|bx| std::array::from_fn(|a| bx[a].start.saturating_sub(pad[a])..(bx[a].end + pad[a]).min(full[a])))
        })
        .collect();
    for b in 1..=n {
        let Some(r) = &boxes[b as usize] else {
            continue;
        };
        let sl = s![r[0].clone(), r[1].clone(), r[2].clone()];
        let s = crate::stage!("bone start");
        let mb = labels.slice(sl).mapv(|l| l == b);
        let d = scaled_sq_distance(&mb, inv);
        let start = bone_start(&mb, &d, inv, max_distance, closing_radius);
        crate::done!(
            s,
            "bone {b}: box {:?} of {dim:?}, {} thresholded voxels, {} in its start",
            mb.dim(),
            mb.iter().filter(|&&v| v).count(),
            start.iter().filter(|&&v| v).count()
        );
        Zip::from(owner.slice_mut(sl))
            .and(best.slice_mut(sl))
            .and(&start)
            .and(&d)
            .par_for_each(|o, bst, &h, &dv| {
                if h && dv < *bst {
                    *bst = dv;
                    *o = b;
                }
            });
    }
    let s = crate::stage!("separate the starts");
    let mut out = separate(&owner, &labels);
    for b in 1..=n {
        // a bone's region lies within its padded box
        if let Some(r) = &boxes[b as usize] {
            let sl = s![r[0].clone(), r[1].clone(), r[2].clone()];
            drop_fragments(out.slice_mut(sl), labels.slice(sl), b);
        }
    }
    s.end();
    out
}

/// The bounding box (index ranges per axis) of every label `1..=n`, in one
/// pass; `None` for a label that does not occur.
fn bounding_boxes(labels: &ArrayView3<u16>, n: u16) -> Vec<Option<[Range<usize>; 3]>> {
    let mut lo = vec![[usize::MAX; 3]; n as usize + 1];
    let mut hi = vec![[0usize; 3]; n as usize + 1];
    for ((i, j, k), &l) in labels.indexed_iter() {
        if l != 0 {
            let (a, b) = (&mut lo[l as usize], &mut hi[l as usize]);
            for (axis, p) in [i, j, k].into_iter().enumerate() {
                a[axis] = a[axis].min(p);
                b[axis] = b[axis].max(p + 1);
            }
        }
    }
    (0..=n as usize)
        .map(|l| (lo[l][0] != usize::MAX).then(|| std::array::from_fn(|a| lo[l][a]..hi[l][a])))
        .collect()
}

/// One bone's start (`mb`, squared distance map `d`): its slice hulls or
/// closing, optionally trimmed to `max_distance` of the bone with the slice
/// holes filled.
fn bone_start(
    mb: &Array3<bool>,
    d: &Array3<f32>,
    inv: [f64; 3],
    max_distance: Option<f64>,
    closing_radius: Option<f64>,
) -> Array3<bool> {
    let mut start = match closing_radius {
        Some(r) => closing(mb, d, inv, r),
        None => slice_hulls(mb),
    };
    if let Some(r) = max_distance {
        let r2 = (r * r) as f32;
        Zip::from(&mut start).and(d).par_for_each(|h, &dv| *h &= dv <= r2);
        start = outer_region(&start.mapv(|b| !b), true).mapv(|b| !b);
    }
    start
}

/// Clear the voxels of each bone's region that touch another bone's
/// (6-neighbourhood), except its own thresholded bone (`labels`).
fn separate(owner: &Array3<u16>, labels: &ArrayView3<u16>) -> Array3<u16> {
    let mut out = owner.clone();
    Zip::indexed(&mut out).par_for_each(|(i, j, k), o| {
        let b = owner[[i, j, k]];
        if b == 0 || labels[[i, j, k]] == b {
            return;
        }
        let touches = neighbours6([i, j, k], owner.dim()).any(|q| owner[q] != 0 && owner[q] != b);
        if touches {
            *o = 0;
        }
    });
    out
}

/// Keep only the pieces of bone `b`'s region that contain some of its
/// thresholded bone (assigning and separating can cut off small fragments).
fn drop_fragments(mut out: ArrayViewMut3<u16>, labels: ArrayView3<u16>, b: u16) {
    let mb = out.mapv(|v| v == b);
    let (comp, sizes) = label_components(&mb, false);
    if sizes.len() <= 2 {
        return;
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

/// Morphological closing of `m` by a ball of radius `r` (physical units)
/// with its holes filled: dilate by `r`, fill every background region not
/// connected to the in-plane faces of the volume (voids inside the bone,
/// e.g. sparse trabecular or fatty marrow regions below the threshold), then
/// erode by `r` (as in the periosteal mask of Buie et al. 2007). `d` is the
/// squared distance map of `m` (physical units²; `inv` = 1 / spacing).
/// Slice holes left after the erosion are filled too.
fn closing(m: &Array3<bool>, d: &Array3<f32>, inv: [f64; 3], r: f64) -> Array3<bool> {
    let r2 = (r * r) as f32;
    let s = crate::stage!("closing: dilation and enclosed voids");
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
    s.end();
    let s = crate::stage!("closing: erosion");
    let outside = filled.mapv(|b| !b);
    // erosion: voxels farther than r from the outside (the image border
    // counts as inside)
    let e = scaled_sq_distance(&outside, inv);
    let mut c = e.mapv(|v| v > r2);
    Zip::from(&mut c).and(m).par_for_each(|c, &b| *c |= b);
    let out = outer_region(&c.mapv(|b| !b), true).mapv(|b| !b);
    s.end();
    out
}

#[cfg(test)]
mod crop_tests {
    use super::*;
    use crate::util::Rng;

    /// [`bone_hulls`] as it was before cropping: every bone on the whole image.
    fn whole_image(
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
            let start = bone_start(&mb, &d, inv, max_distance, closing_radius);
            Zip::from(&mut owner)
                .and(&mut best)
                .and(&start)
                .and(&d)
                .for_each(|o, bst, &h, &dv| {
                    if h && dv < *bst {
                        *bst = dv;
                        *o = b;
                    }
                });
        }
        let mut out = separate(&owner, &labels);
        for b in 1..=n {
            drop_fragments(out.view_mut(), labels, b);
        }
        out
    }

    /// Hollow, dented blobs (marrow, concavities, a break), some cut by the
    /// image's edges, close enough for their starts to compete.
    fn scene(seed: u64) -> Array3<u16> {
        let mut rng = Rng::new(seed);
        let dim = (14, 48, 52);
        let blobs: Vec<([f64; 3], f64, [f64; 3])> = (0..4)
            .map(|_| {
                let c = [
                    rng.uniform() * 14.0,
                    8.0 + rng.uniform() * 32.0,
                    8.0 + rng.uniform() * 36.0,
                ];
                let r = 5.0 + rng.uniform() * 5.0;
                let dent = [rng.normal(), rng.normal(), rng.normal()];
                (c, r, dent)
            })
            .collect();
        Array3::from_shape_fn(dim, |(i, j, k)| {
            let p = [i as f64, j as f64, k as f64];
            for (b, (c, r, dent)) in blobs.iter().enumerate() {
                let v = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                let d = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                let dented = r - 2.5 * (v[1] * dent[1] + v[2] * dent[2]).max(0.0) / r;
                let in_break = v[1].abs() < 1.5 && v[2] > 0.0;
                if d <= dented && d >= 0.6 * dented && !in_break {
                    return b as u16 + 1;
                }
            }
            0
        })
    }

    #[test]
    fn cropping_to_each_bone_changes_nothing() {
        for seed in 1..6 {
            let labels = scene(seed);
            for spacing in [[1.0, 1.0, 1.0], [1.3, 0.8, 1.0]] {
                for (md, cr) in [(None, None), (Some(3.0), None), (None, Some(3.0)), (None, Some(5.5))] {
                    let got = bone_hulls(labels.view(), spacing, md, cr);
                    let want = whole_image(labels.view(), spacing, md, cr);
                    assert!(
                        got == want,
                        "seed {seed}, spacing {spacing:?}, trim {md:?}, closing {cr:?}"
                    );
                }
            }
        }
    }
}
