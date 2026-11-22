//! Binary morphology on 3D masks: median, exact (anisotropic) distance
//! transforms, ellipsoidal dilation/erosion, connected components, region
//! and hole fills, and VTK-style truncated Gaussian smoothing.
//!
//! * **Binary median**: a voxel is set if more than half of the voxels of the
//!   box kernel around it are (a majority vote; edges replicated), which
//!   removes speckle smaller than half the kernel without moving straight
//!   edges.
//! * **Distance transform**: the exact squared Euclidean distance to the
//!   nearest set voxel with per-axis scales, computed separably, axis by axis,
//!   as the lower envelope of parabolas (Felzenszwalb & Huttenlocher 2012):
//!   after processing axes `0 … a−1`, `D_a(q) = min_p [w_a (q − p)² + D_{a−1}(p)]`
//!   along each line of axis `a`. The minimisation over a line is linear in
//!   its length, so the transform is O(voxels) whatever the distances, and
//!   exact (a squared distance is a sum of per-axis terms, so separable
//!   minimisation is exact).
//! * **Dilation / erosion** with an ellipsoidal footprint of semi-axes
//!   `r_a = (k_a − 1)/2` voxels, as `vtkImageDilateErode3D`: a voxel is in the
//!   dilation if `Σ_a (Δ_a / r_a)² ≤ 1` for some set voxel, i.e. if the
//!   distance transform with weights `1/r_a²` is ≤ 1; erosion is the dilation
//!   of the complement (Serra 1982). Cost is independent of the kernel size.
//! * **Connected components**: breadth-first flood fill over 6-neighbours (4
//!   within a slice), one pass over the voxels (Rosenfeld & Pfaltz 1966).
//! * **Outer region**: the components touching the in-plane faces of the
//!   volume (axes 1 and 2), which is the background around a bone in an
//!   axial stack even when the bone is cut by the first or last slice.
//!   Filling holes is taking the complement of the outer region of the
//!   complement.
//! * **Truncated Gaussian smoothing** of a 0/1 mask, as
//!   `vtkImageGaussianSmooth`: kernel radius `r` voxels per axis, weights
//!   normalised over the truncated support, edges replicated.
//!
//! # References
//!
//! * P. F. Felzenszwalb, D. P. Huttenlocher, *Distance transforms of sampled
//!   functions*, Theory of Computing 8:415–428, 2012.
//! * J. Serra, *Image Analysis and Mathematical Morphology*, Academic Press,
//!   1982.
//! * A. Rosenfeld, J. L. Pfaltz, *Sequential operations in digital picture
//!   processing*, Journal of the ACM 13(4):471–494, 1966.

use std::collections::VecDeque;

use ndarray::{Array3, Axis, Zip};
use rayon::prelude::*;

/// Binary median (majority vote) over a box kernel of `k` voxels per axis,
/// edges replicated: set where more than half of the kernel's voxels are.
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

/// The 1D lower envelope of parabolas `D(q) = min_p [w (q − p)² + f(p)]`
/// (Felzenszwalb & Huttenlocher 2012, algorithm DT(f)): parabolas are added
/// left to right, dropping those hidden by the new one, using the abscissa
/// `s` where two parabolas intersect; then each `q` reads off the parabola
/// whose interval contains it. Linear in the length of the line. Infinite
/// `f(p)` (no set voxel) contribute no parabola.
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

/// Semi-axes `(k − 1)/2` of the ellipsoidal footprint of a `k`-voxel kernel
/// (as `vtkImageDilateErode3D`).
pub(super) fn radii(k: [usize; 3]) -> [f64; 3] {
    k.map(|v| (v.max(1) - 1) as f64 / 2.0)
}

/// Binary dilation with an ellipsoidal footprint of `kernel` voxels per axis
/// (as `vtkImageDilateErode3D`): the voxels whose scaled squared distance
/// `Σ_a (Δ_a / r_a)²` to a set voxel is at most 1.
pub fn dilate(m: &Array3<bool>, kernel: [usize; 3]) -> Array3<bool> {
    let d = scaled_sq_distance(m, radii(kernel));
    d.mapv(|v| v <= 1.0 + 1e-6)
}

/// Binary erosion with an ellipsoidal footprint, the dual of dilation:
/// `erode(m) = ¬dilate(¬m)` (voxels outside the image do not erode).
pub fn erode(m: &Array3<bool>, kernel: [usize; 3]) -> Array3<bool> {
    let inv = m.mapv(|b| !b);
    dilate(&inv, kernel).mapv(|b| !b)
}

/// The 6-connected neighbours of voxel `p` in an array of shape `dim`, in
/// the order −axis 0, +axis 0, −axis 1, +axis 1, −axis 2, +axis 2.
pub(super) fn neighbours6(p: [usize; 3], dim: (usize, usize, usize)) -> impl Iterator<Item = [usize; 3]> {
    let d = [dim.0, dim.1, dim.2];
    (0..6).filter_map(move |n| {
        let (axis, up) = (n / 2, n % 2 == 1);
        let mut q = p;
        if up && q[axis] + 1 < d[axis] {
            q[axis] += 1;
        } else if !up && q[axis] > 0 {
            q[axis] -= 1;
        } else {
            return None;
        }
        Some(q)
    })
}

/// Label 6-connected components (4-connected within slices if `two_d`) by
/// breadth-first flood fill. Returns labels (0 = background) in standard
/// layout and component sizes (index = label), labels in order of first
/// voxel.
pub(super) fn label_components(m: &Array3<bool>, two_d: bool) -> (Vec<u32>, Vec<usize>) {
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
            let at = [p / (d1 * d2), (p / d2) % d1, p % d2];
            for q in neighbours6(at, m.dim()).filter(|q| !two_d || q[0] == at[0]) {
                let q = (q[0] * d1 + q[1]) * d2 + q[2];
                if flat[q] && label[q] == 0 {
                    label[q] = id;
                    queue.push_back(q);
                }
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
pub(super) fn fill_slice_holes(m: &Array3<bool>, axis: usize) -> Array3<bool> {
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

/// Apply a 3D operation to every slice along axis 0 separately (in
/// parallel), each as a one-slice volume.
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
/// (voxels) and kernel radius `r` voxels per axis (0 = no smoothing along that
/// axis), weights normalised over the truncated kernel and edges clamped, as
/// `vtkImageGaussianSmooth` (step 12 of Buie et al. 2007).
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
