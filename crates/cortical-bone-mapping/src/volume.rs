//! Image geometry, voxel types, trilinear sampling and Gaussian filtering.
//!
//! Coordinates follow the array axis order (axis 0, axis 1, axis 2), i.e.
//! `(z, y, x)` for a NumPy array. A voxel with index `(i, j, k)` has its
//! centre at `origin + (i, j, k) ⊙ spacing`.
//!
//! * **Trilinear interpolation**: the tensor product of linear interpolation
//!   along each axis, exact for fields linear in each coordinate; positions
//!   outside the image are clamped to the edge voxels (profiles that leave the
//!   image see the edge value, not zeros, which would look like a tissue
//!   boundary).
//! * **Gaussian filtering**: a Gaussian is separable, so a 3D filter is three
//!   1D convolutions. Kernels are sampled at integer offsets, truncated at
//!   `4σ` (where the tail beyond is below 7·10⁻⁵ of the mass) and normalised to
//!   sum to one, so constants are preserved; edges are replicated.

use ndarray::{Array3, ArrayView3, Axis, Zip};
use rayon::prelude::*;

use crate::util::V3;

/// A scalar voxel type the image may be stored as. The image is never
/// converted or copied: values are converted to `f64` as they are sampled.
pub trait Voxel: Copy + Send + Sync + 'static {
    /// The value as `f64` (`true` = 1).
    fn to_f64(self) -> f64;
}

macro_rules! voxel {
    ($($t:ty),*) => {$(impl Voxel for $t { #[inline] fn to_f64(self) -> f64 { self as f64 } })*};
}
voxel!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
impl Voxel for bool {
    #[inline]
    fn to_f64(self) -> f64 {
        self as u8 as f64
    }
}

/// Physical placement of the voxel grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// Voxel size along each array axis (e.g. mm).
    pub spacing: V3,
    /// Physical position of the centre of voxel `(0, 0, 0)`.
    pub origin: V3,
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            spacing: [1.0; 3],
            origin: [0.0; 3],
        }
    }
}

impl Geometry {
    /// A grid with the given spacing and its origin at zero.
    pub fn new(spacing: V3) -> Self {
        Geometry {
            spacing,
            origin: [0.0; 3],
        }
    }
    /// Continuous index position of the physical point `p`.
    #[inline]
    pub fn to_index(&self, p: V3) -> V3 {
        [
            (p[0] - self.origin[0]) / self.spacing[0],
            (p[1] - self.origin[1]) / self.spacing[1],
            (p[2] - self.origin[2]) / self.spacing[2],
        ]
    }
    /// Physical position of the continuous index `idx`.
    #[inline]
    pub fn to_physical(&self, idx: V3) -> V3 {
        [
            self.origin[0] + idx[0] * self.spacing[0],
            self.origin[1] + idx[1] * self.spacing[1],
            self.origin[2] + idx[2] * self.spacing[2],
        ]
    }
    /// The smallest voxel spacing.
    pub fn min_spacing(&self) -> f64 {
        self.spacing.iter().cloned().fold(f64::INFINITY, f64::min)
    }
}

/// Trilinear interpolation at a continuous index position: linear
/// interpolation along axis 2, then axis 1, then axis 0 between the 8
/// surrounding voxels. Positions outside the image are clamped to the nearest
/// edge voxel.
#[inline]
pub fn trilinear<T: Voxel>(img: &ArrayView3<T>, idx: V3) -> f64 {
    let (d0, d1, d2) = img.dim();
    let mut i0 = [0usize; 3];
    let mut f = [0.0f64; 3];
    let dims = [d0, d1, d2];
    for a in 0..3 {
        let maxc = (dims[a] - 1) as f64;
        let c = idx[a].clamp(0.0, maxc);
        let fl = c.floor();
        let mut i = fl as usize;
        let mut fr = c - fl;
        if i >= dims[a] - 1 {
            // at (or clamped to) the upper edge
            i = dims[a].saturating_sub(2);
            fr = if dims[a] == 1 { 0.0 } else { 1.0 };
        }
        i0[a] = i;
        f[a] = fr;
    }
    let get = |a: usize, b: usize, c: usize| -> f64 {
        let ia = (i0[0] + a).min(d0 - 1);
        let ib = (i0[1] + b).min(d1 - 1);
        let ic = (i0[2] + c).min(d2 - 1);
        // SAFETY: indices clamped to the array bounds above.
        unsafe { img.uget((ia, ib, ic)).to_f64() }
    };
    let c00 = get(0, 0, 0) * (1.0 - f[2]) + get(0, 0, 1) * f[2];
    let c01 = get(0, 1, 0) * (1.0 - f[2]) + get(0, 1, 1) * f[2];
    let c10 = get(1, 0, 0) * (1.0 - f[2]) + get(1, 0, 1) * f[2];
    let c11 = get(1, 1, 0) * (1.0 - f[2]) + get(1, 1, 1) * f[2];
    let c0 = c00 * (1.0 - f[1]) + c01 * f[1];
    let c1 = c10 * (1.0 - f[1]) + c11 * f[1];
    c0 * (1.0 - f[0]) + c1 * f[0]
}

/// Normalised 1D Gaussian kernel with standard deviation `sigma` (in
/// samples), sampled at integer offsets and truncated at `4σ`; `[1]` for
/// `sigma ≤ 0`.
pub fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    if !(sigma > 0.0) {
        return vec![1.0];
    }
    let r = (4.0 * sigma).ceil() as isize;
    let mut k: Vec<f64> = (-r..=r).map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp()).collect();
    let s: f64 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Convolve `src` with a symmetric kernel into `dst`, replicating edges.
pub fn convolve_1d(src: &[f64], kernel: &[f64], dst: &mut [f64]) {
    let r = (kernel.len() / 2) as isize;
    let n = src.len() as isize;
    for i in 0..n {
        let mut acc = 0.0;
        for (t, &w) in kernel.iter().enumerate() {
            let j = (i + t as isize - r).clamp(0, n - 1);
            acc += w * src[j as usize];
        }
        dst[i as usize] = acc;
    }
}

/// Separable anisotropic Gaussian filter of a 3D array (edges replicated).
/// `sigma` is in voxels per axis; zero skips that axis.
pub fn gaussian_filter(img: &Array3<f64>, sigma: V3) -> Array3<f64> {
    let mut out = img.clone();
    for (axis, &s) in sigma.iter().enumerate() {
        if !(s > 0.0) {
            continue;
        }
        let k = gaussian_kernel(s);
        let mut next = Array3::<f64>::zeros(out.dim());
        Zip::from(next.lanes_mut(Axis(axis)))
            .and(out.lanes(Axis(axis)))
            .par_for_each(|mut dst, src| {
                let s: Vec<f64> = src.iter().cloned().collect();
                let mut d = vec![0.0; s.len()];
                convolve_1d(&s, &k, &mut d);
                dst.iter_mut().zip(d).for_each(|(o, v)| *o = v);
            });
        out = next;
    }
    out
}

/// Whether the voxel nearest to the physical point `p` is set (non-zero);
/// points outside the image are not.
pub fn is_set_at<V: Voxel>(img: &ArrayView3<V>, geom: &Geometry, p: V3) -> bool {
    is_set_in_box(img, [0; 3], geom, p)
}

/// [`is_set_at`] for `img` holding the image's voxels from index `lo` on (a
/// crop): the voxel nearest to `p` in the image's grid `geom`, looked up in
/// the crop; voxels outside it are not set.
pub fn is_set_in_box<V: Voxel>(img: &ArrayView3<V>, lo: [usize; 3], geom: &Geometry, p: V3) -> bool {
    let r = geom.to_index(p).map(|v| v.round());
    if r[0] < lo[0] as f64 || r[1] < lo[1] as f64 || r[2] < lo[2] as f64 {
        return false;
    }
    let (d0, d1, d2) = img.dim();
    let (a, b, c) = (r[0] as usize - lo[0], r[1] as usize - lo[1], r[2] as usize - lo[2]);
    a < d0 && b < d1 && c < d2 && img[[a, b, c]].to_f64() != 0.0
}

/// Bounding box `(lo, hi)` (inclusive) of the non-zero voxels, or `None`.
pub fn bounding_box<T: Voxel>(mask: &ArrayView3<T>) -> Option<([usize; 3], [usize; 3])> {
    let (d0, _, _) = mask.dim();
    let per_slice: Vec<Option<([usize; 3], [usize; 3])>> = (0..d0)
        .into_par_iter()
        .map(|i| {
            let s = mask.index_axis(Axis(0), i);
            let mut lo = [usize::MAX; 3];
            let mut hi = [0usize; 3];
            let mut any = false;
            for ((j, k), v) in s.indexed_iter() {
                if v.to_f64() != 0.0 {
                    any = true;
                    lo = [i, lo[1].min(j), lo[2].min(k)];
                    hi = [i, hi[1].max(j), hi[2].max(k)];
                }
            }
            any.then_some((lo, hi))
        })
        .collect();
    per_slice.into_iter().flatten().reduce(|a, b| {
        (
            [a.0[0].min(b.0[0]), a.0[1].min(b.0[1]), a.0[2].min(b.0[2])],
            [a.1[0].max(b.1[0]), a.1[1].max(b.1[1]), a.1[2].max(b.1[2])],
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trilinear_is_exact_for_linear_fields() {
        let img = Array3::from_shape_fn((5, 6, 7), |(i, j, k)| {
            1.0 + 2.0 * i as f64 - 0.5 * j as f64 + 3.0 * k as f64
        });
        let v = img.view();
        for &p in &[[0.0, 0.0, 0.0], [1.3, 2.7, 4.1], [4.0, 5.0, 6.0], [3.99, 0.01, 5.5]] {
            let want = 1.0 + 2.0 * p[0] - 0.5 * p[1] + 3.0 * p[2];
            assert!((trilinear(&v, p) - want).abs() < 1e-12, "{p:?}");
        }
        // clamped outside
        assert!((trilinear(&v, [-3.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn trilinear_integer_types() {
        let img = Array3::from_shape_fn((2, 2, 2), |(i, _, _)| (i as i16) * 100 - 50);
        assert!((trilinear(&img.view(), [0.25, 0.5, 0.5]) - (-25.0)).abs() < 1e-12);
    }

    #[test]
    fn gaussian_preserves_constants_and_mass() {
        let mut img = Array3::<f64>::zeros((21, 21, 21));
        img[[10, 10, 10]] = 1.0;
        let out = gaussian_filter(&img, [1.5, 1.0, 0.0]);
        assert!((out.sum() - 1.0).abs() < 1e-9);
        let c = gaussian_filter(&Array3::from_elem((4, 5, 6), 3.0), [2.0, 2.0, 2.0]);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-12));
    }

    #[test]
    fn bbox() {
        let mut m = Array3::<u8>::zeros((5, 5, 5));
        m[[1, 2, 3]] = 1;
        m[[3, 0, 4]] = 1;
        assert_eq!(bounding_box(&m.view()), Some(([1, 0, 3], [3, 2, 4])));
        assert_eq!(bounding_box(&Array3::<u8>::zeros((2, 2, 2)).view()), None);
    }
}
