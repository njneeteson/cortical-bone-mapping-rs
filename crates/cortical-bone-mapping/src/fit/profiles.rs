//! Sampling density profiles along surface normals.

use ndarray::ArrayView3;
use rayon::prelude::*;

use crate::util::{add, dot, norm, scale, sub, V3};
use crate::volume::{convolve_1d, gaussian_kernel, is_set_at, trilinear, Geometry, Voxel};

/// Restrict the direction of the sampling lines.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum NormalConstraint {
    /// Use the surface normals as they are.
    #[default]
    None,
    /// Project normals into the plane perpendicular to the given array axis
    /// (e.g. the transverse plane of a long-bone HR-pQCT scan).
    InPlane(usize),
    /// Replace normals by ± the given array axis (e.g. the subchondral plate
    /// of the tibial plateau).
    AlongAxis(usize),
}

impl NormalConstraint {
    /// Apply the constraint to a unit normal. Returns `None` when the
    /// constrained direction is degenerate (e.g. an axial normal projected into
    /// the transverse plane): such a vertex has no meaningful profile.
    pub fn apply(&self, n: V3) -> Option<V3> {
        let out = match *self {
            NormalConstraint::None => n,
            NormalConstraint::InPlane(a) => {
                let mut v = n;
                v[a] = 0.0;
                v
            }
            NormalConstraint::AlongAxis(a) => {
                if n[a].abs() < 0.33 {
                    return None;
                }
                let mut v = [0.0; 3];
                v[a] = n[a].signum();
                v
            }
        };
        let l = norm(out);
        // reject directions more than ~70° from the true normal
        if l < 0.33 {
            return None;
        }
        Some(scale(out, 1.0 / l))
    }
}

/// The sample positions along every profile.
#[derive(Clone, Debug)]
pub struct SampleLine {
    /// Signed distances `xj` from the surface node (negative = outside).
    pub x: Vec<f64>,
}

impl SampleLine {
    /// Samples from `-outside` to `+inside` every `resolution`.
    pub fn new(outside: f64, inside: f64, resolution: f64) -> Self {
        assert!(resolution > 0.0 && outside >= 0.0 && inside > 0.0);
        let m = ((outside + inside) / resolution + 1e-9).floor() as usize + 1;
        SampleLine {
            x: (0..m).map(|j| -outside + j as f64 * resolution).collect(),
        }
    }
    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// Sampled profiles, stored contiguously as `f32` (`n × m`, row per node).
#[derive(Clone, Debug)]
pub struct Profiles {
    pub n: usize,
    pub m: usize,
    pub values: Vec<f32>,
    /// Per sample (`n × m`): `false` for samples to ignore (weight zero),
    /// e.g. inside another bone. `None` = every sample is used.
    pub used: Option<Vec<bool>>,
}

impl Profiles {
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.values[i * self.m..(i + 1) * self.m]
    }

    /// Which samples of row `i` are used (`None` = all).
    #[inline]
    pub fn used_row(&self, i: usize) -> Option<&[bool]> {
        self.used.as_ref().map(|u| &u[i * self.m..(i + 1) * self.m])
    }

    /// The residual weights `w` of row `i`, with ignored samples set to 0.
    pub fn row_weights<'a>(&self, i: usize, w: &'a [f64]) -> std::borrow::Cow<'a, [f64]> {
        match self.used_row(i) {
            None => std::borrow::Cow::Borrowed(w),
            Some(u) => std::borrow::Cow::Owned(w.iter().zip(u).map(|(&v, &k)| if k { v } else { 0.0 }).collect()),
        }
    }

    /// Ignore the inside samples beyond half the object's width along each
    /// normal: the line leaves `region` (the object's interior) at `x_exit`,
    /// and samples with `x > max(x_exit / 2, min_inside)` get zero weight.
    /// Lines that do not leave the region within the profile are unchanged.
    pub fn limit_inside<V: Voxel>(
        &mut self,
        region: &ArrayView3<V>,
        geom: &Geometry,
        points: &[V3],
        normals: &[V3],
        line: &SampleLine,
        min_inside: f64,
    ) {
        self.unuse(points, normals, |u, p, nrm| {
            // first inside sample (x > 0) that is outside the region, after
            // the line has entered it
            let mut entered = false;
            let mut exit = None;
            for &x in line.x.iter().filter(|&&x| x > 0.0) {
                if is_set_at(region, geom, add(p, scale(nrm, x))) {
                    entered = true;
                } else if entered {
                    exit = Some(x);
                    break;
                }
            }
            if let Some(xe) = exit {
                let limit = (0.5 * xe).max(min_inside);
                for (j, &x) in line.x.iter().enumerate() {
                    if x > limit {
                        u[j] = false;
                    }
                }
            }
        });
    }

    /// Ignore the samples that fall in voxels marked in `excluded` (nearest
    /// voxel; outside the image counts as not excluded).
    pub fn exclude<V: Voxel>(
        &mut self,
        excluded: &ArrayView3<V>,
        geom: &Geometry,
        points: &[V3],
        normals: &[V3],
        line: &SampleLine,
    ) {
        self.unuse(points, normals, |u, p, nrm| {
            for (j, &x) in line.x.iter().enumerate() {
                if is_set_at(excluded, geom, add(p, scale(nrm, x))) {
                    u[j] = false;
                }
            }
        });
    }

    /// Clear the used flags of each row with `f(used_row, point, normal)`.
    fn unuse(&mut self, points: &[V3], normals: &[V3], f: impl Fn(&mut [bool], V3, V3) + Sync) {
        let m = self.m;
        let mut used = self.used.take().unwrap_or_else(|| vec![true; self.n * m]);
        used.par_chunks_mut(m)
            .zip(points.par_iter().zip(normals.par_iter()))
            .for_each(|(u, (&p, &nrm))| f(u, p, nrm));
        self.used = Some(used);
    }

    /// Sample `img` along `p + x·n` for every point/normal pair, optionally
    /// smoothing each profile with a 1D Gaussian of `smoothing` samples.
    pub fn sample<T: Voxel>(
        img: &ArrayView3<T>,
        geom: &Geometry,
        points: &[V3],
        normals: &[V3],
        line: &SampleLine,
        smoothing: f64,
    ) -> Profiles {
        let m = line.len();
        let n = points.len();
        let mut values = vec![0f32; n * m];
        let kernel = gaussian_kernel(smoothing);
        values
            .par_chunks_mut(m)
            .zip(points.par_iter().zip(normals.par_iter()))
            .for_each_init(
                || (vec![0.0; m], vec![0.0; m]),
                |(raw, smooth), (out, (&p, &nrm))| {
                    for (j, &x) in line.x.iter().enumerate() {
                        raw[j] = trilinear(img, geom.to_index(add(p, scale(nrm, x))));
                    }
                    if kernel.len() > 1 {
                        convolve_1d(raw, &kernel, smooth);
                        out.iter_mut().zip(smooth.iter()).for_each(|(o, &v)| *o = v as f32);
                    } else {
                        out.iter_mut().zip(raw.iter()).for_each(|(o, &v)| *o = v as f32);
                    }
                },
            );
        Profiles {
            n,
            m,
            values,
            used: None,
        }
    }
}

/// Inward sampling normals for `outward` vertex normals under `constraint`,
/// or `None` for vertices without a valid direction.
pub fn sampling_normals(outward: &[V3], constraint: NormalConstraint) -> Vec<Option<V3>> {
    outward
        .par_iter()
        .map(|&n| {
            if norm(n) == 0.0 {
                return None;
            }
            constraint.apply(scale(n, -1.0))
        })
        .collect()
}

/// Component of `a` orthogonal to unit `n`.
#[inline]
pub fn tangential(a: V3, n: V3) -> V3 {
    sub(a, scale(n, dot(a, n)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn sample_line() {
        let l = SampleLine::new(1.0, 2.0, 0.5);
        assert_eq!(l.x, vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn samples_linear_field_exactly() {
        let img = Array3::from_shape_fn((20, 20, 20), |(i, j, k)| (i + 2 * j + 3 * k) as f32);
        let geom = Geometry::new([0.5, 0.5, 0.5]);
        let line = SampleLine::new(1.0, 1.0, 0.25);
        let p = [[5.0, 5.0, 5.0]];
        let n = [[0.0, 0.6, 0.8]];
        let prof = Profiles::sample(&img.view(), &geom, &p, &n, &line, 0.0);
        for (j, &x) in line.x.iter().enumerate() {
            let q = add(p[0], scale(n[0], x));
            let want = (q[0] + 2.0 * q[1] + 3.0 * q[2]) / 0.5;
            assert!((prof.row(0)[j] as f64 - want).abs() < 1e-3);
        }
    }

    #[test]
    fn inside_samples_stop_at_half_the_width() {
        // a 2-voxel-thick plate (axis 1 in 10..12) and a 20-voxel slab
        let geom = Geometry::new([1.0, 1.0, 1.0]);
        let line = SampleLine::new(1.0, 8.0, 0.5);
        let img = Array3::<f32>::zeros((30, 40, 30));
        for (lo, hi, want_last) in [(10usize, 12usize, Some(1.0)), (10, 30, None)] {
            let region = Array3::from_shape_fn((30, 40, 30), |(_, j, _)| (lo..hi).contains(&j) as u8);
            let p = [[15.0, lo as f64 - 0.5, 15.0]];
            let n = [[0.0, 1.0, 0.0]];
            let mut prof = Profiles::sample(&img.view(), &geom, &p, &n, &line, 0.0);
            prof.limit_inside(&region.view(), &geom, &p, &n, &line, 1.0);
            let used = prof.used_row(0).unwrap();
            let last = line
                .x
                .iter()
                .zip(used)
                .filter(|(_, &u)| u)
                .map(|(&x, _)| x)
                .fold(f64::MIN, f64::max);
            match want_last {
                // exits at x = 2.5 (index 12 rounds out): limit max(1.25, 1) = 1.25 -> last sample 1.0
                Some(w) => assert_eq!(last, w, "plate"),
                None => assert_eq!(last, 8.0, "thick slab: unchanged"),
            }
        }
    }

    #[test]
    fn constraints() {
        let n = [0.6, 0.8, 0.0];
        assert_eq!(NormalConstraint::InPlane(0).apply(n), Some([0.0, 1.0, 0.0]));
        assert_eq!(NormalConstraint::AlongAxis(0).apply(n), Some([1.0, 0.0, 0.0]));
        assert_eq!(NormalConstraint::InPlane(0).apply([1.0, 0.0, 0.0]), None);
        assert_eq!(NormalConstraint::AlongAxis(0).apply([0.0, 1.0, 0.0]), None);
    }
}
