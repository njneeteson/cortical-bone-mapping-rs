//! A realistic long-bone phantom with known cortical geometry.
//!
//! [`BonePhantom`] is a bone shaft along array axis 0 whose
//!
//! * periosteal cross-section is a random star-shaped curve
//!   `R(θ, z) = R̄ [1 + Σ_{k=2..K} a_k cos(kθ + φ_k + ω_k z)]` (amplitudes
//!   decaying like `k^-1.5`, phases drifting along the axis), with a centre
//!   that wanders along the axis;
//! * cortical (radial) thickness is an independent random field
//!   `t(θ, z) = t̄ [1 + Σ_{k=1..Kt} b_k cos(kθ + ψ_k + ν_k z)] · (1 + c sin(2π z/L))`,
//!   clipped below at `t_min`;
//! * marrow space holds a trabecular network: the thin level-set shell
//!   `|f(p)| < w` of a band-limited random field `f` (random plane waves of
//!   wavelength `λ`, elongated along the axis), with `w` chosen for a target
//!   bone volume fraction;
//! * cortex contains pores: the super-level set of another random field;
//! * an endocortical transition zone: cortical porosity increases towards the
//!   endosteal surface (`porosity + endo_porosity · exp(−u / 0.25)`, `u` the
//!   relative depth from the endosteum) and trabecular bone volume fraction
//!   increases towards it (`bvtv + (endo_bvtv − bvtv) · exp(−d / transition_width)`,
//!   `d` the distance into the marrow space). The *true* endosteal surface is
//!   still the smooth surface `r = R − t`, which is what makes it ambiguous
//!   in the image, as in real trabecularised cortex;
//!
//! integrated over voxels by supersampling, blurred with an anisotropic
//! Gaussian PSF and given Gaussian noise. Ground truth is exact: the
//! periosteal and endosteal surfaces are implicit, so the true edge positions
//! along any line are found by stepping plus bisection
//! ([`BonePhantom::edges_along`]), i.e. the reference thickness is measured
//! along the same line as CBM measures it.

use ndarray::Array3;
use rayon::prelude::*;

use crate::util::{Rng, V3};
use crate::volume::{gaussian_filter, Geometry};

#[derive(Clone, Debug)]
pub struct BonePhantom {
    pub dims: (usize, usize, usize),
    pub spacing: V3,
    /// Mean periosteal radius.
    pub radius: f64,
    /// Number of shape harmonics and relative amplitude of the first one.
    pub shape_harmonics: usize,
    pub shape_amplitude: f64,
    /// Typical axial wavelength of shape and thickness phase drift.
    pub axial_wavelength: f64,
    /// Amplitude of the axial wander of the centre.
    pub center_wander: f64,
    pub thickness_mean: f64,
    pub thickness_harmonics: usize,
    /// Relative amplitude of thickness variation.
    pub thickness_variation: f64,
    pub thickness_min: f64,
    /// Trabecular bone volume fraction (0 = empty marrow).
    pub bvtv: f64,
    /// Trabecular pattern wavelength (≈ 2 × trabecular separation).
    pub trabecular_wavelength: f64,
    /// Cortical porosity (volume fraction of pores in the cortex).
    pub porosity: f64,
    pub pore_wavelength: f64,
    /// Extra porosity at the endosteal side of the cortex (trabecularisation).
    pub endo_porosity: f64,
    /// Trabecular bone volume fraction next to the endosteal surface.
    pub endo_bvtv: f64,
    /// Width of the endocortical transition zone (physical units).
    pub transition_width: f64,
    pub rho_soft: f64,
    pub rho_bone: f64,
    pub rho_marrow: f64,
    /// Gaussian PSF standard deviation per axis (physical units).
    pub blur: V3,
    pub noise: f64,
    pub seed: u64,
    pub supersample: usize,
}

impl Default for BonePhantom {
    /// A small distal-tibia-like segment at HR-pQCT resolution.
    fn default() -> Self {
        BonePhantom {
            dims: (32, 192, 192),
            spacing: [0.0607; 3],
            radius: 4.5,
            shape_harmonics: 6,
            shape_amplitude: 0.12,
            axial_wavelength: 8.0,
            center_wander: 0.2,
            thickness_mean: 0.8,
            thickness_harmonics: 6,
            thickness_variation: 0.35,
            thickness_min: 0.15,
            bvtv: 0.2,
            trabecular_wavelength: 1.0,
            porosity: 0.03,
            pore_wavelength: 0.35,
            endo_porosity: 0.15,
            endo_bvtv: 0.4,
            transition_width: 0.3,
            rho_soft: 50.0,
            rho_bone: 1000.0,
            rho_marrow: 30.0,
            blur: [0.08; 3],
            noise: 40.0,
            seed: 1,
            supersample: 2,
        }
    }
}

/// Random coefficients derived from the seed (fixed for a phantom).
struct Coeffs {
    shape: Vec<(f64, f64, f64)>,  // (amplitude, phase, axial rate) for k = 2..
    thick: Vec<(f64, f64, f64)>,  // for k = 1..
    center: [(f64, f64, f64); 2], // (amplitude, rate, phase) per in-plane axis
    thick_axial: (f64, f64),      // (amplitude, phase)
    trab: Vec<(V3, f64)>,         // wave vectors and phases
    pores: Vec<(V3, f64)>,
}

/// Inverse of erf for the level-set widths (Giles' approximation).
fn erfinv(x: f64) -> f64 {
    let w = -((1.0 - x) * (1.0 + x)).ln();
    if w < 5.0 {
        let w = w - 2.5;
        let mut p = 2.810_226_36e-08;
        for c in [
            3.432_739_39e-07,
            -3.523_387_7e-06,
            -4.391_506_54e-06,
            0.000_218_580_87,
            -0.001_253_725_03,
            -0.004_177_681_64,
            0.246_640_727,
            1.501_409_41,
        ] {
            p = c + p * w;
        }
        p * x
    } else {
        let w = w.sqrt() - 3.0;
        let mut p = -0.000_200_214_257;
        for c in [
            0.000_100_950_558,
            0.001_349_343_22,
            -0.003_673_428_44,
            0.005_739_507_73,
            -0.007_622_461_3,
            0.009_438_870_47,
            1.001_674_06,
            2.832_976_82,
        ] {
            p = c + p * w;
        }
        p * x
    }
}

fn waves(rng: &mut Rng, n: usize, wavelength: f64, axial_stretch: f64) -> Vec<(V3, f64)> {
    let k = 2.0 * std::f64::consts::PI / wavelength;
    (0..n)
        .map(|_| {
            let v = [rng.normal() / axial_stretch, rng.normal(), rng.normal()];
            let l = crate::util::norm(v).max(1e-12);
            (crate::util::scale(v, k / l), 2.0 * std::f64::consts::PI * rng.uniform())
        })
        .collect()
}

#[inline]
fn field(w: &[(V3, f64)], p: V3) -> f64 {
    let s: f64 = w.iter().map(|(k, ph)| (crate::util::dot(*k, p) + ph).cos()).sum();
    s * (2.0 / w.len() as f64).sqrt() // unit variance
}

impl BonePhantom {
    pub fn geometry(&self) -> Geometry {
        Geometry::new(self.spacing)
    }

    fn coeffs(&self) -> Coeffs {
        let mut r = Rng::new(self.seed);
        let tau = 2.0 * std::f64::consts::PI;
        let rate = |r: &mut Rng| tau / self.axial_wavelength * r.normal() * 0.5;
        let shape = (2..2 + self.shape_harmonics)
            .map(|k| {
                let a = self.shape_amplitude * r.normal() * (k as f64 / 2.0).powf(-1.5);
                (a, tau * r.uniform(), rate(&mut r))
            })
            .collect();
        let thick = (1..1 + self.thickness_harmonics)
            .map(|k| {
                let a = self.thickness_variation * r.normal() * (k as f64).powf(-1.0) / 1.5;
                (a, tau * r.uniform(), rate(&mut r))
            })
            .collect();
        let center = [0, 1].map(|_| {
            (
                self.center_wander * r.normal(),
                tau / self.axial_wavelength * r.uniform(),
                tau * r.uniform(),
            )
        });
        let thick_axial = (0.15 * self.thickness_variation * r.normal(), tau * r.uniform());
        let trab = waves(&mut r, 48, self.trabecular_wavelength, 2.0);
        let pores = waves(&mut r, 32, self.pore_wavelength, 2.5);
        // Level-set widths are computed per sample in `density`:
        // P(|f| < w) = erf(w/√2) for a unit-variance Gaussian field.
        Coeffs {
            shape,
            thick,
            center,
            thick_axial,
            trab,
            pores,
        }
    }

    /// Physical centre of the cross-section at axial position `z`.
    fn center_at(&self, c: &Coeffs, z: f64) -> [f64; 2] {
        let (_, d1, d2) = self.dims;
        let base = [
            (d1 as f64 - 1.0) * 0.5 * self.spacing[1],
            (d2 as f64 - 1.0) * 0.5 * self.spacing[2],
        ];
        [
            base[0] + c.center[0].0 * (c.center[0].1 * z + c.center[0].2).sin(),
            base[1] + c.center[1].0 * (c.center[1].1 * z + c.center[1].2).sin(),
        ]
    }

    /// (r, θ, R(θ,z), t(θ,z)) of a physical point.
    fn polar(&self, c: &Coeffs, p: V3) -> (f64, f64, f64, f64) {
        let z = p[0];
        let cc = self.center_at(c, z);
        let (dy, dx) = (p[1] - cc[0], p[2] - cc[1]);
        let r = (dy * dy + dx * dx).sqrt();
        let th = dx.atan2(dy);
        let mut s = 1.0;
        for (i, &(a, ph, w)) in c.shape.iter().enumerate() {
            s += a * ((i + 2) as f64 * th + ph + w * z).cos();
        }
        let big_r = self.radius * s;
        let mut tf = 1.0;
        for (i, &(a, ph, w)) in c.thick.iter().enumerate() {
            tf += a * ((i + 1) as f64 * th + ph + w * z).cos();
        }
        let len = self.dims.0 as f64 * self.spacing[0];
        tf *= 1.0 + c.thick_axial.0 * (2.0 * std::f64::consts::PI * z / len + c.thick_axial.1).sin();
        let t = (self.thickness_mean * tf).max(self.thickness_min).min(0.8 * big_r);
        (r, th, big_r, t)
    }

    fn density(&self, c: &Coeffs, p: V3) -> f64 {
        let (r, _, big_r, t) = self.polar(c, p);
        let re = big_r - t;
        if r >= big_r {
            self.rho_soft
        } else if r >= re {
            // relative depth from the endosteal surface (0) to the periosteum (1)
            let u = (r - re) / t;
            let por = (self.porosity + self.endo_porosity * (-u / 0.25).exp()).min(0.49);
            let tau = if por > 0.0 {
                2f64.sqrt() * erfinv(1.0 - 2.0 * por)
            } else {
                f64::INFINITY
            };
            if field(&c.pores, p) > tau {
                self.rho_marrow
            } else {
                self.rho_bone
            }
        } else {
            let d = re - r;
            let bv = if self.transition_width > 0.0 {
                self.bvtv + (self.endo_bvtv - self.bvtv) * (-d / self.transition_width).exp()
            } else {
                self.bvtv
            };
            if bv > 0.0 && field(&c.trab, p).abs() < 2f64.sqrt() * erfinv(bv.min(0.99)) {
                self.rho_bone
            } else {
                self.rho_marrow
            }
        }
    }

    /// Generate `(image, periosteal mask, endosteal mask)`; the cortical mask
    /// is `periosteal & !endosteal`. Masks are evaluated at voxel centres.
    pub fn generate(&self) -> (Array3<f32>, Array3<bool>, Array3<bool>) {
        let c = self.coeffs();
        let g = self.geometry();
        let ss = self.supersample.max(1);
        let (d0, d1, d2) = self.dims;
        let mut truth = Array3::<f64>::zeros(self.dims);
        truth
            .as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(lin, v)| {
                let (i, j, k) = (lin / (d1 * d2), (lin / d2) % d1, lin % d2);
                let off = |q: usize| (q as f64 + 0.5) / ss as f64 - 0.5;
                let mut acc = 0.0;
                for a in 0..ss {
                    for b in 0..ss {
                        for e in 0..ss {
                            acc += self.density(
                                &c,
                                g.to_physical([i as f64 + off(a), j as f64 + off(b), k as f64 + off(e)]),
                            );
                        }
                    }
                }
                *v = acc / (ss * ss * ss) as f64;
            });
        let sig = [0, 1, 2].map(|a| self.blur[a] / self.spacing[a]);
        let blurred = gaussian_filter(&truth, sig);
        drop(truth);
        let n = d0 * d1 * d2;
        let mut noise = vec![0f32; n];
        noise.par_chunks_mut(65536).enumerate().for_each(|(ci, ch)| {
            let mut r = Rng::new(self.seed ^ 0xABCD ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            ch.iter_mut().for_each(|v| *v = r.normal() as f32);
        });
        let image = Array3::from_shape_vec(
            self.dims,
            blurred
                .iter()
                .zip(noise)
                .map(|(&v, e)| (v + self.noise * e as f64) as f32)
                .collect(),
        )
        .unwrap();
        let mut peri = Array3::from_elem(self.dims, false);
        let mut endo = Array3::from_elem(self.dims, false);
        peri.as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .zip(endo.as_slice_mut().unwrap().par_iter_mut())
            .enumerate()
            .for_each(|(lin, (pv, ev))| {
                let (i, j, k) = (lin / (d1 * d2), (lin / d2) % d1, lin % d2);
                let (r, _, big_r, t) = self.polar(&c, g.to_physical([i as f64, j as f64, k as f64]));
                *pv = r < big_r;
                *ev = r < big_r - t;
            });
        (image, peri, endo)
    }

    /// The true periosteal and endosteal crossing positions `(x_p, x_e)`
    /// along each line `p + x n`, `x ∈ [x0, x1]`: `x_p` is the first entry
    /// into the bone and `x_e` the first entry into the marrow space after it
    /// (NaN where the line misses).
    pub fn edges_along(&self, points: &[V3], normals: &[V3], x0: f64, x1: f64) -> Vec<(f64, f64)> {
        let c = self.coeffs();
        let h = 0.25 * self.spacing.iter().cloned().fold(f64::INFINITY, f64::min);
        points
            .par_iter()
            .zip(normals.par_iter())
            .map(|(&p, &n)| {
                let at = |x: f64| crate::util::add(p, crate::util::scale(n, x));
                let outer = |x: f64| {
                    let (r, _, big_r, _) = self.polar(&c, at(x));
                    r - big_r
                };
                let inner = |x: f64| {
                    let (r, _, big_r, t) = self.polar(&c, at(x));
                    r - (big_r - t)
                };
                let find = |f: &dyn Fn(f64) -> f64, from: f64| -> f64 {
                    let mut a = from;
                    let mut fa = f(a);
                    if fa < 0.0 {
                        return f64::NAN; // already inside
                    }
                    while a < x1 {
                        let b = (a + h).min(x1);
                        let fb = f(b);
                        if fb < 0.0 {
                            let (mut lo, mut hi) = (a, b);
                            for _ in 0..50 {
                                let m = 0.5 * (lo + hi);
                                if f(m) < 0.0 {
                                    hi = m
                                } else {
                                    lo = m
                                }
                            }
                            return 0.5 * (lo + hi);
                        }
                        a = b;
                        fa = fb;
                    }
                    let _ = fa;
                    f64::NAN
                };
                let xp = find(&outer, x0);
                if xp.is_nan() {
                    return (f64::NAN, f64::NAN);
                }
                (xp, find(&inner, xp))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfinv_is_inverse() {
        for &x in &[-0.9, -0.3, 0.0, 0.2, 0.6, 0.95, 0.999] {
            assert!((libm::erf(erfinv(x)) - x).abs() < 1e-6, "{x}");
        }
    }

    #[test]
    fn phantom_statistics_and_truth() {
        let ph = BonePhantom {
            dims: (12, 160, 160),
            noise: 0.0,
            ..BonePhantom::default()
        };
        let (img, peri, endo) = ph.generate();
        // deep marrow (away from the transition zone)
        let c0 = ph.coeffs();
        let g0 = ph.geometry();
        let deep = Array3::from_shape_fn(ph.dims, |(i, j, k)| {
            let (r, _, big_r, t) = ph.polar(&c0, g0.to_physical([i as f64, j as f64, k as f64]));
            r < big_r - t - 4.0 * ph.transition_width
        });
        let marrow: Vec<f32> = img
            .iter()
            .zip(deep.iter())
            .filter(|(_, &e)| e)
            .map(|(&v, _)| v)
            .collect();
        assert!(endo.iter().filter(|&&b| b).count() > marrow.len());
        // marrow-space mean ≈ ρm + BV/TV (ρb − ρm)
        let mean = marrow.iter().map(|&v| v as f64).sum::<f64>() / marrow.len() as f64;
        let want = ph.rho_marrow + ph.bvtv * (ph.rho_bone - ph.rho_marrow);
        assert!((mean - want).abs() < 0.08 * want, "marrow mean {mean} vs {want}");
        // the cortex exists everywhere and is not a circle
        assert!(peri.iter().filter(|&&b| b).count() > endo.iter().filter(|&&b| b).count());
        let c = ph.coeffs();
        let g = ph.geometry();
        let mid = [6.0 * g.spacing[0], 0.0, 0.0];
        let rs: Vec<f64> = (0..36)
            .map(|i| {
                let th = i as f64 * 10f64.to_radians();
                let cc = ph.center_at(&c, mid[0]);
                ph.polar(&c, [mid[0], cc[0] + th.cos(), cc[1] + th.sin()]).2
            })
            .collect();
        let (lo, hi) = rs
            .iter()
            .fold((f64::INFINITY, 0.0f64), |a, &r| (a.0.min(r), a.1.max(r)));
        assert!(hi - lo > 0.3, "radius range {lo}..{hi}");
        // edges along a radial line from outside agree with the implicit surfaces
        let cc = ph.center_at(&c, mid[0]);
        let p = [mid[0], cc[0] + 7.0, cc[1]];
        let e = ph.edges_along(&[p], &[[0.0, -1.0, 0.0]], 0.0, 7.0)[0];
        let (_, _, big_r, t) = ph.polar(&c, [mid[0], cc[0] + 4.0, cc[1]]);
        assert!(((7.0 - e.0) - big_r).abs() < 1e-6, "{e:?} R {big_r}");
        assert!(((e.1 - e.0) - t).abs() < 1e-6);
    }
}
