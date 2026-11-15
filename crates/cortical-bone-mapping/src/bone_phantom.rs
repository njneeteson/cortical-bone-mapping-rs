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
    /// A neighbouring bone (fibula / ulna) next to the main bone.
    pub neighbour: Option<NeighbourBone>,
    /// Cortical breaks (channels of soft tissue through the cortex).
    pub breaks: Vec<CorticalBreak>,
}

/// A second, simpler bone beside the main one: a circular cross-section with
/// a solid cortex and the same trabecular pattern, placed so that the gap
/// between the two periosteal surfaces along the direction `angle` is `gap`
/// in every slice (it follows the main bone's surface). In vivo there is
/// always soft tissue between the radius and ulna or the tibia and fibula
/// (interosseous membrane, joint cartilage), typically 1–5 mm at the
/// standard HR-pQCT sites; `gap` should be positive.
#[derive(Clone, Debug, PartialEq)]
pub struct NeighbourBone {
    pub radius: f64,
    pub thickness: f64,
    pub gap: f64,
    /// Direction from the main bone (radians; 0 = array axis 1).
    pub angle: f64,
}

/// A cortical break: a channel of soft tissue through the full cortical
/// depth, elliptical on the periosteal surface with axes `width` (along the
/// arc, centred on `angle`) and `length` (along the axis, centred on
/// `z_center`), in physical units. Real breaks in HR-pQCT are small: the
/// detectable ones are ≳ 0.25 mm across, healthy finger joints show a median
/// break volume of ~0.2 mm³ and rheumatoid arthritis ~0.5 mm³ (Peters et al.
/// 2017, 2019), i.e. diameters of roughly 0.3–1.2 mm through a ~0.5–1 mm
/// cortex. At the radius and tibia the analogues are transcortical vascular
/// foramina and, in thin osteoporotic cortices, local loss of the shell.
#[derive(Clone, Debug, PartialEq)]
pub struct CorticalBreak {
    pub angle: f64,
    pub width: f64,
    pub z_center: f64,
    pub length: f64,
}

impl Default for BonePhantom {
    /// A small long-bone segment at HR-pQCT resolution (scaled down: a real
    /// distal radius has a periosteal radius of ~8.5–10.5 mm, a distal tibia
    /// ~13.5–16.5 mm).
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
            neighbour: None,
            breaks: Vec::new(),
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
pub(crate) fn erfinv(x: f64) -> f64 {
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

pub(crate) fn waves(rng: &mut Rng, n: usize, wavelength: f64, axial_stretch: f64) -> Vec<(V3, f64)> {
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
pub(crate) fn field(w: &[(V3, f64)], p: V3) -> f64 {
    let s: f64 = w.iter().map(|(k, ph)| (crate::util::dot(*k, p) + ph).cos()).sum();
    s * (2.0 / w.len() as f64).sqrt() // unit variance
}

/// Render a phantom: integrate `density` over each voxel by supersampling,
/// blur with a Gaussian PSF (`blur` per axis, physical units) and add
/// Gaussian noise of standard deviation `noise`.
pub(crate) fn render(
    dims: (usize, usize, usize),
    g: &Geometry,
    blur: V3,
    noise: f64,
    seed: u64,
    supersample: usize,
    density: impl Fn(V3) -> f64 + Sync,
) -> Array3<f32> {
    let ss = supersample.max(1);
    let (d0, d1, d2) = dims;
    let mut truth = Array3::<f64>::zeros(dims);
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
                        acc += density(g.to_physical([i as f64 + off(a), j as f64 + off(b), k as f64 + off(e)]));
                    }
                }
            }
            *v = acc / (ss * ss * ss) as f64;
        });
    let sig = [0, 1, 2].map(|a| blur[a] / g.spacing[a]);
    let blurred = gaussian_filter(&truth, sig);
    drop(truth);
    let n = d0 * d1 * d2;
    let mut eps = vec![0f32; n];
    eps.par_chunks_mut(65536).enumerate().for_each(|(ci, ch)| {
        let mut r = Rng::new(seed ^ 0xABCD ^ (ci as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        ch.iter_mut().for_each(|v| *v = r.normal() as f32);
    });
    Array3::from_shape_vec(
        dims,
        blurred
            .iter()
            .zip(eps)
            .map(|(&v, e)| (v + noise * e as f64) as f32)
            .collect(),
    )
    .unwrap()
}

/// The first `x` in `(from, x1]` where `f` becomes negative, by stepping `h`
/// and bisection; NaN if `f(from) < 0` (already inside) or never.
pub(crate) fn first_entry(f: &dyn Fn(f64) -> f64, from: f64, x1: f64, h: f64) -> f64 {
    let mut a = from;
    if f(a) < 0.0 {
        return f64::NAN;
    }
    while a < x1 {
        let b = (a + h).min(x1);
        if f(b) < 0.0 {
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
    }
    f64::NAN
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
        let big_r = self.radius_at(c, th, z);
        let mut tf = 1.0;
        for (i, &(a, ph, w)) in c.thick.iter().enumerate() {
            tf += a * ((i + 1) as f64 * th + ph + w * z).cos();
        }
        let len = self.dims.0 as f64 * self.spacing[0];
        tf *= 1.0 + c.thick_axial.0 * (2.0 * std::f64::consts::PI * z / len + c.thick_axial.1).sin();
        let t = (self.thickness_mean * tf).max(self.thickness_min).min(0.8 * big_r);
        (r, th, big_r, t)
    }

    /// Periosteal radius `R(θ, z)` of the main bone.
    fn radius_at(&self, c: &Coeffs, th: f64, z: f64) -> f64 {
        let mut s = 1.0;
        for (i, &(a, ph, w)) in c.shape.iter().enumerate() {
            s += a * ((i + 2) as f64 * th + ph + w * z).cos();
        }
        self.radius * s
    }

    /// In-plane distance from the neighbouring bone's axis, if there is one.
    fn neighbour_distance(&self, c: &Coeffs, p: V3) -> Option<(f64, &NeighbourBone)> {
        let nb = self.neighbour.as_ref()?;
        let cc = self.center_at(c, p[0]);
        let d = self.radius_at(c, nb.angle, p[0]) + nb.gap + nb.radius;
        let (ny, nx) = (cc[0] + d * nb.angle.cos(), cc[1] + d * nb.angle.sin());
        Some((((p[1] - ny).powi(2) + (p[2] - nx).powi(2)).sqrt(), nb))
    }

    /// Whether the point (taken to be in the cortex shell) lies in a break.
    fn in_break(&self, th: f64, big_r: f64, z: f64) -> bool {
        let tau = 2.0 * std::f64::consts::PI;
        self.breaks.iter().any(|b| {
            let dth = (th - b.angle + 0.5 * tau).rem_euclid(tau) - 0.5 * tau;
            let u = dth * big_r / (0.5 * b.width);
            let v = (z - b.z_center) / (0.5 * b.length);
            u * u + v * v < 1.0
        })
    }

    fn density(&self, c: &Coeffs, p: V3) -> f64 {
        let (r, th, big_r, t) = self.polar(c, p);
        let re = big_r - t;
        if r >= big_r {
            match self.neighbour_distance(c, p) {
                Some((d, nb)) if d < nb.radius => {
                    let solid = d >= nb.radius - nb.thickness;
                    let trab = self.bvtv > 0.0 && field(&c.trab, p).abs() < 2f64.sqrt() * erfinv(self.bvtv.min(0.99));
                    if solid || trab {
                        self.rho_bone
                    } else {
                        self.rho_marrow
                    }
                }
                _ => self.rho_soft,
            }
        } else if r >= re && self.in_break(th, big_r, p[0]) {
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
        let image = render(self.dims, &g, self.blur, self.noise, self.seed, self.supersample, |p| {
            self.density(&c, p)
        });
        let (_, d1, d2) = self.dims;
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

    /// Feature labels at voxel centres: 1 = neighbouring bone, 2 = cortical
    /// break (soft tissue inside the main bone's cortex shell), 0 otherwise.
    pub fn labels(&self) -> Array3<u8> {
        let c = self.coeffs();
        let g = self.geometry();
        let (_, d1, d2) = self.dims;
        let mut out = Array3::<u8>::zeros(self.dims);
        out.as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .enumerate()
            .for_each(|(lin, v)| {
                let (i, j, k) = (lin / (d1 * d2), (lin / d2) % d1, lin % d2);
                let p = g.to_physical([i as f64, j as f64, k as f64]);
                let (r, th, big_r, t) = self.polar(&c, p);
                *v = if r >= big_r {
                    u8::from(matches!(self.neighbour_distance(&c, p), Some((d, nb)) if d < nb.radius))
                } else if r >= big_r - t && self.in_break(th, big_r, p[0]) {
                    2
                } else {
                    0
                };
            });
        out
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
                let find = |f: &dyn Fn(f64) -> f64, from: f64| first_entry(f, from, x1, h);
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

    #[test]
    fn neighbour_and_breaks() {
        let gap = 0.5;
        let ph = BonePhantom {
            dims: (6, 200, 200),
            noise: 0.0,
            neighbour: Some(NeighbourBone {
                radius: 1.2,
                thickness: 0.4,
                gap,
                angle: std::f64::consts::FRAC_PI_4,
            }),
            breaks: vec![CorticalBreak {
                angle: -2.0,
                width: 1.0,
                z_center: 0.15,
                length: 0.2,
            }],
            ..BonePhantom::default()
        };
        let (_, peri, _) = ph.generate();
        let lab = ph.labels();
        let c = ph.coeffs();
        let g = ph.geometry();
        // the neighbour is outside the main bone, at the requested gap
        assert!(lab.iter().zip(peri.iter()).all(|(&l, &p)| !(l == 1 && p)));
        let z = 3.0 * g.spacing[0];
        let a = std::f64::consts::FRAC_PI_4;
        let cc = ph.center_at(&c, z);
        let r = ph.radius_at(&c, a, z);
        let at = |d: f64| [z, cc[0] + d * a.cos(), cc[1] + d * a.sin()];
        assert!(ph.neighbour_distance(&c, at(r + 0.5 * gap)).unwrap().0 > 1.2);
        assert!(ph.neighbour_distance(&c, at(r + gap + 0.05)).unwrap().0 < 1.2);
        // breaks: soft tissue through the cortex in the labelled slices only
        let nb = lab.iter().filter(|&&l| l == 2).count();
        assert!(nb > 0);
        // |z − 0.15| < 0.1 mm: slices 1–4 at 60.7 µm
        for (idx, &l) in lab.indexed_iter() {
            if l == 2 {
                assert!((1..=4).contains(&idx.0), "break at slice {}", idx.0);
            }
        }
        let mid = 2.0 * g.spacing[0];
        let cc = ph.center_at(&c, mid);
        let (_, _, big_r, t) = ph.polar(&c, [mid, cc[0] + (-2f64).cos(), cc[1] + (-2f64).sin()]);
        let p = [
            mid,
            cc[0] + (big_r - 0.5 * t) * (-2f64).cos(),
            cc[1] + (big_r - 0.5 * t) * (-2f64).sin(),
        ];
        assert_eq!(ph.density(&c, p), ph.rho_soft);
    }
}
