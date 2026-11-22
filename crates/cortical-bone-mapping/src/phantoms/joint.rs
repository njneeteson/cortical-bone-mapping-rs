//! A knee-joint phantom: the articular ends of two bones (proximal tibia
//! below, distal femur above, along array axis 0) facing each other across a
//! joint space (cartilage).
//!
//! Each bone is the intersection of a star-shaped cylinder (its in-plane
//! footprint, a random harmonic curve) and a half-space bounded by its
//! articular surface, with the rim rounded by a smooth maximum:
//!
//! * tibia: `φ_T = smax(r − R_T(θ), z − z_T(y, x))`, with the plateau
//!   `z_T = z_j + q²/(2 ρ_T) + b(y, x)` a shallow dish around the contact point
//!   (`q` the in-plane distance from it) plus random bumps `b`;
//! * femur: `φ_F = smax(r − R_F(θ), z_F(y, x) − z)`, with the condyle
//!   `z_F = z_T + g + q² (1/(2 ρ_F) − 1/(2 ρ_T))`, i.e. a more curved surface
//!   touching the tibia nowhere: the joint space is `g` at the contact point
//!   and widens towards the rim.
//!
//! The cortex is a subchondral plate of thickness `t_p(y, x)` (a random field)
//! under the articular surface and a thin shell of thickness `t_s` on the
//! sides: the marrow space is `smax(r − (R − t_s), ±(z − z_*) + t_p) < 0`.
//! Marrow holds a trabecular network whose bone volume fraction increases
//! towards the subchondral plate; the plate has pores. The joint space is
//! filled with cartilage / synovial fluid (`rho_cartilage`), the rest with
//! soft tissue.
//!
//! The phantom is a scaled-down joint (a real tibial plateau is ~70 mm wide):
//! it reproduces the local geometry of the contact region (plate, curvature,
//! a cartilage-filled joint space) in a volume small enough for tests. The
//! default joint space, 4 mm, is a healthy bone-to-bone distance (radiographic
//! medial joint space width ~4.8 mm in women and ~5.7 mm in men; tibial
//! cartilage ~1.5 mm thick); narrowing to ~2 mm is moderate osteoarthritis,
//! and bone-on-bone contact only occurs in end-stage disease. Ground truth is exact as for
//! [`BonePhantom`](crate::phantoms::bone::BonePhantom).
//!
//! The smooth maximum is the log-sum-exp
//! `smax(a, b; k) = k ln(e^{a/k} + e^{b/k}) = max(a, b) + k ln(1 + e^{−|a−b|/k})`,
//! which exceeds `max(a, b)` by at most `k ln 2` and rounds the rim over a width
//! of about `k` while keeping the implicit functions smooth for the bisection
//! of the ground truth. The phantom is ours; the joint-space and cartilage
//! dimensions follow measurements in healthy adults (the knee reference below).
//!
//! # References
//!
//! * K. A. Beattie et al., *Minimum joint space width and tibial cartilage
//!   morphology in the knees of healthy individuals: a cross-sectional study*,
//!   BMC Musculoskeletal Disorders 9:119, 2008, doi:10.1186/1471-2474-9-119.

use ndarray::Array3;
use rayon::prelude::*;

use super::render::{erfinv, field, first_entry, render, waves};
use crate::util::{Rng, V3};
use crate::volume::Geometry;

#[derive(Clone, Debug)]
pub struct JointPhantom {
    pub dims: (usize, usize, usize),
    pub spacing: V3,
    /// Mean footprint radius (tibia; the femur is `femur_scale` × larger).
    pub radius: f64,
    pub femur_scale: f64,
    pub shape_harmonics: usize,
    pub shape_amplitude: f64,
    /// Height of the joint line (the tibial plateau at the contact point),
    /// as a fraction of the volume height.
    pub joint_level: f64,
    /// Minimum joint space (at the contact point).
    pub joint_space: f64,
    /// In-plane offset of the contact point from the bone axis.
    pub contact_offset: [f64; 2],
    /// Radii of curvature of the tibial dish and the femoral condyle.
    pub tibia_curvature: f64,
    pub femur_curvature: f64,
    /// Amplitude and wavelength of the random bumps of the articular surfaces.
    pub bump_amplitude: f64,
    pub bump_wavelength: f64,
    /// Subchondral plate thickness: mean, relative variation, minimum.
    pub plate_thickness: f64,
    pub plate_variation: f64,
    pub plate_min: f64,
    /// Thickness of the cortical shell on the sides.
    pub shell_thickness: f64,
    /// Rounding of the rim (smooth-maximum width).
    pub rim_rounding: f64,
    pub bvtv: f64,
    /// Trabecular BV/TV next to the subchondral plate.
    pub endo_bvtv: f64,
    pub transition_width: f64,
    pub trabecular_wavelength: f64,
    pub porosity: f64,
    pub pore_wavelength: f64,
    pub rho_soft: f64,
    pub rho_cartilage: f64,
    pub rho_bone: f64,
    pub rho_marrow: f64,
    pub blur: V3,
    pub noise: f64,
    pub seed: u64,
    pub supersample: usize,
}

impl Default for JointPhantom {
    fn default() -> Self {
        JointPhantom {
            dims: (180, 200, 200),
            spacing: [0.0607; 3],
            radius: 5.0,
            femur_scale: 1.05,
            shape_harmonics: 5,
            shape_amplitude: 0.08,
            joint_level: 0.35,
            joint_space: 4.0,
            contact_offset: [0.8, -0.5],
            tibia_curvature: 20.0,
            femur_curvature: 9.0,
            bump_amplitude: 0.08,
            bump_wavelength: 4.0,
            plate_thickness: 0.45,
            plate_variation: 0.3,
            plate_min: 0.15,
            shell_thickness: 0.3,
            rim_rounding: 0.4,
            bvtv: 0.22,
            endo_bvtv: 0.4,
            transition_width: 0.4,
            trabecular_wavelength: 1.0,
            porosity: 0.03,
            pore_wavelength: 0.35,
            rho_soft: 50.0,
            rho_cartilage: 80.0,
            rho_bone: 1000.0,
            rho_marrow: 30.0,
            blur: [0.08; 3],
            noise: 40.0,
            seed: 1,
            supersample: 2,
        }
    }
}

/// Which bone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bone {
    Tibia = 1,
    Femur = 2,
}

struct Coeffs {
    shape: [Vec<(f64, f64)>; 2], // per bone: (amplitude, phase) for k = 2..
    bumps: Vec<(V3, f64)>,       // 2D waves (axis-0 component zero)
    plate: [Vec<(V3, f64)>; 2],
    trab: Vec<(V3, f64)>,
    pores: Vec<(V3, f64)>,
}

/// Smooth maximum `k ln(e^{a/k} + e^{b/k})` with width `k` (`max` for
/// `k = 0`), written as `max + k ln(1 + e^{−|a−b|/k})` to avoid overflow.
fn smax(a: f64, b: f64, k: f64) -> f64 {
    if k <= 0.0 {
        return a.max(b);
    }
    let m = a.max(b);
    m + k * (-(a - b).abs() / k).exp().ln_1p()
}

impl JointPhantom {
    /// The phantom's voxel grid (origin at zero).
    pub fn geometry(&self) -> Geometry {
        Geometry::new(self.spacing)
    }

    /// The random coefficients of the phantom, drawn from its seed: footprints,
    /// articular bumps, plate thickness, trabecular and pore fields.
    fn coeffs(&self) -> Coeffs {
        let mut r = Rng::new(self.seed);
        let tau = 2.0 * std::f64::consts::PI;
        let mut shape = || -> Vec<(f64, f64)> {
            (2..2 + self.shape_harmonics)
                .map(|k| {
                    (
                        self.shape_amplitude * r.normal() * (k as f64 / 2.0).powf(-1.5),
                        tau * r.uniform(),
                    )
                })
                .collect()
        };
        let shape = [shape(), shape()];
        let flat = |w: Vec<(V3, f64)>| -> Vec<(V3, f64)> {
            w.into_iter()
                .map(|(k, ph)| {
                    let l = (k[1] * k[1] + k[2] * k[2]).sqrt().max(1e-12);
                    let s = crate::util::norm(k) / l;
                    ([0.0, k[1] * s, k[2] * s], ph)
                })
                .collect()
        };
        let bumps = flat(waves(&mut r, 12, self.bump_wavelength, 1.0));
        let plate = [flat(waves(&mut r, 12, 3.0, 1.0)), flat(waves(&mut r, 12, 3.0, 1.0))];
        let trab = waves(&mut r, 48, self.trabecular_wavelength, 1.0);
        let pores = waves(&mut r, 32, self.pore_wavelength, 1.0);
        Coeffs {
            shape,
            bumps,
            plate,
            trab,
            pores,
        }
    }

    /// In-plane position of the bones' common axis (the volume's centre).
    fn axis(&self) -> [f64; 2] {
        let (_, d1, d2) = self.dims;
        [
            (d1 as f64 - 1.0) * 0.5 * self.spacing[1],
            (d2 as f64 - 1.0) * 0.5 * self.spacing[2],
        ]
    }

    /// Height of the joint line (physical).
    pub fn joint_height(&self) -> f64 {
        self.joint_level * (self.dims.0 as f64 - 1.0) * self.spacing[0]
    }

    /// Articular surface heights `(z_T, z_F)` at in-plane position `(y, x)`.
    fn surfaces(&self, c: &Coeffs, p: V3) -> (f64, f64) {
        let a = self.axis();
        let q2 = (p[1] - a[0] - self.contact_offset[0]).powi(2) + (p[2] - a[1] - self.contact_offset[1]).powi(2);
        let b = self.bump_amplitude * field(&c.bumps, [0.0, p[1], p[2]]);
        let zt = self.joint_height() + q2 / (2.0 * self.tibia_curvature) + b;
        let zf = zt + self.joint_space + q2 * (1.0 / (2.0 * self.femur_curvature) - 1.0 / (2.0 * self.tibia_curvature));
        (zt, zf)
    }

    /// `(r, R_bone(θ))` for a bone.
    fn footprint(&self, c: &Coeffs, p: V3, bone: Bone) -> (f64, f64) {
        let a = self.axis();
        let (dy, dx) = (p[1] - a[0], p[2] - a[1]);
        let th = dx.atan2(dy);
        let i = bone as usize - 1;
        let mut s = 1.0;
        for (k, &(amp, ph)) in c.shape[i].iter().enumerate() {
            s += amp * ((k + 2) as f64 * th + ph).cos();
        }
        let scale = if bone == Bone::Femur { self.femur_scale } else { 1.0 };
        ((dy * dy + dx * dx).sqrt(), self.radius * scale * s)
    }

    /// Subchondral plate thickness of a bone at in-plane position `p`:
    /// `t̄ (1 + v f)` for a unit random field `f`, clipped at the minimum.
    fn plate_at(&self, c: &Coeffs, p: V3, bone: Bone) -> f64 {
        let f = field(&c.plate[bone as usize - 1], [0.0, p[1], p[2]]);
        (self.plate_thickness * (1.0 + self.plate_variation * f)).max(self.plate_min)
    }

    /// `(φ_outer, φ_marrow)` of a bone: negative inside.
    fn implicit(&self, c: &Coeffs, p: V3, bone: Bone) -> (f64, f64) {
        let (zt, zf) = self.surfaces(c, p);
        let (r, big_r) = self.footprint(c, p, bone);
        let along = match bone {
            Bone::Tibia => p[0] - zt,
            Bone::Femur => zf - p[0],
        };
        let k = self.rim_rounding;
        let tp = self.plate_at(c, p, bone);
        (
            smax(r - big_r, along, k),
            smax(r - (big_r - self.shell_thickness), along + tp, k),
        )
    }

    /// The true density at `p`: inside a bone, cortex (with pores, the level set
    /// `f > √2 erf⁻¹(1 − 2P)`) or marrow with trabeculae whose BV/TV rises towards
    /// the plate as `BV + (BV_endo − BV) exp(−d / w)`; cartilage between the two
    /// articular surfaces within the tibial footprint; soft tissue elsewhere.
    fn density(&self, c: &Coeffs, p: V3) -> f64 {
        for bone in [Bone::Tibia, Bone::Femur] {
            let (outer, marrow) = self.implicit(c, p, bone);
            if outer >= 0.0 {
                continue;
            }
            if marrow >= 0.0 {
                let por = self.porosity.min(0.49);
                let tau = if por > 0.0 {
                    2f64.sqrt() * erfinv(1.0 - 2.0 * por)
                } else {
                    f64::INFINITY
                };
                return if field(&c.pores, p) > tau {
                    self.rho_marrow
                } else {
                    self.rho_bone
                };
            }
            let d = -marrow; // ≈ depth into the marrow space
            let bv = if self.transition_width > 0.0 {
                self.bvtv + (self.endo_bvtv - self.bvtv) * (-d / self.transition_width).exp()
            } else {
                self.bvtv
            };
            return if bv > 0.0 && field(&c.trab, p).abs() < 2f64.sqrt() * erfinv(bv.min(0.99)) {
                self.rho_bone
            } else {
                self.rho_marrow
            };
        }
        let (zt, zf) = self.surfaces(c, p);
        let (r, big_r) = self.footprint(c, p, Bone::Tibia);
        if p[0] > zt && p[0] < zf && r < big_r {
            self.rho_cartilage
        } else {
            self.rho_soft
        }
    }

    /// Generate `(image, periosteal labels, endosteal labels)`: labels are
    /// 0 outside, [`Bone::Tibia`] (1) or [`Bone::Femur`] (2) inside the bone's
    /// periosteal surface / its marrow space, at voxel centres.
    pub fn generate(&self) -> (Array3<f32>, Array3<u8>, Array3<u8>) {
        let c = self.coeffs();
        let g = self.geometry();
        let image = render(self.dims, &g, self.blur, self.noise, self.seed, self.supersample, |p| {
            self.density(&c, p)
        });
        let (_, d1, d2) = self.dims;
        let mut peri = Array3::<u8>::zeros(self.dims);
        let mut endo = Array3::<u8>::zeros(self.dims);
        peri.as_slice_mut()
            .unwrap()
            .par_iter_mut()
            .zip(endo.as_slice_mut().unwrap().par_iter_mut())
            .enumerate()
            .for_each(|(lin, (pv, ev))| {
                let (i, j, k) = (lin / (d1 * d2), (lin / d2) % d1, lin % d2);
                let p = g.to_physical([i as f64, j as f64, k as f64]);
                for bone in [Bone::Tibia, Bone::Femur] {
                    let (o, m) = self.implicit(&c, p, bone);
                    if o < 0.0 {
                        *pv = bone as u8;
                        *ev = if m < 0.0 { bone as u8 } else { 0 };
                    }
                }
            });
        (image, peri, endo)
    }

    /// True `(x_p, x_e, bone)` along each line `p + x n`, `x ∈ [x0, x1]`: the
    /// first entry into either bone, and the first entry into that bone's
    /// marrow space after it (NaN / 0 where the line misses).
    pub fn edges_along(&self, points: &[V3], normals: &[V3], x0: f64, x1: f64) -> Vec<(f64, f64, u8)> {
        let c = self.coeffs();
        let h = 0.25 * self.spacing.iter().cloned().fold(f64::INFINITY, f64::min);
        points
            .par_iter()
            .zip(normals.par_iter())
            .map(|(&p, &n)| {
                let at = |x: f64| crate::util::add(p, crate::util::scale(n, x));
                let outer = |x: f64| {
                    let q = at(x);
                    self.implicit(&c, q, Bone::Tibia)
                        .0
                        .min(self.implicit(&c, q, Bone::Femur).0)
                };
                let xp = first_entry(&outer, x0, x1, h);
                if xp.is_nan() {
                    return (f64::NAN, f64::NAN, 0);
                }
                let q = at(xp + 1e-6);
                let bone = if self.implicit(&c, q, Bone::Tibia).0 <= self.implicit(&c, q, Bone::Femur).0 {
                    Bone::Tibia
                } else {
                    Bone::Femur
                };
                let inner = |x: f64| self.implicit(&c, at(x), bone).1;
                (xp, first_entry(&inner, xp, x1, h), bone as u8)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_geometry_and_truth() {
        let ph = JointPhantom {
            dims: (100, 120, 120),
            radius: 3.0,
            joint_space: 2.0,
            joint_level: 0.4,
            noise: 0.0,
            ..JointPhantom::default()
        };
        let (img, peri, endo) = ph.generate();
        let n = |m: &Array3<u8>, v: u8| m.iter().filter(|&&b| b == v).count();
        assert!(n(&peri, 1) > n(&endo, 1) && n(&endo, 1) > 0);
        assert!(n(&peri, 2) > n(&endo, 2) && n(&endo, 2) > 0);
        // the bones never touch: a vertical line through the contact point
        // meets tibia, joint space, femur
        let c = ph.coeffs();
        let a = ph.axis();
        let p = [0.0, a[0] + ph.contact_offset[0], a[1] + ph.contact_offset[1]];
        let (zt, zf) = ph.surfaces(&c, p);
        assert!((zf - zt - ph.joint_space).abs() < 1e-9);
        // a line going down from inside the joint space finds the tibial plateau
        let top = [0.5 * (zt + zf), p[1], p[2]];
        let e = ph.edges_along(&[top], &[[-1.0, 0.0, 0.0]], 0.0, 2.0)[0];
        assert_eq!(e.2, Bone::Tibia as u8);
        assert!((e.0 - 0.5 * ph.joint_space).abs() < 0.01, "{e:?}"); // rim rounding
        let tp = ph.plate_at(&c, p, Bone::Tibia);
        assert!((e.1 - e.0 - tp).abs() < 0.01, "{e:?} plate {tp}");
        // and up, the femoral condyle
        let e = ph.edges_along(&[top], &[[1.0, 0.0, 0.0]], 0.0, 2.0)[0];
        assert_eq!(e.2, Bone::Femur as u8);
        assert!((e.0 - 0.5 * ph.joint_space).abs() < 0.01, "{e:?}"); // rim rounding
                                                                     // image: the plate is bright, the joint space dark
        let g = ph.geometry();
        let vox = |q: V3| {
            let i = g.to_index(q).map(|v| v.round() as usize);
            img[[i[0], i[1], i[2]]]
        };
        assert!(vox([zt - 0.5 * tp, p[1], p[2]]) > 500.0);
        assert!(vox([0.5 * (zt + zf), p[1], p[2]]) < 400.0);
    }
}
