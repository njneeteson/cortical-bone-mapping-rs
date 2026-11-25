//! A proximal femur phantom at clinical CT scale, with the acetabulum across
//! the hip joint space.
//!
//! The periosteal surface is the zero level of a smooth union of four
//! approximately distance-like implicit functions (negative inside), all in
//! physical units with array axis 0 the scanner axis, pointing superior:
//!
//! * the shaft, a vertical cylinder `r − R_s` closed above the lesser
//!   trochanter level and open below (it leaves the image there);
//! * the neck, an elliptical cylinder along the neck axis `d`, inclined by
//!   the neck–shaft angle and turned anteriorly by the anteversion, from the
//!   shaft axis to the head centre: `(ρ − 1)/|∇ρ|` with
//!   `ρ² = (u/a)² + (v/b)²` in the neck's superior (`u`) and anterior (`v`)
//!   directions, the first-order distance to the ellipse (exact on its axes,
//!   so the cortex there is `t` thick);
//! * the head, a sphere `|p − c| − R_h` whose centre lies at the femoral
//!   offset from the shaft axis;
//! * the greater trochanter, an ellipsoid (the same first-order distance)
//!   lateral to the top of the shaft.
//!
//! The union is the smooth minimum `φ = −k ln Σ_i exp(−φ_i/k)` (a
//! log-sum-exp), which fillets the junctions over a width of about `k`. The endosteal surface is the zero
//! level of `φ/|∇φ| + t(p)`, the first-order distance offset by a thickness
//! field `t` blended from the parts with the partition of unity
//! `w_i ∝ exp(−φ_i/k)` (the part a point is nearest to dominates). In the neck, `t` varies around the neck axis from
//! `t_sup` superiorly to `t_inf` inferiorly, as `t_sup + (t_inf − t_sup)(1 −
//! cos θ)/2`. Every part is multiplied by `1 + v f(p)` for a smooth unit
//! random field `f`, and `t` is clipped below at `t_min`. Since `φ` is close
//! to a distance, the cortex is about `t` thick along the normal; the
//! reference thickness is in any case measured along the same line as CBM
//! measures it ([`FemurPhantom::edges_along`]).
//!
//! The acetabulum is a spherical shell around the head, separated from it by
//! the joint space (cartilage), covering the head within a half-angle of its
//! superomedial axis, with plates of bone on both faces and trabeculae
//! between, and a rounded rim. It is a second bone, labelled 2: real hips
//! always have the joint space between them.
//!
//! The marrow space holds trabeculae as in [`BonePhantom`](crate::phantoms::bone::BonePhantom),
//! with BV/TV rising towards the endosteum, and the cortex has pores (both
//! well below clinical resolution; they set the mean densities and the
//! texture).
//!
//! Parameters ([`FemurPhantom::default`]) follow measurements of adult
//! femora: femoral head diameter 46.3 mm, neck–shaft angle 126.4°,
//! horizontal offset 37.9 mm and anteversion 17.5° (Soodmand et al. 2019,
//! 169 femora from CT); femoral neck cortical thickness 0.77–0.95 mm in the
//! superior quadrants and 2.39–3.20 mm in the inferior ones, in women aged
//! about 79 (Johannesdottir et al. 2011, controls, clinical QCT); cortical
//! thicknesses of 0.3–4 mm over the proximal femur (Treece et al. 2010); a
//! radiographic hip joint space of about 4 mm. The neck and shaft radii, the
//! trochanter and the acetabulum are typical, not measured, values.
//!
//! # References
//!
//! * E. Soodmand, G. Zheng, W. Steens, R. Bader, L.-P. Nolte, D. Kluess,
//!   *Surgically relevant morphological parameters of proximal human femur:
//!   a statistical analysis based on 3D reconstruction of CT data*,
//!   Orthopaedic Surgery 11(1):135–142, 2019, doi:10.1111/os.12416.
//! * F. Johannesdottir, K. E. S. Poole, J. Reeve, K. Siggeirsdottir, T.
//!   Aspelund, B. Mogensen, B. Y. Jonsson, S. Sigurdsson, T. B. Harris, V. G.
//!   Gudnason, G. Sigurdsson, *Distribution of cortical bone in the femoral
//!   neck and hip fracture: a prospective case-control analysis of 143
//!   incident hip fractures; the AGES-REYKJAVIK Study*, Bone
//!   48(6):1268–1276, 2011, doi:10.1016/j.bone.2011.03.776.
//! * G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
//!   cortical bone thickness measurement from clinical CT data*, Medical
//!   Image Analysis 14(3):276–290, 2010.

use ndarray::Array3;
use rayon::prelude::*;

use super::render::smax;
use super::render::{erfinv, field, first_entry, render, waves};
use crate::util::{add, cross, dot, norm, scale, sub, Rng, V3};
use crate::volume::Geometry;

#[derive(Clone, Debug)]
pub struct FemurPhantom {
    pub dims: (usize, usize, usize),
    pub spacing: V3,
    pub head_radius: f64,
    /// Neck–shaft angle (degrees).
    pub neck_shaft_angle: f64,
    /// Anteversion of the neck (degrees).
    pub anteversion: f64,
    /// Horizontal distance from the shaft axis to the head centre.
    pub offset: f64,
    /// Semi-axes of the neck cross-section (superoinferior, anteroposterior).
    pub neck_radii: [f64; 2],
    pub shaft_radius: f64,
    /// Semi-axes of the greater trochanter (superoinferior, anteroposterior,
    /// mediolateral).
    pub trochanter: V3,
    /// Fillet width of the smooth union.
    pub rounding: f64,
    /// Cortical thickness: neck superior and inferior, head, trochanter, shaft.
    pub neck_thickness: [f64; 2],
    pub head_thickness: f64,
    pub trochanter_thickness: f64,
    pub shaft_thickness: f64,
    /// Relative random variation of the thickness and its wavelength.
    pub thickness_variation: f64,
    pub thickness_wavelength: f64,
    pub thickness_min: f64,
    pub acetabulum: Option<Acetabulum>,
    pub bvtv: f64,
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

/// The acetabulum: a spherical shell around the femoral head across the
/// joint space `gap`, `thickness` thick with bone plates of `plate` on both
/// faces, covering the head within `coverage` degrees of its axis (tilted
/// `tilt` degrees medially from vertical).
#[derive(Clone, Debug, PartialEq)]
pub struct Acetabulum {
    pub gap: f64,
    pub thickness: f64,
    pub plate: f64,
    pub coverage: f64,
    pub tilt: f64,
}

impl Default for Acetabulum {
    fn default() -> Self {
        Acetabulum {
            gap: 4.0,
            thickness: 8.0,
            plate: 1.0,
            coverage: 75.0,
            tilt: 45.0,
        }
    }
}

impl Default for FemurPhantom {
    /// A proximal femur in a hip CT: 0.75 mm pixels and 1 mm slices, a PSF of
    /// σ = 0.5 mm in-plane and 0.6 mm along the axis, noise 30 HU (module
    /// documentation and `phantoms::sites`).
    fn default() -> Self {
        FemurPhantom {
            dims: (110, 104, 140),
            spacing: [1.0, 0.75, 0.75],
            head_radius: 23.15,
            neck_shaft_angle: 126.4,
            anteversion: 17.5,
            offset: 37.9,
            neck_radii: [15.0, 12.5],
            shaft_radius: 15.0,
            trochanter: [16.0, 14.0, 11.0],
            rounding: 4.0,
            neck_thickness: [0.86, 2.79],
            head_thickness: 0.8,
            trochanter_thickness: 0.8,
            shaft_thickness: 3.5,
            thickness_variation: 0.15,
            thickness_wavelength: 25.0,
            thickness_min: 0.3,
            acetabulum: Some(Acetabulum::default()),
            bvtv: 0.18,
            endo_bvtv: 0.35,
            transition_width: 2.0,
            trabecular_wavelength: 1.6,
            porosity: 0.03,
            pore_wavelength: 0.35,
            rho_soft: 57.0,
            rho_cartilage: 80.0,
            rho_bone: 1600.0,
            rho_marrow: -30.0,
            blur: [0.6, 0.5, 0.5],
            noise: 30.0,
            seed: 1,
            supersample: 3,
        }
    }
}

/// Which bone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bone {
    Femur = 1,
    Pelvis = 2,
}

/// The phantom's frame and random fields, derived once from its parameters.
struct Frame {
    /// Shaft axis (in-plane) and the neck–shaft junction height.
    shaft: [f64; 2],
    junction: V3,
    /// Neck axis, its superior and anterior directions, and its length.
    neck: [V3; 3],
    neck_length: f64,
    head: V3,
    trochanter: V3,
    shaft_top: f64,
    cup_axis: V3,
    thickness: Vec<(V3, f64)>,
    trab: Vec<(V3, f64)>,
    pores: Vec<(V3, f64)>,
}

/// `(φ, t)` of each part of the femur at a point.
struct Parts {
    phi: [f64; 4],
    t: [f64; 4],
}

impl FemurPhantom {
    /// The phantom's voxel grid (origin at zero).
    pub fn geometry(&self) -> Geometry {
        Geometry::new(self.spacing)
    }

    /// Place the parts: the shaft in the lateral part of the volume, the head
    /// medially, the head centre `offset` from the shaft axis, and the
    /// acetabulum's dome `cup radius + 2 mm` below the top of the volume.
    fn frame(&self) -> Frame {
        let (d0, d1, d2) = self.dims;
        let ext = [d0, d1, d2].map(|d| d as f64 - 1.0);
        let size = [0, 1, 2].map(|a| ext[a] * self.spacing[a]);
        let rad = |deg: f64| deg.to_radians();
        let incl = rad(self.neck_shaft_angle - 90.0);
        let av = rad(self.anteversion);
        // neck axis: up by the inclination, medial (+axis 2), anterior (+axis 1)
        let d = [incl.sin(), incl.cos() * av.sin(), incl.cos() * av.cos()];
        let up = [1.0, 0.0, 0.0];
        let sup = normalise(sub(up, scale(d, dot(up, d))));
        let ant = cross(d, sup);
        let ant = if ant[1] < 0.0 { scale(ant, -1.0) } else { ant };
        let neck_length = self.offset / (d[1] * d[1] + d[2] * d[2]).sqrt();
        let cup_out = self.head_radius + self.acetabulum.as_ref().map_or(2.0, |a| a.gap + a.thickness);
        let head_z = size[0] - cup_out - 2.0;
        let head_x = size[2] - cup_out - 2.0;
        let junction = sub([head_z, 0.5 * size[1], head_x], scale(d, neck_length));
        let shaft = [junction[1], junction[2]];
        let tilt = rad(self.acetabulum.as_ref().map_or(45.0, |a| a.tilt));
        let mut r = Rng::new(self.seed);
        Frame {
            shaft,
            junction,
            neck: [d, sup, ant],
            neck_length,
            head: add(junction, scale(d, neck_length)),
            trochanter: [
                junction[0] + 0.6 * self.trochanter[0],
                junction[1] - 0.15 * self.trochanter[1],
                junction[2] - (self.shaft_radius - 0.6 * self.trochanter[2]),
            ],
            shaft_top: junction[0] + 0.5 * self.shaft_radius,
            cup_axis: [tilt.cos(), 0.0, tilt.sin()],
            thickness: waves(&mut r, 16, self.thickness_wavelength, 1.0),
            trab: waves(&mut r, 48, self.trabecular_wavelength, 1.5),
            pores: waves(&mut r, 32, self.pore_wavelength, 2.5),
        }
    }

    /// The implicit function and the local thickness of each part.
    fn parts(&self, f: &Frame, p: V3) -> Parts {
        let k = self.rounding;
        // shaft
        let r = ((p[1] - f.shaft[0]).powi(2) + (p[2] - f.shaft[1]).powi(2)).sqrt();
        let shaft = smax(r - self.shaft_radius, p[0] - f.shaft_top, k);
        // neck
        let q = sub(p, f.junction);
        let (s, u, v) = (dot(q, f.neck[0]), dot(q, f.neck[1]), dot(q, f.neck[2]));
        let [a, b] = self.neck_radii;
        let neck = ellipsoid(&[(u, a), (v, b)]).max(-s).max(s - f.neck_length);
        let cos_th = u / (u * u + v * v).sqrt().max(1e-12);
        let [ts, ti] = self.neck_thickness;
        let t_neck = ts + (ti - ts) * 0.5 * (1.0 - cos_th);
        // head
        let head = norm(sub(p, f.head)) - self.head_radius;
        // greater trochanter
        let e = sub(p, f.trochanter);
        let tr = self.trochanter;
        let troch = ellipsoid(&[(e[0], tr[0]), (e[1], tr[1]), (e[2], tr[2])]);
        Parts {
            phi: [shaft, neck, head, troch],
            t: [
                self.shaft_thickness,
                t_neck,
                self.head_thickness,
                self.trochanter_thickness,
            ],
        }
    }

    /// `(φ_periosteal, φ_marrow)` of the femur: negative inside.
    ///
    /// The union is not a distance: where parts mix, `|∇φ| < 1`, and an
    /// offset of `t` in `φ` would be deeper than `t`. So the endosteum is
    /// offset from the first-order distance `φ/|∇φ|` (central differences).
    fn femur(&self, f: &Frame, p: V3) -> (f64, f64) {
        let (outer, t) = self.union(f, p);
        let h = 0.01 * self.rounding;
        let grad = [0, 1, 2].map(|a| {
            let (mut lo, mut hi) = (p, p);
            lo[a] -= h;
            hi[a] += h;
            (self.union(f, hi).0 - self.union(f, lo).0) / (2.0 * h)
        });
        (outer, outer / norm(grad).max(1e-6) + t)
    }

    /// The smooth union `φ = m − k ln Σ_i exp(−(φ_i − m)/k)` (`m = min φ_i`)
    /// and the thickness, blended with the same weights
    /// `w_i ∝ exp(−φ_i/k)`, the part a point is nearest to dominating.
    fn union(&self, f: &Frame, p: V3) -> (f64, f64) {
        let parts = self.parts(f, p);
        let k = self.rounding;
        let m = parts.phi.iter().cloned().fold(f64::INFINITY, f64::min);
        let w = parts.phi.map(|x| (-(x - m) / k).exp());
        let sw: f64 = w.iter().sum();
        let t_mix = w.iter().zip(&parts.t).map(|(w, t)| w * t).sum::<f64>() / sw;
        let t = (t_mix * (1.0 + self.thickness_variation * field(&f.thickness, p))).max(self.thickness_min);
        (m - k * sw.ln(), t)
    }

    /// `(φ_outer, φ_marrow)` of the acetabulum, if there is one.
    fn pelvis(&self, f: &Frame, p: V3) -> Option<(f64, f64)> {
        let a = self.acetabulum.as_ref()?;
        let q = sub(p, f.head);
        let d = norm(q);
        let inner = self.head_radius + a.gap;
        let shell = (d - (inner + 0.5 * a.thickness)).abs() - 0.5 * a.thickness;
        let angle = (dot(q, f.cup_axis) / d.max(1e-12)).clamp(-1.0, 1.0).acos();
        let cone = (angle - a.coverage.to_radians()) * d;
        let k = 0.5 * a.plate;
        Some((smax(shell, cone, k), smax(shell + a.plate, cone + a.plate, k)))
    }

    /// The bone (if any) whose periosteal surface contains `p`, with its
    /// `(φ_outer, φ_marrow)`.
    fn bone_at(&self, f: &Frame, p: V3) -> Option<(Bone, f64, f64)> {
        let (o, m) = self.femur(f, p);
        if o < 0.0 {
            return Some((Bone::Femur, o, m));
        }
        match self.pelvis(f, p) {
            Some((o, m)) if o < 0.0 => Some((Bone::Pelvis, o, m)),
            _ => None,
        }
    }

    /// Cartilage fills the joint space within the acetabulum's coverage.
    fn in_joint(&self, f: &Frame, p: V3) -> bool {
        let Some(a) = &self.acetabulum else { return false };
        let q = sub(p, f.head);
        let d = norm(q);
        let angle = (dot(q, f.cup_axis) / d.max(1e-12)).clamp(-1.0, 1.0).acos();
        d < self.head_radius + a.gap && angle < a.coverage.to_radians()
    }

    /// The true density at `p`: cortex with pores between the surfaces,
    /// marrow with trabeculae (BV/TV rising towards the endosteum) inside,
    /// cartilage in the joint space, soft tissue elsewhere.
    fn density(&self, f: &Frame, p: V3) -> f64 {
        let Some((_, _, marrow)) = self.bone_at(f, p) else {
            return if self.in_joint(f, p) {
                self.rho_cartilage
            } else {
                self.rho_soft
            };
        };
        if marrow >= 0.0 {
            let por = self.porosity.min(0.49);
            let tau = if por > 0.0 {
                2f64.sqrt() * erfinv(1.0 - 2.0 * por)
            } else {
                f64::INFINITY
            };
            return if field(&f.pores, p) > tau {
                self.rho_marrow
            } else {
                self.rho_bone
            };
        }
        let depth = -marrow;
        let bv = self.bvtv + (self.endo_bvtv - self.bvtv) * (-depth / self.transition_width.max(1e-12)).exp();
        if bv > 0.0 && field(&f.trab, p).abs() < 2f64.sqrt() * erfinv(bv.min(0.99)) {
            self.rho_bone
        } else {
            self.rho_marrow
        }
    }

    /// Generate `(image, periosteal labels, endosteal labels)`: 0 outside,
    /// [`Bone::Femur`] (1) or [`Bone::Pelvis`] (2) inside the bone's
    /// periosteal surface / its marrow space, at voxel centres.
    pub fn generate(&self) -> (Array3<f32>, Array3<u8>, Array3<u8>) {
        let f = self.frame();
        let g = self.geometry();
        let image = render(self.dims, &g, self.blur, self.noise, self.seed, self.supersample, |p| {
            self.density(&f, p)
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
                if let Some((bone, _, m)) = self.bone_at(&f, g.to_physical([i as f64, j as f64, k as f64])) {
                    *pv = bone as u8;
                    *ev = if m < 0.0 { bone as u8 } else { 0 };
                }
            });
        (image, peri, endo)
    }

    /// True `(x_p, x_e, bone)` along each line `p + x n`, `x ∈ [x0, x1]`: the
    /// first entry into either bone, and the first entry into that bone's
    /// marrow space after it (NaN / 0 where the line misses).
    pub fn edges_along(&self, points: &[V3], normals: &[V3], x0: f64, x1: f64) -> Vec<(f64, f64, u8)> {
        let f = self.frame();
        let h = 0.25 * self.spacing.iter().cloned().fold(f64::INFINITY, f64::min);
        points
            .par_iter()
            .zip(normals.par_iter())
            .map(|(&p, &n)| {
                let at = |x: f64| add(p, scale(n, x));
                let outer = |x: f64| {
                    let q = at(x);
                    let pel = self.pelvis(&f, q).map_or(f64::INFINITY, |v| v.0);
                    self.femur(&f, q).0.min(pel)
                };
                let xp = first_entry(&outer, x0, x1, h);
                if xp.is_nan() {
                    return (f64::NAN, f64::NAN, 0);
                }
                let q = at(xp + 1e-6);
                let pel = self.pelvis(&f, q).map_or(f64::INFINITY, |v| v.0);
                let bone = if self.femur(&f, q).0 <= pel {
                    Bone::Femur
                } else {
                    Bone::Pelvis
                };
                let inner = |x: f64| match bone {
                    Bone::Femur => self.femur(&f, at(x)).1,
                    Bone::Pelvis => self.pelvis(&f, at(x)).map_or(f64::INFINITY, |v| v.1),
                };
                (xp, first_entry(&inner, xp, x1, h), bone as u8)
            })
            .collect()
    }
}

/// The first-order distance `(ρ − 1)/|∇ρ|` to the ellipsoid `ρ = 1`,
/// `ρ² = Σ (x_i/a_i)²`, for coordinates and semi-axes `(x_i, a_i)`.
fn ellipsoid(c: &[(f64, f64)]) -> f64 {
    let rho = c.iter().map(|(x, a)| (x / a).powi(2)).sum::<f64>().sqrt();
    let grad = c.iter().map(|(x, a)| (x / (a * a)).powi(2)).sum::<f64>().sqrt() / rho.max(1e-300);
    (rho - 1.0) / grad.max(1e-300)
}

fn normalise(v: V3) -> V3 {
    scale(v, 1.0 / norm(v).max(1e-300))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> FemurPhantom {
        FemurPhantom {
            spacing: [2.0, 1.5, 1.5],
            dims: (56, 52, 70),
            supersample: 1,
            noise: 0.0,
            ..FemurPhantom::default()
        }
    }

    #[test]
    fn parts_are_placed_and_inside_the_volume() {
        let ph = FemurPhantom::default();
        let f = ph.frame();
        let (d0, d1, d2) = ph.dims;
        let size = [d0, d1, d2]
            .iter()
            .zip(ph.spacing)
            .map(|(&d, s)| (d as f64 - 1.0) * s)
            .collect::<Vec<_>>();
        // the acetabulum (superior, medial, anterior and posterior extents),
        // the trochanter and the shaft are inside the volume
        let cup = ph.head_radius + 12.0;
        assert!(f.head[0] + cup < size[0] && f.head[2] + cup < size[2]);
        assert!(f.head[1] - cup > 0.0 && f.head[1] + cup < size[1]);
        assert!(f.trochanter[2] - ph.trochanter[2] > 0.0);
        assert!(f.shaft[0] - ph.shaft_radius > 0.0 && f.shaft[1] - ph.shaft_radius > 0.0);
        // the offset and the neck-shaft angle
        let off = ((f.head[1] - f.shaft[0]).powi(2) + (f.head[2] - f.shaft[1]).powi(2)).sqrt();
        assert!((off - ph.offset).abs() < 1e-9);
        assert!((f.neck[0][0].asin().to_degrees() + 90.0 - ph.neck_shaft_angle).abs() < 1e-9);
        // the joint space separates the bones along the cup axis
        let top = add(f.head, scale(f.cup_axis, ph.head_radius + 2.0));
        assert!(ph.bone_at(&f, top).is_none() && ph.in_joint(&f, top));
    }

    #[test]
    fn truth_matches_the_thickness_field() {
        let ph = FemurPhantom::default();
        let f = ph.frame();
        // a line into the head from the joint space, along the cup axis
        let start = add(f.head, scale(f.cup_axis, ph.head_radius + 2.0));
        let n = scale(f.cup_axis, -1.0);
        let e = ph.edges_along(&[start], &[n], 0.0, 6.0)[0];
        assert_eq!(e.2, Bone::Femur as u8);
        // the smooth union bulges the head out by a few hundredths
        assert!((e.0 - 2.0).abs() < 0.03, "{e:?}");
        let t = ph.femur(&f, add(start, scale(n, e.0))).1 - ph.femur(&f, add(start, scale(n, e.0))).0;
        assert!((e.1 - e.0 - t).abs() < 0.1, "{e:?} thickness {t}");
        // and outwards, the acetabulum's plate
        let e = ph.edges_along(&[start], &[f.cup_axis], 0.0, 6.0)[0];
        assert_eq!(e.2, Bone::Pelvis as u8);
        assert!((e.0 - 2.0).abs() < 0.01 && (e.1 - e.0 - 1.0).abs() < 0.05, "{e:?}");
    }

    #[test]
    fn neck_is_thin_superiorly_and_thick_inferiorly() {
        let ph = FemurPhantom {
            thickness_variation: 0.0,
            ..FemurPhantom::default()
        };
        let f = ph.frame();
        // mid-neck, away from the head and the shaft
        let mid = add(f.junction, scale(f.neck[0], 0.45 * f.neck_length));
        let along = |dir: V3| {
            let start = add(mid, scale(dir, ph.neck_radii[0] + 5.0));
            let e = ph.edges_along(&[start], &[scale(dir, -1.0)], 0.0, 12.0)[0];
            e.1 - e.0
        };
        let (sup, inf) = (along(f.neck[1]), along(scale(f.neck[1], -1.0)));
        assert!((sup - ph.neck_thickness[0]).abs() < 0.15, "superior {sup}");
        assert!((inf - ph.neck_thickness[1]).abs() < 0.3, "inferior {inf}");
    }

    #[test]
    fn image_and_labels() {
        let ph = small();
        let (img, peri, endo) = ph.generate();
        let n = |m: &Array3<u8>, v: u8| m.iter().filter(|&&b| b == v).count();
        assert!(n(&peri, 1) > n(&endo, 1) && n(&endo, 1) > 0);
        assert!(n(&peri, 2) > 0);
        // bright cortex somewhere, soft tissue in the corner
        assert!(img.iter().cloned().fold(f32::MIN, f32::max) > 1000.0);
        assert!((img[[0, 0, 0]] - ph.rho_soft as f32).abs() < 1.0);
    }
}
