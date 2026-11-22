//! Synthetic phantoms with known cortical geometry, for tests, validation
//! and examples.
//!
//! A [`ShellPhantom`] is a cylinder (along array axis 0) or a sphere whose
//! outer (periosteal) radius is `R(θ)` and whose cortical thickness is
//! `t(θ) = t̄ + a cos(kθ)`. Densities are piecewise constant (soft tissue,
//! cortex, trabecular/marrow), integrated over each voxel by supersampling,
//! blurred with an anisotropic Gaussian PSF and corrupted with Gaussian noise.
//! The companion segmentation can be perturbed (offset, bumps) to emulate a
//! poor outer-surface segmentation.

use ndarray::Array3;
use rayon::prelude::*;

use crate::util::{Rng, V3};
use crate::volume::{gaussian_filter, Geometry};

/// Shape of the phantom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// Infinite cylinder along array axis 0 (θ: angle in the axis-1/axis-2 plane).
    Cylinder,
    /// Sphere (θ: polar angle from axis 0).
    Sphere,
}

#[derive(Clone, Debug)]
pub struct ShellPhantom {
    pub dims: (usize, usize, usize),
    pub spacing: V3,
    pub shape: Shape,
    /// Outer radius `R` (physical units).
    pub radius: f64,
    /// Radial bump on the true outer surface: `R(θ) = R + bump·cos(bump_lobes θ)`.
    pub radius_bump: f64,
    pub radius_lobes: f64,
    pub thickness_mean: f64,
    pub thickness_amplitude: f64,
    pub thickness_lobes: f64,
    pub rho_s: f64,
    pub rho_c: f64,
    pub rho_b: f64,
    /// Gaussian PSF standard deviation along each axis (physical units).
    pub blur: V3,
    pub noise: f64,
    pub seed: u64,
    /// Sub-samples per axis per voxel for partial-volume integration.
    pub supersample: usize,
    /// Segmentation radius offset and bumps (0 = perfect segmentation).
    pub mask_offset: f64,
    pub mask_bump: f64,
    pub mask_lobes: f64,
}

impl Default for ShellPhantom {
    fn default() -> Self {
        ShellPhantom {
            dims: (24, 96, 96),
            spacing: [0.1; 3],
            shape: Shape::Cylinder,
            radius: 3.5,
            radius_bump: 0.0,
            radius_lobes: 3.0,
            thickness_mean: 0.8,
            thickness_amplitude: 0.3,
            thickness_lobes: 2.0,
            rho_s: 50.0,
            rho_c: 1000.0,
            rho_b: 250.0,
            blur: [0.12; 3],
            noise: 20.0,
            seed: 0,
            supersample: 3,
            mask_offset: 0.0,
            mask_bump: 0.0,
            mask_lobes: 5.0,
        }
    }
}

impl ShellPhantom {
    pub fn geometry(&self) -> Geometry {
        Geometry::new(self.spacing)
    }

    /// Physical centre of the shape.
    pub fn center(&self) -> V3 {
        let (d0, d1, d2) = self.dims;
        [
            (d0 as f64 - 1.0) * 0.5 * self.spacing[0],
            (d1 as f64 - 1.0) * 0.5 * self.spacing[1],
            (d2 as f64 - 1.0) * 0.5 * self.spacing[2],
        ]
    }

    /// (radial distance, angle θ) of a physical point.
    pub fn polar(&self, p: V3) -> (f64, f64) {
        let c = self.center();
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        match self.shape {
            Shape::Cylinder => ((d[1] * d[1] + d[2] * d[2]).sqrt(), d[2].atan2(d[1])),
            Shape::Sphere => {
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                (
                    r,
                    if r > 0.0 {
                        (d[0] / r).clamp(-1.0, 1.0).acos()
                    } else {
                        0.0
                    },
                )
            }
        }
    }

    pub fn outer_radius(&self, theta: f64) -> f64 {
        self.radius + self.radius_bump * (self.radius_lobes * theta).cos()
    }

    pub fn thickness(&self, theta: f64) -> f64 {
        self.thickness_mean + self.thickness_amplitude * (self.thickness_lobes * theta).cos()
    }

    pub fn mask_radius(&self, theta: f64) -> f64 {
        self.outer_radius(theta) + self.mask_offset + self.mask_bump * (self.mask_lobes * theta).cos()
    }

    /// True thickness at (the angle of) a physical point.
    pub fn true_thickness(&self, p: V3) -> f64 {
        self.thickness(self.polar(p).1)
    }

    /// Signed distance of `p` outside the true periosteal surface,
    /// measured radially.
    pub fn radial_offset(&self, p: V3) -> f64 {
        let (r, th) = self.polar(p);
        r - self.outer_radius(th)
    }

    fn density(&self, p: V3) -> f64 {
        let (r, th) = self.polar(p);
        let ro = self.outer_radius(th);
        if r > ro {
            self.rho_s
        } else if r > ro - self.thickness(th) {
            self.rho_c
        } else {
            self.rho_b
        }
    }

    /// Generate `(image, mask)`.
    pub fn generate(&self) -> (Array3<f32>, Array3<u8>) {
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
                let i = lin / (d1 * d2);
                let j = (lin / d2) % d1;
                let k = lin % d2;
                let mut acc = 0.0;
                for a in 0..ss {
                    for b in 0..ss {
                        for c in 0..ss {
                            let off = |q: usize| (q as f64 + 0.5) / ss as f64 - 0.5;
                            let p = g.to_physical([i as f64 + off(a), j as f64 + off(b), k as f64 + off(c)]);
                            acc += self.density(p);
                        }
                    }
                }
                *v = acc / (ss * ss * ss) as f64;
            });
        let sig = [
            self.blur[0] / self.spacing[0],
            self.blur[1] / self.spacing[1],
            self.blur[2] / self.spacing[2],
        ];
        let blurred = gaussian_filter(&truth, sig);
        drop(truth);
        let mut rng = Rng::new(self.seed);
        let noise: Vec<f64> = (0..d0 * d1 * d2).map(|_| rng.normal()).collect();
        let image = Array3::from_shape_vec(
            self.dims,
            blurred
                .iter()
                .zip(noise)
                .map(|(&v, n)| (v + self.noise * n) as f32)
                .collect(),
        )
        .unwrap();
        let mask = Array3::from_shape_fn(self.dims, |(i, j, k)| {
            let p = g.to_physical([i as f64, j as f64, k as f64]);
            let (r, th) = self.polar(p);
            (r <= self.mask_radius(th)) as u8
        });
        (image, mask)
    }
}
