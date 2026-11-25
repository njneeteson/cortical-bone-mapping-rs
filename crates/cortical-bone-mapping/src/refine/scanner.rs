//! The scanner's parameters during refinement: the blur (the volumetric
//! model's PSF, or a global `σ`) and the cortical density `ρc` (ours).
//!
//! They belong to the scan, not to a node, but each fit estimates them from
//! the surface it is given, and a surface that is still wrong biases them
//! (the blur and the edge position trade off). Re-estimating them in every
//! fit makes the edges the surface moves toward change with the surface.
//! Refinement therefore alternates between the two, as the segregated
//! solvers of fluid dynamics alternate between their sets of unknowns
//! (Patankar & Spalding 1972):
//!
//! 1. refine the surface with the scanner parameters held (one *pass*);
//! 2. re-fit them on the refined surface and move them part of the way,
//!    `p ← p + β (p_fit − p)` (under-relaxation, `β` =
//!    [`RefineOptions::scanner_relaxation`](super::RefineOptions::scanner_relaxation));
//! 3. repeat until a re-fit changes no parameter by more than
//!    [`RefineOptions::scanner_tolerance`](super::RefineOptions::scanner_tolerance)
//!    (relative), or for at most
//!    [`RefineOptions::scanner_passes`](super::RefineOptions::scanner_passes)
//!    passes.
//!
//! Parameters the options fix (a given PSF or `ρc`) re-fit to themselves, so
//! refinement then stops after one pass. The densities of soft tissue and
//! trabecular bone, when global, are held only with regional convergence,
//! where partial fits must use the whole surface's values.
//!
//! # References
//!
//! * S. V. Patankar, D. B. Spalding, *A calculation procedure for heat, mass
//!   and momentum transfer in three-dimensional parabolic flows*,
//!   International Journal of Heat and Mass Transfer 15(10):1787–1806, 1972.

use ndarray::ArrayView3;

use crate::fit::FitConstraints;
use crate::model::{RHO_B, RHO_S, SIGMA};
use crate::options::{Method, Options, RhoCMode};
use crate::result::CbmResult;

/// The scanner parameters held during one refinement pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScannerState {
    /// The volumetric model's PSF `(σ_xy, σ_z)`.
    pub psf: Option<[f64; 2]>,
    /// A global `σ` (regularised method).
    pub sigma: Option<f64>,
    /// The cortical density (`None`: one per node, `RhoCMode::ProfileMax`).
    pub rho_c: Option<f64>,
}

impl ScannerState {
    /// The parameters a fit estimated, those this state holds.
    fn of(&self, r: &CbmResult) -> ScannerState {
        ScannerState {
            psf: self.psf.and(r.volumetric.as_ref().map(|v| [v.sigma_xy, v.sigma_z])),
            sigma: self.sigma.map(|_| r.mean_sigma()),
            rho_c: self.rho_c.map(|_| r.rho_c),
        }
    }

    /// The values of `other` for the parameters this state holds.
    #[cfg(test)]
    fn of_values(&self, other: &ScannerState) -> ScannerState {
        ScannerState {
            psf: self.psf.and(other.psf),
            sigma: self.sigma.and(other.sigma),
            rho_c: self.rho_c.and(other.rho_c),
        }
    }

    /// Every held value, in a fixed order.
    fn values(&self) -> Vec<f64> {
        let mut v = Vec::new();
        v.extend(self.psf.iter().flatten());
        v.extend(self.sigma);
        v.extend(self.rho_c);
        v
    }

    /// `self + β (fit − self)`.
    fn relaxed(&self, fit: &ScannerState, beta: f64) -> ScannerState {
        let mix = |a: f64, b: f64| a + beta * (b - a);
        ScannerState {
            psf: self.psf.zip(fit.psf).map(|(a, b)| [mix(a[0], b[0]), mix(a[1], b[1])]),
            sigma: self.sigma.zip(fit.sigma).map(|(a, b)| mix(a, b)),
            rho_c: self.rho_c.zip(fit.rho_c).map(|(a, b)| mix(a, b)),
        }
    }

    /// The largest relative difference from `other`.
    fn change(&self, other: &ScannerState) -> f64 {
        self.values()
            .iter()
            .zip(other.values())
            .map(|(a, b)| (b - a).abs() / a.abs().max(1e-12))
            .fold(0.0, f64::max)
    }
}

/// The densities and blur shared by the fits of one refinement.
pub(super) struct Shared {
    /// The scanner parameters held in the current pass.
    pub scanner: ScannerState,
    /// Global densities, held with regional convergence.
    densities: Option<[f64; 2]>,
    density_start: [f64; 2],
}

impl Shared {
    /// From the first, full fit: hold its scanner parameters and, with
    /// regional convergence, its global densities (regularised method) or
    /// density values (volumetric model with global densities).
    pub fn new(first: &CbmResult, options: &Options, freezing: bool) -> Self {
        let regularized = matches!(options.method, Method::Regularized { .. });
        let global_densities = options.model.global[RHO_S] || options.model.global[RHO_B];
        let volumetric_global = matches!(&options.method, Method::Volumetric(v) if v.global_densities);
        let held_densities = freezing && (regularized && global_densities || volumetric_global);
        Shared {
            scanner: ScannerState {
                psf: first.volumetric.as_ref().map(|v| [v.sigma_xy, v.sigma_z]),
                sigma: (options.model.global[SIGMA] && regularized).then(|| first.mean_sigma()),
                rho_c: (options.model.rho_c_mode != RhoCMode::ProfileMax).then_some(first.rho_c),
            },
            densities: held_densities.then(|| first.mean_densities()),
            density_start: first.mean_densities(),
        }
    }

    /// Carry the densities over from a fit as a warm start.
    pub fn update(&mut self, r: &CbmResult) {
        if self.densities.is_none() {
            self.density_start = r.mean_densities();
        }
    }

    /// Constraints of a fit with the scanner parameters held: `full`
    /// (verifying, whole surface) or partial, with nodes `held` fixed and
    /// inner solves stopped at `edge_tol`.
    pub fn constraints<'a>(
        &self,
        exclude: Option<ArrayView3<'a, bool>>,
        full: bool,
        held: Option<&'a [bool]>,
        edge_tol: f64,
    ) -> FitConstraints<'a> {
        let s = &self.scanner;
        FitConstraints {
            exclude,
            fixed: held,
            rho_c: s.rho_c,
            densities: self.densities,
            density_start: Some(self.density_start),
            edge_tol: if full { 0.0 } else { edge_tol },
            sigma: s.sigma,
            sigma_start: s.sigma,
            psf: s.psf,
            psf_start: s.psf,
        }
    }

    /// Constraints of a fit that re-estimates the scanner parameters (from
    /// the held values), for the whole surface.
    pub fn rescan_constraints<'a>(&self, exclude: Option<ArrayView3<'a, bool>>) -> FitConstraints<'a> {
        let s = &self.scanner;
        FitConstraints {
            exclude,
            density_start: Some(self.density_start),
            sigma_start: s.sigma,
            psf_start: s.psf,
            ..FitConstraints::default()
        }
    }

    /// Move the held parameters toward those of `fit` by `beta`; returns the
    /// largest relative change the re-fit asked for (before relaxation).
    pub fn relax(&mut self, fit: &CbmResult, beta: f64) -> f64 {
        let estimated = self.scanner.of(fit);
        let change = self.scanner.change(&estimated);
        self.scanner = self.scanner.relaxed(&estimated, beta);
        change
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxation_moves_part_of_the_way_and_reports_the_full_change() {
        let held = ScannerState {
            psf: Some([0.5, 0.8]),
            sigma: None,
            rho_c: Some(1500.0),
        };
        let fit = ScannerState {
            psf: Some([0.6, 0.8]),
            sigma: Some(9.0), // not held: ignored
            rho_c: Some(1500.0),
        };
        let mut fit_masked = held.of_values(&fit);
        assert_eq!(fit_masked.sigma, None);
        assert!((held.change(&fit_masked) - 0.2).abs() < 1e-12);
        fit_masked = held.relaxed(&fit_masked, 0.5);
        assert_eq!(fit_masked.psf, Some([0.55, 0.8]));
        assert_eq!(fit_masked.rho_c, Some(1500.0));
    }

    #[test]
    fn nothing_held_never_changes() {
        let none = ScannerState {
            psf: None,
            sigma: None,
            rho_c: None,
        };
        assert_eq!(none.change(&none), 0.0);
    }
}
