//! Physical units, checked at compile time.
//!
//! The physical computations of the crate (the profile model and its fits,
//! sampling, geometry, surfaces) involve two base dimensions: **length** (the
//! unit of the image's spacing, usually mm) and **image intensity** (the unit
//! of its values: HU, mg HA/cm³, ...). Their units are whatever the input
//! uses; the types only make sure that every expression combines them
//! consistently, so a formula that adds a length to an intensity, or compares
//! a thickness with a density, does not compile.
//!
//! * [`Quantity<L, I>`] is a linear quantity of dimensions
//!   `Length^L · IntensityContrast^I`, `L`, `I` type-level integers
//!   (`typenum`). Sums, differences and comparisons need equal dimensions;
//!   products and quotients add and subtract the exponents, so every
//!   combination is a type without being written out. The names below are
//!   only aliases.
//! * Two **affine** types hold values whose zero is arbitrary:
//!   [`Intensity`] (an absolute image value) and [`Position`] (a point).
//!   Their differences are linear ([`IntensityContrast`], [`Displacement`]);
//!   they cannot be added together, scaled or squared, since the result
//!   would depend on where the units put their zero. (Squaring an absolute
//!   intensity is how a penalty weight `ρc²` comes to depend on an offset of
//!   the image values; `(ρc − ρs)²` is a contrast squared.)
//!
//! Every type is a `#[repr(transparent)]` wrapper of `f64` with inlined
//! operations: the checks cost nothing at run time. Numbers leave the system
//! explicitly with `.value()`, at the boundary of arrays and of the solver,
//! whose variables are dimensionless.
//!
//! ```
//! use cortical_bone_mapping::units::*;
//! let t = Length::new(0.8);
//! let sigma = Length::new(0.1);
//! let z: Dimensionless = t / sigma; // a ratio of lengths
//! let (rho_c, rho_s) = (Intensity::new(1000.0), Intensity::new(50.0));
//! let contrast: IntensityContrast = rho_c - rho_s;
//! let weight: IntensityContrastSq = contrast * contrast;
//! let slope: Quantity<N1, P1> = contrast / t; // intensity per length
//! # let _ = (z, weight, slope);
//! ```
//!
//! Inconsistent expressions do not compile:
//!
//! ```compile_fail
//! use cortical_bone_mapping::units::*;
//! let _ = Length::new(1.0) + IntensityContrast::new(1.0);
//! ```
//!
//! ```compile_fail
//! use cortical_bone_mapping::units::*;
//! let _ = Length::new(1.0) < Area::new(1.0);
//! ```
//!
//! ```compile_fail
//! use cortical_bone_mapping::units::*;
//! let _ = Intensity::new(1000.0) + Intensity::new(50.0); // depends on the zero
//! ```
//!
//! ```compile_fail
//! use cortical_bone_mapping::units::*;
//! let rho_c = Intensity::new(1000.0);
//! let _ = rho_c * rho_c; // depends on the zero
//! ```

mod affine;
mod quantity;

pub use affine::{Displacement, Intensity, Position};
pub use quantity::Quantity;
pub use typenum::{N1, N2, P1, P2, P3, Z0};

/// A plain number: a ratio of quantities of the same dimensions.
pub type Dimensionless = Quantity<Z0, Z0>;
pub type Length = Quantity<P1, Z0>;
pub type Area = Quantity<P2, Z0>;
pub type Volume = Quantity<P3, Z0>;
/// Curvature.
pub type InvLength = Quantity<N1, Z0>;
pub type InvArea = Quantity<N2, Z0>;
/// A difference of image intensities.
pub type IntensityContrast = Quantity<Z0, P1>;
pub type IntensityContrastSq = Quantity<Z0, P2>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_combine() {
        let a: Area = Length::new(2.0) * Length::new(3.0);
        assert_eq!(a.value(), 6.0);
        let l: Length = a / Length::new(2.0);
        assert_eq!(l.value(), 3.0);
        let r: Dimensionless = Length::new(1.0) / Length::new(4.0);
        assert_eq!(r.value(), 0.25);
        let k: InvLength = Dimensionless::new(1.0) / Length::new(0.5);
        assert_eq!(k.value(), 2.0);
        assert_eq!(Area::new(9.0).sqrt(), Length::new(3.0));
        assert_eq!(Length::new(3.0).squared(), Area::new(9.0));
        let s: Length = [Length::new(1.0), Length::new(2.0)].iter().sum();
        assert_eq!(s.value(), 3.0);
    }

    #[test]
    fn intensities_are_affine() {
        let (c, s) = (Intensity::new(1000.0), Intensity::new(50.0));
        let d: IntensityContrast = c - s;
        assert_eq!(d.value(), 950.0);
        assert_eq!(s + d, c);
        assert_eq!(s.lerp(c, 0.5).value(), 525.0);
        // an offset of the units changes no contrast
        let off = IntensityContrast::new(1024.0);
        assert_eq!((c + off) - (s + off), d);
    }

    #[test]
    fn positions_are_affine() {
        let p = Position([Length::new(1.0), Length::new(2.0), Length::new(2.0)]);
        let q = Position([Length::new(1.0), Length::new(0.0), Length::new(0.0)]);
        let d = p - q;
        assert_eq!(d.norm(), Length::new(8f64.sqrt()));
        assert_eq!(q + d, p);
        assert_eq!(d.along([0.0, 1.0, 0.0]), Length::new(2.0));
    }

    #[test]
    fn a_quantity_is_a_plain_f64() {
        assert_eq!(std::mem::size_of::<Length>(), std::mem::size_of::<f64>());
        assert_eq!(std::mem::size_of::<Intensity>(), std::mem::size_of::<f64>());
        assert_eq!(std::mem::size_of::<Position>(), 3 * std::mem::size_of::<f64>());
    }
}
