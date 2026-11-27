//! Quantities whose zero is arbitrary: an absolute image intensity, a
//! position. Only their differences are linear quantities; they cannot be
//! added to each other, scaled or squared.

use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use super::{IntensityContrast, Length};

/// An absolute image intensity (HU, mg HA/cm³, ...). The difference of two is
/// an [`IntensityContrast`]; adding a contrast gives an intensity again.
/// Nothing else: a sum, a multiple or a square of absolute intensities
/// depends on where the units put their zero.
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
pub struct Intensity(f64);

impl Intensity {
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Intensity(value)
    }

    /// The number in the image's units.
    #[inline(always)]
    pub const fn value(self) -> f64 {
        self.0
    }

    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Intensity(self.0.min(o.0))
    }

    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Intensity(self.0.max(o.0))
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Intensity(self.0.clamp(lo.0, hi.0))
    }

    /// The point a fraction `s` of the way from `self` to `other` (an
    /// interpolation, which does not depend on the zero).
    #[inline(always)]
    pub fn lerp(self, other: Self, s: f64) -> Self {
        self + (other - self) * s
    }
}

impl fmt::Debug for Intensity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} [intensity]", self.0)
    }
}

impl fmt::Display for Intensity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl Sub for Intensity {
    type Output = IntensityContrast;
    #[inline(always)]
    fn sub(self, o: Self) -> IntensityContrast {
        IntensityContrast::new(self.0 - o.0)
    }
}

impl Add<IntensityContrast> for Intensity {
    type Output = Intensity;
    #[inline(always)]
    fn add(self, c: IntensityContrast) -> Intensity {
        Intensity(self.0 + c.value())
    }
}

impl Sub<IntensityContrast> for Intensity {
    type Output = Intensity;
    #[inline(always)]
    fn sub(self, c: IntensityContrast) -> Intensity {
        Intensity(self.0 - c.value())
    }
}

impl AddAssign<IntensityContrast> for Intensity {
    #[inline(always)]
    fn add_assign(&mut self, c: IntensityContrast) {
        self.0 += c.value();
    }
}

impl SubAssign<IntensityContrast> for Intensity {
    #[inline(always)]
    fn sub_assign(&mut self, c: IntensityContrast) {
        self.0 -= c.value();
    }
}

/// A displacement: a vector of [`Length`]s.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct Displacement(pub [Length; 3]);

/// A point in physical coordinates (array-axis order). The difference of two
/// is a [`Displacement`]; adding a displacement gives a point again.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct Position(pub [Length; 3]);

impl Displacement {
    /// The Euclidean length.
    #[inline(always)]
    pub fn norm(self) -> Length {
        let [a, b, c] = self.0;
        (a * a + b * b + c * c).sqrt()
    }

    /// The component along a unit vector `n` (dimensionless).
    #[inline(always)]
    pub fn along(self, n: [f64; 3]) -> Length {
        self.0[0] * n[0] + self.0[1] * n[1] + self.0[2] * n[2]
    }

    /// `d` along a unit vector `n`.
    #[inline(always)]
    pub fn of(d: Length, n: [f64; 3]) -> Self {
        Displacement(n.map(|c| d * c))
    }
}

impl fmt::Debug for Displacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} [displacement]", self.0.map(|l| l.value()))
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} [position]", self.0.map(|l| l.value()))
    }
}

impl Sub for Position {
    type Output = Displacement;
    #[inline(always)]
    fn sub(self, o: Self) -> Displacement {
        Displacement([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
    }
}

impl Add<Displacement> for Position {
    type Output = Position;
    #[inline(always)]
    fn add(self, d: Displacement) -> Position {
        Position([self.0[0] + d.0[0], self.0[1] + d.0[1], self.0[2] + d.0[2]])
    }
}

impl Sub<Displacement> for Position {
    type Output = Position;
    #[inline(always)]
    fn sub(self, d: Displacement) -> Position {
        Position([self.0[0] - d.0[0], self.0[1] - d.0[1], self.0[2] - d.0[2]])
    }
}

impl Add for Displacement {
    type Output = Displacement;
    #[inline(always)]
    fn add(self, d: Displacement) -> Displacement {
        Displacement([self.0[0] + d.0[0], self.0[1] + d.0[1], self.0[2] + d.0[2]])
    }
}

impl Sub for Displacement {
    type Output = Displacement;
    #[inline(always)]
    fn sub(self, d: Displacement) -> Displacement {
        Displacement([self.0[0] - d.0[0], self.0[1] - d.0[1], self.0[2] - d.0[2]])
    }
}
