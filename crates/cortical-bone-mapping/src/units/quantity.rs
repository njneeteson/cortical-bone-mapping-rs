//! Linear quantities: a value with the dimensions `Length^L · IntensityContrast^I`.

use std::fmt;
use std::iter::Sum as IterSum;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use typenum::{Diff, Integer, PartialDiv, PartialQuot, Sum, P2};

/// A value with the dimensions `Length^L · IntensityContrast^I`, `L` and `I`
/// type-level integers. Compiles to a plain `f64`.
#[repr(transparent)]
pub struct Quantity<L: Integer, I: Integer> {
    value: f64,
    dims: PhantomData<(L, I)>,
}

impl<L: Integer, I: Integer> Quantity<L, I> {
    /// A quantity of `value` in the crate's units (the length unit of the
    /// image's spacing; the intensity unit of its values).
    #[inline(always)]
    pub const fn new(value: f64) -> Self {
        Quantity {
            value,
            dims: PhantomData,
        }
    }

    /// The number in the crate's units: the way out of the unit system, for
    /// arrays, the solver's dimensionless variables and output.
    #[inline(always)]
    pub const fn value(self) -> f64 {
        self.value
    }

    pub const ZERO: Self = Self::new(0.0);

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new(self.value.abs())
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self::new(self.value.min(other.value))
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self::new(self.value.max(other.value))
    }

    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::new(self.value.clamp(lo.value, hi.value))
    }

    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.value.is_finite()
    }

    /// The square root, for even powers only.
    #[inline(always)]
    #[allow(clippy::type_complexity)]
    pub fn sqrt(self) -> Quantity<PartialQuot<L, P2>, PartialQuot<I, P2>>
    where
        L: PartialDiv<P2>,
        I: PartialDiv<P2>,
        PartialQuot<L, P2>: Integer,
        PartialQuot<I, P2>: Integer,
    {
        Quantity::new(self.value.sqrt())
    }

    /// The square.
    #[inline(always)]
    pub fn squared(self) -> Quantity<Sum<L, L>, Sum<I, I>>
    where
        L: Add<L>,
        I: Add<I>,
        Sum<L, L>: Integer,
        Sum<I, I>: Integer,
    {
        Quantity::new(self.value * self.value)
    }
}

impl<L: Integer, I: Integer> Clone for Quantity<L, I> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<L: Integer, I: Integer> Copy for Quantity<L, I> {}

impl<L: Integer, I: Integer> Default for Quantity<L, I> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<L: Integer, I: Integer> PartialEq for Quantity<L, I> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<L: Integer, I: Integer> PartialOrd for Quantity<L, I> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<L: Integer, I: Integer> fmt::Debug for Quantity<L, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} [L^{} I^{}]", self.value, L::to_i32(), I::to_i32())
    }
}

impl<L: Integer, I: Integer> fmt::Display for Quantity<L, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

impl<L: Integer, I: Integer> Add for Quantity<L, I> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self::new(self.value + o.value)
    }
}

impl<L: Integer, I: Integer> Sub for Quantity<L, I> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self::new(self.value - o.value)
    }
}

impl<L: Integer, I: Integer> Neg for Quantity<L, I> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}

impl<L: Integer, I: Integer> AddAssign for Quantity<L, I> {
    #[inline(always)]
    fn add_assign(&mut self, o: Self) {
        self.value += o.value;
    }
}

impl<L: Integer, I: Integer> SubAssign for Quantity<L, I> {
    #[inline(always)]
    fn sub_assign(&mut self, o: Self) {
        self.value -= o.value;
    }
}

impl<L: Integer, I: Integer> Mul<f64> for Quantity<L, I> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, k: f64) -> Self {
        Self::new(self.value * k)
    }
}

impl<L: Integer, I: Integer> Mul<Quantity<L, I>> for f64 {
    type Output = Quantity<L, I>;
    #[inline(always)]
    fn mul(self, q: Quantity<L, I>) -> Quantity<L, I> {
        Quantity::new(self * q.value)
    }
}

impl<L: Integer, I: Integer> Div<f64> for Quantity<L, I> {
    type Output = Self;
    #[inline(always)]
    fn div(self, k: f64) -> Self {
        Self::new(self.value / k)
    }
}

impl<L: Integer, I: Integer> MulAssign<f64> for Quantity<L, I> {
    #[inline(always)]
    fn mul_assign(&mut self, k: f64) {
        self.value *= k;
    }
}

impl<L: Integer, I: Integer> DivAssign<f64> for Quantity<L, I> {
    #[inline(always)]
    fn div_assign(&mut self, k: f64) {
        self.value /= k;
    }
}

/// Dimensions multiply: their exponents add.
impl<L1, I1, L2, I2> Mul<Quantity<L2, I2>> for Quantity<L1, I1>
where
    L1: Integer + Add<L2>,
    I1: Integer + Add<I2>,
    L2: Integer,
    I2: Integer,
    Sum<L1, L2>: Integer,
    Sum<I1, I2>: Integer,
{
    type Output = Quantity<Sum<L1, L2>, Sum<I1, I2>>;
    #[inline(always)]
    fn mul(self, o: Quantity<L2, I2>) -> Self::Output {
        Quantity::new(self.value * o.value)
    }
}

/// Dimensions divide: their exponents subtract.
impl<L1, I1, L2, I2> Div<Quantity<L2, I2>> for Quantity<L1, I1>
where
    L1: Integer + Sub<L2>,
    I1: Integer + Sub<I2>,
    L2: Integer,
    I2: Integer,
    Diff<L1, L2>: Integer,
    Diff<I1, I2>: Integer,
{
    type Output = Quantity<Diff<L1, L2>, Diff<I1, I2>>;
    #[inline(always)]
    fn div(self, o: Quantity<L2, I2>) -> Self::Output {
        Quantity::new(self.value / o.value)
    }
}

impl<L: Integer, I: Integer> IterSum for Quantity<L, I> {
    fn sum<It: Iterator<Item = Self>>(iter: It) -> Self {
        Self::new(iter.map(|q| q.value).sum())
    }
}

impl<'a, L: Integer, I: Integer> IterSum<&'a Quantity<L, I>> for Quantity<L, I> {
    fn sum<It: Iterator<Item = &'a Self>>(iter: It) -> Self {
        Self::new(iter.map(|q| q.value).sum())
    }
}
