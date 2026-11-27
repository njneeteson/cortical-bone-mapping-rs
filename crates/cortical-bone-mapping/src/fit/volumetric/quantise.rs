//! Compact storage of the band's per-voxel values.
//!
//! The band holds tens to hundreds of millions of voxels, and none of their
//! values needs more than 16 bits: each is stored as an integer code `q` of a
//! linear scale, `x = lo + q · step`, rounded to the nearest code, and the
//! computation reads it back as `f64`. The rounding error is at most
//! `step / 2`: for the intensities, 16 bits over the image's range (an SD of
//! `step / √12`, far below the noise of any CT image); for depths, 16 bits
//! over the band's depth range (0.15 µm for a 10 mm band); for barycentric
//! coordinates, 16 bits over `[0, 1]`; and for the squared axial component
//! of a normal, which only mixes the in-plane and axial blur, 8 bits.

/// A linear scale of integer codes: `x = lo + q · step`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Linear {
    pub lo: f64,
    pub step: f64,
}

impl Linear {
    /// The scale with `levels` steps from `lo` to `hi` (one step of 1 if the
    /// range is empty).
    pub fn over(lo: f64, hi: f64, levels: u32) -> Self {
        let step = if hi > lo { (hi - lo) / levels as f64 } else { 1.0 };
        Linear { lo, step }
    }

    /// 16 bits from `lo` to `hi`.
    pub fn u16(lo: f64, hi: f64) -> Self {
        Self::over(lo, hi, u16::MAX as u32)
    }

    /// The nearest code (clamped to the scale's range), as a `u32`.
    #[inline]
    fn code(&self, x: f64, max: u32) -> u32 {
        ((x - self.lo) / self.step).round().clamp(0.0, max as f64) as u32
    }

    #[inline]
    pub fn encode16(&self, x: f64) -> u16 {
        self.code(x, u16::MAX as u32) as u16
    }

    #[inline]
    pub fn encode8(&self, x: f64) -> u8 {
        self.code(x, u8::MAX as u32) as u8
    }

    #[inline]
    pub fn decode(&self, q: u32) -> f64 {
        self.lo + q as f64 * self.step
    }
}

/// Barycentric coordinates in `[0, 1]`, 16 bits each.
pub(crate) const BARY: Linear = Linear {
    lo: 0.0,
    step: 1.0 / u16::MAX as f64,
};

/// A squared normal component in `[0, 1]`, 8 bits.
pub(crate) const UNIT8: Linear = Linear {
    lo: 0.0,
    step: 1.0 / u8::MAX as f64,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_within_half_a_step() {
        let s = Linear::u16(-5.0, 7.5);
        for k in 0..=1000 {
            let x = -5.0 + 12.5 * k as f64 / 1000.0;
            let back = s.decode(s.encode16(x) as u32);
            assert!((back - x).abs() <= 0.5 * s.step + 1e-12, "{x} -> {back}");
        }
        // the ends are exact, and values beyond them clamp
        assert_eq!(s.decode(s.encode16(-5.0) as u32), -5.0);
        assert!((s.decode(s.encode16(7.5) as u32) - 7.5).abs() < 1e-12);
        assert_eq!(s.encode16(100.0), u16::MAX);
        assert_eq!(s.encode16(-100.0), 0);
        let a = UNIT8;
        assert!((a.decode(a.encode8(0.3) as u32) - 0.3).abs() <= 0.5 / 255.0);
        assert_eq!(
            Linear::u16(2.0, 2.0).decode(Linear::u16(2.0, 2.0).encode16(2.0) as u32),
            2.0
        );
    }
}
