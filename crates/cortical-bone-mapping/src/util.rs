//! Small shared helpers: a deterministic PRNG, 3-vectors, tiny dense solves,
//! deterministic parallel sums and robust statistics.
//!
//! * [`Rng`]: SplitMix64 (Steele et al. 2014), a 64-bit counter passed through
//!   a bijective mixing function; it passes BigCrush, needs no state beyond
//!   one word, and is reproducible across platforms. Uniforms take the top 53
//!   bits; normals use the Box–Muller transform (Box & Muller 1958); shuffles
//!   are Fisher–Yates (Durstenfeld 1964).
//! * [`cholesky_solve`]: `A = LLᵀ` by the Cholesky–Banachiewicz order, then two
//!   triangular solves (Golub & Van Loan 2013). Pivots below `10⁻¹⁴ max diag`
//!   are floored instead of failing: callers solve damped Gauss–Newton blocks
//!   that are positive semi-definite by construction, where a vanishing pivot
//!   means an unidentifiable direction, which the floor sets to (nearly) zero.
//! * [`det_sum`]: floating-point addition is not associative, so a parallel
//!   sum whose grouping follows the thread schedule differs from run to run.
//!   Summing fixed-size chunks in parallel and the chunk sums in order gives a
//!   result independent of the thread count, which is what makes every output
//!   of the crate bitwise reproducible.
//! * [`quantile`]: linear interpolation between order statistics at position
//!   `q (n − 1)`, the "type 7" definition of Hyndman & Fan (1996) used by
//!   NumPy and R by default.
//!
//! # References
//!
//! * G. L. Steele, D. Lea, C. H. Flood, *Fast splittable pseudorandom number
//!   generators*, Proceedings of OOPSLA 2014, ACM, pp. 453–472.
//! * G. E. P. Box, M. E. Muller, *A note on the generation of random normal
//!   deviates*, Annals of Mathematical Statistics 29(2):610–611, 1958.
//! * R. Durstenfeld, *Algorithm 235: Random permutation*, Communications of
//!   the ACM 7(7):420, 1964.
//! * G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns
//!   Hopkins University Press, 2013.
//! * R. J. Hyndman, Y. Fan, *Sample quantiles in statistical packages*, The
//!   American Statistician 50(4):361–365, 1996.

/// SplitMix64: a tiny, fast, deterministic generator (no external crates,
/// reproducible across platforms).
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// A generator whose state is `seed` XOR the golden-ratio constant.
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    /// The next SplitMix64 output (Steele et al. 2014): add the golden-ratio
    /// increment `0x9E3779B97F4A7C15` to the state and pass it through the
    /// xor-shift/multiply finaliser.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`: the top 53 bits as a fraction, every value equally
    /// likely and exactly representable.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    /// Standard normal by the Box–Muller transform (Box & Muller 1958):
    /// `√(−2 ln u₁) cos(2π u₂)` for independent uniforms (`u₁ > 0`).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Fisher–Yates shuffle (Durstenfeld 1964): every permutation equally likely
    /// (up to the negligible modulo bias of a 64-bit generator).
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

pub type V3 = [f64; 3];

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
pub fn dist2(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

/// Solve the symmetric positive (semi-)definite system `A x = b` of size
/// `n ≤ 8` in place: Cholesky factorisation `A = LLᵀ` (row by row), then the
/// triangular solves `L y = b` and `Lᵀ x = y` (Golub & Van Loan 2013). `a` is
/// row-major `n × n`. Pivots below `10⁻¹⁴ · max(diag)` are floored, so the solve
/// never fails; it returns `false` if any pivot needed it.
pub fn cholesky_solve(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    let maxd = (0..n).map(|i| a[i * n + i].abs()).fold(0.0, f64::max);
    let floor = 1e-14 * maxd.max(1e-300);
    let mut ok = true;
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= floor {
            d = floor.max(1e-300);
            ok = false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
    }
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in i + 1..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    ok
}

/// Rows per chunk of the deterministic parallel reductions.
pub const CHUNK: usize = 4096;

/// Parallel sum of `f(i)` for `i < n` whose result does not depend on the
/// number of threads or on scheduling: fixed-size chunks are summed in
/// parallel and the chunk sums are added in order.
pub fn det_sum(n: usize, f: impl Fn(usize) -> f64 + Sync) -> f64 {
    use rayon::prelude::*;
    let parts: Vec<f64> = (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| (c * CHUNK..((c + 1) * CHUNK).min(n)).map(&f).sum::<f64>())
        .collect();
    parts.iter().sum()
}

/// [`det_sum`] for 5-vectors.
pub fn det_sum5(n: usize, f: impl Fn(usize) -> [f64; 5] + Sync) -> [f64; 5] {
    use rayon::prelude::*;
    let parts: Vec<[f64; 5]> = (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| {
            let mut acc = [0.0; 5];
            for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
                let v = f(i);
                for k in 0..5 {
                    acc[k] += v[k];
                }
            }
            acc
        })
        .collect();
    let mut out = [0.0; 5];
    for p in parts {
        for k in 0..5 {
            out[k] += p[k];
        }
    }
    out
}

/// Index into a packed upper-triangular 5×5 symmetric matrix (15 entries).
#[inline]
pub const fn sym5(i: usize, j: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    // rows: 0 → 0..5, 1 → 5..9, 2 → 9..12, 3 → 12..14, 4 → 14
    i * 5 - i * (i + 1) / 2 + j
}

/// The `q` quantile of the finite values by linear interpolation between
/// order statistics at `q (n − 1)` (type 7 of Hyndman & Fan 1996, the NumPy
/// default); `NaN` if there are none.
pub fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let pos = q.clamp(0.0, 1.0) * (v.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    v[lo] + (pos - lo as f64) * (v[hi] - v[lo])
}

/// Median of the finite values (the mean of the two middle values for an even
/// count) by selection, O(n); `NaN` if there are none.
pub fn median(v: &[f64]) -> f64 {
    let mut w: Vec<f64> = v.iter().cloned().filter(|x| x.is_finite()).collect();
    if w.is_empty() {
        return f64::NAN;
    }
    let k = w.len() / 2;
    w.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    let hi = w[k];
    if w.len() % 2 == 1 {
        hi
    } else {
        let lo = w[..k].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        0.5 * (lo + hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sym5_packing() {
        let mut seen = [false; 15];
        for i in 0..5 {
            for j in i..5 {
                let k = sym5(i, j);
                assert!(!seen[k]);
                seen[k] = true;
                assert_eq!(sym5(j, i), k);
            }
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn cholesky() {
        let mut a = [4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let a0 = a;
        let x = [1.0, -2.0, 0.5];
        let mut b = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                b[i] += a0[i * 3 + j] * x[j];
            }
        }
        assert!(cholesky_solve(&mut a, &mut b, 3));
        for i in 0..3 {
            assert!((b[i] - x[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn rng_is_deterministic_and_normal() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
        let n = 20000;
        let xs: Vec<f64> = (0..n).map(|_| a.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.03 && (var - 1.0).abs() < 0.05);
    }

    #[test]
    fn median_even_odd() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}
