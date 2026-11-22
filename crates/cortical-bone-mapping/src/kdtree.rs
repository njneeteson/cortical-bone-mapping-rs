//! k-nearest-neighbour and radius queries over 3D points, on the k-d tree
//! of the `kiddo` crate (Bentley 1975; Friedman et al. 1977).
//!
//! `kiddo`'s immutable tree (a balanced k-d tree in Eytzinger layout with
//! leaf buckets) finds the candidates. This wrapper makes the results
//! *canonical*, i.e. independent of the tree's layout and of how the library
//! computes distances (it may use SIMD or fused multiply-adds):
//!
//! * every candidate's squared distance is recomputed as
//!   `Σ_a (q_a − p_a)²` in axis order ([`crate::util::dist2`]);
//! * a k-NN query returns the `k` smallest by (squared distance, index), so
//!   ties, which are common on voxel-lattice points, are resolved by index;
//!   the library is asked for a few more neighbours than needed, and for more
//!   until the last one it returns is farther than the `k`-th by more than
//!   the library's possible rounding, so no point that belongs in the result
//!   can be missing;
//! * a radius query asks for a radius larger by the same margin, keeps the
//!   points with `d² ≤ r²` by the recomputed distance, and sorts them by
//!   index.
//!
//! # References
//!
//! * J. L. Bentley, *Multidimensional binary search trees used for associative
//!   searching*, Communications of the ACM 18(9):509–517, 1975.
//! * J. H. Friedman, J. L. Bentley, R. A. Finkel, *An algorithm for finding
//!   best matches in logarithmic expected time*, ACM Transactions on
//!   Mathematical Software 3(3):209–226, 1977.

use std::num::NonZeroUsize;

use kiddo::{ImmutableKdTree, SquaredEuclidean};

use crate::util::{dist2, V3};

/// Relative margin for the library's rounding of squared distances (a few
/// ulps in practice).
const MARGIN: f64 = 1e-12;

pub struct KdTree {
    tree: Option<ImmutableKdTree<f64, 3>>,
    pts: Vec<V3>,
}

impl KdTree {
    /// Build the tree (in parallel for large inputs).
    pub fn new(points: &[V3]) -> Self {
        let tree = (!points.is_empty())
            .then(|| ImmutableKdTree::new_from_slice_parallel(points).expect("fewer than 2³² points"));
        KdTree {
            tree,
            pts: points.to_vec(),
        }
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.pts.len()
    }

    /// Whether there are no points.
    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    /// The `k` nearest points to `q` as `(index, squared distance)`: the `k`
    /// smallest by (squared distance, index), in that order, so that ties are
    /// resolved by index whatever the tree's layout. Points with
    /// `index == exclude` are skipped.
    pub fn knn(&self, q: V3, k: usize, exclude: Option<u32>) -> Vec<(u32, f64)> {
        let (Some(tree), n) = (&self.tree, self.pts.len()) else {
            return Vec::new();
        };
        if k == 0 {
            return Vec::new();
        }
        if k == 1 && exclude.is_none() {
            return vec![self.nearest(tree, q)];
        }
        let mut ask = (k + exclude.is_some() as usize + 2).min(n);
        loop {
            let found = tree
                .query(&q)
                .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(ask).unwrap())
                .execute();
            let farthest = found.last().map_or(0.0, |r| r.distance);
            let mut out: Vec<(u32, f64)> = found
                .iter()
                .filter(|r| Some(r.item) != exclude)
                .map(|r| (r.item, dist2(q, self.pts[r.item as usize])))
                .collect();
            out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            // complete if every point is in, or if no point left out can be as
            // close as the k-th
            let complete = ask == n || (out.len() > k && farthest > out[k - 1].1 * (1.0 + MARGIN) + f64::MIN_POSITIVE);
            if complete {
                out.truncate(k);
                return out;
            }
            ask = (2 * ask).min(n);
        }
    }

    /// The nearest point, smallest index among exact ties: the library's
    /// nearest, unless another point is as close within its rounding, in
    /// which case the general query decides.
    fn nearest(&self, tree: &ImmutableKdTree<f64, 3>, q: V3) -> (u32, f64) {
        let r = tree.query(&q).nearest_one::<SquaredEuclidean<f64>>().execute();
        let d = dist2(q, self.pts[r.item as usize]);
        let mut near = Vec::new();
        self.within(q, (d * (1.0 + 4.0 * MARGIN)).sqrt(), &mut near);
        near.into_iter()
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .unwrap_or((r.item, d))
    }

    /// All points within distance `r` of `q` (`d² ≤ r²`), as
    /// `(index, squared distance)`, sorted by index (an order that does not
    /// depend on the tree's layout).
    pub fn within(&self, q: V3, r: f64, out: &mut Vec<(u32, f64)>) {
        out.clear();
        let Some(tree) = &self.tree else {
            return;
        };
        let r2 = r * r;
        let pts = &self.pts;
        tree.query(&q)
            .within::<SquaredEuclidean<f64>>(r2 * (1.0 + MARGIN) + f64::MIN_POSITIVE)
            .unsorted()
            .visit(|c| {
                let d = dist2(q, pts[c.item as usize]);
                if d <= r2 {
                    out.push((c.item, d));
                }
            });
        out.sort_unstable_by_key(|c| c.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Rng;

    fn cloud(n: usize, seed: u64) -> Vec<V3> {
        let mut r = Rng::new(seed);
        (0..n)
            .map(|_| [r.uniform() * 10.0, r.uniform() * 3.0, r.uniform() * 7.0])
            .collect()
    }

    #[test]
    fn knn_matches_brute_force() {
        let pts = cloud(3000, 1);
        let tree = KdTree::new(&pts);
        let mut r = Rng::new(2);
        for _ in 0..50 {
            let q = [r.uniform() * 10.0, r.uniform() * 3.0, r.uniform() * 7.0];
            let k = 1 + (r.next_u64() % 40) as usize;
            let got = tree.knn(q, k, None);
            let mut all: Vec<(u32, f64)> = pts
                .iter()
                .enumerate()
                .map(|(i, &p)| (i as u32, crate::util::dist2(q, p)))
                .collect();
            all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            assert_eq!(got.len(), k);
            for (g, a) in got.iter().zip(&all) {
                assert!((g.1 - a.1).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn results_are_canonical_with_ties() {
        // a lattice: many points at exactly equal distances
        let pts: Vec<V3> = (0..12)
            .flat_map(|i| (0..12).flat_map(move |j| (0..6).map(move |k| [i as f64, j as f64, 0.5 * k as f64])))
            .collect();
        let tree = KdTree::new(&pts);
        let mut r = Rng::new(5);
        for _ in 0..100 {
            let q = pts[(r.next_u64() % pts.len() as u64) as usize];
            let k = 1 + (r.next_u64() % 30) as usize;
            let ex = r
                .next_u64()
                .is_multiple_of(2)
                .then(|| (r.next_u64() % pts.len() as u64) as u32);
            let mut all: Vec<(u32, f64)> = (0..pts.len() as u32)
                .filter(|&i| Some(i) != ex)
                .map(|i| (i, crate::util::dist2(q, pts[i as usize])))
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            all.truncate(k);
            assert_eq!(tree.knn(q, k, ex), all);
            let mut out = Vec::new();
            tree.within(q, 1.5, &mut out);
            let want: Vec<(u32, f64)> = (0..pts.len() as u32)
                .map(|i| (i, crate::util::dist2(q, pts[i as usize])))
                .filter(|c| c.1 <= 2.25)
                .collect();
            assert_eq!(out, want);
        }
    }

    #[test]
    fn exclusion_and_radius() {
        let pts = cloud(500, 3);
        let tree = KdTree::new(&pts);
        let nn = tree.knn(pts[10], 3, Some(10));
        assert!(nn.iter().all(|&(i, _)| i != 10));
        let mut out = Vec::new();
        tree.within(pts[10], 1.5, &mut out);
        let brute = pts.iter().filter(|&&p| crate::util::dist2(p, pts[10]) <= 2.25).count();
        assert_eq!(out.len(), brute);
    }
}
