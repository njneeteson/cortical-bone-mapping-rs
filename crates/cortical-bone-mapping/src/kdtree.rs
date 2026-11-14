//! A compact static k-d tree over 3D points for k-nearest-neighbour and
//! radius queries.
//!
//! The tree is stored implicitly: points are permuted so that each node is a
//! contiguous range whose median (the pivot) sits in the middle, so the only memory beyond the input
//! is one permutation (`u32` per point) and a copy of the coordinates in tree
//! order (for cache-friendly leaf scans).

use std::collections::BinaryHeap;

use crate::util::V3;

const LEAF: usize = 16;

pub struct KdTree {
    pts: Vec<V3>,
    idx: Vec<u32>,
}

/// Candidate neighbour (squared distance, index), ordered by distance then
/// index so that ties are broken deterministically.
struct Cand(f64, u32);
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == std::cmp::Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0).then(self.1.cmp(&o.1))
    }
}

impl KdTree {
    pub fn new(points: &[V3]) -> Self {
        let mut idx: Vec<u32> = (0..points.len() as u32).collect();
        build(points, &mut idx, 0);
        let pts = idx.iter().map(|&i| points[i as usize]).collect();
        KdTree { pts, idx }
    }

    pub fn len(&self) -> usize {
        self.pts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    /// The `k` nearest points to `q` as `(index, squared distance)`, sorted
    /// by distance. Points with `index == exclude` are skipped.
    pub fn knn(&self, q: V3, k: usize, exclude: Option<u32>) -> Vec<(u32, f64)> {
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        if k > 0 && !self.pts.is_empty() {
            self.knn_rec(q, k, exclude, 0, self.pts.len(), 0, &mut heap);
        }
        let mut out: Vec<(u32, f64)> = heap.into_iter().map(|c| (c.1, c.0)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn knn_rec(
        &self,
        q: V3,
        k: usize,
        ex: Option<u32>,
        lo: usize,
        hi: usize,
        depth: usize,
        heap: &mut BinaryHeap<Cand>,
    ) {
        if hi - lo <= LEAF {
            for t in lo..hi {
                let id = self.idx[t];
                if Some(id) == ex {
                    continue;
                }
                let d = crate::util::dist2(q, self.pts[t]);
                if heap.len() < k {
                    heap.push(Cand(d, id));
                } else if d < heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push(Cand(d, id));
                }
            }
            return;
        }
        let axis = depth % 3;
        let mid = lo + (hi - lo) / 2;
        let split = self.pts[mid][axis];
        let diff = q[axis] - split;
        // the pivot itself
        let id = self.idx[mid];
        if Some(id) != ex {
            let d = crate::util::dist2(q, self.pts[mid]);
            if heap.len() < k {
                heap.push(Cand(d, id));
            } else if d < heap.peek().unwrap().0 {
                heap.pop();
                heap.push(Cand(d, id));
            }
        }
        let (first, second) = if diff < 0.0 {
            ((lo, mid), (mid + 1, hi))
        } else {
            ((mid + 1, hi), (lo, mid))
        };
        self.knn_rec(q, k, ex, first.0, first.1, depth + 1, heap);
        if heap.len() < k || diff * diff < heap.peek().unwrap().0 {
            self.knn_rec(q, k, ex, second.0, second.1, depth + 1, heap);
        }
    }

    /// All points within distance `r` of `q` (unsorted), as
    /// `(index, squared distance)`.
    pub fn within(&self, q: V3, r: f64, out: &mut Vec<(u32, f64)>) {
        out.clear();
        if !self.pts.is_empty() {
            self.within_rec(q, r * r, 0, self.pts.len(), 0, out);
        }
    }

    fn within_rec(&self, q: V3, r2: f64, lo: usize, hi: usize, depth: usize, out: &mut Vec<(u32, f64)>) {
        if hi - lo <= LEAF {
            for t in lo..hi {
                let d = crate::util::dist2(q, self.pts[t]);
                if d <= r2 {
                    out.push((self.idx[t], d));
                }
            }
            return;
        }
        let axis = depth % 3;
        let mid = lo + (hi - lo) / 2;
        let diff = q[axis] - self.pts[mid][axis];
        let d = crate::util::dist2(q, self.pts[mid]);
        if d <= r2 {
            out.push((self.idx[mid], d));
        }
        if diff < 0.0 || diff * diff <= r2 {
            self.within_rec(q, r2, lo, mid, depth + 1, out);
        }
        if diff >= 0.0 || diff * diff <= r2 {
            self.within_rec(q, r2, mid + 1, hi, depth + 1, out);
        }
    }
}

fn build(points: &[V3], idx: &mut [u32], depth: usize) {
    if idx.len() <= LEAF {
        return;
    }
    let axis = depth % 3;
    let mid = idx.len() / 2;
    idx.select_nth_unstable_by(mid, |&a, &b| {
        points[a as usize][axis]
            .partial_cmp(&points[b as usize][axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (l, r) = idx.split_at_mut(mid);
    let r = &mut r[1..]; // the pivot stays at `mid`
    if l.len() + r.len() > 4096 {
        rayon::join(|| build(points, l, depth + 1), || build(points, r, depth + 1));
    } else {
        build(points, l, depth + 1);
        build(points, r, depth + 1);
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
