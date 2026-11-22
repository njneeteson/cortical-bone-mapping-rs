//! Multigrid control points and the interpolation matrix `A` from control
//! points to surface nodes (Neeteson & Boyd 2025).
//!
//! **Control points.** Visit the nodes in a random order and accept a node if
//! no accepted node lies within `s`: random sequential selection, also known as
//! dart throwing for Poisson-disk sampling (Cook 1986). The result is
//! *maximal*: every node lies within `s` of a control point (else it would have
//! been accepted), and no two control points are closer than `s`, so the
//! control points cover the surface evenly at spacing `s` whatever the node
//! density. A hash grid of cell size `s` makes each test O(1), so selection is
//! O(N); the order comes from a seeded generator, so it is deterministic.
//!
//! **Interpolation.** Node `i` takes the normalised Gaussian-weighted mean of
//! its `k` nearest control points,
//! `A_iq = exp(−d_iq²/2s²) / Σ_q' exp(−d_iq'²/2s²)`, so the rows of `A` sum to
//! one (constants are reproduced exactly) and the interpolated fields vary on
//! the scale `s`.
//!
//! # References
//!
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.
//! * R. L. Cook, *Stochastic sampling in computer graphics*, ACM Transactions
//!   on Graphics 5(1):51–72, 1986.

use std::collections::HashMap;

use rayon::prelude::*;

use super::{gaussian_weights, SparseRows};
use crate::kdtree::KdTree;
use crate::util::{Rng, V3};

/// The multigrid interpolation matrix `A` (N × Q): each point takes a
/// normalised Gaussian-weighted (σ = `spacing`) average of its `k` nearest
/// control points (Neeteson & Boyd 2025). Rows sum to one.
pub fn interpolation_matrix(points: &[V3], controls: &[V3], k: usize, spacing: f64) -> SparseRows<f64> {
    let tree = KdTree::new(controls);
    let k = k.max(1).min(controls.len());
    let rows: Vec<Vec<(u32, f64)>> = points
        .par_iter()
        .map(|&p| {
            let nn = tree.knn(p, k, None);
            let d2: Vec<f64> = nn.iter().map(|x| x.1).collect();
            let w = gaussian_weights(&d2, spacing);
            nn.iter().zip(w).map(|(&(i, _), w)| (i, w)).collect()
        })
        .collect();
    SparseRows::from_rows(rows, controls.len())
}

/// Random sequential selection of control points (dart throwing; Cook 1986):
/// in a seeded random order, accept every point with no accepted point within
/// `spacing`. No two control points are closer than `spacing`, and every point
/// is within `spacing` of one (the selection is maximal). Returns indices into
/// `points`. A hash grid of cell size `spacing` limits each test to the 27
/// surrounding cells.
pub fn select_control_points(points: &[V3], spacing: f64, seed: u64) -> Vec<u32> {
    let mut order: Vec<u32> = (0..points.len() as u32).collect();
    Rng::new(seed).shuffle(&mut order);
    let cell = |p: V3| -> (i64, i64, i64) {
        (
            (p[0] / spacing).floor() as i64,
            (p[1] / spacing).floor() as i64,
            (p[2] / spacing).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
    let mut chosen = Vec::new();
    let s2 = spacing * spacing;
    for &i in &order {
        let p = points[i as usize];
        let c = cell(p);
        let mut free = true;
        'search: for a in -1..=1 {
            for b in -1..=1 {
                for d in -1..=1 {
                    if let Some(v) = grid.get(&(c.0 + a, c.1 + b, c.2 + d)) {
                        if v.iter().any(|&q| crate::util::dist2(points[q as usize], p) <= s2) {
                            free = false;
                            break 'search;
                        }
                    }
                }
            }
        }
        if free {
            grid.entry(c).or_default().push(i);
            chosen.push(i);
        }
    }
    chosen
}
