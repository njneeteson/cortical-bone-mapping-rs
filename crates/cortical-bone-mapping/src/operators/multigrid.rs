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

use super::neighbours::Neighbours;
use super::{gaussian_weights, SparseRows};
use crate::kdtree::KdTree;
use crate::util::{Rng, V3};

/// The multigrid interpolation matrix `A` (N × Q): each node takes a
/// normalised Gaussian-weighted (σ = `spacing`) average of its `k` nearest
/// control points (Neeteson & Boyd 2025), by the distance of `neighbours`.
/// `controls` are node indices. Rows sum to one.
pub fn interpolation_matrix(neighbours: &Neighbours, controls: &[u32], k: usize, spacing: f64) -> SparseRows<f64> {
    let k = k.max(1).min(controls.len());
    let nearest = match neighbours {
        Neighbours::Euclidean { points, .. } => nearest_controls_euclidean(points, controls, k),
        Neighbours::Surface(_) => nearest_controls_surface(neighbours, controls, k, spacing),
    };
    let rows: Vec<Vec<(u32, f64)>> = nearest
        .into_par_iter()
        .map(|nn| {
            let d2: Vec<f64> = nn.iter().map(|x| x.1).collect();
            let w = gaussian_weights(&d2, spacing);
            nn.iter().zip(w).map(|(&(i, _), w)| (i, w)).collect()
        })
        .collect();
    SparseRows::from_rows(rows, controls.len())
}

/// Per node, its `k` nearest control points in space, as (control number,
/// squared distance).
fn nearest_controls_euclidean(points: &[V3], controls: &[u32], k: usize) -> Vec<Vec<(u32, f64)>> {
    let cpts: Vec<V3> = controls.iter().map(|&i| points[i as usize]).collect();
    let tree = KdTree::new(&cpts);
    points.par_iter().map(|&p| tree.knn(p, k, None)).collect()
}

/// Per node, its `k` nearest control points along the surface, as (control
/// number, squared distance), ties by control number. The searches run from
/// each control out to `3 · spacing` (where a Gaussian of width `spacing` has
/// fallen to 1%); every node is within `spacing` of a control (the selection
/// is maximal), so every row has at least one.
fn nearest_controls_surface(neighbours: &Neighbours, controls: &[u32], k: usize, spacing: f64) -> Vec<Vec<(u32, f64)>> {
    let reach: Vec<Vec<(u32, f64)>> = controls
        .par_iter()
        .map_init(
            || (Vec::new(), neighbours.scratch()),
            |(out, s), &c| {
                neighbours.within(c as usize, 3.0 * spacing, out, s);
                out.clone()
            },
        )
        .collect();
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); neighbours.points().len()];
    for (q, near) in reach.iter().enumerate() {
        for &(v, d2) in near {
            rows[v as usize].push((q as u32, d2));
        }
    }
    rows.par_iter_mut().for_each(|r| {
        r.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        r.truncate(k);
    });
    rows
}

/// Random sequential selection of control points (dart throwing; Cook 1986):
/// in a seeded random order, accept every node with no accepted node within
/// `spacing` (by the distance of `neighbours`). No two control points are
/// closer than `spacing`, and every node is within `spacing` of one (the
/// selection is maximal). Returns node indices.
pub fn select_control_points(neighbours: &Neighbours, spacing: f64, seed: u64) -> Vec<u32> {
    let mut order: Vec<u32> = (0..neighbours.points().len() as u32).collect();
    Rng::new(seed).shuffle(&mut order);
    match neighbours {
        Neighbours::Euclidean { points, .. } => select_euclidean(points, &order, spacing),
        Neighbours::Surface(_) => select_surface(neighbours, &order, spacing),
    }
}

/// [`select_control_points`] in space: a hash grid of cell size `spacing`
/// limits each test to the 27 surrounding cells.
fn select_euclidean(points: &[V3], order: &[u32], spacing: f64) -> Vec<u32> {
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
    for &i in order {
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

/// [`select_control_points`] along the surface: an accepted node marks every
/// node within `spacing` of it as covered (one truncated search per control),
/// and only uncovered nodes are accepted.
fn select_surface(neighbours: &Neighbours, order: &[u32], spacing: f64) -> Vec<u32> {
    let mut covered = vec![false; neighbours.points().len()];
    let mut chosen = Vec::new();
    let (mut near, mut s) = (Vec::new(), neighbours.scratch());
    for &i in order {
        if covered[i as usize] {
            continue;
        }
        chosen.push(i);
        neighbours.within(i as usize, spacing, &mut near, &mut s);
        for &(v, _) in &near {
            covered[v as usize] = true;
        }
    }
    chosen
}
