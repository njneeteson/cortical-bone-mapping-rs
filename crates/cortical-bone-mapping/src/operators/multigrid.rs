//! Multigrid control points and the interpolation matrix `A` from control
//! points to surface nodes.

use std::collections::HashMap;

use rayon::prelude::*;

use super::{gaussian_weights, SparseRows};
use crate::kdtree::KdTree;
use crate::util::{Rng, V3};

/// The multigrid interpolation matrix `A` (N × Q): each point takes a
/// Gaussian-weighted (σ = `spacing`) average of its `k` nearest control
/// points.
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

/// Random sequential selection of control points such that no two are
/// closer than `spacing` and every point is within `spacing` of a control
/// point. Returns indices into `points`. Deterministic for a seed.
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
