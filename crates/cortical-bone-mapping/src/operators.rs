//! Sparse linear operators used by the global solver:
//!
//! * [`SparseRows`]: a row-compressed sparse matrix (used for the multigrid
//!   interpolation matrix `A`).
//! * [`SmoothingOperator`]: the spatial regularisation matrix `ω` of
//!   Neeteson & Boyd (2025), `(ω x)_i = x_i − Σ_k g(d_ik) x_k / Σ_k g(d_ik)`
//!   over the k nearest neighbours.
//! * [`CurvatureOperator`]: curvature penalties on the fitted periosteal and
//!   endosteal surfaces (an extension in this crate).
//! * [`select_control_points`]: random sequential selection of control points
//!   with a minimum separation.
//!
//! Transposed products use one accumulation buffer per worker thread instead
//! of storing the transpose, which halves the memory of the large
//! neighbourhood matrices.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::kdtree::KdTree;
use crate::mesh::{CotanLaplacian, Mesh};
use crate::util::{add, dot, scale, Rng, V3};

/// Row-compressed sparse matrix with `u32` column indices.
#[derive(Clone, Debug)]
pub struct SparseRows<V> {
    pub offsets: Vec<usize>,
    pub cols: Vec<u32>,
    pub vals: Vec<V>,
    pub ncols: usize,
}

impl<V: Copy + Into<f64> + Send + Sync> SparseRows<V> {
    pub fn nrows(&self) -> usize {
        self.offsets.len() - 1
    }
    pub fn nnz(&self) -> usize {
        self.cols.len()
    }

    /// Assemble from per-row entry lists produced in parallel.
    pub fn from_rows(rows: Vec<Vec<(u32, V)>>, ncols: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        let mut total = 0;
        for r in &rows {
            total += r.len();
            offsets.push(total);
        }
        let mut cols = Vec::with_capacity(total);
        let mut vals = Vec::with_capacity(total);
        for r in rows {
            for (c, v) in r {
                cols.push(c);
                vals.push(v);
            }
        }
        SparseRows {
            offsets,
            cols,
            vals,
            ncols,
        }
    }

    /// `out = M x`.
    pub fn apply(&self, x: &[f64], out: &mut [f64]) {
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let mut s = 0.0;
            for e in self.offsets[i]..self.offsets[i + 1] {
                s += self.vals[e].into() * x[self.cols[e] as usize];
            }
            *o = s;
        });
    }

    /// `out = Mᵀ y` (scatter into one buffer per worker, then reduce).
    pub fn apply_t(&self, y: &[f64], out: &mut [f64]) {
        let n = self.nrows();
        // A fixed number of chunks (independent of the thread count) keeps
        // the result bitwise reproducible; memory is <= 16 buffers of ncols.
        let chunk = n.div_ceil(16).max(4096);
        let nchunks = n.div_ceil(chunk).max(1);
        if nchunks == 1 {
            out.iter_mut().for_each(|o| *o = 0.0);
            for i in 0..n {
                for e in self.offsets[i]..self.offsets[i + 1] {
                    out[self.cols[e] as usize] += self.vals[e].into() * y[i];
                }
            }
            return;
        }
        let parts: Vec<Vec<f64>> = (0..nchunks)
            .into_par_iter()
            .map(|c| {
                let mut acc = vec![0.0; self.ncols];
                for i in c * chunk..((c + 1) * chunk).min(n) {
                    let yi = y[i];
                    if yi == 0.0 {
                        continue;
                    }
                    for e in self.offsets[i]..self.offsets[i + 1] {
                        acc[self.cols[e] as usize] += self.vals[e].into() * yi;
                    }
                }
                acc
            })
            .collect();
        out.par_iter_mut().enumerate().for_each(|(k, o)| {
            *o = parts.iter().map(|p| p[k]).sum();
        });
    }

    /// `Σ_i M_ik² w_i` for every column `k`.
    pub fn col_sq_weighted(&self, w: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.ncols];
        for i in 0..self.nrows() {
            for e in self.offsets[i]..self.offsets[i + 1] {
                let v: f64 = self.vals[e].into();
                out[self.cols[e] as usize] += v * v * w[i];
            }
        }
        out
    }
}

/// Normalised Gaussian weights `g(d|σ) / Σ g` from squared distances,
/// computed relative to the nearest neighbour so they never underflow.
fn gaussian_weights(d2: &[f64], sigma: f64) -> Vec<f64> {
    let d0 = d2.iter().cloned().fold(f64::INFINITY, f64::min);
    let inv = 1.0 / (2.0 * sigma * sigma);
    let w: Vec<f64> = d2.iter().map(|&d| (-(d - d0) * inv).exp()).collect();
    let s: f64 = w.iter().sum();
    w.into_iter().map(|v| v / s).collect()
}

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

/// The spatial regularisation operator `ω` with the off-diagonal
/// weights stored as `f32`: `(ω x)_i = x_i − Σ_k w_ik x_k`, `Σ_k w_ik = 1`.
pub struct SmoothingOperator {
    pub rows: SparseRows<f32>,
}

impl SmoothingOperator {
    /// Build from the `k` nearest neighbours of each point (excluding the
    /// point itself), keeping only neighbours within `cutoff · σ` of the
    /// point but always at least the nearest one. `cutoff = ∞` reproduces the
    /// plain k-nearest-neighbour operator of the paper.
    ///
    /// Memory: 8 bytes per stored neighbour (`u32` index + `f32` weight). The
    /// matrix is filled in place at a fixed stride of `k` and then compacted,
    /// so construction never holds a second copy.
    pub fn new(points: &[V3], k: usize, sigma: f64, cutoff: f64) -> Self {
        let n = points.len();
        let tree = KdTree::new(points);
        let k = k.max(1).min(n.saturating_sub(1));
        let r2 = (cutoff * sigma).powi(2);
        if k == 0 {
            return SmoothingOperator {
                rows: SparseRows {
                    offsets: vec![0; n + 1],
                    cols: Vec::new(),
                    vals: Vec::new(),
                    ncols: n,
                },
            };
        }
        let mut cols = vec![0u32; n * k];
        let mut vals = vec![0f32; n * k];
        let mut counts = vec![0usize; n];
        cols.par_chunks_mut(k)
            .zip(vals.par_chunks_mut(k))
            .zip(counts.par_iter_mut())
            .enumerate()
            .for_each(|(i, ((c, v), cnt))| {
                let mut nn = tree.knn(points[i], k, Some(i as u32));
                let keep = nn.iter().filter(|x| x.1 <= r2).count().max(1);
                nn.truncate(keep);
                let d2: Vec<f64> = nn.iter().map(|x| x.1).collect();
                let w = gaussian_weights(&d2, sigma);
                for (e, (&(j, _), wj)) in nn.iter().zip(w).enumerate() {
                    c[e] = j;
                    v[e] = wj as f32;
                }
                *cnt = nn.len();
            });
        // compact in place
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let mut w = 0;
        for (i, &cnt) in counts.iter().enumerate() {
            cols.copy_within(i * k..i * k + cnt, w);
            vals.copy_within(i * k..i * k + cnt, w);
            w += cnt;
            offsets.push(w);
        }
        cols.truncate(w);
        vals.truncate(w);
        if w < (n * k) * 3 / 4 {
            cols.shrink_to_fit();
            vals.shrink_to_fit();
        }
        SmoothingOperator {
            rows: SparseRows {
                offsets,
                cols,
                vals,
                ncols: n,
            },
        }
    }

    pub fn apply(&self, x: &[f64], out: &mut [f64]) {
        self.rows.apply(x, out);
        out.par_iter_mut().zip(x.par_iter()).for_each(|(o, &xi)| *o = xi - *o);
    }

    pub fn apply_t(&self, y: &[f64], out: &mut [f64]) {
        self.rows.apply_t(y, out);
        out.par_iter_mut().zip(y.par_iter()).for_each(|(o, &yi)| *o = yi - *o);
    }

    /// Diagonal of `ωᵀω`: `1 + Σ_l w_li²` (the cross term `−2 w_ii` is zero
    /// since points are not their own neighbours).
    pub fn gram_diag(&self) -> Vec<f64> {
        let ones = vec![1.0; self.rows.nrows()];
        let mut d = self.rows.col_sq_weighted(&ones);
        d.iter_mut().for_each(|v| *v += 1.0);
        d
    }

    /// Bytes used by the operator.
    pub fn memory_bytes(&self) -> usize {
        self.rows.nnz() * (4 + 4) + self.rows.offsets.len() * 8
    }
}

/// How fitted-surface curvature is penalised.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CurvatureMode {
    /// Penalise curvature *added* by the fit: the mean curvature of the
    /// fitted surface `S = V + d n` in excess of that of the parallel
    /// surface of `V` at the local offset `d_i`. To first order,
    /// `H(S) − H(V_d) ≈ ½ Δ_S d`, computed as `n_i · ½ Σ_k L_ik (d_k − d_i) n_k`. A uniform offset (any parallel surface)
    /// costs nothing, so the natural shape of the bone and a uniform cortex
    /// are free and only bumps are resisted. Quadratic in `(m, t)`.
    Change,
    /// Penalise absolute mean curvature above a threshold:
    /// `max(0, |H(S)| − κ)²` with `H(S) ≈ n · ½ L S`, leaving curvature below
    /// `κ` (1/length) free.
    Excess { kappa: f64 },
}

/// A discrete surface Laplace–Beltrami operator in the form
/// `(L x)_i = diag_i x_i + Σ_k L_ik x_k` (off-diagonal entries row-wise),
/// such that `½ L S ≈ H n` (mean-curvature normal) for surface positions.
pub struct SurfaceLaplacian {
    pub rows: SparseRows<f64>,
    pub diag: Vec<f64>,
    /// Rows that are valid (e.g. not on the boundary of an open mesh).
    pub interior: Vec<bool>,
}

impl SurfaceLaplacian {
    /// Cotangent Laplacian of the mesh (one-ring, mesh resolution).
    pub fn cotangent(mesh: &Mesh) -> Self {
        let c = CotanLaplacian::new(mesh);
        let n = mesh.vertices.len();
        SurfaceLaplacian {
            rows: SparseRows {
                offsets: c.offsets,
                cols: c.cols,
                vals: c.vals,
                ncols: n,
            },
            diag: c.diag,
            interior: c.interior,
        }
    }

    /// Point-cloud Laplacian at physical scale `sigma`.
    ///
    /// The Gaussian-weighted neighbourhood average `Σ_k w_ik f_k` over all
    /// points within `cutoff · σ` is estimated by importance sampling `k`
    /// neighbours with probability ∝ `exp(−d²/2σ²)` (deterministic per node),
    /// so the operator acts at the requested physical scale whatever the node
    /// density, with `k` entries per row. (Taking the `k` *nearest*
    /// neighbours instead would shrink the scale to the radius they cover,
    /// e.g. 0.27 mm for k = 64 at 60 µm node spacing, and make the
    /// penalty stiff at the node scale.) Rows are normalised by the weighted
    /// second moment `M2_i = Σ w_ik |p_k − p_i|²` so that
    /// `(L f)_i = 4 (Σ w_ik f_k − f_i) / M2_i ≈ Δf` on a surface.
    pub fn gaussian(points: &[V3], k: usize, sigma: f64, cutoff: f64) -> Self {
        let n = points.len();
        let tree = KdTree::new(points);
        let k = k.max(1);
        let radius = cutoff * sigma;
        let inv = 1.0 / (2.0 * sigma * sigma);
        let rows: Vec<Vec<(u32, f64)>> = (0..n)
            .into_par_iter()
            .map_init(Vec::new, |cand, i| {
                tree.within(points[i], radius, cand);
                cand.retain(|c| c.0 as usize != i);
                if cand.is_empty() {
                    let nn = tree.knn(points[i], 1, Some(i as u32));
                    return nn.into_iter().map(|(j, _)| (j, 1.0)).collect();
                }
                let g: Vec<f64> = cand.iter().map(|c| (-c.1 * inv).exp()).collect();
                if cand.len() <= k {
                    let s: f64 = g.iter().sum();
                    return cand.iter().zip(&g).map(|(c, w)| (c.0, w / s)).collect();
                }
                // importance sampling with replacement from the Gaussian
                let mut cdf = Vec::with_capacity(g.len());
                let mut acc = 0.0;
                for w in &g {
                    acc += w;
                    cdf.push(acc);
                }
                let mut rng = Rng::new(0x5eed ^ (i as u64).wrapping_mul(0x9E37_79B9));
                let mut picks: Vec<u32> = (0..k)
                    .map(|_| {
                        let u = rng.uniform() * acc;
                        let j = cdf.partition_point(|&c| c < u).min(cand.len() - 1);
                        cand[j].0
                    })
                    .collect();
                picks.sort_unstable();
                let mut out: Vec<(u32, f64)> = Vec::new();
                for p in picks {
                    match out.last_mut() {
                        Some(last) if last.0 == p => last.1 += 1.0 / k as f64,
                        _ => out.push((p, 1.0 / k as f64)),
                    }
                }
                out
            })
            .collect();
        let mut rows = SparseRows::from_rows(rows, n);
        let mut diag = vec![0.0; points.len()];
        for i in 0..points.len() {
            let mut m2 = 0.0;
            for e in rows.offsets[i]..rows.offsets[i + 1] {
                m2 += rows.vals[e] * crate::util::dist2(points[i], points[rows.cols[e] as usize]);
            }
            let s = 4.0 / m2.max(1e-300);
            for e in rows.offsets[i]..rows.offsets[i + 1] {
                rows.vals[e] *= s;
            }
            diag[i] = -s;
        }
        SurfaceLaplacian {
            rows,
            diag,
            interior: vec![true; points.len()],
        }
    }

    /// `out_k = Σ_i L_ik e_i` over off-diagonal entries, for 3-vectors.
    fn apply_t3(&self, e: &[V3], out: &mut [V3]) {
        let mut tmp_in = vec![0.0; e.len()];
        let mut tmp_out = vec![0.0; out.len()];
        for c in 0..3 {
            tmp_in.par_iter_mut().zip(e.par_iter()).for_each(|(t, v)| *t = v[c]);
            self.rows.apply_t(&tmp_in, &mut tmp_out);
            out.par_iter_mut().zip(tmp_out.par_iter()).for_each(|(o, &v)| o[c] = v);
        }
    }
}

/// Curvature penalty on the fitted periosteal (`S = V + (m − t/2) n`) and/or
/// endosteal (`S = V + (m + t/2) n`) surfaces, using a fixed Laplacian of the
/// input surface `V`: `H(S) ≈ n · ½ L S` (the normal component, which is
/// robust to the tangential drift of a point-cloud Laplacian on irregular
/// sampling). Each node contributes one scalar residual per surface.
pub struct CurvatureOperator {
    pub lap: SurfaceLaplacian,
    pub normals: Vec<V3>,
    /// `½ L V` of the input surface.
    pub base: Vec<V3>,
    /// `u_i = Σ_{k≠i} L_ik n_k`.
    pub u: Vec<V3>,
    pub mode: CurvatureMode,
    /// Residual scale `c`: each residual is `c · (curvature measure)`.
    pub coef: f64,
    /// Penalise the periosteal / endosteal surface.
    pub surfaces: [bool; 2],
}

/// Per-row Gauss–Newton state from the last evaluation (unit curvature
/// directions for the excess mode, zero where inactive).
pub struct CurvatureState {
    pub dir: [Vec<V3>; 2],
}

impl CurvatureOperator {
    pub fn new(
        lap: SurfaceLaplacian,
        points: &[V3],
        normals: &[V3],
        mode: CurvatureMode,
        coef: f64,
        surfaces: [bool; 2],
    ) -> Self {
        let n = points.len();
        let r = &lap.rows;
        let gather = |x: &[V3], i: usize, diag: f64| -> V3 {
            let mut acc = scale(x[i], diag);
            for e in r.offsets[i]..r.offsets[i + 1] {
                acc = add(acc, scale(x[r.cols[e] as usize], r.vals[e]));
            }
            acc
        };
        let base = (0..n)
            .into_par_iter()
            .map(|i| scale(gather(points, i, lap.diag[i]), 0.5))
            .collect();
        let u = (0..n).into_par_iter().map(|i| gather(normals, i, 0.0)).collect();
        CurvatureOperator {
            lap,
            normals: normals.to_vec(),
            base,
            u,
            mode,
            coef,
            surfaces,
        }
    }

    pub fn n(&self) -> usize {
        self.normals.len()
    }

    fn relative(&self) -> bool {
        matches!(self.mode, CurvatureMode::Change)
    }

    /// `q_i = ½ (L (d n))_i` (excess mode) or `q_i = ½ Σ_k L_ik (d_k − d_i) n_k`
    /// (change mode).
    fn q(&self, d: &[f64], out: &mut [V3]) {
        let l = &self.lap;
        let r = &l.rows;
        let rel = self.relative();
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let mut acc = if rel {
                scale(self.u[i], -d[i])
            } else {
                scale(self.normals[i], l.diag[i] * d[i])
            };
            for e in r.offsets[i]..r.offsets[i + 1] {
                let k = r.cols[e] as usize;
                acc = add(acc, scale(self.normals[k], r.vals[e] * d[k]));
            }
            *o = scale(acc, 0.5);
        });
    }

    /// Adjoint of `c · q`.
    fn jt(&self, e: &[V3], out: &mut [f64]) {
        let l = &self.lap;
        let c = self.coef;
        let rel = self.relative();
        let mut lt = vec![[0.0; 3]; e.len()];
        l.apply_t3(e, &mut lt);
        out.par_iter_mut().enumerate().for_each(|(k, o)| {
            let diag = if rel {
                -dot(self.u[k], e[k])
            } else {
                l.diag[k] * dot(self.normals[k], e[k])
            };
            *o = 0.5 * c * (dot(self.normals[k], lt[k]) + diag);
        });
    }

    /// Cost and gradient with respect to `(m, t)`; updates `state`.
    /// Gradients are *added* to `gm`, `gt`.
    pub fn eval(&self, m: &[f64], t: &[f64], state: &mut CurvatureState, gm: &mut [f64], gt: &mut [f64]) -> f64 {
        let n = self.n();
        let mut cost = 0.0;
        let mut d = vec![0.0; n];
        let mut q = vec![[0.0; 3]; n];
        let mut e = vec![[0.0; 3]; n];
        let mut g = vec![0.0; n];
        for (s, sign) in [(0usize, -0.5f64), (1, 0.5)] {
            if !self.surfaces[s] {
                state.dir[s].iter_mut().for_each(|v| *v = [0.0; 3]);
                continue;
            }
            d.par_iter_mut().enumerate().for_each(|(i, v)| *v = m[i] + sign * t[i]);
            self.q(&d, &mut q);
            let c = self.coef;
            let dir = &mut state.dir[s];
            let partial: Vec<f64> = e
                .par_iter_mut()
                .zip(dir.par_iter_mut())
                .enumerate()
                .map(|(i, (ei, di))| {
                    *ei = [0.0; 3];
                    *di = [0.0; 3];
                    if !self.lap.interior[i] {
                        return 0.0;
                    }
                    match self.mode {
                        CurvatureMode::Change => {
                            let nrm = self.normals[i];
                            let r = c * dot(nrm, q[i]);
                            *ei = scale(nrm, r);
                            *di = nrm;
                            0.5 * r * r
                        }
                        CurvatureMode::Excess { kappa } => {
                            let nrm = self.normals[i];
                            let h = dot(nrm, add(self.base[i], q[i]));
                            if h.abs() > kappa {
                                let u = scale(nrm, h.signum());
                                let r = c * (h.abs() - kappa);
                                *ei = scale(u, r);
                                *di = u;
                                0.5 * r * r
                            } else {
                                0.0
                            }
                        }
                    }
                })
                .collect();
            cost += partial.iter().sum::<f64>();
            self.jt(&e, &mut g);
            gm.par_iter_mut()
                .zip(gt.par_iter_mut())
                .zip(g.par_iter())
                .for_each(|((a, b), &gi)| {
                    *a += gi;
                    *b += sign * gi;
                });
        }
        cost
    }

    pub fn new_state(&self) -> CurvatureState {
        CurvatureState {
            dir: [vec![[0.0; 3]; self.n()], vec![[0.0; 3]; self.n()]],
        }
    }

    /// Gauss–Newton product: adds `Jᵀ J (δm, δt)` to `(ym, yt)`.
    pub fn gn_product(&self, state: &CurvatureState, dm: &[f64], dt: &[f64], ym: &mut [f64], yt: &mut [f64]) {
        let n = self.n();
        let mut d = vec![0.0; n];
        let mut q = vec![[0.0; 3]; n];
        let mut e = vec![[0.0; 3]; n];
        let mut g = vec![0.0; n];
        for (s, sign) in [(0usize, -0.5f64), (1, 0.5)] {
            if !self.surfaces[s] {
                continue;
            }
            d.par_iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = dm[i] + sign * dt[i]);
            self.q(&d, &mut q);
            let c = self.coef;
            let dir = &state.dir[s];
            e.par_iter_mut().enumerate().for_each(|(i, ei)| {
                *ei = if !self.lap.interior[i] {
                    [0.0; 3]
                } else {
                    // both modes: one scalar residual along dir (n or ±n)
                    scale(dir[i], c * dot(dir[i], q[i]))
                };
            });
            self.jt(&e, &mut g);
            ym.par_iter_mut()
                .zip(yt.par_iter_mut())
                .zip(g.par_iter())
                .for_each(|((a, b), &gi)| {
                    *a += gi;
                    *b += sign * gi;
                });
        }
    }

    /// Upper bound of the diagonal of `JᵀJ` with respect to `d` for one
    /// surface.
    pub fn gram_diag(&self) -> Vec<f64> {
        let l = &self.lap;
        let c2 = self.coef * self.coef * 0.25;
        let w: Vec<f64> = l.interior.iter().map(|&b| b as u8 as f64).collect();
        let col = l.rows.col_sq_weighted(&w);
        let rel = self.relative();
        (0..self.n())
            .map(|k| {
                let own = if !l.interior[k] {
                    0.0
                } else if rel {
                    dot(self.u[k], self.u[k])
                } else {
                    l.diag[k].powi(2)
                };
                c2 * (own + col[k])
            })
            .collect()
    }

    /// Mean curvature magnitude `|n · ½ L S|` of the surface `V + d n`.
    pub fn mean_curvature(&self, d: &[f64]) -> Vec<f64> {
        let l = &self.lap;
        let r = &l.rows;
        (0..self.n())
            .into_par_iter()
            .map(|i| {
                let mut acc = scale(self.normals[i], l.diag[i] * d[i]);
                for e in r.offsets[i]..r.offsets[i + 1] {
                    let k = r.cols[e] as usize;
                    acc = add(acc, scale(self.normals[k], r.vals[e] * d[k]));
                }
                dot(self.normals[i], add(self.base[i], scale(acc, 0.5))).abs()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_points(n: usize, h: f64) -> Vec<V3> {
        let mut v = Vec::new();
        for i in 0..n {
            for j in 0..n {
                v.push([i as f64 * h, j as f64 * h, 0.0]);
            }
        }
        v
    }

    #[test]
    fn sparse_transpose_matches_dense() {
        let pts = grid_points(90, 0.1); // 8100 rows, multiple chunks
        let ctrl_idx = select_control_points(&pts, 0.5, 3);
        let ctrl: Vec<V3> = ctrl_idx.iter().map(|&i| pts[i as usize]).collect();
        let a = interpolation_matrix(&pts, &ctrl, 5, 0.5);
        let mut rng = Rng::new(1);
        let x: Vec<f64> = (0..ctrl.len()).map(|_| rng.normal()).collect();
        let y: Vec<f64> = (0..pts.len()).map(|_| rng.normal()).collect();
        let mut ax = vec![0.0; pts.len()];
        let mut aty = vec![0.0; ctrl.len()];
        a.apply(&x, &mut ax);
        a.apply_t(&y, &mut aty);
        let lhs: f64 = ax.iter().zip(&y).map(|(a, b)| a * b).sum();
        let rhs: f64 = aty.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0));
        // rows are convex combinations
        for i in 0..a.nrows() {
            let s: f64 = (a.offsets[i]..a.offsets[i + 1]).map(|e| a.vals[e]).sum();
            assert!((s - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn control_points_cover_and_separate() {
        let pts = grid_points(40, 0.1);
        let s = 0.45;
        let c = select_control_points(&pts, s, 9);
        for (a, &i) in c.iter().enumerate() {
            for &j in &c[a + 1..] {
                assert!(crate::util::dist2(pts[i as usize], pts[j as usize]) > s * s);
            }
        }
        for p in &pts {
            assert!(c.iter().any(|&i| crate::util::dist2(pts[i as usize], *p) <= s * s));
        }
        assert_eq!(c, select_control_points(&pts, s, 9));
    }

    #[test]
    fn change_mode_ignores_uniform_offsets_but_not_bumps() {
        let mesh = Mesh::uv_sphere(3.0, 48, 32);
        let normals: Vec<V3> = mesh.vertex_normals().iter().map(|v| scale(*v, -1.0)).collect();
        let op = CurvatureOperator::new(
            SurfaceLaplacian::cotangent(&mesh),
            &mesh.vertices,
            &normals,
            CurvatureMode::Change,
            1.0,
            [true, false],
        );
        let n = mesh.vertices.len();
        let mut st = op.new_state();
        let (mut gm, mut gt) = (vec![0.0; n], vec![0.0; n]);
        let flat = op.eval(&vec![0.7; n], &vec![0.0; n], &mut st, &mut gm, &mut gt);
        assert!(flat < 1e-20, "uniform offset cost {flat}");
        let bumpy: Vec<f64> = mesh.vertices.iter().map(|v| 0.7 + 0.1 * (4.0 * v[0]).sin()).collect();
        let c = op.eval(&bumpy, &vec![0.0; n], &mut st, &mut gm, &mut gt);
        assert!(c > 1e-3);
        // the excess mode sees the curvature 1/R of the sphere itself
        for lap in [
            SurfaceLaplacian::cotangent(&mesh),
            SurfaceLaplacian::gaussian(&mesh.vertices, 40, 0.4, 3.0),
        ] {
            let ex = CurvatureOperator::new(
                lap,
                &mesh.vertices,
                &normals,
                CurvatureMode::Excess { kappa: 0.2 },
                1.0,
                [true, false],
            );
            let h = ex.mean_curvature(&vec![0.0; n]);
            let mid = h
                .iter()
                .zip(&mesh.vertices)
                .filter(|(_, v)| v[2].abs() < 2.0)
                .map(|(h, _)| *h);
            for hv in mid {
                assert!((hv * 3.0 - 1.0).abs() < 0.08, "H*R = {}", hv * 3.0);
            }
        }
        // the Gaussian change operator also ignores uniform offsets
        let g = CurvatureOperator::new(
            SurfaceLaplacian::gaussian(&mesh.vertices, 40, 0.4, 3.0),
            &mesh.vertices,
            &normals,
            CurvatureMode::Change,
            1.0,
            [true, false],
        );
        assert!(g.eval(&vec![0.7; n], &vec![0.0; n], &mut st, &mut gm, &mut gt) < 1e-20);
    }

    #[test]
    fn smoothing_operator_annihilates_constants() {
        let pts = grid_points(20, 0.1);
        let w = SmoothingOperator::new(&pts, 12, 0.2, f64::INFINITY);
        let mut out = vec![0.0; pts.len()];
        w.apply(&vec![3.0; pts.len()], &mut out);
        assert!(out.iter().all(|v| v.abs() < 1e-6));
        // transposed product is the adjoint
        let mut rng = Rng::new(5);
        let x: Vec<f64> = (0..pts.len()).map(|_| rng.normal()).collect();
        let y: Vec<f64> = (0..pts.len()).map(|_| rng.normal()).collect();
        let mut wx = vec![0.0; pts.len()];
        let mut wty = vec![0.0; pts.len()];
        w.apply(&x, &mut wx);
        w.apply_t(&y, &mut wty);
        let l: f64 = wx.iter().zip(&y).map(|(a, b)| a * b).sum();
        let r: f64 = wty.iter().zip(&x).map(|(a, b)| a * b).sum();
        assert!((l - r).abs() < 1e-5 * l.abs().max(1.0));
        // radius cut-off keeps at least one neighbour
        let w2 = SmoothingOperator::new(&pts, 12, 0.01, 1.0);
        assert!((0..pts.len()).all(|i| w2.rows.offsets[i + 1] > w2.rows.offsets[i]));
    }
}
