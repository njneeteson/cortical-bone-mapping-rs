//! Curvature penalties on the fitted periosteal and endosteal surfaces (an
//! extension in this crate), and the surface Laplacians they use.

use rayon::prelude::*;

use super::SparseRows;
use crate::kdtree::KdTree;
use crate::surface::cotan::CotanLaplacian;
use crate::surface::mesh::Mesh;
use crate::util::{add, dot, scale, Rng, V3};

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
                sampled_row(cand, &g, k, i)
            })
            .collect();
        let mut rows = SparseRows::from_rows(rows, n);
        let diag = normalise_by_second_moment(&mut rows, points);
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

/// Row `i` of the Gaussian average from `k` neighbours drawn with
/// replacement from the candidates `cand` with probability ∝ `g`
/// (deterministic per row): weight `1/k` per draw, repeats merged.
fn sampled_row(cand: &[(u32, f64)], g: &[f64], k: usize, i: usize) -> Vec<(u32, f64)> {
    let mut cdf = Vec::with_capacity(g.len());
    let mut acc = 0.0;
    for w in g {
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
}

/// Scale each row of the averages to `4 / M2_i` (`M2_i = Σ w_ik |p_k −
/// p_i|²`); returns the diagonal `−4 / M2_i`.
fn normalise_by_second_moment(rows: &mut SparseRows<f64>, points: &[V3]) -> Vec<f64> {
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
    diag
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

    /// Node `i`'s residual `r` and its unit direction `u` (the residual
    /// vector is `r u`), given `q_i`; zero at boundary nodes and, in the
    /// excess mode, where the curvature is within `κ`.
    fn residual(&self, i: usize, q: V3) -> (f64, V3) {
        if !self.lap.interior[i] {
            return (0.0, [0.0; 3]);
        }
        let (nrm, c) = (self.normals[i], self.coef);
        match self.mode {
            CurvatureMode::Change => (c * dot(nrm, q), nrm),
            CurvatureMode::Excess { kappa } => {
                let h = dot(nrm, add(self.base[i], q));
                if h.abs() > kappa {
                    (c * (h.abs() - kappa), scale(nrm, h.signum()))
                } else {
                    (0.0, [0.0; 3])
                }
            }
        }
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
            let dir = &mut state.dir[s];
            let partial: Vec<f64> = e
                .par_iter_mut()
                .zip(dir.par_iter_mut())
                .enumerate()
                .map(|(i, (ei, di))| {
                    let (r, u) = self.residual(i, q[i]);
                    *ei = scale(u, r);
                    *di = u;
                    0.5 * r * r
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
