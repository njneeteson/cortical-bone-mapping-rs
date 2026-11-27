//! Curvature penalties on the fitted periosteal and endosteal surfaces (an
//! extension in this crate), and the surface Laplacians they use.
//!
//! The fitted surfaces are `S = V + d n` with `d = m − t/2` (periosteal) or
//! `d = m + t/2` (endosteal), `V` the node positions and `n` the (inward)
//! sampling normals. For a discrete Laplace–Beltrami operator `L` of `V`, the
//! mean-curvature normal is `½ ΔS = H n_S` (Meyer et al. 2003), so the scalar
//! mean curvature of `S` is approximated by its normal component
//!
//! ```text
//! H_i(S) ≈ n_i · ½ (L S)_i = n_i · ½ [(L V)_i + Σ_k L_ik d_k n_k + L_ii d_i n_i]
//! ```
//!
//! using the fixed operator and normals of the input surface (a first-order
//! approximation that keeps the penalty linear in `(m, t)`). Taking the normal
//! component discards the tangential drift that point-cloud Laplacians show on
//! irregular sampling.
//!
//! Modes ([`CurvatureMode`]):
//!
//! * **change**: the curvature *added by the fit*, in excess of the parallel
//!   surface `V_d` at the local offset. A uniform offset moves every point of a
//!   surface to its parallel surface, whose curvature changes in a way fixed by
//!   the geometry of `V`; to first order the difference is
//!   `H(V + d n) − H(V_d) ≈ ½ Δ_S d`, computed as
//!   `n_i · ½ Σ_k L_ik (d_k − d_i) n_k`. A uniform offset costs nothing, so the
//!   shape of the bone and a uniform cortex are free and only bumps are
//!   resisted. (Penalising `H(S) − H(V)` instead biases thickness: the
//!   endosteal surface of a cylinder is *meant* to be more curved than the
//!   periosteal one.) With `q_i = ½ Σ_k L_ik (d_k − d_i) n_k`, the residual is
//!   `r_i = c · n_i · q_i` (`c` the residual scale of
//!   [`crate::fit::regularized`]), linear in `(m, t)`.
//!   Kept for comparison: its reference is the surface being fitted, so over
//!   a refinement, where that surface is the previous fit's, curvature added
//!   in one fit is free in the next and dents can deepen without bound.
//! * **absolute**: the curvature of the fitted surface itself above a
//!   threshold, `r_i = c · max(0, K_i(S) − κ)`, leaving curvature below `κ`
//!   free (`κ = 0`: a bending energy, which prefers a flat surface). The
//!   measure `K` ([`CurvatureMeasure`]) is `|κ₁ + κ₂| = 2|H|` (mean) or
//!   `√(κ₁² + κ₂²)` (total, [`super::hessian`]), which unlike `H` is not zero
//!   at a saddle. Both equal the curvature `1/R` of the contour of a surface
//!   that is straight in one direction (a shaft), so `κ` is the curvature of
//!   the sharpest bend left free, as in a slice: our HR-pQCT segmentation
//!   network penalised contour curvature above 5 mm⁻¹ (0.005 µm⁻¹, from the
//!   surfaces of real radius and tibia segmentations; Neeteson et al. 2023),
//!   the default. The
//!   squared hinge is continuous with a continuous slope at `κ`, so the
//!   Gauss–Newton solver sees no jump as nodes cross the threshold; the
//!   active set and the direction are re-evaluated at every linearisation.
//!   This penalty does not depend on the surface the fit starts from, so a
//!   refinement converges to one balance of data and smoothness.
//!
//! Laplacians ([`SurfaceLaplacian`]):
//!
//! * **cotangent** (Pinkall & Polthier 1993; Meyer et al. 2003): curvature at
//!   mesh resolution ([`crate::surface::cotan`]);
//! * **Gaussian point cloud** at physical scale `σ` (after Belkin et al. 2009,
//!   ours in the details): `(L f)_i = 4 (Σ_k w_ik f_k − f_i) / M2_i` with
//!   `w_ik ∝ exp(−d_ik²/2σ²)` over the points within `cutoff · σ` and
//!   `M2_i = Σ_k w_ik d_ik²`. Justification: on a locally flat surface a
//!   second-order Taylor expansion gives
//!   `Σ_k w_ik f_k − f_i ≈ ∇f · Σ w d + ½ Σ_k w_ik d_kᵀ ∇²f d_k`; for an
//!   isotropic neighbourhood the first moment vanishes and the second is
//!   `(M2/2) I` in the tangent plane, so the average minus the value is
//!   `(M2/4) Δf`. Normalising by the actual `M2_i` makes this hold for any
//!   truncation and sampling. When there are more than `k` candidates, the
//!   weighted mean is estimated by importance sampling `k` neighbours with
//!   probability `∝ w_ik` (each draw weighing `1/k`), deterministically per
//!   node, so every row has at most `k` entries and the operator keeps its
//!   physical scale whatever the node density (taking the `k` nearest
//!   instead would shrink the scale to the radius they cover).
//!
//! # References
//!
//! * N. J. Neeteson, B. A. Besler, D. E. Whittier, S. K. Boyd, *Automatic
//!   segmentation of trabecular and cortical compartments in HR-pQCT images
//!   using an embedding-predicting U-Net and morphological post-processing*,
//!   Scientific Reports 13:252, 2023, doi:10.1038/s41598-022-27350-0.
//! * U. Pinkall, K. Polthier, *Computing discrete minimal surfaces and their
//!   conjugates*, Experimental Mathematics 2(1):15–36, 1993.
//! * M. Meyer, M. Desbrun, P. Schröder, A. H. Barr, *Discrete
//!   differential-geometry operators for triangulated 2-manifolds*, in
//!   H.-C. Hege, K. Polthier (eds.), Visualization and Mathematics III,
//!   Springer, 2003, pp. 35–57.
//! * M. Belkin, J. Sun, Y. Wang, *Constructing Laplace operator from point
//!   clouds in Rᵈ*, Proceedings of the 20th ACM-SIAM Symposium on Discrete
//!   Algorithms (SODA), 2009, pp. 1031–1040.

use rayon::prelude::*;

use super::hessian::HeightHessian;
use super::neighbours::Neighbours;
use super::variation::CurvatureGradient;
use super::SparseRows;
use crate::surface::cotan::CotanLaplacian;
use crate::surface::mesh::Mesh;
use crate::util::{add, dot, norm, scale, Rng, V3};

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
    /// Penalise the curvature of the fitted surface above a threshold:
    /// `max(0, K(S) − κ)²`, `K` the [`CurvatureMeasure`], leaving curvature
    /// below `κ` (1/length) free.
    Absolute { kappa: f64 },
    /// Penalise how fast the curvature `κ₁ + κ₂` changes along the fitted
    /// surface, `½ ∫ |∇K|²` ([`CurvatureGradient`]): smoothly curved bone is
    /// free, bumps and dents are not. Mean curvature only.
    Variation,
}

/// Which curvature the absolute penalty measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurvatureMeasure {
    /// `|κ₁ + κ₂| = 2|H|` (from the surface Laplacian).
    Mean,
    /// `√(κ₁² + κ₂²)` (from a local quadric fit, [`HeightHessian`]): not
    /// zero at a saddle, where `H` is.
    Total,
}

/// The smallest second moment of a Gaussian Laplacian's row, relative to that
/// of a flat, isotropic neighbourhood at its scale (`2σ²`), for the row to
/// measure curvature: below it the neighbourhood (e.g. a few nearly
/// coincident nodes of a small fragment) samples less than half the scale,
/// and the row's entries (`∝ 1/M2`) would measure curvature at a far finer
/// scale, with gradients no representable step can follow.
const MIN_SECOND_MOMENT: f64 = 0.25;

/// A discrete surface Laplace–Beltrami operator in the form
/// `(L x)_i = diag_i x_i + Σ_k L_ik x_k` (off-diagonal entries row-wise),
/// such that `½ L S ≈ H n` (mean-curvature normal) for surface positions.
pub struct SurfaceLaplacian {
    pub rows: SparseRows<f64>,
    pub diag: Vec<f64>,
    /// Rows that are valid (e.g. not on the boundary of an open mesh).
    pub interior: Vec<bool>,
    /// The off-diagonal entries are proportional to neighbourhood weights
    /// (Gaussian), not geometric coefficients that may vanish or be negative
    /// (cotangent).
    pub weighted: bool,
}

impl SurfaceLaplacian {
    /// Cotangent Laplacian of the mesh (one-ring, mesh resolution; Pinkall &
    /// Polthier 1993, Meyer et al. 2003).
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
            weighted: false,
        }
    }

    /// Point-cloud Laplacian at physical scale `sigma` (module documentation):
    /// `(L f)_i = 4 (Σ_k w_ik f_k − f_i) / M2_i`, with the Gaussian-weighted
    /// neighbourhood average over the points within `cutoff · σ` taken exactly
    /// when there are at most `k` of them and estimated by importance sampling `k`
    /// of them otherwise (`sampled_row`). Isolated points use their nearest
    /// neighbour.
    pub fn gaussian(neighbours: &Neighbours, k: usize, sigma: f64, cutoff: f64) -> Self {
        let points = neighbours.points();
        let n = points.len();
        let k = k.max(1);
        let radius = cutoff * sigma;
        let inv = 1.0 / (2.0 * sigma * sigma);
        let rows: Vec<Vec<(u32, f64)>> = (0..n)
            .into_par_iter()
            .map_init(
                || (Vec::new(), neighbours.scratch()),
                |(cand, s), i| {
                    neighbours.within(i, radius, cand, s);
                    cand.retain(|c| c.0 as usize != i);
                    if cand.is_empty() {
                        let nn = neighbours.nearest(i, 1, s);
                        return nn.into_iter().map(|(j, _)| (j, 1.0)).collect();
                    }
                    let g: Vec<f64> = cand.iter().map(|c| (-c.1 * inv).exp()).collect();
                    if cand.len() <= k {
                        let s: f64 = g.iter().sum();
                        return cand.iter().zip(&g).map(|(c, w)| (c.0, w / s)).collect();
                    }
                    sampled_row(cand, &g, k, i)
                },
            )
            .collect();
        let mut rows = SparseRows::from_rows(rows, n);
        let diag = normalise_by_second_moment(&mut rows, points);
        // a row's second moment relative to that of a flat, isotropic
        // neighbourhood at σ (2σ²): about 1 for a sound row, ½ at an open edge
        let rel: Vec<f64> = diag.iter().map(|&d| 2.0 / (-d * sigma * sigma)).collect();
        let interior: Vec<bool> = rel.iter().map(|&r| r >= MIN_SECOND_MOMENT).collect();
        let dropped = interior.iter().filter(|&&a| !a).count();
        if dropped > 0 {
            let lo = rel.iter().copied().fold(f64::INFINITY, f64::min);
            log::debug!(
                "gaussian Laplacian at σ {sigma:.4}: {dropped} of {n} rows sample less than the scale (second \
                 moment below {MIN_SECOND_MOMENT} × 2σ², smallest {lo:.3e}); they have no curvature residual"
            );
        }
        SurfaceLaplacian {
            rows,
            diag,
            interior,
            weighted: true,
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

/// The curvature measure at every vertex of `mesh` (`2|H|` or `√(κ₁² +
/// κ₂²)`, 1/length), with the Laplacian of a curvature penalty: Gaussian
/// at `sigma` over `neighbours` points with `distance`, or cotangent (`None`).
/// For choosing the threshold `κ` from reference surfaces.
pub fn mesh_curvature(
    mesh: &Mesh,
    measure: CurvatureMeasure,
    sigma: Option<f64>,
    neighbours: usize,
    distance: super::Distance,
) -> Vec<f64> {
    let lap = match sigma {
        Some(s) => SurfaceLaplacian::gaussian(&Neighbours::new(mesh, distance), neighbours, s, 3.0),
        None => SurfaceLaplacian::cotangent(mesh),
    };
    let normals: Vec<V3> = mesh.vertex_normals().iter().map(|v| scale(*v, -1.0)).collect();
    let mode = CurvatureMode::Absolute { kappa: 0.0 };
    let op = CurvatureOperator::new(lap, &mesh.vertices, &normals, mode, measure, [1.0, 0.0]);
    op.curvature(&vec![0.0; mesh.vertices.len()])
}

/// `gm += g`, `gt += sign · g` (`d = m + sign · t`).
fn add_to(gm: &mut [f64], gt: &mut [f64], g: &[f64], sign: f64) {
    gm.par_iter_mut()
        .zip(gt.par_iter_mut())
        .zip(g.par_iter())
        .for_each(|((a, b), &gi)| {
            *a += gi;
            *b += sign * gi;
        });
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
/// endosteal (`S = V + (m + t/2) n`) surfaces, using fixed operators of the
/// input surface `V`. Each node contributes one scalar residual per surface,
/// `r_i = c · (measure)`, through a per-node curvature vector that is linear
/// in the offsets `d`: `½ L S` (the mean-curvature normal; its normal
/// component, `H`, is robust to the tangential drift of a point-cloud
/// Laplacian on irregular sampling) or the second fundamental form `w`
/// ([`HeightHessian`]).
pub struct CurvatureOperator {
    pub lap: SurfaceLaplacian,
    pub normals: Vec<V3>,
    /// The curvature vector of the input surface: `½ L V`, or `B` (total).
    pub base: Vec<V3>,
    /// `u_i = Σ_{k≠i} L_ik n_k` (change mode).
    pub u: Vec<V3>,
    /// The second fundamental form operator (total curvature only).
    hessian: Option<HeightHessian>,
    /// The gradient operator of the curvature (variation mode only).
    variation: Option<CurvatureGradient>,
    pub mode: CurvatureMode,
    pub measure: CurvatureMeasure,
    /// Residual scales `c` of the periosteal and endosteal surfaces: each
    /// residual is `c · (curvature measure)`; 0 = that surface is not
    /// penalised.
    pub coef: [f64; 2],
}

/// Per-row Gauss–Newton state from the last evaluation: the derivative of
/// each residual's measure with respect to the node's curvature vector (zero
/// where inactive).
pub struct CurvatureState {
    pub dir: [Vec<V3>; 2],
}

impl CurvatureOperator {
    /// Precompute, for the input surface `V` with normals `n`, the curvature
    /// vector of `V` and the operators that give its change for any offsets
    /// `d` in one pass over the rows. The change mode always measures mean
    /// curvature.
    pub fn new(
        lap: SurfaceLaplacian,
        points: &[V3],
        normals: &[V3],
        mode: CurvatureMode,
        measure: CurvatureMeasure,
        coef: [f64; 2],
    ) -> Self {
        let n = points.len();
        log::debug!("curvature operator: {n} nodes, mode {mode:?}, measure {measure:?}, weights {coef:?}");
        let r = &lap.rows;
        let gather = |x: &[V3], i: usize, diag: f64| -> V3 {
            let mut acc = scale(x[i], diag);
            for e in r.offsets[i]..r.offsets[i + 1] {
                acc = add(acc, scale(x[r.cols[e] as usize], r.vals[e]));
            }
            acc
        };
        let measure = match mode {
            CurvatureMode::Absolute { .. } => measure,
            CurvatureMode::Change | CurvatureMode::Variation => CurvatureMeasure::Mean,
        };
        let variation = (mode == CurvatureMode::Variation).then(|| CurvatureGradient::new(&lap));
        let hessian = (measure == CurvatureMeasure::Total).then(|| HeightHessian::new(&lap, points, normals));
        let base = match &hessian {
            Some(h) => h.base.clone(),
            None => (0..n)
                .into_par_iter()
                .map(|i| scale(gather(points, i, lap.diag[i]), 0.5))
                .collect(),
        };
        let u = match mode {
            CurvatureMode::Change => (0..n).into_par_iter().map(|i| gather(normals, i, 0.0)).collect(),
            CurvatureMode::Absolute { .. } | CurvatureMode::Variation => Vec::new(),
        };
        CurvatureOperator {
            lap,
            normals: normals.to_vec(),
            base,
            u,
            hessian,
            variation,
            mode,
            measure,
            coef,
        }
    }

    /// Number of nodes.
    pub fn n(&self) -> usize {
        self.normals.len()
    }

    /// The change mode, which measures curvature relative to the parallel surface.
    fn relative(&self) -> bool {
        matches!(self.mode, CurvatureMode::Change)
    }

    /// Whether node `i` has a residual: an interior Laplacian row and, for
    /// the total curvature, a well-posed quadric fit.
    fn active_row(&self, i: usize) -> bool {
        self.lap.interior[i] && self.hessian.as_ref().is_none_or(|h| h.valid[i])
    }

    /// The change of the curvature vector for offsets `d`: `q_i = ½ (L (d
    /// n))_i` (mean), `q_i = ½ Σ_k L_ik (d_k − d_i) n_k` (change mode) or the
    /// linear part of `w_i` (total).
    fn q(&self, d: &[f64], out: &mut [V3]) {
        if let Some(h) = &self.hessian {
            h.apply(d, out);
            return;
        }
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
    fn jt(&self, c: f64, e: &[V3], out: &mut [f64]) {
        if let Some(h) = &self.hessian {
            h.apply_t(e, out);
            out.par_iter_mut().for_each(|o| *o *= c);
            return;
        }
        let l = &self.lap;
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

    /// The curvature measure of node `i` from its full curvature vector `w`,
    /// and the derivative of the measure with respect to `w`.
    fn measure_of(&self, i: usize, w: V3) -> (f64, V3) {
        match self.measure {
            CurvatureMeasure::Mean => {
                // w = ½ L S = H n_S
                let h = dot(self.normals[i], w);
                (2.0 * h.abs(), scale(self.normals[i], 2.0 * h.signum()))
            }
            CurvatureMeasure::Total => {
                // |w| = ‖II‖_F
                let len = norm(w);
                if len > 0.0 {
                    (len, scale(w, 1.0 / len))
                } else {
                    (0.0, [0.0; 3])
                }
            }
        }
    }

    /// Node `i`'s residual `r` and the derivative `u` of its measure with
    /// respect to the curvature vector (the residual vector is `r u`), given
    /// `q_i`; zero at inactive rows and, in the absolute mode, where the
    /// curvature is within `κ`.
    fn residual(&self, c: f64, i: usize, q: V3) -> (f64, V3) {
        if !self.active_row(i) {
            return (0.0, [0.0; 3]);
        }
        match self.mode {
            CurvatureMode::Change => {
                let nrm = self.normals[i];
                (c * dot(nrm, q), nrm)
            }
            CurvatureMode::Absolute { kappa } => {
                let (k, u) = self.measure_of(i, add(self.base[i], q));
                if k > kappa {
                    (c * (k - kappa), u)
                } else {
                    (0.0, [0.0; 3])
                }
            }
            // not a per-node residual (`eval_variation`)
            CurvatureMode::Variation => (0.0, [0.0; 3]),
        }
    }

    /// The signed curvature `K_i = κ₁ + κ₂ = 2 n_i · ½ (L S)_i` from the
    /// change `q` of the curvature vector, with the input surface's part
    /// (`with_base`) or without it (for a change of the offsets).
    fn signed(&self, q: &[V3], with_base: bool) -> Vec<f64> {
        q.par_iter()
            .enumerate()
            .map(|(i, qi)| {
                let w = if with_base { add(self.base[i], *qi) } else { *qi };
                2.0 * dot(self.normals[i], w)
            })
            .collect()
    }

    /// `out = Jᵀ y` for a field `y` on the signed curvatures `K`.
    fn signed_adjoint(&self, y: &[f64], out: &mut [f64]) {
        let e: Vec<V3> = y
            .par_iter()
            .zip(&self.normals)
            .map(|(&v, n)| scale(*n, 2.0 * v))
            .collect();
        self.jt(1.0, &e, out);
    }

    /// The variation penalty: the cost `½ c² Kᵀ G K` of each surface, its
    /// gradient added to `gm`, `gt`.
    fn eval_variation(&self, g_op: &CurvatureGradient, m: &[f64], t: &[f64], gm: &mut [f64], gt: &mut [f64]) -> f64 {
        let n = self.n();
        let mut cost = 0.0;
        let (mut q, mut gk, mut g) = (vec![[0.0; 3]; n], vec![0.0; n], vec![0.0; n]);
        for (s, sign) in [(0usize, -0.5f64), (1, 0.5)] {
            let c2 = self.coef[s] * self.coef[s];
            if c2 == 0.0 {
                continue;
            }
            let d: Vec<f64> = (0..n).map(|i| m[i] + sign * t[i]).collect();
            self.q(&d, &mut q);
            let k = self.signed(&q, true);
            g_op.apply(&k, &mut gk);
            cost += 0.5 * c2 * crate::util::det_sum(n, |i| k[i] * gk[i]);
            gk.par_iter_mut().for_each(|v| *v *= c2);
            self.signed_adjoint(&gk, &mut g);
            add_to(gm, gt, &g, sign);
        }
        cost
    }

    /// The variation penalty's Gauss–Newton product (exact: it is quadratic).
    fn gn_variation(&self, g_op: &CurvatureGradient, dm: &[f64], dt: &[f64], ym: &mut [f64], yt: &mut [f64]) {
        let n = self.n();
        let (mut q, mut gk, mut g) = (vec![[0.0; 3]; n], vec![0.0; n], vec![0.0; n]);
        for (s, sign) in [(0usize, -0.5f64), (1, 0.5)] {
            let c2 = self.coef[s] * self.coef[s];
            if c2 == 0.0 {
                continue;
            }
            let d: Vec<f64> = (0..n).map(|i| dm[i] + sign * dt[i]).collect();
            self.q(&d, &mut q);
            let k = self.signed(&q, false);
            g_op.apply(&k, &mut gk);
            gk.par_iter_mut().for_each(|v| *v *= c2);
            self.signed_adjoint(&gk, &mut g);
            add_to(ym, yt, &g, sign);
        }
    }

    /// Cost `½ Σ_surfaces Σ_i r_i²` and its gradient with respect to `(m, t)`
    /// (`d = m ∓ t/2`, so `∂/∂m = ∂/∂d` and `∂/∂t = ∓½ ∂/∂d`), via the adjoint of
    /// `q`; records each row's residual direction in `state` for the Gauss–Newton
    /// products. Gradients are *added* to `gm`, `gt`.
    pub fn eval(&self, m: &[f64], t: &[f64], state: &mut CurvatureState, gm: &mut [f64], gt: &mut [f64]) -> f64 {
        if let Some(g_op) = &self.variation {
            return self.eval_variation(g_op, m, t, gm, gt);
        }
        let n = self.n();
        let mut cost = 0.0;
        let mut d = vec![0.0; n];
        let mut q = vec![[0.0; 3]; n];
        let mut e = vec![[0.0; 3]; n];
        let mut g = vec![0.0; n];
        for (s, sign) in [(0usize, -0.5f64), (1, 0.5)] {
            let c = self.coef[s];
            if c == 0.0 {
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
                    let (r, u) = self.residual(c, i, q[i]);
                    *ei = scale(u, r);
                    *di = u;
                    0.5 * r * r
                })
                .collect();
            cost += partial.iter().sum::<f64>();
            self.jt(c, &e, &mut g);
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

    /// An empty Gauss–Newton state (no active rows).
    pub fn new_state(&self) -> CurvatureState {
        CurvatureState {
            dir: [vec![[0.0; 3]; self.n()], vec![[0.0; 3]; self.n()]],
        }
    }

    /// Gauss–Newton product: adds `Jᵀ J (δm, δt)` to `(ym, yt)`.
    pub fn gn_product(&self, state: &CurvatureState, dm: &[f64], dt: &[f64], ym: &mut [f64], yt: &mut [f64]) {
        if let Some(g_op) = &self.variation {
            return self.gn_variation(g_op, dm, dt, ym, yt);
        }
        let n = self.n();
        let mut d = vec![0.0; n];
        let mut q = vec![[0.0; 3]; n];
        let mut e = vec![[0.0; 3]; n];
        let mut g = vec![0.0; n];
        for (s, sign) in [(0usize, -0.5f64), (1, 0.5)] {
            let c = self.coef[s];
            if c == 0.0 {
                continue;
            }
            d.par_iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = dm[i] + sign * dt[i]);
            self.q(&d, &mut q);
            let dir = &state.dir[s];
            e.par_iter_mut().enumerate().for_each(|(i, ei)| {
                *ei = if !self.active_row(i) {
                    [0.0; 3]
                } else {
                    // one scalar residual per row, with derivative dir
                    scale(dir[i], c * dot(dir[i], q[i]))
                };
            });
            self.jt(c, &e, &mut g);
            ym.par_iter_mut()
                .zip(yt.par_iter_mut())
                .zip(g.par_iter())
                .for_each(|((a, b), &gi)| {
                    *a += gi;
                    *b += sign * gi;
                });
        }
    }

    /// Upper bounds of the diagonal of `JᵀJ` with respect to `m` and to `t`
    /// (`d = m ∓ t/2`: each surface adds `c² D` to `m` and `c² D / 4` to `t`,
    /// the cross terms cancelling when both are penalised alike).
    pub fn gram_diag(&self) -> (Vec<f64>, Vec<f64>) {
        let mut unit = self.unit_gram_diag();
        if let Some(g_op) = &self.variation {
            // an estimate: the curvature diagonal times that of G
            unit.iter_mut().zip(g_op.diagonal()).for_each(|(u, g)| *u *= g);
        }
        let c2: f64 = self.coef.iter().map(|c| c * c).sum();
        let m = unit.iter().map(|v| c2 * v).collect();
        let t = unit.iter().map(|v| 0.25 * c2 * v).collect();
        (m, t)
    }

    /// Upper bound of the diagonal of `JᵀJ` with respect to `d` for one
    /// surface with `c = 1`.
    fn unit_gram_diag(&self) -> Vec<f64> {
        if let Some(h) = &self.hessian {
            // the derivative of ‖II‖_F has unit length
            return h.col_sq();
        }
        let l = &self.lap;
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
                // q carries ½ and the mean measure 2: unit factor
                let f = if rel { 0.25 } else { 1.0 };
                f * (own + col[k])
            })
            .collect()
    }

    /// The curvature measure (`2|H|`, or `√(κ₁² + κ₂²)` for the total
    /// curvature) of the surface `V + d n` at every node; 0 where inactive.
    /// The change mode measures `2|H|` too.
    pub fn curvature(&self, d: &[f64]) -> Vec<f64> {
        let n = self.n();
        let mut q = vec![[0.0; 3]; n];
        if self.relative() {
            // ½ L (d n), not the change mode's relative q
            let l = &self.lap;
            let r = &l.rows;
            q.par_iter_mut().enumerate().for_each(|(i, o)| {
                let mut acc = scale(self.normals[i], l.diag[i] * d[i]);
                for e in r.offsets[i]..r.offsets[i + 1] {
                    let k = r.cols[e] as usize;
                    acc = add(acc, scale(self.normals[k], r.vals[e] * d[k]));
                }
                *o = scale(acc, 0.5);
            });
        } else {
            self.q(d, &mut q);
        }
        (0..n)
            .into_par_iter()
            .map(|i| {
                if self.active_row(i) {
                    self.measure_of(i, add(self.base[i], q[i])).0
                } else {
                    0.0
                }
            })
            .collect()
    }
}
