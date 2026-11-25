//! The volumetric least-squares problem: cost, gradient, Gauss–Newton
//! products and preconditioner blocks.
//!
//! **Parameters.** Per node: `m`, `t`, `ρs`, `ρb` (each a field or one global
//! value); globally the PSF `σ_xy`, `σ_z` (`σ_z` fitted, fixed, or tied to
//! `σ_xy`). A voxel `x` on face `(a, b, c)` with barycentric coordinates `β`
//! takes the interpolated parameters `θ(x) = Σ_s β_s θ_s` (a convex
//! combination, so the parameter bounds hold at every voxel) and the blur of
//! its normal, `σ_n² = σ_xy² (1 − n_a²) + σ_z² n_a²`. Its model value is the
//! 1D model at its depth: `f(d; m, t, σ_n, ρs, ρb)` ([`crate::model`]).
//!
//! **Cost.** `L = 1/(2V) Σ_x w(d) r(x)²` over the `V` band voxels (`r` the
//! residual, `w` the residual weight), plus the curvature penalty on `m`, `t`
//! ([`CurvatureOperator`]) and the smoothing penalty `½ Σ κ_c ‖ω v_c‖²` on
//! the density fields.
//!
//! **Gauss–Newton.** A voxel's residual depends on the face's local vector
//! `z = (θ_a, θ_b, θ_c, globals)`, with Jacobian row
//! `J = (β_a ∇θ f, β_b ∇θ f, β_c ∇θ f, ∇_g f)`, where the PSF derivatives
//! follow from `∂σ_n/∂σ_xy = σ_xy (1 − n_a²)/σ_n` and
//! `∂σ_n/∂σ_z = σ_z n_a²/σ_n` (`σ_xy/σ_n` for a tied `σ_z`). Each face keeps
//! `G_f = Σ_x w J Jᵀ / V` (packed, up to 16 × 16): its diagonal blocks
//! couple a vertex's own parameters, its off-diagonal ones the face's
//! vertices (the edge blocks) and the globals. The product `JᵀJ v` is then
//! `y_f = G_f z_f(v)` per face, gathered back to the vertices and summed for
//! the globals, so the voxels are visited only when the blocks are formed.
//! The block-Jacobi preconditioner uses the vertex blocks
//! `Σ_{f ∋ v} G_f[v, v]` plus the penalties' diagonals.
//!
//! Every sum runs face by face in a fixed order, so results do not depend
//! on the number of threads.

use rayon::prelude::*;

use super::band::Band;
use crate::fit::global::penalty::{smoothing_cost, smoothing_product};
use crate::fit::global::{Costs, LeastSquares, Role};
use crate::model::{eval, eval_grad, Params, M as PM, RHO_B as PRB, RHO_S as PRS, SIGMA as PS, T as PT};
use crate::operators::{CurvatureOperator, CurvatureState, SmoothingOperator};
use crate::util::det_sum;

/// Index of `m` in the volumetric parameters.
pub(crate) const M: usize = 0;
/// Index of `t`.
pub(crate) const T: usize = 1;
/// Index of `ρs`.
pub(crate) const RHO_S: usize = 2;
/// Index of `ρb`.
pub(crate) const RHO_B: usize = 3;
/// Index of the in-plane blur `σ_xy`.
pub(crate) const SXY: usize = 4;
/// Index of the blur along the scanner axis `σ_z`.
pub(crate) const SZ: usize = 5;
/// Number of volumetric parameters.
pub(crate) const NV: usize = 6;
/// The node parameters (`m`, `t`, `ρs`, `ρb`) and their place in the 1D
/// model's parameter vector.
const NODE_PARAMS: [(usize, usize); 4] = [(M, PM), (T, PT), (RHO_S, PRS), (RHO_B, PRB)];

/// The fields the smoothing penalty can act on, in the order of its weights.
const SMOOTHED: [usize; 4] = [M, T, RHO_S, RHO_B];

/// A volumetric problem on a fixed surface.
pub(crate) struct VolumetricProblem<'a> {
    pub band: &'a Band,
    /// The fitted faces (node indices).
    pub faces: &'a [[u32; 3]],
    pub nodes: usize,
    pub rho_c: f64,
    /// Roles of `[m, t, ρs, ρb, σ_xy, σ_z]`; only the node parameters may be
    /// fields.
    pub roles: [Role; NV],
    /// `σ_z = σ_xy` (then `σ_z` is not a variable).
    pub tied: bool,
    /// Values of the fixed parameters.
    pub fixed: [f64; NV],
    /// Nodes held at their values (boundary conditions).
    pub frozen: Option<&'a [bool]>,
    /// The smoothing operator and the weights of the `m`, `t`, `ρs`, `ρb`
    /// fields.
    pub smoothing: Option<(&'a SmoothingOperator, [f64; 4])>,
    pub curvature: Option<&'a CurvatureOperator>,
    pub lower: [f64; NV],
    pub upper: [f64; NV],
}

/// Which parameters are variables, and where they sit.
#[derive(Clone)]
pub(crate) struct Layout {
    /// Field parameters, in order.
    pub fields: Vec<usize>,
    /// Global parameters, in order.
    pub globals: Vec<usize>,
    pub nodes: usize,
}

impl Layout {
    fn nvars(&self) -> usize {
        self.fields.len() * self.nodes + self.globals.len()
    }
    /// Size of a face's local vector.
    fn nz(&self) -> usize {
        3 * self.fields.len() + self.globals.len()
    }
}

/// What a linearisation keeps: every face's packed Gram matrix, and the
/// curvature penalty's state.
pub(crate) struct Lin {
    gram: Vec<f64>,
    curv: Option<CurvatureState>,
}

/// One face's contribution: cost, gradient over its local vector, and Gram.
struct FaceTerms {
    cost: f64,
    grad: Vec<f64>,
    gram: Vec<f64>,
}

/// Where a parameter's value comes from.
#[derive(Clone, Copy)]
enum Slot {
    /// Field channel: values at `x[offset + node]`.
    Field(usize),
    /// One value at `x[index]`.
    Global(usize),
    Fixed(f64),
}

/// The parameter values a voxel sees.
struct Values<'x> {
    lay: &'x Layout,
    x: &'x [f64],
    slots: [Slot; NV],
}

impl<'x> Values<'x> {
    fn new(lay: &'x Layout, x: &'x [f64], fixed: &[f64; NV], tied: bool) -> Self {
        let n = lay.nodes;
        let off = lay.fields.len() * n;
        let mut slots = [Slot::Fixed(0.0); NV];
        for (c, slot) in slots.iter_mut().enumerate() {
            let c = if c == SZ && tied { SXY } else { c };
            *slot = if let Some(fi) = lay.fields.iter().position(|&f| f == c) {
                Slot::Field(fi * n)
            } else if let Some(gi) = lay.globals.iter().position(|&g| g == c) {
                Slot::Global(off + gi)
            } else {
                Slot::Fixed(fixed[c])
            };
        }
        Values { lay, x, slots }
    }

    /// Parameter `c` at node `v`.
    #[inline]
    fn node(&self, c: usize, v: usize) -> f64 {
        match self.slots[c] {
            Slot::Field(off) => self.x[off + v],
            Slot::Global(i) => self.x[i],
            Slot::Fixed(val) => val,
        }
    }

    /// A global or fixed parameter.
    #[inline]
    fn scalar(&self, c: usize) -> f64 {
        self.node(c, 0)
    }

    /// The node values of parameter `c`.
    fn nodes(&self, c: usize) -> Vec<f64> {
        (0..self.lay.nodes).map(|v| self.node(c, v)).collect()
    }
}

impl<'a> VolumetricProblem<'a> {
    pub fn layout(&self) -> Layout {
        let tied = |c: usize| c == SZ && self.tied;
        Layout {
            fields: (0..NV).filter(|&c| self.roles[c] == Role::Field).collect(),
            globals: (0..NV).filter(|&c| self.roles[c] == Role::Global && !tied(c)).collect(),
            nodes: self.nodes,
        }
    }

    fn values<'x>(&'x self, lay: &'x Layout, x: &'x [f64]) -> Values<'x> {
        Values::new(lay, x, &self.fixed, self.tied)
    }

    /// The model parameters `[m, t, σ_n, ρs, ρb]` of voxel `e` on face `f`,
    /// the derivatives of `σ_n` with respect to `σ_xy` and `σ_z`, and the
    /// barycentric weights.
    #[inline]
    fn voxel(&self, val: &Values, f: usize, e: usize) -> (Params, [f64; 2], [f64; 3]) {
        let b = self.band.bary[e];
        let beta = [b[0] as f64, b[1] as f64, 1.0 - b[0] as f64 - b[1] as f64];
        let tri = self.faces[f];
        let mut p = [0.0; 5];
        for &(c, pc) in &NODE_PARAMS {
            p[pc] = (0..3).map(|s| beta[s] * val.node(c, tri[s] as usize)).sum();
        }
        let (sxy, sz) = (val.scalar(SXY), val.scalar(SZ));
        let a2 = self.band.axial2[e] as f64;
        let sn = (sxy * sxy * (1.0 - a2) + sz * sz * a2).sqrt();
        p[PS] = sn;
        let dsn = if self.tied {
            [sxy / sn, 0.0]
        } else {
            [sxy * (1.0 - a2) / sn, sz * a2 / sn]
        };
        (p, dsn, beta)
    }

    /// One face's cost, and with `derivs` its gradient and Gram matrix over
    /// the local vector `(θ_a, θ_b, θ_c, globals)`.
    fn face_terms(&self, lay: &Layout, val: &Values, f: usize, derivs: bool, scale: f64) -> FaceTerms {
        let nfc = lay.fields.len();
        let nz = lay.nz();
        let mut out = FaceTerms {
            cost: 0.0,
            grad: if derivs { vec![0.0; nz] } else { Vec::new() },
            gram: if derivs {
                vec![0.0; nz * (nz + 1) / 2]
            } else {
                Vec::new()
            },
        };
        let mut j = vec![0.0; nz];
        for e in self.band.of(f) {
            let (p, dsn, beta) = self.voxel(val, f, e);
            let (d, y, w) = (
                self.band.depth[e] as f64,
                self.band.value[e] as f64,
                self.band.weight[e] as f64 * scale,
            );
            if !derivs {
                let r = eval(d, &p, self.rho_c) - y;
                out.cost += 0.5 * w * r * r;
                continue;
            }
            let (fv, g) = eval_grad(d, &p, self.rho_c);
            let r = fv - y;
            out.cost += 0.5 * w * r * r;
            for (fi, &c) in lay.fields.iter().enumerate() {
                let gc = g[NODE_PARAMS[c].1];
                for s in 0..3 {
                    j[s * nfc + fi] = beta[s] * gc;
                }
            }
            for (gi, &c) in lay.globals.iter().enumerate() {
                j[3 * nfc + gi] = match c {
                    SXY => g[PS] * dsn[0],
                    SZ => g[PS] * dsn[1],
                    _ => g[NODE_PARAMS[c].1],
                };
            }
            let mut k = 0;
            for a in 0..nz {
                let wa = w * j[a];
                out.grad[a] += wa * r;
                for &jb in &j[a..nz] {
                    out.gram[k] += wa * jb;
                    k += 1;
                }
            }
        }
        out
    }

    /// Every face's terms.
    fn all_faces(&self, lay: &Layout, x: &[f64], derivs: bool) -> Vec<FaceTerms> {
        let val = self.values(lay, x);
        let scale = 1.0 / self.band.len().max(1) as f64;
        (0..self.faces.len())
            .into_par_iter()
            .map(|f| self.face_terms(lay, &val, f, derivs, scale))
            .collect()
    }

    /// The penalties' cost; with `grads` (one per variable), their gradient
    /// is added there. Returns `(smoothing, curvature, curvature state)`.
    fn penalties(&self, lay: &Layout, x: &[f64], grads: Option<&mut [f64]>) -> (f64, f64, Option<CurvatureState>) {
        let val = self.values(lay, x);
        let n = lay.nodes;
        let field = |c: usize| lay.fields.iter().position(|&f| f == c);
        let mut gnode: Vec<Vec<f64>> = vec![vec![0.0; n]; lay.fields.len()];
        let derivs = grads.is_some();
        // smoothing of the fields
        let mut smooth = 0.0;
        if let Some((op, coef)) = self.smoothing {
            let ch: Vec<(usize, f64)> = SMOOTHED
                .into_iter()
                .zip(coef)
                .filter_map(|(c, k)| field(c).filter(|_| k != 0.0).map(|fi| (fi, k)))
                .collect();
            if !ch.is_empty() {
                let xs: Vec<&[f64]> = ch.iter().map(|&(fi, _)| &x[fi * n..(fi + 1) * n]).collect();
                let coefs: Vec<f64> = ch.iter().map(|&(_, k)| k).collect();
                if derivs {
                    let mut gs: Vec<&mut Vec<f64>> = gnode
                        .iter_mut()
                        .enumerate()
                        .filter(|(fi, _)| ch.iter().any(|&(f, _)| f == *fi))
                        .map(|(_, g)| g)
                        .collect();
                    smooth = smoothing_cost(op, &xs, &coefs, Some(&mut gs));
                } else {
                    smooth = smoothing_cost(op, &xs, &coefs, None);
                }
            }
        }
        // curvature of the fitted surfaces (m and t fields)
        let (mut curv, mut state) = (0.0, None);
        if let (Some(cv), Some(fm), Some(ft)) = (self.curvature, field(M), field(T)) {
            let mut st = cv.new_state();
            let (mut gm, mut gt) = (vec![0.0; n], vec![0.0; n]);
            curv = cv.eval(&val.nodes(M), &val.nodes(T), &mut st, &mut gm, &mut gt);
            if derivs {
                add(&mut gnode[fm], &gm);
                add(&mut gnode[ft], &gt);
            }
            state = Some(st);
        }
        if let Some(g) = grads {
            for (fi, gn) in gnode.iter().enumerate() {
                add(&mut g[fi * n..(fi + 1) * n], gn);
            }
        }
        (smooth, curv, state)
    }

    /// The gradient over the variables from the faces' local gradients.
    fn gather_gradient(&self, lay: &Layout, incident: &Incident, terms: &[FaceTerms]) -> Vec<f64> {
        let (n, nfc) = (lay.nodes, lay.fields.len());
        let mut grad = vec![0.0; lay.nvars()];
        for fi in 0..nfc {
            grad[fi * n..(fi + 1) * n]
                .par_iter_mut()
                .enumerate()
                .for_each(|(v, g)| {
                    *g = incident
                        .of(v)
                        .iter()
                        .map(|&(f, s)| terms[f as usize].grad[s as usize * nfc + fi])
                        .sum()
                });
        }
        for gi in 0..lay.globals.len() {
            grad[nfc * n + gi] = det_sum(terms.len(), |f| terms[f].grad[3 * nfc + gi]);
        }
        grad
    }

    /// `JᵀJ v` for the data term and the penalties.
    fn product(&self, lay: &Layout, incident: &Incident, lin: &Lin, v: &[f64], out: &mut [f64]) {
        let (n, nfc, nz) = (lay.nodes, lay.fields.len(), lay.nz());
        let np = nz * (nz + 1) / 2;
        let faces = self.faces;
        // y_f = G_f z_f(v), per face
        let y: Vec<Vec<f64>> = (0..faces.len())
            .into_par_iter()
            .map(|f| {
                let mut z = vec![0.0; nz];
                for (s, &vtx) in faces[f].iter().enumerate() {
                    for fi in 0..nfc {
                        z[s * nfc + fi] = v[fi * n + vtx as usize];
                    }
                }
                z[3 * nfc..].copy_from_slice(&v[nfc * n..]);
                sym_matvec(&lin.gram[f * np..(f + 1) * np], &z, nz)
            })
            .collect();
        for fi in 0..nfc {
            out[fi * n..(fi + 1) * n]
                .par_iter_mut()
                .enumerate()
                .for_each(|(vtx, o)| {
                    *o = incident
                        .of(vtx)
                        .iter()
                        .map(|&(f, s)| y[f as usize][s as usize * nfc + fi])
                        .sum()
                });
        }
        for gi in 0..lay.globals.len() {
            out[nfc * n + gi] = det_sum(y.len(), |f| y[f][3 * nfc + gi]);
        }
        self.penalty_product(lay, lin, v, out);
    }

    /// Add the penalties' Gauss–Newton product to `out`.
    fn penalty_product(&self, lay: &Layout, lin: &Lin, v: &[f64], out: &mut [f64]) {
        let n = lay.nodes;
        let field = |c: usize| lay.fields.iter().position(|&f| f == c);
        if let Some((op, coef)) = self.smoothing {
            for (c, k) in SMOOTHED.into_iter().zip(coef) {
                if let (Some(fi), true) = (field(c), k != 0.0) {
                    let mut y = vec![0.0; n];
                    smoothing_product(op, &[&v[fi * n..(fi + 1) * n]], &[k], &mut [&mut y]);
                    add(&mut out[fi * n..(fi + 1) * n], &y);
                }
            }
        }
        if let (Some(cv), Some(st), Some(fm), Some(ft)) = (self.curvature, &lin.curv, field(M), field(T)) {
            let (mut ym, mut yt) = (vec![0.0; n], vec![0.0; n]);
            cv.gn_product(st, &v[fm * n..(fm + 1) * n], &v[ft * n..(ft + 1) * n], &mut ym, &mut yt);
            add(&mut out[fm * n..(fm + 1) * n], &ym);
            add(&mut out[ft * n..(ft + 1) * n], &yt);
        }
    }

    /// The vertex blocks (data term, `nfc × nfc` per node) and the global
    /// block, without the penalties.
    pub fn data_blocks(&self, lay: &Layout, incident: &Incident, gram: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (n, nfc, ng, nz) = (lay.nodes, lay.fields.len(), lay.globals.len(), lay.nz());
        let np = nz * (nz + 1) / 2;
        let bs = nfc * nfc;
        let mut field = vec![0.0; bs * n];
        field.par_chunks_mut(bs.max(1)).enumerate().for_each(|(vtx, blk)| {
            if bs == 0 {
                return;
            }
            for &(f, s) in incident.of(vtx) {
                let g = &gram[f as usize * np..(f as usize + 1) * np];
                for a in 0..nfc {
                    for b in 0..nfc {
                        blk[a * nfc + b] += g[packed(s as usize * nfc + a, s as usize * nfc + b, nz)];
                    }
                }
            }
        });
        let mut glob = vec![0.0; ng * ng];
        for a in 0..ng {
            for b in 0..ng {
                let (ia, ib) = (3 * nfc + a, 3 * nfc + b);
                glob[a * ng + b] = det_sum(self.faces.len(), |f| gram[f * np + packed(ia, ib, nz)]);
            }
        }
        (field, glob)
    }

    /// The penalties' Gram diagonals, per node and field channel (`n × nfc`).
    pub fn penalty_diagonal(&self, lay: &Layout) -> Vec<f64> {
        let (n, nfc) = (lay.nodes, lay.fields.len());
        let mut d = vec![0.0; n * nfc];
        if let Some((op, coef)) = self.smoothing {
            let sd = op.gram_diag();
            for (fi, &c) in lay.fields.iter().enumerate() {
                let k = SMOOTHED.iter().position(|&s| s == c).map_or(0.0, |i| coef[i]);
                for v in 0..n {
                    d[v * nfc + fi] += k * sd[v];
                }
            }
        }
        if let Some(cv) = self.curvature {
            let (cm, ct) = cv.gram_diag();
            for (fi, &c) in lay.fields.iter().enumerate() {
                let cd = match c {
                    M => &cm,
                    T => &ct,
                    _ => continue,
                };
                for v in 0..n {
                    d[v * nfc + fi] += cd[v];
                }
            }
        }
        d
    }
}

/// The volumetric problem with its layout and penalty diagonals, for the
/// solver.
pub(crate) struct Prepared<'p, 'a> {
    pub problem: &'p VolumetricProblem<'a>,
    pub lay: Layout,
    pub incident: Incident,
    pen_diag: Vec<f64>,
}

impl<'p, 'a> Prepared<'p, 'a> {
    pub fn new(problem: &'p VolumetricProblem<'a>) -> Self {
        let lay = problem.layout();
        Prepared {
            pen_diag: problem.penalty_diagonal(&lay),
            incident: Incident::new(problem.faces, problem.nodes),
            problem,
            lay,
        }
    }

    /// The model's value at every band voxel (in band order) at `x`.
    #[cfg(test)]
    pub fn model_values(&self, x: &[f64]) -> Vec<f64> {
        let p = self.problem;
        let val = p.values(&self.lay, x);
        let per_face: Vec<Vec<f64>> = (0..p.faces.len())
            .into_par_iter()
            .map(|f| {
                p.band
                    .of(f)
                    .map(|e| eval(p.band.depth[e] as f64, &p.voxel(&val, f, e).0, p.rho_c))
                    .collect()
            })
            .collect();
        per_face.into_iter().flatten().collect()
    }

    /// The node values of every parameter at `x` (`NV × nodes`).
    pub fn node_values(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let val = self.problem.values(&self.lay, x);
        (0..NV).map(|c| val.nodes(c)).collect()
    }

    /// The model's value at the band voxels `positions`.
    pub fn values_at(&self, x: &[f64], positions: &[u32]) -> Vec<f64> {
        let p = self.problem;
        let val = p.values(&self.lay, x);
        positions
            .par_iter()
            .map(|&e| {
                let e = e as usize;
                let f = p.band.offsets.partition_point(|&o| o <= e) - 1;
                eval(p.band.depth[e] as f64, &p.voxel(&val, f, e).0, p.rho_c)
            })
            .collect()
    }

    /// Per node, from the voxels of its faces at `x`, with `g_i` the
    /// gradient of voxel `i`'s model value in the node's own parameters
    /// (`β_i ∇f`): the data-only Gauss–Newton block `B = Σ w_i g_i g_iᵀ`,
    /// the noise covariance of the weighted gradient
    /// `C = Σ_ij ρ_ij (w_i g_i)(w_j g_j)ᵀ` for noise correlated as `noise`
    /// (unit variance), and `Σ β w r²`, `Σ β w` (the node's share of the
    /// weighted squared residual).
    pub fn node_moments(&self, x: &[f64], noise: &Correlation) -> NodeMoments {
        let (n, nfc) = (self.lay.nodes, self.lay.fields.len());
        let val = self.problem.values(&self.lay, x);
        let per_node: Vec<NodeMoment> = (0..n)
            .into_par_iter()
            .map(|v| self.node_moment(&val, v, noise))
            .collect();
        let mut out = NodeMoments {
            b: Vec::with_capacity(n * nfc * nfc),
            c: Vec::with_capacity(n * nfc * nfc),
            wr2: Vec::with_capacity(n),
            w: Vec::with_capacity(n),
        };
        for m in per_node {
            out.b.extend(m.b);
            out.c.extend(m.c);
            out.wr2.push(m.wr2);
            out.w.push(m.w);
        }
        out
    }

    /// [`Prepared::node_moments`] of node `v`.
    fn node_moment(&self, val: &Values, v: usize, noise: &Correlation) -> NodeMoment {
        let p = self.problem;
        let nfc = self.lay.fields.len();
        let mut m = NodeMoment {
            b: vec![0.0; nfc * nfc],
            c: vec![0.0; nfc * nfc],
            wr2: 0.0,
            w: 0.0,
        };
        // the node's voxels: grid index and weighted gradient w g
        let (mut grids, mut wg) = (Vec::new(), Vec::new());
        for &(f, s) in self.incident.of(v) {
            let (f, s) = (f as usize, s as usize);
            for e in p.band.of(f) {
                let (q, _, beta) = p.voxel(val, f, e);
                let (d, y, w) = (p.band.depth[e] as f64, p.band.value[e] as f64, p.band.weight[e] as f64);
                let (fv, grad) = eval_grad(d, &q, p.rho_c);
                let g: Vec<f64> = self
                    .lay
                    .fields
                    .iter()
                    .map(|&c| beta[s] * grad[NODE_PARAMS[c].1])
                    .collect();
                for i in 0..nfc {
                    for j in 0..nfc {
                        m.b[i * nfc + j] += w * g[i] * g[j];
                    }
                }
                m.wr2 += beta[s] * w * (fv - y).powi(2);
                m.w += beta[s] * w;
                grids.push(p.band.grid[e]);
                wg.extend(g.iter().map(|gi| w * gi));
            }
        }
        noise.covariance(&grids, &wg, nfc, &mut m.c);
        m
    }

    /// The data's share of the curvature of every field parameter at every
    /// node, `B / (B + penalty)` from the diagonals (`n × nfc`).
    pub fn data_share(&self, lin: &Lin) -> Vec<f64> {
        let (field, _) = self.problem.data_blocks(&self.lay, &self.incident, &lin.gram);
        let (n, nfc) = (self.lay.nodes, self.lay.fields.len());
        (0..n * nfc)
            .map(|k| {
                let (v, a) = (k / nfc, k % nfc);
                let b = field[v * nfc * nfc + a * nfc + a];
                let total = b + self.pen_diag[k];
                if total > 0.0 {
                    b / total
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
}

impl LeastSquares for Prepared<'_, '_> {
    type Lin = Lin;

    fn shape(&self) -> (usize, usize, usize) {
        (self.lay.fields.len(), self.lay.nodes, self.lay.globals.len())
    }

    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let p = self.problem;
        let per = |b: &[f64; NV]| -> Vec<f64> {
            let mut out = Vec::with_capacity(self.lay.nvars());
            for &c in &self.lay.fields {
                out.extend(std::iter::repeat_n(b[c], self.lay.nodes));
            }
            out.extend(self.lay.globals.iter().map(|&c| b[c]));
            out
        };
        (per(&p.lower), per(&p.upper))
    }

    fn held(&self, dof: usize) -> bool {
        self.problem.frozen.is_some_and(|fz| fz[dof])
    }

    fn cost(&self, x: &[f64]) -> f64 {
        let terms = self.problem.all_faces(&self.lay, x, false);
        let data = det_sum(terms.len(), |f| terms[f].cost);
        let (s, c, _) = self.problem.penalties(&self.lay, x, None);
        data + s + c
    }

    fn linearise(&self, x: &[f64]) -> (Costs, Vec<f64>, Lin) {
        let terms = self.problem.all_faces(&self.lay, x, true);
        let data = det_sum(terms.len(), |f| terms[f].cost);
        let mut grad = self.problem.gather_gradient(&self.lay, &self.incident, &terms);
        let (smoothing, curvature, curv) = self.problem.penalties(&self.lay, x, Some(&mut grad));
        let gram: Vec<f64> = terms.into_iter().flat_map(|t| t.gram).collect();
        let costs = Costs {
            total: data + smoothing + curvature,
            data,
            smoothing,
            curvature,
        };
        (costs, grad, Lin { gram, curv })
    }

    fn diag_blocks(&self, lin: &Lin) -> (Vec<f64>, Vec<f64>) {
        let (mut field, glob) = self.problem.data_blocks(&self.lay, &self.incident, &lin.gram);
        let nfc = self.lay.fields.len();
        for v in 0..self.lay.nodes {
            for a in 0..nfc {
                field[v * nfc * nfc + a * nfc + a] += self.pen_diag[v * nfc + a];
            }
        }
        (field, glob)
    }

    fn hess_vec(&self, lin: &Lin, v: &[f64], out: &mut [f64]) {
        self.problem.product(&self.lay, &self.incident, lin, v, out)
    }

    fn edges(&self, x: &[f64]) -> Vec<f64> {
        let val = self.problem.values(&self.lay, x);
        (0..self.lay.nodes)
            .map(|v| val.node(M, v) - 0.5 * val.node(T, v))
            .collect()
    }
}

/// Per-node data moments ([`Prepared::node_moments`]).
pub(crate) struct NodeMoments {
    /// `B` and `C` over the node's own parameters (`nfc × nfc` per node).
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    /// `Σ β w r²` and `Σ β w`.
    pub wr2: Vec<f64>,
    pub w: Vec<f64>,
}

/// One node's [`NodeMoments`].
struct NodeMoment {
    b: Vec<f64>,
    c: Vec<f64>,
    wr2: f64,
    w: f64,
}

/// Separable exponential noise correlation: `ρ_ij = Π_a ρ_a^{|Δ_a|/step}`
/// between voxels whose grid indices differ by `Δ` (the band's voxels are
/// `step` apart).
pub(crate) struct Correlation {
    pub rho: [f64; 3],
    pub step: u32,
}

impl Correlation {
    /// Pairs whose correlation is below this are left out.
    const NEGLIGIBLE: f64 = 1e-3;

    /// The correlation of two voxels.
    fn between(&self, a: [u32; 3], b: [u32; 3]) -> f64 {
        (0..3)
            .map(|k| {
                let lag = a[k].abs_diff(b[k]) / self.step.max(1);
                if lag == 0 {
                    1.0
                } else {
                    self.rho[k].powi(lag as i32)
                }
            })
            .product()
    }

    /// `C += Σ_ij ρ_ij u_i u_jᵀ` for the vectors `u_i` (rows of `u`, length
    /// `n`) at grid indices `grids` (white noise: the diagonal only).
    fn covariance(&self, grids: &[[u32; 3]], u: &[f64], n: usize, c: &mut [f64]) {
        let white = self.rho.iter().all(|&r| r < Self::NEGLIGIBLE);
        for (i, gi) in grids.iter().enumerate() {
            let ui = &u[i * n..(i + 1) * n];
            let partners: Box<dyn Iterator<Item = usize>> = if white {
                Box::new(std::iter::once(i))
            } else {
                Box::new(0..grids.len())
            };
            for j in partners {
                let r = self.between(*gi, grids[j]);
                if r < Self::NEGLIGIBLE {
                    continue;
                }
                let uj = &u[j * n..(j + 1) * n];
                for a in 0..n {
                    for b in 0..n {
                        c[a * n + b] += r * ui[a] * uj[b];
                    }
                }
            }
        }
    }
}

/// For every node, the faces it belongs to and its slot in each.
pub(crate) struct Incident {
    offsets: Vec<usize>,
    entries: Vec<(u32, u8)>,
}

impl Incident {
    fn new(faces: &[[u32; 3]], n: usize) -> Self {
        let mut offsets = vec![0usize; n + 1];
        for f in faces {
            for &v in f {
                offsets[v as usize + 1] += 1;
            }
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut fill = offsets.clone();
        let mut entries = vec![(0u32, 0u8); offsets[n]];
        for (fi, f) in faces.iter().enumerate() {
            for (s, &v) in f.iter().enumerate() {
                entries[fill[v as usize]] = (fi as u32, s as u8);
                fill[v as usize] += 1;
            }
        }
        Incident { offsets, entries }
    }

    fn of(&self, v: usize) -> &[(u32, u8)] {
        &self.entries[self.offsets[v]..self.offsets[v + 1]]
    }
}

/// Index of `(i, j)` in a packed upper-triangular `n × n` matrix.
#[inline]
pub(crate) fn packed(i: usize, j: usize, n: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    i * n - i * (i + 1) / 2 + j
}

/// `y = G z` for a packed symmetric `G`.
fn sym_matvec(g: &[f64], z: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    let mut k = 0;
    for a in 0..n {
        for b in a..n {
            let v = g[k];
            y[a] += v * z[b];
            if b != a {
                y[b] += v * z[a];
            }
            k += 1;
        }
    }
    y
}

/// `dst += src`.
fn add(dst: &mut [f64], src: &[f64]) {
    dst.iter_mut().zip(src).for_each(|(a, &b)| *a += b);
}
