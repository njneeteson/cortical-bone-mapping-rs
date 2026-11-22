//! The global problem: cost, gradient, Gauss–Newton products and diagonal
//! blocks of the data term and the penalties.
//!
//! Notation: node `i` has parameters `p_i` (fields from the variables, the
//! rest global or fixed), residuals `r_ij = f̂(x_j; p_i) − f_ij` with weights
//! `w_ij = γ_j / (NM)` (zero for unused samples) and Jacobian rows
//! `∇f̂_ij = ∂f̂(x_j; p_i)/∂p_i`. For a field channel `c` the node values are
//! `v_c = A u_c` (multigrid) or `v_c = u_c` (one dof per node).
//!
//! **Data term.** `D = ½ Σ_ij w_ij r_ij²`; per node the gradient
//! `g_i = Σ_j w_ij r_ij ∇f̂_ij` and the Gauss–Newton block
//! `B_i = Σ_j w_ij ∇f̂_ij ∇f̂_ijᵀ` (5×5 symmetric, packed in 15 entries).
//!
//! **Smoothing.** `S = ½ Σ_c κ_c ‖ω v_c‖²`; gradient `κ_c ωᵀω v_c` and
//! Gauss–Newton (here exact) Hessian `κ_c ωᵀω`.
//!
//! **Curvature.** Residuals linear (change mode) or piecewise linear (excess
//! mode) in `(m, t)`; see [`crate::operators::CurvatureOperator`].
//!
//! **Mapping to variables.** Node-space gradients of field channel `c` map to
//! its dofs by `Aᵀ` (or the identity); a global parameter's gradient is the sum
//! over nodes of its node-space gradient (chain rule: every node uses the same
//! value). The Gauss–Newton product `JᵀJ v` maps `v` to node space the same
//! way (`A v`, and the global values broadcast), applies `B_i` per node and the
//! penalty Hessians, and maps back. Its diagonal blocks (for the
//! preconditioner) are `Σ_i a_ik² B_i` per dof (multigrid; the off-diagonal
//! couplings of `A` are dropped, which is what a block-Jacobi preconditioner
//! does) plus the diagonals of the penalty Hessians, and for the globals the
//! sums `Σ_i B_i` restricted to the global channels.
//!
//! All sums over nodes use fixed-size chunks in a fixed order
//! ([`crate::util::det_sum`]), so results do not depend on the thread count.

use rayon::prelude::*;

use super::{Evaluation, Layout, Problem, Role, Smoothing, State};
use crate::model::{eval, eval_grad, Params, M, NPARAM, T};
use crate::operators::{CurvatureState, SmoothingOperator};
use crate::util::{det_sum, det_sum5, sym5, CHUNK};

/// Data-term derivatives per node: Gauss–Newton block (packed symmetric
/// 5×5) and gradient.
type NodeDerivatives = (Vec<[f64; 15]>, Vec<[f64; NPARAM]>);
/// Node-space gradients being accumulated: per field channel, and per node.
type NodeGradientsMut<'g> = (&'g mut Vec<Vec<f64>>, &'g mut Vec<[f64; NPARAM]>);

impl<'a> Problem<'a> {
    pub(super) fn layout(&self) -> Layout {
        let fields = (0..NPARAM).filter(|&c| self.roles[c] == Role::Field).collect();
        let globals = (0..NPARAM).filter(|&c| self.roles[c] == Role::Global).collect();
        let n = self.profiles.n;
        let ndof = self.interp.map(|a| a.ncols).unwrap_or(n);
        Layout {
            fields,
            globals,
            ndof,
            n,
        }
    }

    /// Node values of one field channel's dofs (`A v` or `v`).
    fn to_nodes(&self, lay: &Layout, dofs: &[f64]) -> Vec<f64> {
        match self.interp {
            Some(a) => {
                let mut out = vec![0.0; lay.n];
                a.apply(dofs, &mut out);
                out
            }
            None => dofs.to_vec(),
        }
    }

    /// Node values of every field channel.
    fn node_fields(&self, lay: &Layout, field: &[Vec<f64>]) -> Vec<Vec<f64>> {
        field.iter().map(|f| self.to_nodes(lay, f)).collect()
    }

    /// Node-space values to dofs (`Aᵀ u` or `u`).
    fn to_dofs(&self, node: &[f64], out: &mut [f64]) {
        match self.interp {
            Some(a) => a.apply_t(node, out),
            None => out.copy_from_slice(node),
        }
    }

    /// Model parameters of every node.
    pub fn node_params(&self, state: &State) -> Vec<Params> {
        let lay = self.layout();
        let nf = self.node_fields(&lay, &state.field);
        (0..lay.n)
            .into_par_iter()
            .map(|i| self.params_at(&lay, &nf, &state.scalars, i))
            .collect()
    }

    #[inline]
    fn params_at(&self, lay: &Layout, nf: &[Vec<f64>], scalars: &Params, i: usize) -> Params {
        let mut p = *scalars;
        for (fi, &c) in lay.fields.iter().enumerate() {
            p[c] = nf[fi][i];
        }
        for c in 0..NPARAM {
            if self.roles[c] == Role::Fixed {
                if let Some(a) = self.fixed_per_node[c] {
                    p[c] = a[i];
                }
            }
        }
        p
    }

    /// The residual weights scaled by `1/(N M)`.
    fn scaled_weights(&self) -> Vec<f64> {
        let sc = 1.0 / (self.profiles.n as f64 * self.profiles.m as f64);
        self.weights.iter().map(|g| g * sc).collect()
    }

    /// Cost of `state` and, with `derivs`, its gradient and the data term's
    /// Gauss–Newton blocks.
    pub(super) fn evaluate(&self, lay: &Layout, state: &State, derivs: bool) -> Evaluation {
        let nf = self.node_fields(lay, &state.field);
        let w = self.scaled_weights();
        let sc = &state.scalars;
        let (data_cost, blocks, mut gnode) = if derivs {
            let (cost, (blocks, gnode)) = self.data_derivatives(lay, &nf, sc, &w);
            (cost, blocks, gnode)
        } else {
            (self.data_cost(lay, &nf, sc, &w), Vec::new(), Vec::new())
        };
        let mut gfield: Vec<Vec<f64>> = if derivs {
            lay.fields
                .iter()
                .map(|&c| gnode.par_iter().map(|g| g[c]).collect())
                .collect()
        } else {
            Vec::new()
        };
        let smoothing_cost = self.smoothing_terms(lay, &nf, derivs.then_some(&mut gfield));
        let grads = if derivs { Some((&mut gfield, &mut gnode)) } else { None };
        let (curvature_cost, curv_state) = self.curvature_terms(lay, &nf, sc, grads);
        let grad = if derivs {
            self.variable_gradient(lay, &gfield, &gnode)
        } else {
            Vec::new()
        };
        Evaluation {
            cost: data_cost + smoothing_cost + curvature_cost,
            data_cost,
            smoothing_cost,
            curvature_cost,
            grad,
            blocks,
            curv_state,
        }
    }

    /// The data cost alone.
    fn data_cost(&self, lay: &Layout, nf: &[Vec<f64>], scalars: &Params, w: &[f64]) -> f64 {
        let x = self.x;
        det_sum(lay.n, |i| {
            let p = self.params_at(lay, nf, scalars, i);
            let y = self.profiles.row(i);
            let w = self.profiles.row_weights(i, w);
            let mut c = 0.0;
            for j in 0..x.len() {
                let r = eval(x[j], &p, self.rho_c) - y[j] as f64;
                c += 0.5 * w[j] * r * r;
            }
            c
        })
    }

    /// The data cost with its per-node Gauss–Newton blocks and gradients.
    fn data_derivatives(&self, lay: &Layout, nf: &[Vec<f64>], scalars: &Params, w: &[f64]) -> (f64, NodeDerivatives) {
        let x = self.x;
        let mut blocks = vec![[0.0; 15]; lay.n];
        let mut gnode = vec![[0.0; NPARAM]; lay.n];
        let parts: Vec<f64> = blocks
            .par_chunks_mut(CHUNK)
            .zip(gnode.par_chunks_mut(CHUNK))
            .enumerate()
            .map(|(ci, (bc, gc))| {
                let mut total = 0.0;
                for (k, (b, g)) in bc.iter_mut().zip(gc.iter_mut()).enumerate() {
                    let i = ci * CHUNK + k;
                    let p = self.params_at(lay, nf, scalars, i);
                    let y = self.profiles.row(i);
                    let w = self.profiles.row_weights(i, w);
                    for j in 0..x.len() {
                        if w[j] == 0.0 {
                            continue;
                        }
                        let (f, d) = eval_grad(x[j], &p, self.rho_c);
                        let r = f - y[j] as f64;
                        total += 0.5 * w[j] * r * r;
                        for a in 0..NPARAM {
                            let wd = w[j] * d[a];
                            g[a] += wd * r;
                            for bb in a..NPARAM {
                                b[sym5(a, bb)] += wd * d[bb];
                            }
                        }
                    }
                }
                total
            })
            .collect();
        (parts.iter().sum(), (blocks, gnode))
    }

    /// The smoothing cost; with `gfield`, its gradient is added there.
    fn smoothing_terms(&self, lay: &Layout, nf: &[Vec<f64>], gfield: Option<&mut Vec<Vec<f64>>>) -> f64 {
        let Some(sm) = &self.smoothing else {
            return 0.0;
        };
        let ch = smoothed_channels(lay, sm);
        let xs: Vec<&[f64]> = ch.iter().map(|&(fi, _)| nf[fi].as_slice()).collect();
        let us = smooth_each(sm.op, &xs, lay.n, false);
        let mut cost = 0.0;
        for (u, &(_, c)) in us.iter().zip(&ch) {
            cost += 0.5 * sm.coef[c] * det_sum(lay.n, |i| u[i] * u[i]);
        }
        if let Some(gfield) = gfield {
            let ys: Vec<&[f64]> = us.iter().map(|u| u.as_slice()).collect();
            let ts = smooth_each(sm.op, &ys, lay.n, true);
            for (t, &(fi, c)) in ts.iter().zip(&ch) {
                add_scaled(&mut gfield[fi], t, sm.coef[c]);
            }
        }
        cost
    }

    /// The curvature cost and the operator's state; with `grads`, its
    /// gradient is added there.
    fn curvature_terms(
        &self,
        lay: &Layout,
        nf: &[Vec<f64>],
        scalars: &Params,
        grads: Option<NodeGradientsMut>,
    ) -> (f64, Option<CurvatureState>) {
        let Some(cv) = self.curvature else {
            return (0.0, None);
        };
        let mvals = self.channel_node_values(lay, nf, scalars, M);
        let tvals = self.channel_node_values(lay, nf, scalars, T);
        let mut st = cv.new_state();
        let mut gm = vec![0.0; lay.n];
        let mut gt = vec![0.0; lay.n];
        let cost = cv.eval(&mvals, &tvals, &mut st, &mut gm, &mut gt);
        if let Some((gfield, gnode)) = grads {
            self.add_node_grad(lay, gfield, gnode, M, &gm);
            self.add_node_grad(lay, gfield, gnode, T, &gt);
        }
        (cost, Some(st))
    }

    /// The gradient over the variables from the node-space gradients.
    fn variable_gradient(&self, lay: &Layout, gfield: &[Vec<f64>], gnode: &[[f64; NPARAM]]) -> Vec<f64> {
        let mut grad = vec![0.0; lay.nvars()];
        for (fi, gf) in gfield.iter().enumerate() {
            self.to_dofs(gf, &mut grad[fi * lay.ndof..(fi + 1) * lay.ndof]);
        }
        let off = lay.fields.len() * lay.ndof;
        for (gi, &c) in lay.globals.iter().enumerate() {
            grad[off + gi] = det_sum(lay.n, |i| gnode[i][c]);
        }
        grad
    }

    /// Node values of parameter `c`, whatever its role.
    fn channel_node_values(&self, lay: &Layout, nf: &[Vec<f64>], scalars: &Params, c: usize) -> Vec<f64> {
        if let Some(fi) = lay.fields.iter().position(|&f| f == c) {
            return nf[fi].clone();
        }
        if self.roles[c] == Role::Fixed {
            if let Some(a) = self.fixed_per_node[c] {
                return a.to_vec();
            }
        }
        vec![scalars[c]; lay.n]
    }

    /// Add a node-space gradient for parameter `c` to the right place.
    fn add_node_grad(&self, lay: &Layout, gfield: &mut [Vec<f64>], gnode: &mut [[f64; NPARAM]], c: usize, g: &[f64]) {
        if let Some(fi) = lay.fields.iter().position(|&f| f == c) {
            gfield[fi].par_iter_mut().zip(g.par_iter()).for_each(|(a, &b)| *a += b);
        } else if self.roles[c] == Role::Global {
            gnode.par_iter_mut().zip(g.par_iter()).for_each(|(a, &b)| a[c] += b);
        }
    }

    /// Gauss–Newton Hessian–vector product `out = JᵀJ v`.
    pub(super) fn hess_vec(&self, lay: &Layout, ev: &Evaluation, v: &[f64], out: &mut [f64]) {
        let nfc = lay.fields.len();
        let off = nfc * lay.ndof;
        // node-space perturbations of the field channels, and of the globals
        let dp: Vec<Vec<f64>> = (0..nfc)
            .map(|fi| self.to_nodes(lay, &v[fi * lay.ndof..(fi + 1) * lay.ndof]))
            .collect();
        let mut dglob = [0.0; NPARAM];
        for (gi, &c) in lay.globals.iter().enumerate() {
            dglob[c] = v[off + gi];
        }
        let (mut y, gsum) = block_products(lay, ev, &dp, &dglob);
        self.add_smoothing_product(lay, &dp, &mut y);
        let gextra = self.add_curvature_product(lay, ev, &dp, &dglob, &mut y);
        for fi in 0..nfc {
            self.to_dofs(&y[fi], &mut out[fi * lay.ndof..(fi + 1) * lay.ndof]);
        }
        for (gi, &c) in lay.globals.iter().enumerate() {
            out[off + gi] = gsum[c] + gextra[c];
        }
    }

    /// Add `κ ωᵀω δp` to the field channels' products.
    fn add_smoothing_product(&self, lay: &Layout, dp: &[Vec<f64>], y: &mut [Vec<f64>]) {
        let Some(sm) = &self.smoothing else {
            return;
        };
        let ch = smoothed_channels(lay, sm);
        let xs: Vec<&[f64]> = ch.iter().map(|&(fi, _)| dp[fi].as_slice()).collect();
        let us = smooth_each(sm.op, &xs, lay.n, false);
        let ys: Vec<&[f64]> = us.iter().map(|u| u.as_slice()).collect();
        let ts = smooth_each(sm.op, &ys, lay.n, true);
        for (t, &(fi, c)) in ts.iter().zip(&ch) {
            add_scaled(&mut y[fi], t, sm.coef[c]);
        }
    }

    /// Add the curvature penalty's Gauss–Newton product: to the field
    /// channels' products, and returned for the globals.
    fn add_curvature_product(
        &self,
        lay: &Layout,
        ev: &Evaluation,
        dp: &[Vec<f64>],
        dglob: &Params,
        y: &mut [Vec<f64>],
    ) -> Params {
        let mut gextra = [0.0; NPARAM];
        let (Some(cv), Some(st)) = (self.curvature, &ev.curv_state) else {
            return gextra;
        };
        let n = lay.n;
        let node_delta = |c: usize| -> Vec<f64> {
            match lay.fields.iter().position(|&f| f == c) {
                Some(fi) => dp[fi].clone(),
                None => vec![dglob[c]; n],
            }
        };
        let mut ym = vec![0.0; n];
        let mut yt = vec![0.0; n];
        cv.gn_product(st, &node_delta(M), &node_delta(T), &mut ym, &mut yt);
        for (c, yc) in [(M, &ym), (T, &yt)] {
            if let Some(fi) = lay.fields.iter().position(|&f| f == c) {
                y[fi].par_iter_mut().zip(yc.par_iter()).for_each(|(a, &b)| *a += b);
            } else if self.roles[c] == Role::Global {
                gextra[c] += det_sum(n, |i| yc[i]);
            }
        }
        gextra
    }

    /// Diagonal blocks of `JᵀJ`: one `nfc × nfc` block per dof plus the
    /// global block (row-major).
    pub(super) fn diag_blocks(
        &self,
        lay: &Layout,
        ev: &Evaluation,
        smooth_diag: &Option<Vec<f64>>,
        curv_diag: &Option<Vec<f64>>,
    ) -> (Vec<f64>, Vec<f64>) {
        let node_block = |i: usize| self.node_block(lay, &ev.blocks[i], i, smooth_diag, curv_diag);
        let bs = lay.fields.len() * lay.fields.len();
        let mut field_blocks = vec![0.0; bs * lay.ndof];
        match self.interp {
            None => {
                field_blocks.par_chunks_mut(bs.max(1)).enumerate().for_each(|(i, fb)| {
                    if bs > 0 {
                        fb.copy_from_slice(&node_block(i));
                    }
                });
            }
            Some(a) => {
                // Σ_i a_ik² B_i
                for i in 0..lay.n {
                    let nb = node_block(i);
                    for e in a.offsets[i]..a.offsets[i + 1] {
                        let k = a.cols[e] as usize;
                        let w2 = a.vals[e] * a.vals[e];
                        for q in 0..bs {
                            field_blocks[k * bs + q] += w2 * nb[q];
                        }
                    }
                }
            }
        }
        (field_blocks, global_block(lay, ev))
    }

    /// Node `i`'s field block (data term `b`), including the penalties' diagonals.
    fn node_block(
        &self,
        lay: &Layout,
        b: &[f64; 15],
        i: usize,
        smooth_diag: &Option<Vec<f64>>,
        curv_diag: &Option<Vec<f64>>,
    ) -> Vec<f64> {
        let nfc = lay.fields.len();
        let mut out = vec![0.0; nfc * nfc];
        for (a, &ca) in lay.fields.iter().enumerate() {
            for (bb, &cb) in lay.fields.iter().enumerate() {
                out[a * nfc + bb] = b[sym5(ca, cb)];
            }
            if let (Some(sm), Some(sd)) = (&self.smoothing, smooth_diag) {
                out[a * nfc + a] += sm.coef[ca] * sd[i];
            }
            if let Some(cd) = curv_diag {
                // two surfaces: m gets 2D, t gets 2D/4 (cross terms cancel)
                if ca == M {
                    out[a * nfc + a] += 2.0 * cd[i];
                } else if ca == T {
                    out[a * nfc + a] += 0.5 * cd[i];
                }
            }
        }
        out
    }
}

/// `y_i = B_i δp_i` for the field channels, and its sum over nodes for the
/// globals.
fn block_products(lay: &Layout, ev: &Evaluation, dp: &[Vec<f64>], dglob: &Params) -> (Vec<Vec<f64>>, Params) {
    let n = lay.n;
    let mut y: Vec<Vec<f64>> = vec![vec![0.0; n]; lay.fields.len()];
    let ptrs: Vec<SendPtr> = y.iter_mut().map(|s| SendPtr(s.as_mut_ptr())).collect();
    let fields = &lay.fields;
    let gsum = det_sum5(n, |i| {
        let mut d = *dglob;
        for (fi, &c) in fields.iter().enumerate() {
            d[c] = dp[fi][i];
        }
        let b = &ev.blocks[i];
        let mut r = [0.0; NPARAM];
        for a in 0..NPARAM {
            let mut s = 0.0;
            for bb in 0..NPARAM {
                s += b[sym5(a, bb)] * d[bb];
            }
            r[a] = s;
        }
        for (fi, &c) in fields.iter().enumerate() {
            // SAFETY: each i is written by exactly one task.
            unsafe { *ptrs[fi].0.add(i) = r[c] };
        }
        r
    });
    (y, gsum)
}

/// The global block of `JᵀJ` (data term).
fn global_block(lay: &Layout, ev: &Evaluation) -> Vec<f64> {
    let ng = lay.globals.len();
    let mut gb = vec![0.0; ng * ng];
    let sums: Vec<f64> = (0..15).map(|q| det_sum(ev.blocks.len(), |i| ev.blocks[i][q])).collect();
    for (a, &ca) in lay.globals.iter().enumerate() {
        for (bb, &cb) in lay.globals.iter().enumerate() {
            gb[a * ng + bb] = sums[sym5(ca, cb)];
        }
    }
    gb
}

/// The field channels with a smoothing penalty: (field index, parameter).
fn smoothed_channels(lay: &Layout, sm: &Smoothing) -> Vec<(usize, usize)> {
    lay.fields
        .iter()
        .enumerate()
        .filter(|(_, &c)| sm.coef[c] != 0.0)
        .map(|(fi, &c)| (fi, c))
        .collect()
}

/// `ω x` (or `ωᵀ x`) for each of `xs`, in one pass over the operator.
fn smooth_each(op: &SmoothingOperator, xs: &[&[f64]], n: usize, transpose: bool) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; n]; xs.len()];
    if transpose {
        op.apply_t_many(xs, &mut out);
    } else {
        op.apply_many(xs, &mut out);
    }
    out
}

/// `dst += c src`.
fn add_scaled(dst: &mut [f64], src: &[f64], c: f64) {
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(a, &b)| *a += c * b);
}

struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
