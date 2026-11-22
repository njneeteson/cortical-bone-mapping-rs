//! The global problem: cost, gradient, Gauss–Newton products and diagonal
//! blocks of the data term and the penalties.

use rayon::prelude::*;

use super::{Evaluation, Layout, Problem, Role, State};
use crate::model::{eval, eval_grad, Params, M, NPARAM, T};
use crate::util::{det_sum, det_sum5, sym5, CHUNK};

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

    /// Node values of every field channel (`A v` or `v`).
    pub(super) fn node_fields(&self, lay: &Layout, field: &[Vec<f64>]) -> Vec<Vec<f64>> {
        field
            .iter()
            .map(|f| match self.interp {
                Some(a) => {
                    let mut out = vec![0.0; lay.n];
                    a.apply(f, &mut out);
                    out
                }
                None => f.clone(),
            })
            .collect()
    }

    pub(super) fn to_dofs(&self, node: &[f64], out: &mut [f64]) {
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
    pub(super) fn params_at(&self, lay: &Layout, nf: &[Vec<f64>], scalars: &Params, i: usize) -> Params {
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

    pub(super) fn data_scale(&self) -> f64 {
        1.0 / (self.profiles.n as f64 * self.profiles.m as f64)
    }

    pub(super) fn evaluate(&self, lay: &Layout, state: &State, derivs: bool) -> Evaluation {
        let n = lay.n;
        let nf = self.node_fields(lay, &state.field);
        let sc = self.data_scale();
        let w: Vec<f64> = self.weights.iter().map(|g| g * sc).collect();
        let rho_c = self.rho_c;
        let x = self.x;
        let mut blocks = if derivs { vec![[0.0; 15]; n] } else { Vec::new() };
        let mut gnode = if derivs { vec![[0.0; NPARAM]; n] } else { Vec::new() };
        let data_cost: f64 = if derivs {
            let parts: Vec<f64> = blocks
                .par_chunks_mut(CHUNK)
                .zip(gnode.par_chunks_mut(CHUNK))
                .enumerate()
                .map(|(ci, (bc, gc))| {
                    let mut total = 0.0;
                    for (k, (b, g)) in bc.iter_mut().zip(gc.iter_mut()).enumerate() {
                        let i = ci * CHUNK + k;
                        let p = self.params_at(lay, &nf, &state.scalars, i);
                        let y = self.profiles.row(i);
                        let w = self.profiles.row_weights(i, &w);
                        for j in 0..x.len() {
                            if w[j] == 0.0 {
                                continue;
                            }
                            let (f, d) = eval_grad(x[j], &p, rho_c);
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
            parts.iter().sum()
        } else {
            det_sum(n, |i| {
                let p = self.params_at(lay, &nf, &state.scalars, i);
                let y = self.profiles.row(i);
                let w = self.profiles.row_weights(i, &w);
                let mut c = 0.0;
                for j in 0..x.len() {
                    let r = eval(x[j], &p, rho_c) - y[j] as f64;
                    c += 0.5 * w[j] * r * r;
                }
                c
            })
        };

        // Node-space gradients of field channels (data + penalties).
        let mut gfield: Vec<Vec<f64>> = if derivs {
            lay.fields
                .iter()
                .map(|&c| gnode.par_iter().map(|g| g[c]).collect())
                .collect()
        } else {
            Vec::new()
        };
        let mut smoothing_cost = 0.0;
        if let Some(sm) = &self.smoothing {
            // all smoothed channels in one pass over the operator
            let ch: Vec<(usize, usize)> = lay
                .fields
                .iter()
                .enumerate()
                .filter(|(_, &c)| sm.coef[c] != 0.0)
                .map(|(fi, &c)| (fi, c))
                .collect();
            let xs: Vec<&[f64]> = ch.iter().map(|&(fi, _)| nf[fi].as_slice()).collect();
            let mut us = vec![vec![0.0; n]; ch.len()];
            sm.op.apply_many(&xs, &mut us);
            for (u, &(_, c)) in us.iter().zip(&ch) {
                smoothing_cost += 0.5 * sm.coef[c] * det_sum(n, |i| u[i] * u[i]);
            }
            if derivs {
                let ys: Vec<&[f64]> = us.iter().map(|u| u.as_slice()).collect();
                let mut ts = vec![vec![0.0; n]; ch.len()];
                sm.op.apply_t_many(&ys, &mut ts);
                for (t, &(fi, c)) in ts.iter().zip(&ch) {
                    gfield[fi]
                        .par_iter_mut()
                        .zip(t.par_iter())
                        .for_each(|(g, &t)| *g += sm.coef[c] * t);
                }
            }
        }
        let mut curvature_cost = 0.0;
        let mut curv_state = None;
        if let Some(cv) = self.curvature {
            let mvals = self.channel_node_values(lay, &nf, &state.scalars, M);
            let tvals = self.channel_node_values(lay, &nf, &state.scalars, T);
            let mut st = cv.new_state();
            let mut gm = vec![0.0; n];
            let mut gt = vec![0.0; n];
            curvature_cost = cv.eval(&mvals, &tvals, &mut st, &mut gm, &mut gt);
            if derivs {
                self.add_node_grad(lay, &mut gfield, &mut gnode, M, &gm);
                self.add_node_grad(lay, &mut gfield, &mut gnode, T, &gt);
            }
            curv_state = Some(st);
        }

        let mut grad = Vec::new();
        if derivs {
            grad = vec![0.0; lay.nvars()];
            for (fi, gf) in gfield.iter().enumerate() {
                self.to_dofs(gf, &mut grad[fi * lay.ndof..(fi + 1) * lay.ndof]);
            }
            let off = lay.fields.len() * lay.ndof;
            for (gi, &c) in lay.globals.iter().enumerate() {
                grad[off + gi] = det_sum(n, |i| gnode[i][c]);
            }
        }
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

    pub(super) fn channel_node_values(&self, lay: &Layout, nf: &[Vec<f64>], scalars: &Params, c: usize) -> Vec<f64> {
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
    pub(super) fn add_node_grad(
        &self,
        lay: &Layout,
        gfield: &mut [Vec<f64>],
        gnode: &mut [[f64; NPARAM]],
        c: usize,
        g: &[f64],
    ) {
        if let Some(fi) = lay.fields.iter().position(|&f| f == c) {
            gfield[fi].par_iter_mut().zip(g.par_iter()).for_each(|(a, &b)| *a += b);
        } else if self.roles[c] == Role::Global {
            gnode.par_iter_mut().zip(g.par_iter()).for_each(|(a, &b)| a[c] += b);
        }
    }

    /// Gauss–Newton Hessian–vector product `out = JᵀJ v`.
    pub(super) fn hess_vec(&self, lay: &Layout, ev: &Evaluation, v: &[f64], out: &mut [f64]) {
        let n = lay.n;
        let nfc = lay.fields.len();
        let off = nfc * lay.ndof;
        // node-space perturbations of the field channels
        let dp: Vec<Vec<f64>> = (0..nfc)
            .map(|fi| {
                let vf = &v[fi * lay.ndof..(fi + 1) * lay.ndof];
                match self.interp {
                    Some(a) => {
                        let mut o = vec![0.0; n];
                        a.apply(vf, &mut o);
                        o
                    }
                    None => vf.to_vec(),
                }
            })
            .collect();
        let mut dglob = [0.0; NPARAM];
        for (gi, &c) in lay.globals.iter().enumerate() {
            dglob[c] = v[off + gi];
        }
        // y_i = B_i δp_i
        let mut y: Vec<Vec<f64>> = vec![vec![0.0; n]; nfc];
        let gsum: Params = {
            let ys: Vec<&mut [f64]> = y.iter_mut().map(|v| v.as_mut_slice()).collect();
            let ptrs: Vec<SendPtr> = ys.into_iter().map(|s| SendPtr(s.as_mut_ptr())).collect();
            let fields = &lay.fields;
            det_sum5(n, |i| {
                let mut d = dglob;
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
            })
        };
        // smoothing: κ ωᵀω δp
        if let Some(sm) = &self.smoothing {
            let ch: Vec<(usize, usize)> = lay
                .fields
                .iter()
                .enumerate()
                .filter(|(_, &c)| sm.coef[c] != 0.0)
                .map(|(fi, &c)| (fi, c))
                .collect();
            let xs: Vec<&[f64]> = ch.iter().map(|&(fi, _)| dp[fi].as_slice()).collect();
            let mut us = vec![vec![0.0; n]; ch.len()];
            sm.op.apply_many(&xs, &mut us);
            let ys: Vec<&[f64]> = us.iter().map(|u| u.as_slice()).collect();
            let mut ts = vec![vec![0.0; n]; ch.len()];
            sm.op.apply_t_many(&ys, &mut ts);
            for (t, &(fi, c)) in ts.iter().zip(&ch) {
                y[fi]
                    .par_iter_mut()
                    .zip(t.par_iter())
                    .for_each(|(a, &b)| *a += sm.coef[c] * b);
            }
        }
        // curvature
        let mut gextra = [0.0; NPARAM];
        if let (Some(cv), Some(st)) = (self.curvature, &ev.curv_state) {
            let node_delta = |c: usize| -> Vec<f64> {
                if let Some(fi) = lay.fields.iter().position(|&f| f == c) {
                    dp[fi].clone()
                } else {
                    vec![dglob[c]; n]
                }
            };
            let dm = node_delta(M);
            let dt = node_delta(T);
            let mut ym = vec![0.0; n];
            let mut yt = vec![0.0; n];
            cv.gn_product(st, &dm, &dt, &mut ym, &mut yt);
            for (c, yc) in [(M, &ym), (T, &yt)] {
                if let Some(fi) = lay.fields.iter().position(|&f| f == c) {
                    y[fi].par_iter_mut().zip(yc.par_iter()).for_each(|(a, &b)| *a += b);
                } else if self.roles[c] == Role::Global {
                    gextra[c] += det_sum(n, |i| yc[i]);
                }
            }
        }
        for fi in 0..nfc {
            self.to_dofs(&y[fi], &mut out[fi * lay.ndof..(fi + 1) * lay.ndof]);
        }
        for (gi, &c) in lay.globals.iter().enumerate() {
            out[off + gi] = gsum[c] + gextra[c];
        }
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
        let nfc = lay.fields.len();
        let ng = lay.globals.len();
        let bs = nfc * nfc;
        let mut field_blocks = vec![0.0; bs * lay.ndof];
        // per-node field block, including penalty diagonals
        let node_block = |i: usize| -> Vec<f64> {
            let b = &ev.blocks[i];
            let mut out = vec![0.0; bs];
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
        };
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
        let mut gb = vec![0.0; ng * ng];
        let sums: Vec<f64> = (0..15).map(|q| det_sum(ev.blocks.len(), |i| ev.blocks[i][q])).collect();
        for (a, &ca) in lay.globals.iter().enumerate() {
            for (bb, &cb) in lay.globals.iter().enumerate() {
                gb[a * ng + bb] = sums[sym5(ca, cb)];
            }
        }
        (field_blocks, gb)
    }
}

struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
