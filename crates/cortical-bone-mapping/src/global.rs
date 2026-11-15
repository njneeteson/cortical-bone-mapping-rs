//! The global model-fitting engine.
//!
//! One weighted least-squares problem over all profiles at once:
//!
//! ```text
//! L = 1/(2NM) Σ_ij γj (f̂ij − fij)²                         data
//!   + ½ Σ_c κ_c Σ_i ((ω v_c)_i)²                           smoothing
//!   + ½ Σ_surfaces Σ_i ‖r_curv,i‖²                         curvature (extension)
//! ```
//!
//! Each of the five model parameters plays one of three roles
//! ([`Role`]): fixed, a single global value, or a *field* with one value per
//! degree of freedom. Field degrees of freedom are the nodes themselves
//! (regularised method) or control points interpolated to the nodes by a
//! sparse matrix `A` (multigrid method). The original global initialisation
//! (one model for all profiles) is the special case with every parameter
//! global.
//!
//! The problem is solved with a bounded, damped Gauss–Newton (Levenberg–
//! Marquardt) iteration. Each node's 5×5 Gauss–Newton block `Bᵢ = Σⱼ wⱼ ∇f̂ᵢⱼ
//! ∇f̂ᵢⱼᵀ` is formed once per outer iteration; the linear system
//! `(JᵀJ + μD) δ = −g` is then solved matrix-free by preconditioned conjugate
//! gradients (block-Jacobi preconditioner), so the Hessian is never
//! assembled. Memory is ~25 doubles per node beyond the profiles, and every
//! step is parallel. This replaces the L-BFGS-B solver used by Neeteson &
//! Boyd: it exploits the least-squares structure and typically converges in
//! tens rather than hundreds of iterations.

use rayon::prelude::*;

use crate::local::Status;
use crate::model::{eval, eval_grad, Params, M, NPARAM, T};
use crate::operators::{CurvatureOperator, CurvatureState, SmoothingOperator, SparseRows};
use crate::profiles::Profiles;
use crate::util::{cholesky_solve, det_sum, det_sum5, sym5, CHUNK};

/// The role a model parameter plays in a global fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Held at its value in the state (or at a per-node fixed array).
    Fixed,
    /// One value shared by all nodes, optimised.
    Global,
    /// One value per degree of freedom, optimised.
    Field,
}

/// Quadratic smoothing penalty `½ κ_c ‖ω v_c‖²` per field channel.
pub struct Smoothing<'a> {
    pub op: &'a SmoothingOperator,
    pub coef: Params,
}

/// A global least-squares problem; see the module documentation.
pub struct Problem<'a> {
    pub profiles: &'a Profiles,
    pub x: &'a [f64],
    /// Residual weights `γj` (the solver divides by `N M`).
    pub weights: &'a [f64],
    pub rho_c: f64,
    pub roles: [Role; NPARAM],
    /// Per-node values for fixed parameters (e.g. σ from a PSF).
    pub fixed_per_node: [Option<&'a [f64]>; NPARAM],
    /// `A` (N × Q) for multigrid; `None` means one dof per node.
    pub interp: Option<&'a SparseRows<f64>>,
    pub smoothing: Option<Smoothing<'a>>,
    pub curvature: Option<&'a CurvatureOperator>,
    pub lower: Params,
    pub upper: Params,
}

/// The optimisation variables.
#[derive(Clone, Debug)]
pub struct State {
    /// Field values, channel-major: `field[c][k]` for each field channel `c`
    /// (in parameter order), `k` over degrees of freedom.
    pub field: Vec<Vec<f64>>,
    /// Values of global and (scalar) fixed parameters.
    pub scalars: Params,
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub max_iterations: usize,
    pub ftol: f64,
    pub gtol: f64,
    pub xtol: f64,
    pub cg_max_iterations: usize,
    pub cg_tolerance: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            max_iterations: 100,
            ftol: 1e-6,
            gtol: 1e-6,
            xtol: 1e-10,
            cg_max_iterations: 200,
            cg_tolerance: 1e-3,
        }
    }
}

/// Summary of a global solve.
#[derive(Clone, Copy, Debug)]
pub struct SolveInfo {
    pub iterations: usize,
    pub cg_iterations: usize,
    pub status: Status,
    pub cost: f64,
    pub data_cost: f64,
    pub smoothing_cost: f64,
    pub curvature_cost: f64,
}

struct Layout {
    fields: Vec<usize>,  // parameter indices with Role::Field
    globals: Vec<usize>, // parameter indices with Role::Global
    ndof: usize,
    n: usize,
}

impl Layout {
    fn nvars(&self) -> usize {
        self.fields.len() * self.ndof + self.globals.len()
    }
    /// (parameter index) of variable `v`.
    fn param_of(&self, v: usize) -> usize {
        let nf = self.fields.len() * self.ndof;
        if v < nf {
            self.fields[v / self.ndof]
        } else {
            self.globals[v - nf]
        }
    }
}

struct Evaluation {
    cost: f64,
    data_cost: f64,
    smoothing_cost: f64,
    curvature_cost: f64,
    grad: Vec<f64>,
    blocks: Vec<[f64; 15]>,
    curv_state: Option<CurvatureState>,
}

impl<'a> Problem<'a> {
    fn layout(&self) -> Layout {
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
    fn node_fields(&self, lay: &Layout, field: &[Vec<f64>]) -> Vec<Vec<f64>> {
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

    fn data_scale(&self) -> f64 {
        1.0 / (self.profiles.n as f64 * self.profiles.m as f64)
    }

    fn evaluate(&self, lay: &Layout, state: &State, derivs: bool) -> Evaluation {
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
                        for j in 0..x.len() {
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
            let mut u = vec![0.0; n];
            let mut tmp = vec![0.0; n];
            for (fi, &c) in lay.fields.iter().enumerate() {
                if sm.coef[c] == 0.0 {
                    continue;
                }
                sm.op.apply(&nf[fi], &mut u);
                smoothing_cost += 0.5 * sm.coef[c] * det_sum(n, |i| u[i] * u[i]);
                if derivs {
                    sm.op.apply_t(&u, &mut tmp);
                    gfield[fi]
                        .par_iter_mut()
                        .zip(tmp.par_iter())
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
    fn hess_vec(&self, lay: &Layout, ev: &Evaluation, v: &[f64], out: &mut [f64]) {
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
            let mut u = vec![0.0; n];
            let mut t = vec![0.0; n];
            for (fi, &c) in lay.fields.iter().enumerate() {
                if sm.coef[c] == 0.0 {
                    continue;
                }
                sm.op.apply(&dp[fi], &mut u);
                sm.op.apply_t(&u, &mut t);
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
    fn diag_blocks(
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

/// Block-Jacobi preconditioner (factored, damped blocks).
struct Precond {
    nfc: usize,
    ng: usize,
    ndof: usize,
    field: Vec<f64>, // damped blocks, row-major per dof
    glob: Vec<f64>,
}

impl Precond {
    fn apply(&self, r: &[f64], free: &[bool], z: &mut [f64]) {
        let nfc = self.nfc;
        let bs = nfc * nfc;
        let ndof = self.ndof;
        // gather per-dof, solve, scatter (parallel over dofs)
        let sol: Vec<[f64; NPARAM]> = (0..ndof)
            .into_par_iter()
            .map(|k| {
                let mut a = [0.0; 25];
                let mut b = [0.0; NPARAM];
                let mut idx = [0usize; NPARAM];
                let mut nv = 0;
                for f in 0..nfc {
                    let v = f * ndof + k;
                    if free[v] {
                        idx[nv] = f;
                        b[nv] = r[v];
                        nv += 1;
                    }
                }
                for i in 0..nv {
                    for j in 0..nv {
                        a[i * nv + j] = self.field[k * bs + idx[i] * nfc + idx[j]];
                    }
                }
                if nv > 0 {
                    cholesky_solve(&mut a, &mut b, nv);
                }
                let mut out = [0.0; NPARAM];
                for i in 0..nv {
                    out[idx[i]] = b[i];
                }
                out
            })
            .collect();
        for f in 0..nfc {
            z[f * ndof..(f + 1) * ndof]
                .par_iter_mut()
                .enumerate()
                .for_each(|(k, zv)| *zv = sol[k][f]);
        }
        let off = nfc * ndof;
        let ng = self.ng;
        if ng > 0 {
            let mut a = vec![0.0; 25];
            let mut b = [0.0; NPARAM];
            let mut idx = [0usize; NPARAM];
            let mut nv = 0;
            for g in 0..ng {
                if free[off + g] {
                    idx[nv] = g;
                    b[nv] = r[off + g];
                    nv += 1;
                }
            }
            for i in 0..nv {
                for j in 0..nv {
                    a[i * nv + j] = self.glob[idx[i] * ng + idx[j]];
                }
            }
            if nv > 0 {
                cholesky_solve(&mut a, &mut b, nv);
            }
            for g in 0..ng {
                z[off + g] = 0.0;
            }
            for i in 0..nv {
                z[off + idx[i]] = b[i];
            }
        }
        z.par_iter_mut().zip(free.par_iter()).for_each(|(v, &f)| {
            if !f {
                *v = 0.0
            }
        });
    }
}

fn dotp(a: &[f64], b: &[f64]) -> f64 {
    det_sum(a.len(), |i| a[i] * b[i])
}

/// Solve the problem, updating `state` in place.
pub fn solve(problem: &Problem, state: &mut State, settings: &Settings) -> SolveInfo {
    let lay = problem.layout();
    let nv = lay.nvars();
    let nfc = lay.fields.len();
    let ndof = lay.ndof;
    let off = nfc * ndof;
    assert_eq!(state.field.len(), nfc, "state has the wrong number of field channels");
    for f in &state.field {
        assert_eq!(f.len(), ndof, "field length must equal the number of dofs");
    }

    let lower: Vec<f64> = (0..nv).map(|v| problem.lower[lay.param_of(v)]).collect();
    let upper: Vec<f64> = (0..nv).map(|v| problem.upper[lay.param_of(v)]).collect();
    let pack = |s: &State| -> Vec<f64> {
        let mut x = Vec::with_capacity(nv);
        for f in &s.field {
            x.extend_from_slice(f);
        }
        for &c in &lay.globals {
            x.push(s.scalars[c]);
        }
        x
    };
    let unpack = |x: &[f64], s: &mut State| {
        for (fi, f) in s.field.iter_mut().enumerate() {
            f.copy_from_slice(&x[fi * ndof..(fi + 1) * ndof]);
        }
        for (gi, &c) in lay.globals.iter().enumerate() {
            s.scalars[c] = x[off + gi];
        }
    };
    let mut xv = pack(state);
    xv.par_iter_mut()
        .zip(lower.par_iter().zip(upper.par_iter()))
        .for_each(|(x, (&l, &u))| *x = x.clamp(l, u));
    unpack(&xv, state);

    let smooth_diag = problem.smoothing.as_ref().map(|s| s.op.gram_diag());
    let curv_diag = problem.curvature.map(|c| c.gram_diag());

    let mut ev = problem.evaluate(&lay, state, true);
    let mut mu = 1e-3;
    let mut nu = 2.0;
    let mut status = Status::MaxIterations;
    let mut iterations = 0;
    let mut cg_total = 0;
    let mut trial = state.clone();
    let mut free = vec![true; nv];
    let mut hv = vec![0.0; nv];

    while iterations < settings.max_iterations {
        iterations += 1;
        // Gauss–Newton diagonal for scaling, damping and the gradient test.
        let (fblocks, gblock) = problem.diag_blocks(&lay, &ev, &smooth_diag, &curv_diag);
        let bs = nfc * nfc;
        let diag: Vec<f64> = (0..nv)
            .map(|v| {
                if v < off {
                    let (f, k) = (v / ndof, v % ndof);
                    fblocks[k * bs + f * nfc + f]
                } else {
                    let g = v - off;
                    gblock[g * lay.globals.len() + g]
                }
            })
            .collect();
        let dmax = diag.iter().cloned().fold(0.0, f64::max);
        let dfloor = 1e-12 * dmax.max(1e-300);
        let rnorm = (2.0 * ev.cost).sqrt();
        let mut gmax: f64 = 0.0;
        for v in 0..nv {
            let g = ev.grad[v];
            let blocked = (xv[v] <= lower[v] && g > 0.0) || (xv[v] >= upper[v] && g < 0.0);
            free[v] = !blocked;
            if !blocked && diag[v] > 0.0 && rnorm > 0.0 {
                gmax = gmax.max(g.abs() / (diag[v].sqrt() * rnorm));
            }
        }
        if gmax <= settings.gtol || !free.iter().any(|&f| f) {
            status = Status::Gtol;
            break;
        }
        // Inner loop: try steps with increasing damping until one is accepted.
        let mut accepted = false;
        while !accepted {
            let damp: Vec<f64> = diag.iter().map(|&d| mu * d.max(dfloor)).collect();
            let mut pf = fblocks.clone();
            for k in 0..ndof {
                for f in 0..nfc {
                    pf[k * bs + f * nfc + f] += damp[f * ndof + k];
                }
            }
            let mut pg = gblock.clone();
            let ng = lay.globals.len();
            for g in 0..ng {
                pg[g * ng + g] += damp[off + g];
            }
            let pre = Precond {
                nfc,
                ng,
                ndof,
                field: pf,
                glob: pg,
            };
            // PCG on (JᵀJ + μD) δ = −g over the free variables
            let mut delta = vec![0.0; nv];
            let mut r: Vec<f64> = (0..nv).map(|v| if free[v] { -ev.grad[v] } else { 0.0 }).collect();
            let bnorm = dotp(&r, &r).sqrt();
            let mut z = vec![0.0; nv];
            pre.apply(&r, &free, &mut z);
            let mut p = z.clone();
            let mut rz = dotp(&r, &z);
            for _ in 0..settings.cg_max_iterations {
                cg_total += 1;
                problem.hess_vec(&lay, &ev, &p, &mut hv);
                hv.par_iter_mut()
                    .enumerate()
                    .for_each(|(v, h)| *h = if free[v] { *h + damp[v] * p[v] } else { 0.0 });
                let pap = dotp(&p, &hv);
                if !(pap > 0.0) {
                    break;
                }
                let alpha = rz / pap;
                delta
                    .par_iter_mut()
                    .zip(p.par_iter())
                    .for_each(|(d, &pv)| *d += alpha * pv);
                r.par_iter_mut()
                    .zip(hv.par_iter())
                    .for_each(|(rv, &h)| *rv -= alpha * h);
                if dotp(&r, &r).sqrt() <= settings.cg_tolerance * bnorm {
                    break;
                }
                pre.apply(&r, &free, &mut z);
                let rz_new = dotp(&r, &z);
                let beta = rz_new / rz;
                rz = rz_new;
                p.par_iter_mut()
                    .zip(z.par_iter())
                    .for_each(|(pv, &zv)| *pv = zv + beta * *pv);
            }
            // projected trial point and actual step
            let xt: Vec<f64> = (0..nv)
                .into_par_iter()
                .map(|v| (xv[v] + delta[v]).clamp(lower[v], upper[v]))
                .collect();
            let step: Vec<f64> = xt.par_iter().zip(xv.par_iter()).map(|(a, b)| a - b).collect();
            let snorm: f64 = (0..nv)
                .map(|v| diag[v].max(dfloor) * step[v] * step[v])
                .sum::<f64>()
                .sqrt();
            let xnorm: f64 = (0..nv).map(|v| diag[v].max(dfloor) * xv[v] * xv[v]).sum::<f64>().sqrt();
            if snorm <= settings.xtol * (xnorm + settings.xtol) {
                status = Status::Xtol;
                unpack(&xv, state);
                return finish(iterations, cg_total, status, &ev);
            }
            problem.hess_vec(&lay, &ev, &step, &mut hv);
            let pred = -(dotp(&ev.grad, &step) + 0.5 * dotp(&step, &hv));
            unpack(&xt, &mut trial);
            let ct = problem.evaluate(&lay, &trial, false).cost;
            let actual = ev.cost - ct;
            let rho = if pred > 0.0 { actual / pred } else { -1.0 };
            if rho > 1e-4 {
                accepted = true;
                let prev = ev.cost;
                xv = xt;
                unpack(&xv, state);
                ev = problem.evaluate(&lay, state, true);
                mu *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
                nu = 2.0;
                if actual <= settings.ftol * prev {
                    return finish(iterations, cg_total, Status::Ftol, &ev);
                }
            } else {
                mu *= nu;
                nu *= 2.0;
                if mu > 1e16 {
                    return finish(iterations, cg_total, Status::Xtol, &ev);
                }
            }
        }
    }
    finish(iterations, cg_total, status, &ev)
}

fn finish(iterations: usize, cg: usize, status: Status, ev: &Evaluation) -> SolveInfo {
    SolveInfo {
        iterations,
        cg_iterations: cg,
        status,
        cost: ev.cost,
        data_cost: ev.data_cost,
        smoothing_cost: ev.smoothing_cost,
        curvature_cost: ev.curvature_cost,
    }
}

/// Evaluate the total cost of a state (for tests and diagnostics).
pub fn cost(problem: &Problem, state: &State) -> f64 {
    let lay = problem.layout();
    problem.evaluate(&lay, state, false).cost
}

/// Gradient of the cost (for tests).
pub fn gradient(problem: &Problem, state: &State) -> Vec<f64> {
    let lay = problem.layout();
    problem.evaluate(&lay, state, true).grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{residual_weights, RHO_B, RHO_S, SIGMA};
    use crate::operators::{interpolation_matrix, select_control_points, CurvatureMode};
    use crate::util::{Rng, V3};

    fn synth(n: usize, seed: u64, noise: f64) -> (Profiles, Vec<f64>, Vec<Params>, Vec<V3>) {
        let x: Vec<f64> = (0..61).map(|j| -1.0 + j as f64 * 0.05).collect();
        let mut rng = Rng::new(seed);
        let mut values = Vec::new();
        let mut truth = Vec::new();
        let mut pts = Vec::new();
        for i in 0..n {
            let s = i as f64 / n as f64;
            let p = [0.5 + 0.1 * (6.0 * s).sin(), 0.6 + 0.3 * s, 0.15, 50.0, 300.0];
            for &xj in &x {
                values.push((eval(xj, &p, 1000.0) + noise * rng.normal()) as f32);
            }
            truth.push(p);
            pts.push([s * 10.0, 0.0, 0.0]);
        }
        (Profiles { n, m: x.len(), values }, x, truth, pts)
    }

    fn bounds() -> (Params, Params) {
        ([-3.0, 0.05, 0.01, -200.0, -200.0], [6.0, 12.0, 2.0, 600.0, 900.0])
    }

    #[test]
    fn all_global_fit_recovers_mean_model() {
        let (prof, x, _, _) = synth(200, 1, 0.0);
        let w = residual_weights(&x, 1.0);
        let (lo, hi) = bounds();
        let pr = Problem {
            profiles: &prof,
            x: &x,
            weights: &w,
            rho_c: 1000.0,
            roles: [Role::Global; 5],
            fixed_per_node: [None; 5],
            interp: None,
            smoothing: None,
            curvature: None,
            lower: lo,
            upper: hi,
        };
        let mut st = State {
            field: vec![],
            scalars: [0.0, 1.0, 0.1, 0.0, 0.0],
        };
        let info = solve(&pr, &mut st, &Settings::default());
        assert!(info.status.converged(), "{info:?}");
        assert!((st.scalars[RHO_S] - 50.0).abs() < 5.0, "{:?}", st.scalars);
        assert!((st.scalars[RHO_B] - 300.0).abs() < 10.0);
        assert!((st.scalars[T] - 0.75).abs() < 0.1);
    }

    #[test]
    fn field_fit_recovers_per_node_parameters() {
        let (prof, x, truth, _) = synth(300, 2, 0.0);
        let w = residual_weights(&x, 2.0);
        let (lo, hi) = bounds();
        let pr = Problem {
            profiles: &prof,
            x: &x,
            weights: &w,
            rho_c: 1000.0,
            roles: [Role::Field, Role::Field, Role::Field, Role::Global, Role::Global],
            fixed_per_node: [None; 5],
            interp: None,
            smoothing: None,
            curvature: None,
            lower: lo,
            upper: hi,
        };
        let mut st = State {
            field: vec![vec![0.4; 300], vec![0.8; 300], vec![0.2; 300]],
            scalars: [0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let info = solve(
            &pr,
            &mut st,
            &Settings {
                max_iterations: 200,
                ftol: 1e-12,
                gtol: 1e-10,
                ..Default::default()
            },
        );
        assert!(info.status.converged(), "{info:?}");
        for i in 0..300 {
            assert!(
                (st.field[1][i] - truth[i][T]).abs() < 2e-3,
                "node {i}: {} vs {}",
                st.field[1][i],
                truth[i][T]
            );
            assert!((st.field[0][i] - truth[i][M]).abs() < 2e-3);
        }
        assert!((st.scalars[RHO_S] - 50.0).abs() < 0.5);
        let _ = SIGMA;
    }

    #[test]
    fn gradient_matches_finite_differences_with_all_terms() {
        // tiny curved surface patch (UV sphere) so curvature is meaningful
        let mesh = crate::mesh::Mesh::uv_sphere(2.0, 10, 8);
        let n = mesh.vertices.len();
        let normals: Vec<V3> = mesh
            .vertex_normals()
            .iter()
            .map(|v| crate::util::scale(*v, -1.0))
            .collect();
        let x: Vec<f64> = (0..31).map(|j| -1.0 + j as f64 * 0.1).collect();
        let mut rng = Rng::new(3);
        let values: Vec<f32> = (0..n * x.len())
            .map(|_| (300.0 + 200.0 * rng.normal()) as f32)
            .collect();
        let prof = Profiles { n, m: x.len(), values };
        let w = residual_weights(&x, 3.0);
        let sm = SmoothingOperator::new(&mesh.vertices, 6, 0.8, f64::INFINITY);
        for mode in [CurvatureMode::Change, CurvatureMode::Excess { kappa: 0.3 }] {
            for lap_kind in 0..2 {
                let lap = if lap_kind == 0 {
                    crate::operators::SurfaceLaplacian::cotangent(&mesh)
                } else {
                    crate::operators::SurfaceLaplacian::gaussian(&mesh.vertices, 8, 0.8, 3.0)
                };
                let cv = CurvatureOperator::new(lap, &mesh.vertices, &normals, mode, 5.0, [true, true]);
                for use_interp in [false, true] {
                    let ctrl_idx = select_control_points(&mesh.vertices, 1.0, 1);
                    let ctrl: Vec<V3> = ctrl_idx.iter().map(|&i| mesh.vertices[i as usize]).collect();
                    let a = interpolation_matrix(&mesh.vertices, &ctrl, 4, 1.0);
                    let ndof = if use_interp { ctrl.len() } else { n };
                    let (lo, hi) = bounds();
                    let pr = Problem {
                        profiles: &prof,
                        x: &x,
                        weights: &w,
                        rho_c: 1000.0,
                        roles: [Role::Field, Role::Field, Role::Field, Role::Global, Role::Global],
                        fixed_per_node: [None; 5],
                        interp: use_interp.then_some(&a),
                        smoothing: Some(Smoothing {
                            op: &sm,
                            coef: [3.0, 2.0, 1.0, 0.0, 0.0],
                        }),
                        curvature: Some(&cv),
                        lower: lo,
                        upper: hi,
                    };
                    let st = State {
                        field: vec![
                            (0..ndof).map(|_| 0.3 + 0.2 * rng.normal()).collect(),
                            (0..ndof).map(|_| 0.8 + 0.2 * rng.normal()).collect(),
                            (0..ndof).map(|_| 0.3 + 0.05 * rng.uniform()).collect(),
                        ],
                        scalars: [0.0, 0.0, 0.0, 60.0, 320.0],
                    };
                    let g = gradient(&pr, &st);
                    let lay = pr.layout();
                    for v in (0..lay.nvars()).step_by(7) {
                        let h = 1e-6;
                        let mut sp = st.clone();
                        let mut sm_ = st.clone();
                        let bump = |s: &mut State, d: f64| {
                            if v < lay.fields.len() * ndof {
                                s.field[v / ndof][v % ndof] += d;
                            } else {
                                s.scalars[lay.globals[v - lay.fields.len() * ndof]] += d;
                            }
                        };
                        bump(&mut sp, h);
                        bump(&mut sm_, -h);
                        let fd = (cost(&pr, &sp) - cost(&pr, &sm_)) / (2.0 * h);
                        assert!(
                            (fd - g[v]).abs() <= 1e-4 * (1.0 + fd.abs()),
                            "{mode:?} interp={use_interp} var {v}: fd {fd} vs {}",
                            g[v]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn smoothing_reduces_noise_in_thickness() {
        let (prof, x, truth, pts) = synth(400, 4, 60.0);
        let w = residual_weights(&x, 2.0);
        let (lo, hi) = bounds();
        let sm = SmoothingOperator::new(&pts, 10, 0.2, 3.0);
        let run = |lambda: f64| {
            let coef = 1000.0f64.powi(2) / (400.0 * 0.8f64.powi(2)) * lambda;
            let pr = Problem {
                profiles: &prof,
                x: &x,
                weights: &w,
                rho_c: 1000.0,
                roles: [Role::Field, Role::Field, Role::Field, Role::Global, Role::Global],
                fixed_per_node: [None; 5],
                interp: None,
                smoothing: Some(Smoothing {
                    op: &sm,
                    coef: [coef, coef, coef * 16.0, 0.0, 0.0],
                }),
                curvature: None,
                lower: lo,
                upper: hi,
            };
            let mut st = State {
                field: vec![vec![0.5; 400], vec![0.75; 400], vec![0.15; 400]],
                scalars: [0.0, 0.0, 0.0, 50.0, 300.0],
            };
            solve(&pr, &mut st, &Settings::default());
            let err: f64 = (0..400).map(|i| (st.field[1][i] - truth[i][T]).powi(2)).sum::<f64>() / 400.0;
            err.sqrt()
        };
        let e0 = run(0.0);
        let e1 = run(1.0);
        assert!(e1 < 0.7 * e0, "rmse without {e0}, with {e1}");
    }
}
