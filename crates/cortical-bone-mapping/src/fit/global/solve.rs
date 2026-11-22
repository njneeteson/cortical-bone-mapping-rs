//! The bounded, damped Gauss–Newton (Levenberg–Marquardt) iteration with
//! matrix-free preconditioned conjugate gradients.
//!
//! Each outer iteration linearises the problem once (gradient, Gauss–Newton
//! diagonal blocks) and then tries steps of increasing damping until one
//! lowers the cost enough; each trial step is one inexact PCG solve of
//! `(JᵀJ + μD) δ = −g` over the variables not held at a bound.

use std::collections::VecDeque;
use std::time::Instant;

use rayon::prelude::*;

use super::{Evaluation, Layout, Problem, Settings, SolveInfo, State};
use crate::fit::local::Status;
use crate::model::{M, NPARAM, T};
use crate::util::{cholesky_solve, det_sum};

/// The last `STALL_WINDOW` accepted steps together reducing the cost by less
/// than `STALL_DECREASE` (relative) stops the solve.
const STALL_WINDOW: usize = 10;
const STALL_DECREASE: f64 = 1e-4;
/// A trial step is accepted above this ratio of actual to predicted decrease.
const ACCEPT_RATIO: f64 = 1e-4;

/// Solve the problem, updating `state` in place.
pub fn solve(problem: &Problem, state: &mut State, settings: &Settings) -> SolveInfo {
    let mut solver = Solver::new(problem, state, settings);
    let status = solver.run(state);
    solver.info(status)
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

macro_rules! timed {
    ($acc:expr, $e:expr) => {{
        let t = Instant::now();
        let r = $e;
        $acc += t.elapsed().as_secs_f64();
        r
    }};
}

/// Wall time per part of the solve.
struct Clock {
    evaluate: f64,
    hessian: f64,
    precond: f64,
    start: Instant,
}

/// The problem linearised at the current point.
struct Linearisation {
    /// Gauss–Newton diagonal blocks: per dof (row-major) and global.
    fblocks: Vec<f64>,
    gblock: Vec<f64>,
    /// Their diagonal, and the floor applied to it for damping and scaling.
    diag: Vec<f64>,
    dfloor: f64,
    /// Largest scaled gradient over the free variables.
    gmax: f64,
    /// Relative residual the linear solves stop at.
    eta: f64,
}

/// The outcome of a trial step.
struct Trial {
    cost: f64,
    /// Actual decrease, and its ratio to the predicted one.
    actual: f64,
    rho: f64,
}

struct Solver<'p, 'a> {
    problem: &'p Problem<'a>,
    settings: &'p Settings,
    lay: Layout,
    lower: Vec<f64>,
    upper: Vec<f64>,
    smooth_diag: Option<Vec<f64>>,
    curv_diag: Option<Vec<f64>>,
    /// The current point (packed) and its evaluation.
    x: Vec<f64>,
    ev: Evaluation,
    /// Scratch: the trial state and Hessian–vector products.
    trial: State,
    hv: Vec<f64>,
    /// Variables not held (at a bound, or frozen).
    free: Vec<bool>,
    /// Damping and its growth factor.
    mu: f64,
    nu: f64,
    iterations: usize,
    cg_total: usize,
    /// Costs after the last accepted steps, for the stall test.
    recent: VecDeque<f64>,
    /// Fitted edges after the last accepted step (with `edge_tol`).
    edges: Vec<f64>,
    edge_stop: bool,
    clock: Clock,
}

impl<'p, 'a> Solver<'p, 'a> {
    /// Clamp `state` into the bounds and evaluate it.
    fn new(problem: &'p Problem<'a>, state: &mut State, settings: &'p Settings) -> Self {
        let lay = problem.layout();
        check_state(&lay, state);
        let nv = lay.nvars();
        let lower: Vec<f64> = (0..nv).map(|v| problem.lower[lay.param_of(v)]).collect();
        let upper: Vec<f64> = (0..nv).map(|v| problem.upper[lay.param_of(v)]).collect();
        let mut x = pack(&lay, state);
        x.par_iter_mut()
            .zip(lower.par_iter().zip(upper.par_iter()))
            .for_each(|(x, (&l, &u))| *x = x.clamp(l, u));
        unpack(&lay, &x, state);
        let smooth_diag = problem.smoothing.as_ref().map(|s| s.op.gram_diag());
        let curv_diag = problem.curvature.map(|c| c.gram_diag());
        let mut clock = Clock {
            evaluate: 0.0,
            hessian: 0.0,
            precond: 0.0,
            start: Instant::now(),
        };
        let ev = timed!(clock.evaluate, problem.evaluate(&lay, state, true));
        let edges = if settings.edge_tol > 0.0 {
            edges(problem, state)
        } else {
            Vec::new()
        };
        Solver {
            problem,
            settings,
            lay,
            lower,
            upper,
            smooth_diag,
            curv_diag,
            x,
            recent: VecDeque::from([ev.cost]),
            ev,
            trial: state.clone(),
            hv: vec![0.0; nv],
            free: vec![true; nv],
            mu: 1e-3,
            nu: 2.0,
            iterations: 0,
            cg_total: 0,
            edges,
            edge_stop: false,
            clock,
        }
    }

    /// Iterate until a stopping test; `state` always holds the current point.
    fn run(&mut self, state: &mut State) -> Status {
        while self.iterations < self.settings.max_iterations {
            let lin = self.linearise();
            if lin.gmax <= self.settings.gtol || !self.free.iter().any(|&f| f) {
                return Status::Gtol;
            }
            if let Some(status) = self.damped_step(state, &lin) {
                return status;
            }
        }
        Status::MaxIterations
    }

    /// Diagonal blocks, the gradient test (which also sets the free
    /// variables) and the forcing term of the linear solves.
    fn linearise(&mut self) -> Linearisation {
        let (lay, ev) = (&self.lay, &self.ev);
        let (fblocks, gblock) = timed!(
            self.clock.precond,
            self.problem.diag_blocks(lay, ev, &self.smooth_diag, &self.curv_diag)
        );
        let diag = diagonal(lay, &fblocks, &gblock);
        let dmax = diag.iter().cloned().fold(0.0, f64::max);
        let dfloor = 1e-12 * dmax.max(1e-300);
        let gmax = self.gradient_test(&diag);
        // inexact Newton (forcing term min(0.1, sqrt(|g|)) on the scaled
        // gradient): loose solves far from the optimum, tight near it
        let eta = gmax.sqrt().min(0.1).max(self.settings.cg_tolerance);
        Linearisation {
            fblocks,
            gblock,
            diag,
            dfloor,
            gmax,
            eta,
        }
    }

    /// Mark the variables held at a bound (or frozen) and return the largest
    /// scaled gradient `|g| / (√D ‖r‖)` over the others.
    fn gradient_test(&mut self, diag: &[f64]) -> f64 {
        let off = self.lay.fields.len() * self.lay.ndof;
        let ndof = self.lay.ndof;
        let rnorm = (2.0 * self.ev.cost).sqrt();
        let mut gmax: f64 = 0.0;
        for v in 0..self.x.len() {
            let g = self.ev.grad[v];
            let held = v < off && self.problem.frozen.is_some_and(|fz| fz[v % ndof]);
            let at_bound = (self.x[v] <= self.lower[v] && g > 0.0) || (self.x[v] >= self.upper[v] && g < 0.0);
            let blocked = held || at_bound;
            self.free[v] = !blocked;
            if !blocked && diag[v] > 0.0 && rnorm > 0.0 {
                gmax = gmax.max(g.abs() / (diag[v].sqrt() * rnorm));
            }
        }
        gmax
    }

    /// Try steps with increasing damping until one is accepted. Every trial
    /// counts towards the iteration budget. `Some` ends the solve.
    fn damped_step(&mut self, state: &mut State, lin: &Linearisation) -> Option<Status> {
        loop {
            if self.iterations >= self.settings.max_iterations || self.cg_total >= self.settings.cg_max_total {
                return Some(Status::MaxIterations);
            }
            self.iterations += 1;
            let damp: Vec<f64> = lin.diag.iter().map(|&d| self.mu * d.max(lin.dfloor)).collect();
            let pre = self.preconditioner(lin, &damp);
            let cg_before = self.cg_total;
            let delta = self.pcg(&pre, &damp, lin.eta);
            let xt = self.project(&delta);
            let step: Vec<f64> = xt.par_iter().zip(self.x.par_iter()).map(|(a, b)| a - b).collect();
            if self.negligible(&step, lin) {
                return Some(Status::Xtol);
            }
            let trial = self.try_step(&xt, &step);
            self.report(&trial, lin.gmax, self.cg_total - cg_before);
            if trial.rho > ACCEPT_RATIO {
                return self.accept(state, xt, &trial);
            }
            self.mu *= self.nu;
            self.nu *= 2.0;
            if self.mu > 1e16 {
                return Some(Status::Xtol);
            }
        }
    }

    /// The block-Jacobi preconditioner of the damped system.
    fn preconditioner(&self, lin: &Linearisation, damp: &[f64]) -> Precond {
        let (nfc, ng, ndof) = (self.lay.fields.len(), self.lay.globals.len(), self.lay.ndof);
        let (bs, off) = (nfc * nfc, nfc * ndof);
        let mut field = lin.fblocks.clone();
        for k in 0..ndof {
            for f in 0..nfc {
                field[k * bs + f * nfc + f] += damp[f * ndof + k];
            }
        }
        let mut glob = lin.gblock.clone();
        for g in 0..ng {
            glob[g * ng + g] += damp[off + g];
        }
        Precond {
            nfc,
            ng,
            ndof,
            field,
            glob,
        }
    }

    /// Preconditioned conjugate gradients on `(JᵀJ + μD) δ = −g` over the
    /// free variables, to the relative residual `eta`.
    fn pcg(&mut self, pre: &Precond, damp: &[f64], eta: f64) -> Vec<f64> {
        let nv = self.x.len();
        let mut delta = vec![0.0; nv];
        let mut r: Vec<f64> = (0..nv)
            .map(|v| if self.free[v] { -self.ev.grad[v] } else { 0.0 })
            .collect();
        let bnorm = dotp(&r, &r).sqrt();
        let mut z = vec![0.0; nv];
        timed!(self.clock.precond, pre.apply(&r, &self.free, &mut z));
        let mut p = z.clone();
        let mut rz = dotp(&r, &z);
        for _ in 0..self.settings.cg_max_iterations {
            if self.cg_total >= self.settings.cg_max_total {
                break;
            }
            self.cg_total += 1;
            self.damped_product(&p, damp);
            let pap = dotp(&p, &self.hv);
            if !(pap > 0.0) {
                break;
            }
            let alpha = rz / pap;
            delta
                .par_iter_mut()
                .zip(p.par_iter())
                .for_each(|(d, &pv)| *d += alpha * pv);
            r.par_iter_mut()
                .zip(self.hv.par_iter())
                .for_each(|(rv, &h)| *rv -= alpha * h);
            if dotp(&r, &r).sqrt() <= eta * bnorm {
                break;
            }
            timed!(self.clock.precond, pre.apply(&r, &self.free, &mut z));
            let rz_new = dotp(&r, &z);
            let beta = rz_new / rz;
            rz = rz_new;
            p.par_iter_mut()
                .zip(z.par_iter())
                .for_each(|(pv, &zv)| *pv = zv + beta * *pv);
        }
        delta
    }

    /// `hv = (JᵀJ + μD) p` on the free variables, zero on the others.
    fn damped_product(&mut self, p: &[f64], damp: &[f64]) {
        timed!(
            self.clock.hessian,
            self.problem.hess_vec(&self.lay, &self.ev, p, &mut self.hv)
        );
        let free = &self.free;
        self.hv
            .par_iter_mut()
            .enumerate()
            .for_each(|(v, h)| *h = if free[v] { *h + damp[v] * p[v] } else { 0.0 });
    }

    /// The trial point: the step projected onto the bounds.
    fn project(&self, delta: &[f64]) -> Vec<f64> {
        (0..self.x.len())
            .into_par_iter()
            .map(|v| (self.x[v] + delta[v]).clamp(self.lower[v], self.upper[v]))
            .collect()
    }

    /// The (scaled) step is negligible next to the point.
    fn negligible(&self, step: &[f64], lin: &Linearisation) -> bool {
        let (diag, dfloor, x) = (&lin.diag, lin.dfloor, &self.x);
        let snorm: f64 = (0..x.len())
            .map(|v| diag[v].max(dfloor) * step[v] * step[v])
            .sum::<f64>()
            .sqrt();
        let xnorm: f64 = (0..x.len())
            .map(|v| diag[v].max(dfloor) * x[v] * x[v])
            .sum::<f64>()
            .sqrt();
        snorm <= self.settings.xtol * (xnorm + self.settings.xtol)
    }

    /// Evaluate the trial point against the decrease the linear model predicts.
    fn try_step(&mut self, xt: &[f64], step: &[f64]) -> Trial {
        timed!(
            self.clock.hessian,
            self.problem.hess_vec(&self.lay, &self.ev, step, &mut self.hv)
        );
        let pred = -(dotp(&self.ev.grad, step) + 0.5 * dotp(step, &self.hv));
        unpack(&self.lay, xt, &mut self.trial);
        let cost = timed!(
            self.clock.evaluate,
            self.problem.evaluate(&self.lay, &self.trial, false)
        )
        .cost;
        let actual = self.ev.cost - cost;
        let rho = if pred > 0.0 { actual / pred } else { -1.0 };
        Trial { cost, actual, rho }
    }

    /// Progress line (every `progress_every` trial steps).
    fn report(&self, trial: &Trial, gmax: f64, cg: usize) {
        let s = self.settings;
        if s.progress_every == 0 || !self.iterations.is_multiple_of(s.progress_every) {
            return;
        }
        eprintln!(
            "  fit iteration {}/{}: cost {:.6e} -> {:.6e} ({}), gradient {gmax:.2e}, CG {cg} \
             ({}/{} total), damping {:.1e}, {:.1} s",
            self.iterations,
            s.max_iterations,
            self.ev.cost,
            trial.cost,
            if trial.rho > ACCEPT_RATIO {
                "accepted"
            } else {
                "rejected"
            },
            self.cg_total,
            s.cg_max_total,
            self.mu,
            self.clock.start.elapsed().as_secs_f64()
        );
    }

    /// Move to the trial point, relax the damping, and test for convergence
    /// (relative decrease, edge change, stall).
    fn accept(&mut self, state: &mut State, xt: Vec<f64>, trial: &Trial) -> Option<Status> {
        let prev = self.ev.cost;
        self.x = xt;
        unpack(&self.lay, &self.x, state);
        self.ev = timed!(self.clock.evaluate, self.problem.evaluate(&self.lay, state, true));
        self.mu *= (1.0 - (2.0 * trial.rho - 1.0).powi(3)).max(1.0 / 3.0);
        self.nu = 2.0;
        if trial.actual <= self.settings.ftol * prev {
            return Some(Status::Ftol);
        }
        if self.settings.edge_tol > 0.0 && self.edges_settled(state) {
            self.edge_stop = true;
            return Some(Status::Ftol);
        }
        self.stalled().then_some(Status::Ftol)
    }

    /// The accepted step changed the free nodes' edges by less than `edge_tol` (RMS).
    fn edges_settled(&mut self, state: &State) -> bool {
        let e = edges(self.problem, state);
        let frozen = self.problem.frozen;
        let (s2, cnt) = (0..e.len())
            .filter(|&i| !frozen.is_some_and(|fz| fz[i]))
            .fold((0.0, 0usize), |(a, c), i| (a + (e[i] - self.edges[i]).powi(2), c + 1));
        self.edges = e;
        cnt == 0 || (s2 / cnt as f64).sqrt() < self.settings.edge_tol
    }

    /// The last `STALL_WINDOW` accepted steps together barely reduced the
    /// cost: happens when steps are cut by the bounds at many nodes
    /// (projected Newton).
    fn stalled(&mut self) -> bool {
        self.recent.push_back(self.ev.cost);
        if self.recent.len() <= STALL_WINDOW {
            return false;
        }
        let old = self.recent.pop_front().unwrap();
        old - self.ev.cost <= STALL_DECREASE * old
    }

    fn info(&self, status: Status) -> SolveInfo {
        let (clock, ev) = (&self.clock, &self.ev);
        SolveInfo {
            edge_stop: self.edge_stop,
            time_evaluate: clock.evaluate,
            time_hessian: clock.hessian,
            time_precond: clock.precond,
            time_total: clock.start.elapsed().as_secs_f64(),
            iterations: self.iterations,
            cg_iterations: self.cg_total,
            status,
            cost: ev.cost,
            data_cost: ev.data_cost,
            smoothing_cost: ev.smoothing_cost,
            curvature_cost: ev.curvature_cost,
        }
    }
}

fn check_state(lay: &Layout, state: &State) {
    assert_eq!(
        state.field.len(),
        lay.fields.len(),
        "state has the wrong number of field channels"
    );
    for f in &state.field {
        assert_eq!(f.len(), lay.ndof, "field length must equal the number of dofs");
    }
}

/// The variables of a state: field channels, then globals.
fn pack(lay: &Layout, s: &State) -> Vec<f64> {
    let mut x = Vec::with_capacity(lay.nvars());
    for f in &s.field {
        x.extend_from_slice(f);
    }
    for &c in &lay.globals {
        x.push(s.scalars[c]);
    }
    x
}

fn unpack(lay: &Layout, x: &[f64], s: &mut State) {
    let (ndof, off) = (lay.ndof, lay.fields.len() * lay.ndof);
    for (fi, f) in s.field.iter_mut().enumerate() {
        f.copy_from_slice(&x[fi * ndof..(fi + 1) * ndof]);
    }
    for (gi, &c) in lay.globals.iter().enumerate() {
        s.scalars[c] = x[off + gi];
    }
}

/// The fitted periosteal edge `m − t/2` of every node.
fn edges(problem: &Problem, s: &State) -> Vec<f64> {
    problem.node_params(s).iter().map(|p| p[M] - 0.5 * p[T]).collect()
}

/// The diagonal of the Gauss–Newton blocks, per variable.
fn diagonal(lay: &Layout, fblocks: &[f64], gblock: &[f64]) -> Vec<f64> {
    let (nfc, ndof) = (lay.fields.len(), lay.ndof);
    let (bs, off) = (nfc * nfc, nfc * ndof);
    (0..lay.nvars())
        .map(|v| {
            if v < off {
                let (f, k) = (v / ndof, v % ndof);
                fblocks[k * bs + f * nfc + f]
            } else {
                let g = v - off;
                gblock[g * lay.globals.len() + g]
            }
        })
        .collect()
}

fn dotp(a: &[f64], b: &[f64]) -> f64 {
    det_sum(a.len(), |i| a[i] * b[i])
}

/// Block-Jacobi preconditioner (factored, damped blocks).
struct Precond {
    nfc: usize,
    ng: usize,
    ndof: usize,
    field: Vec<f64>, // damped blocks, row-major per dof
    glob: Vec<f64>,
}

impl Precond {
    /// `z = P⁻¹ r` on the free variables, zero on the others.
    fn apply(&self, r: &[f64], free: &[bool], z: &mut [f64]) {
        let (nfc, ng, ndof) = (self.nfc, self.ng, self.ndof);
        let (bs, off) = (nfc * nfc, nfc * ndof);
        // per dof (in parallel), then scattered channel by channel
        let sol: Vec<[f64; NPARAM]> = (0..ndof)
            .into_par_iter()
            .map(|k| {
                let at = |f: usize| f * ndof + k;
                solve_free_block(
                    nfc,
                    |f| free[at(f)],
                    |f| r[at(f)],
                    |i, j| self.field[k * bs + i * nfc + j],
                )
            })
            .collect();
        for f in 0..nfc {
            z[f * ndof..(f + 1) * ndof]
                .par_iter_mut()
                .enumerate()
                .for_each(|(k, zv)| *zv = sol[k][f]);
        }
        if ng > 0 {
            let g = solve_free_block(ng, |g| free[off + g], |g| r[off + g], |i, j| self.glob[i * ng + j]);
            z[off..off + ng].copy_from_slice(&g[..ng]);
        }
        z.par_iter_mut().zip(free.par_iter()).for_each(|(v, &f)| {
            if !f {
                *v = 0.0
            }
        });
    }
}

/// Solve an `n × n` block (`n ≤ 5`, entries `entry(i, j)`) restricted to its
/// free rows and columns; zero at the others.
fn solve_free_block(
    n: usize,
    free: impl Fn(usize) -> bool,
    rhs: impl Fn(usize) -> f64,
    entry: impl Fn(usize, usize) -> f64,
) -> [f64; NPARAM] {
    let mut idx = [0usize; NPARAM];
    let mut b = [0.0; NPARAM];
    let mut nv = 0;
    for f in 0..n {
        if free(f) {
            idx[nv] = f;
            b[nv] = rhs(f);
            nv += 1;
        }
    }
    let mut a = [0.0; NPARAM * NPARAM];
    for i in 0..nv {
        for j in 0..nv {
            a[i * nv + j] = entry(idx[i], idx[j]);
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
}
