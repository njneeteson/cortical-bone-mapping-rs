//! The bounded, damped Gauss–Newton (Levenberg–Marquardt) iteration with
//! matrix-free preconditioned conjugate gradients.
//!
//! Each outer iteration linearises the problem once (gradient `g`,
//! Gauss–Newton diagonal blocks) and then tries steps of increasing damping
//! until one lowers the cost enough:
//!
//! 1. **Gradient test and active set.** Variables at a bound with the
//!    gradient pointing out of the box, and frozen nodes, are held (projected
//!    Newton; Bertsekas 1982). The scaled gradient
//!    `max_v |g_v| / (√D_v ‖r‖)` over the others (`D = diag JᵀJ`,
//!    `‖r‖ = √(2L)`), the MINPACK test of the local method generalised to
//!    many variables (Moré 1978), below `gtol` ends the solve.
//! 2. **Inexact Newton step.** Solve `(JᵀJ + μD) δ = −g` over the free
//!    variables by preconditioned conjugate gradients (Hestenes & Stiefel
//!    1952) to a relative residual `η = max(min(0.1, √gmax), cg_tolerance)`.
//!    A forcing term that shrinks with the gradient gives superlinear
//!    convergence while keeping early, far-from-optimal solves cheap (Dembo
//!    et al. 1982; Nocedal & Wright 2006 use `min(0.5, √‖∇f‖)`). The
//!    preconditioner is block Jacobi: one damped `nfc × nfc` block per dof and
//!    one for the globals, each factored by Cholesky on its free rows.
//! 3. **Projection and gain ratio.** `x' = clamp(x + δ, l, u)`; with the actual
//!    step `s = x' − x`, `ρ = (L(x) − L(x')) / (−gᵀs − ½ sᵀ JᵀJ s)`.
//! 4. **Damping** (Nielsen 1999): accept if `ρ > 10⁻⁴`, then
//!    `μ ← μ max(1/3, 1 − (2ρ − 1)³)`; otherwise `μ ← μν`, `ν ← 2ν`.
//!
//! A solve stops converged on the gradient test; on an accepted step that
//! lowers the cost by less than `ftol` (relative); on a step below `xtol`
//! relative to the point (in the `√D` norm); on a stall (the last 10 accepted
//! steps together lower the cost by less than 10⁻⁴, the slow crawl of
//! projected steps when many nodes sit on a bound); or, with `edge_tol`, when an
//! accepted step moves the fitted periosteal edges `m − t/2` of the free nodes
//! by less than `edge_tol` (RMS), which is all refinement needs from an
//! intermediate fit. It stops unconverged when it runs out of trial steps or
//! CG iterations.
//!
//! # References
//!
//! * M. R. Hestenes, E. Stiefel, *Methods of conjugate gradients for solving
//!   linear systems*, Journal of Research of the National Bureau of Standards
//!   49(6):409–436, 1952.
//! * R. S. Dembo, S. C. Eisenstat, T. Steihaug, *Inexact Newton methods*, SIAM
//!   Journal on Numerical Analysis 19(2):400–408, 1982.
//! * J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
//!   2006 (inexact Newton: §7.1; CG: ch. 5; Levenberg–Marquardt: §10.3).
//! * D. P. Bertsekas, *Projected Newton methods for optimization problems with
//!   simple constraints*, SIAM Journal on Control and Optimization
//!   20(2):221–246, 1982.
//! * J. J. Moré, *The Levenberg–Marquardt algorithm: implementation and
//!   theory*, in G. A. Watson (ed.), Numerical Analysis, Lecture Notes in
//!   Mathematics 630, Springer, 1978, pp. 105–116.
//! * H. B. Nielsen, *Damping parameter in Marquardt's method*, Technical
//!   Report IMM-REP-1999-05, Technical University of Denmark, 1999.

use std::collections::VecDeque;
use std::time::Instant;

use rayon::prelude::*;

use super::{Settings, SolveInfo};
use crate::fit::local::Status;
use crate::util::{cholesky_solve, det_sum};

/// The last `STALL_WINDOW` accepted steps together reducing the cost by less
/// than `STALL_DECREASE` (relative) stops the solve.
const STALL_WINDOW: usize = 10;
const STALL_DECREASE: f64 = 1e-4;
/// A trial step is accepted above this ratio of actual to predicted decrease.
const ACCEPT_RATIO: f64 = 1e-4;
/// Largest preconditioner block (field channels per dof, or globals).
const MAX_BLOCK: usize = 8;

/// The parts of a cost.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Costs {
    pub total: f64,
    pub data: f64,
    pub smoothing: f64,
    pub curvature: f64,
}

/// A bounded nonlinear least-squares problem for [`minimise`].
///
/// The variables are `nfc` field channels of `ndof` values each, stored
/// channel by channel, then `ng` globals. The solver needs the cost, its
/// gradient, the Gauss–Newton product `JᵀJ v`, and the Gauss–Newton blocks of
/// every dof (`nfc × nfc`, row-major, dof after dof) and of the globals
/// (`ng × ng`) for its block-Jacobi preconditioner.
pub(crate) trait LeastSquares: Sync {
    /// What a linearisation keeps for the products (e.g. Gauss–Newton blocks).
    type Lin: Sync;
    /// `(nfc, ndof, ng)`.
    fn shape(&self) -> (usize, usize, usize);
    /// Lower and upper bound of every variable.
    fn bounds(&self) -> (Vec<f64>, Vec<f64>);
    /// Dofs whose field values are held (boundary conditions).
    fn held(&self, dof: usize) -> bool;
    /// The cost at `x`.
    fn cost(&self, x: &[f64]) -> f64;
    /// The cost, its gradient and the linearisation at `x`.
    fn linearise(&self, x: &[f64]) -> (Costs, Vec<f64>, Self::Lin);
    /// The diagonal Gauss–Newton blocks of the dofs and of the globals.
    fn diag_blocks(&self, lin: &Self::Lin) -> (Vec<f64>, Vec<f64>);
    /// `out = JᵀJ v`.
    fn hess_vec(&self, lin: &Self::Lin, v: &[f64], out: &mut [f64]);
    /// The fitted periosteal edge of every dof (for `Settings::edge_tol`).
    fn edges(&self, x: &[f64]) -> Vec<f64>;
}

/// Minimise `problem` from `x` (clamped into the bounds first), updating `x`.
pub(crate) fn minimise<P: LeastSquares>(problem: &P, x: &mut Vec<f64>, settings: &Settings) -> SolveInfo {
    let mut solver = Solver::new(problem, std::mem::take(x), settings);
    let status = solver.run();
    let info = solver.info(status);
    *x = solver.x;
    info
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

struct Solver<'p, P: LeastSquares> {
    problem: &'p P,
    settings: &'p Settings,
    /// `(nfc, ndof, ng)`.
    shape: (usize, usize, usize),
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// The current point, its costs, gradient and linearisation.
    x: Vec<f64>,
    costs: Costs,
    grad: Vec<f64>,
    lin: P::Lin,
    /// Scratch: Hessian–vector products.
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

impl<'p, P: LeastSquares> Solver<'p, P> {
    /// Clamp `x` into the bounds and linearise there.
    fn new(problem: &'p P, mut x: Vec<f64>, settings: &'p Settings) -> Self {
        let shape = problem.shape();
        let (lower, upper) = problem.bounds();
        x.par_iter_mut()
            .zip(lower.par_iter().zip(upper.par_iter()))
            .for_each(|(x, (&l, &u))| *x = x.clamp(l, u));
        let mut clock = Clock {
            evaluate: 0.0,
            hessian: 0.0,
            precond: 0.0,
            start: Instant::now(),
        };
        let (costs, grad, lin) = timed!(clock.evaluate, problem.linearise(&x));
        let edges = if settings.edge_tol > 0.0 {
            problem.edges(&x)
        } else {
            Vec::new()
        };
        let nv = x.len();
        Solver {
            problem,
            settings,
            shape,
            lower,
            upper,
            x,
            recent: VecDeque::from([costs.total]),
            costs,
            grad,
            lin,
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

    /// Iterate until a stopping test.
    fn run(&mut self) -> Status {
        while self.iterations < self.settings.max_iterations {
            let lin = self.linearise();
            if lin.gmax <= self.settings.gtol || !self.free.iter().any(|&f| f) {
                return Status::Gtol;
            }
            if let Some(status) = self.damped_step(&lin) {
                return status;
            }
        }
        Status::MaxIterations
    }

    /// Steps 1–2 of the module documentation: diagonal blocks, the gradient test
    /// (which also sets the free variables) and the forcing term
    /// `η = max(min(0.1, √gmax), cg_tolerance)` of the linear solves.
    fn linearise(&mut self) -> Linearisation {
        let (fblocks, gblock) = timed!(self.clock.precond, self.problem.diag_blocks(&self.lin));
        let diag = diagonal(self.shape, &fblocks, &gblock);
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
        let (nfc, ndof, _) = self.shape;
        let off = nfc * ndof;
        let rnorm = (2.0 * self.costs.total).sqrt();
        let mut gmax: f64 = 0.0;
        for v in 0..self.x.len() {
            let g = self.grad[v];
            let held = v < off && self.problem.held(v % ndof);
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
    fn damped_step(&mut self, lin: &Linearisation) -> Option<Status> {
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
                return self.accept(xt, &trial);
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
        let (nfc, ndof, ng) = self.shape;
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

    /// Preconditioned conjugate gradients (Hestenes & Stiefel 1952; Nocedal &
    /// Wright 2006, ch. 5) on `(JᵀJ + μD) δ = −g` over the free
    /// variables, from `δ = 0`, stopping at `‖r‖ ≤ η ‖g‖`, on negative curvature
    /// (`pᵀAp ≤ 0`, which damping prevents in exact arithmetic) or on the CG
    /// budget.
    fn pcg(&mut self, pre: &Precond, damp: &[f64], eta: f64) -> Vec<f64> {
        let nv = self.x.len();
        let mut delta = vec![0.0; nv];
        let mut r: Vec<f64> = (0..nv)
            .map(|v| if self.free[v] { -self.grad[v] } else { 0.0 })
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
        timed!(self.clock.hessian, self.problem.hess_vec(&self.lin, p, &mut self.hv));
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
        timed!(self.clock.hessian, self.problem.hess_vec(&self.lin, step, &mut self.hv));
        let pred = -(dotp(&self.grad, step) + 0.5 * dotp(step, &self.hv));
        let cost = timed!(self.clock.evaluate, self.problem.cost(xt));
        let actual = self.costs.total - cost;
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
            self.costs.total,
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
    fn accept(&mut self, xt: Vec<f64>, trial: &Trial) -> Option<Status> {
        let prev = self.costs.total;
        self.x = xt;
        let (costs, grad, lin) = timed!(self.clock.evaluate, self.problem.linearise(&self.x));
        (self.costs, self.grad, self.lin) = (costs, grad, lin);
        self.mu *= (1.0 - (2.0 * trial.rho - 1.0).powi(3)).max(1.0 / 3.0);
        self.nu = 2.0;
        if trial.actual <= self.settings.ftol * prev {
            return Some(Status::Ftol);
        }
        if self.settings.edge_tol > 0.0 && self.edges_settled() {
            self.edge_stop = true;
            return Some(Status::Ftol);
        }
        self.stalled().then_some(Status::Ftol)
    }

    /// The accepted step changed the free nodes' edges by less than `edge_tol` (RMS).
    fn edges_settled(&mut self) -> bool {
        let e = self.problem.edges(&self.x);
        let (s2, cnt) = (0..e.len())
            .filter(|&i| !self.problem.held(i))
            .fold((0.0, 0usize), |(a, c), i| (a + (e[i] - self.edges[i]).powi(2), c + 1));
        self.edges = e;
        cnt == 0 || (s2 / cnt as f64).sqrt() < self.settings.edge_tol
    }

    /// The last `STALL_WINDOW` accepted steps together barely reduced the
    /// cost: happens when steps are cut by the bounds at many nodes
    /// (projected Newton).
    fn stalled(&mut self) -> bool {
        self.recent.push_back(self.costs.total);
        if self.recent.len() <= STALL_WINDOW {
            return false;
        }
        let old = self.recent.pop_front().unwrap();
        old - self.costs.total <= STALL_DECREASE * old
    }

    /// The summary of the solve, ended with `status`.
    fn info(&self, status: Status) -> SolveInfo {
        let (clock, c) = (&self.clock, &self.costs);
        SolveInfo {
            edge_stop: self.edge_stop,
            time_evaluate: clock.evaluate,
            time_hessian: clock.hessian,
            time_precond: clock.precond,
            time_total: clock.start.elapsed().as_secs_f64(),
            iterations: self.iterations,
            cg_iterations: self.cg_total,
            status,
            cost: c.total,
            data_cost: c.data,
            smoothing_cost: c.smoothing,
            curvature_cost: c.curvature,
        }
    }
}

/// The diagonal of the Gauss–Newton blocks, per variable.
fn diagonal((nfc, ndof, ng): (usize, usize, usize), fblocks: &[f64], gblock: &[f64]) -> Vec<f64> {
    let (bs, off) = (nfc * nfc, nfc * ndof);
    (0..off + ng)
        .map(|v| {
            if v < off {
                let (f, k) = (v / ndof, v % ndof);
                fblocks[k * bs + f * nfc + f]
            } else {
                let g = v - off;
                gblock[g * ng + g]
            }
        })
        .collect()
}

/// Dot product, summed in a fixed order ([`det_sum`]).
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
        let sol: Vec<[f64; MAX_BLOCK]> = (0..ndof)
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

/// Solve an `n × n` block (`n ≤ MAX_BLOCK`, entries `entry(i, j)`) restricted
/// to its free rows and columns; zero at the others.
fn solve_free_block(
    n: usize,
    free: impl Fn(usize) -> bool,
    rhs: impl Fn(usize) -> f64,
    entry: impl Fn(usize, usize) -> f64,
) -> [f64; MAX_BLOCK] {
    assert!(
        n <= MAX_BLOCK,
        "preconditioner blocks are at most {MAX_BLOCK} × {MAX_BLOCK}"
    );
    let mut idx = [0usize; MAX_BLOCK];
    let mut b = [0.0; MAX_BLOCK];
    let mut nv = 0;
    for f in 0..n {
        if free(f) {
            idx[nv] = f;
            b[nv] = rhs(f);
            nv += 1;
        }
    }
    let mut a = [0.0; MAX_BLOCK * MAX_BLOCK];
    for i in 0..nv {
        for j in 0..nv {
            a[i * nv + j] = entry(idx[i], idx[j]);
        }
    }
    if nv > 0 {
        cholesky_solve(&mut a, &mut b, nv);
    }
    let mut out = [0.0; MAX_BLOCK];
    for i in 0..nv {
        out[idx[i]] = b[i];
    }
    out
}
