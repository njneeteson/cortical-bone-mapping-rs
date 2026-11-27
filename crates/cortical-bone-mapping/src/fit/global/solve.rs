//! The bounded, damped Gauss–Newton (Levenberg–Marquardt) iteration with
//! matrix-free preconditioned conjugate gradients.
//!
//! Each outer iteration linearises the problem once (gradient `g`,
//! Gauss–Newton diagonal blocks) and then tries steps of increasing damping
//! until one lowers the cost enough:
//!
//! 1. **Gradient test and active set.** Variables at a bound with the
//!    gradient pointing out of the box, and frozen nodes, are held (projected
//!    Newton; Bertsekas 1982). The scaled gradient (below) over the others
//!    below `gtol` ends the solve.
//! 2. **Inexact Newton step.** Solve `(JᵀJ + μD) δ = −g` over the free
//!    variables by preconditioned conjugate gradients (Hestenes & Stiefel
//!    1952) to a relative residual `η = max(min(cg_forcing, √gmax), cg_tolerance)`.
//!    A forcing term that shrinks with the gradient gives superlinear
//!    convergence while keeping early, far-from-optimal solves cheap (Dembo
//!    et al. 1982; Nocedal & Wright 2006 use `min(0.5, √‖∇f‖)`). The
//!    preconditioner is block Jacobi: one damped `nfc × nfc` block per dof and
//!    one for the globals, each factored by Cholesky on its free rows.
//! 3. **Projection and gain ratio.** `x' = clamp(x + δ, l, u)`; with the actual
//!    step `s = x' − x`, `ρ = (L(x) − L(x')) / (−gᵀs − ½ sᵀ JᵀJ s)`.
//!    **3b. Line search on large residuals.** Gauss–Newton neglects the
//!    residuals' second derivatives; where the residuals are large (model
//!    misfit, not noise), `JᵀJ` overestimates the curvature along the step,
//!    every step falls short by a similar factor, and the iteration
//!    converges only linearly, with `ρ > 1` throughout. So after an accepted
//!    step with `ρ > 1`, the minimum of the parabola through `L(x)`, the
//!    slope `gᵀs` and `L(x')` is tried (one more evaluation, dimensionless
//!    and free of the problem's size).
//! 4. **Damping** (Nielsen 1999): accept if `ρ > 10⁻⁴`, then
//!    `μ ← μ max(1/3, 1 − (2ρ − 1)³)`; otherwise `μ ← μν`, `ν ← 2ν`.
//!
//! **Stopping tests, free of units and of the problem's size.** Each test is
//! a maximum over the variables of a dimensionless measure, never a norm of
//! the whole problem: relative to the whole cost, a node's share shrinks as
//! nodes are added, and relative to the point `x`, a step depends on where
//! the units put their zero (densities in HU or mg HA/cm³). The variables are
//! scaled by their Jacobian, `√D_v` (`D = diag JᵀJ`, the change of the
//! predictions per unit of the variable, as MINPACK's `diag`), and the
//! residuals by the noise, the weighted RMS residual
//! `σ_r = √(Σ w r² / Σ w)`; `W_v` is the weight of the residuals a variable
//! touches (its own samples; all of them for a global):
//!
//! * gradient `max_v |g_v| / (√(D_v W_v) σ_r)`: the cosine between a
//!   variable's Jacobian column and the residuals it touches (MINPACK's test,
//!   Moré 1978, with the variable's own residuals instead of all of them);
//! * step `max_v √(D_v / W_v) |s_v| / σ_r`: the largest change a step makes
//!   in the predictions of any variable's samples, in units of the noise.
//!
//! A solve stops converged on the gradient test (`gtol`); on an accepted step
//! below `xtol` taken where the gradient is within `√gtol` (a rejected step is
//! short because of the damping, and far from the solution the residual is
//! misfit rather than noise, next to which a slow crawl looks negligible); on
//! a stall (the last 10 accepted steps each below `10³ xtol`, the
//! slow crawl of projected steps when many nodes sit on a bound); or, with
//! `edge_tol`, when an accepted step moves the fitted periosteal edges
//! `m − t/2` of the free nodes by less than `edge_tol` (RMS, a length
//! relative to refinement's last move), which is all refinement needs from an
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

/// The last `STALL_WINDOW` accepted steps each below `STALL_FACTOR × xtol`
/// (the step measure of the module documentation) stop the solve.
const STALL_WINDOW: usize = 10;
const STALL_FACTOR: f64 = 10.0;
/// A trial step is accepted above this ratio of actual to predicted decrease.
const ACCEPT_RATIO: f64 = 1e-4;
/// The line search along an accepted step ([`Solver::extend`]) tries
/// lengths `α*` from this (shorter is not worth an evaluation) ...
const MIN_EXTENSION: f64 = 1.25;
/// ... up to this multiple of the step.
const MAX_EXTENSION: f64 = 10.0;
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
    /// The cost at `x` by part (for diagnostics; by default all in `data`).
    fn cost_parts(&self, x: &[f64]) -> Costs {
        let total = self.cost(x);
        Costs {
            total,
            data: total,
            ..Costs::default()
        }
    }
    /// The cost, its gradient and the linearisation at `x`.
    fn linearise(&self, x: &[f64]) -> (Costs, Vec<f64>, Self::Lin);
    /// The diagonal Gauss–Newton blocks of the dofs and of the globals.
    fn diag_blocks(&self, lin: &Self::Lin) -> (Vec<f64>, Vec<f64>);
    /// `out = JᵀJ v`.
    fn hess_vec(&self, lin: &Self::Lin, v: &[f64], out: &mut [f64]);
    /// The fitted periosteal edge of every dof (for `Settings::edge_tol`).
    fn edges(&self, x: &[f64]) -> Vec<f64>;
    /// The weight of the residuals each dof's values touch (`W_v` of the
    /// module documentation, in the normalisation of the cost), and of all
    /// residuals (a global's).
    fn data_weights(&self) -> (Vec<f64>, f64);
}

/// Minimise `problem` from `x` (clamped into the bounds first), updating `x`.
pub(crate) fn minimise<P: LeastSquares>(problem: &P, x: &mut Vec<f64>, settings: &Settings) -> SolveInfo {
    let (nfc, ndof, ng) = problem.shape();
    log::debug!(
        "solve: {} variables ({nfc} field channels x {ndof} dofs + {ng} globals), \
         at most {} iterations and {} CG iterations",
        x.len(),
        settings.max_iterations,
        settings.cg_max_total
    );
    let mut solver = Solver::new(problem, std::mem::take(x), settings);
    let status = solver.run();
    let info = solver.info(status);
    let level = if status == Status::MaxIterations {
        log::Level::Warn
    } else {
        log::Level::Debug
    };
    log::log!(
        level,
        "solve: {status:?} after {} iterations ({} CG) in {:.2} s; cost {:.6e} (data {:.6e}, \
         smoothing {:.6e}, curvature {:.6e}); time: evaluation {:.2} s, products {:.2} s, \
         preconditioner {:.2} s",
        info.iterations,
        info.cg_iterations,
        info.time_total,
        info.cost,
        info.data_cost,
        info.smoothing_cost,
        info.curvature_cost,
        info.time_evaluate,
        info.time_hessian,
        info.time_precond
    );
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
    /// The ratio of the actual to the predicted decrease.
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
    /// The data weight `W_v` of every variable, and of all residuals.
    weight: Vec<f64>,
    weight_total: f64,
    /// The step measure of the last trial step, and of the last accepted
    /// steps (for the stall test).
    last_step: f64,
    recent: VecDeque<f64>,
    /// The scaled gradient where the last trial step started.
    last_gmax: f64,
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
        let (dof_weight, weight_total) = problem.data_weights();
        let (nfc, ndof, _) = shape;
        let weight = (0..nv)
            .map(|v| {
                if v < nfc * ndof {
                    dof_weight[v % ndof]
                } else {
                    weight_total
                }
            })
            .collect();
        Solver {
            problem,
            settings,
            shape,
            lower,
            upper,
            x,
            weight,
            weight_total,
            last_step: f64::INFINITY,
            recent: VecDeque::new(),
            last_gmax: f64::INFINITY,
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
                log::debug!(
                    "solve stops: scaled gradient {:.2e} <= gtol {:.0e}, or no free variable",
                    lin.gmax,
                    self.settings.gtol
                );
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
    /// `η = max(min(cg_forcing, √gmax), cg_tolerance)` of the linear solves.
    fn linearise(&mut self) -> Linearisation {
        let (fblocks, gblock) = timed!(self.clock.precond, self.problem.diag_blocks(&self.lin));
        let diag = diagonal(self.shape, &fblocks, &gblock);
        let dmax = diag.iter().cloned().fold(0.0, f64::max);
        let dfloor = 1e-12 * dmax.max(1e-300);
        let gmax = self.gradient_test(&diag);
        // inexact Newton (forcing term min(0.1, sqrt(|g|)) on the scaled
        // gradient): loose solves far from the optimum, tight near it
        let eta = gmax
            .sqrt()
            .min(self.settings.cg_forcing)
            .max(self.settings.cg_tolerance);
        Linearisation {
            fblocks,
            gblock,
            diag,
            dfloor,
            gmax,
            eta,
        }
    }

    /// The noise level `σ_r`: the weighted RMS residual of the data.
    fn noise(&self) -> f64 {
        (2.0 * self.costs.data / self.weight_total.max(1e-300)).sqrt()
    }

    /// Mark the variables held at a bound (or frozen) and return the largest
    /// scaled gradient `|g_v| / (√(D_v W_v) σ_r)` over the others (module
    /// documentation); 0 for a perfect fit.
    fn gradient_test(&mut self, diag: &[f64]) -> f64 {
        let (nfc, ndof, _) = self.shape;
        let off = nfc * ndof;
        let sigma = self.noise();
        let mut gmax: f64 = 0.0;
        for v in 0..self.x.len() {
            let g = self.grad[v];
            let held = v < off && self.problem.held(v % ndof);
            let at_bound = (self.x[v] <= self.lower[v] && g > 0.0) || (self.x[v] >= self.upper[v] && g < 0.0);
            let blocked = held || at_bound;
            self.free[v] = !blocked;
            let w = self.weight[v];
            if !blocked && diag[v] > 0.0 && w > 0.0 && sigma > 0.0 {
                gmax = gmax.max(g.abs() / ((diag[v] * w).sqrt() * sigma));
            }
        }
        gmax
    }

    /// Try steps with increasing damping until one is accepted. Every trial
    /// counts towards the iteration budget. `Some` ends the solve.
    fn damped_step(&mut self, lin: &Linearisation) -> Option<Status> {
        loop {
            if self.iterations >= self.settings.max_iterations || self.cg_total >= self.settings.cg_max_total {
                log::debug!(
                    "solve stops: budget spent ({} of {} iterations, {} of {} CG iterations)",
                    self.iterations,
                    self.settings.max_iterations,
                    self.cg_total,
                    self.settings.cg_max_total
                );
                return Some(Status::MaxIterations);
            }
            self.iterations += 1;
            let damp: Vec<f64> = lin.diag.iter().map(|&d| self.mu * d.max(lin.dfloor)).collect();
            let pre = self.preconditioner(lin, &damp);
            let cg_before = self.cg_total;
            let delta = self.pcg(&pre, &damp, lin.eta);
            let xt = self.project(&delta);
            let step: Vec<f64> = xt.par_iter().zip(self.x.par_iter()).map(|(a, b)| a - b).collect();
            self.last_step = self.step_measure(&step, lin);
            self.last_gmax = lin.gmax;
            if self.last_step == 0.0 {
                log::debug!("solve stops: no variable can move (all held at their bounds)");
                return Some(Status::Xtol);
            }
            let trial = self.try_step(&xt, &step);
            self.report(&trial, lin.gmax, self.cg_total - cg_before);
            if log::log_enabled!(log::Level::Debug) {
                self.log_measures(&step, lin);
            }
            if trial.rho > ACCEPT_RATIO {
                let (xt, trial) = self.extend(xt, &step, trial, lin);
                return self.accept(xt, &trial);
            }
            if log::log_enabled!(log::Level::Trace) {
                self.check_direction(&step, &trial);
            }
            self.mu *= self.nu;
            self.nu *= 2.0;
            if self.mu > 1e16 {
                log::debug!("solve stops: no acceptable step (damping above 1e16)");
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
    /// variables, from `δ = 0`, stopping at `‖r‖ ≤ η ‖g‖` in the
    /// preconditioned norm `‖r‖ = √(rᵀM⁻¹r)` (each term has the units of the
    /// cost, so the test does not depend on the variables' units, unlike the
    /// Euclidean norm, which adds up gradients of lengths and of densities),
    /// on negative curvature (`pᵀAp ≤ 0`, which damping prevents in exact
    /// arithmetic) or on the CG budget.
    fn pcg(&mut self, pre: &Precond, damp: &[f64], eta: f64) -> Vec<f64> {
        let nv = self.x.len();
        let mut delta = vec![0.0; nv];
        let mut r: Vec<f64> = (0..nv)
            .map(|v| if self.free[v] { -self.grad[v] } else { 0.0 })
            .collect();
        let mut z = vec![0.0; nv];
        timed!(self.clock.precond, pre.apply(&r, &self.free, &mut z));
        let mut p = z.clone();
        let mut rz = dotp(&r, &z);
        let bnorm = rz.max(0.0).sqrt();
        let mut why = "iteration limit";
        let mut k = 0;
        for _ in 0..self.settings.cg_max_iterations {
            if self.cg_total >= self.settings.cg_max_total {
                why = "total CG budget";
                break;
            }
            self.cg_total += 1;
            k += 1;
            self.damped_product(&p, damp);
            let pap = dotp(&p, &self.hv);
            if !(pap > 0.0) {
                // impossible for a damped Gauss–Newton system in exact
                // arithmetic: the products or the preconditioner are broken
                log::warn!(
                    "CG: non-positive curvature pᵀAp = {pap:e} at CG iteration {k} (damping {:.1e}); \
                     the step so far is used",
                    self.mu
                );
                why = "non-positive curvature";
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
            timed!(self.clock.precond, pre.apply(&r, &self.free, &mut z));
            let rz_new = dotp(&r, &z);
            if rz_new.max(0.0).sqrt() <= eta * bnorm {
                rz = rz_new;
                why = "converged";
                break;
            }
            let beta = rz_new / rz;
            rz = rz_new;
            p.par_iter_mut()
                .zip(z.par_iter())
                .for_each(|(pv, &zv)| *pv = zv + beta * *pv);
        }
        log::trace!(
            "CG: {k} iterations, stopped by {why}; relative residual {:.2e} (target {eta:.2e})",
            rz.max(0.0).sqrt() / bnorm.max(1e-300)
        );
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

    /// The step measure `max_v √(D_v / W_v) |s_v| / σ_r` of the module
    /// documentation: the largest change the step makes in the predictions
    /// of any variable's samples, in units of the noise (0 for a perfect fit,
    /// whose steps are all negligible).
    fn step_measure(&self, step: &[f64], lin: &Linearisation) -> f64 {
        let sigma = self.noise();
        if !(sigma > 0.0) {
            return 0.0;
        }
        let (diag, dfloor) = (&lin.diag, lin.dfloor);
        let largest = (0..step.len())
            .into_par_iter()
            .filter(|&v| self.weight[v] > 0.0)
            .map(|v| (diag[v].max(dfloor) / self.weight[v]).sqrt() * step[v].abs())
            .reduce(|| 0.0, f64::max);
        largest / sigma
    }

    /// Step 3b of the module documentation: after an accepted step `s` (to
    /// `xt`) that lowered the cost by more than the Gauss–Newton model
    /// predicted (`ρ > 1`: the model overestimates the curvature along `s`),
    /// the minimum of the parabola through the cost at `x`, its slope `gᵀs`
    /// and the cost at `xt`, at `α* = −gᵀs / κ` with
    /// `κ = 2 (L(xt) − L(x) − gᵀs)`; taken if it lowers the cost further.
    /// The trial keeps the gain ratio of the unit step (for the damping).
    fn extend(&mut self, xt: Vec<f64>, step: &[f64], trial: Trial, lin: &Linearisation) -> (Vec<f64>, Trial) {
        if trial.rho <= 1.0 {
            return (xt, trial);
        }
        let slope = dotp(&self.grad, step);
        let kappa = 2.0 * (trial.cost - self.costs.total - slope);
        let alpha = (-slope / kappa).min(MAX_EXTENSION);
        if !(kappa > 0.0) || alpha < MIN_EXTENSION {
            return (xt, trial);
        }
        let xa: Vec<f64> = (0..self.x.len())
            .into_par_iter()
            .map(|v| (self.x[v] + alpha * step[v]).clamp(self.lower[v], self.upper[v]))
            .collect();
        let cost = timed!(self.clock.evaluate, self.problem.cost(&xa));
        log::debug!(
            "  line search: α {alpha:.3}, cost {cost:.6e} ({} than the unit step's {:.6e})",
            if cost < trial.cost { "lower" } else { "not lower" },
            trial.cost
        );
        if !(cost < trial.cost) {
            return (xt, trial);
        }
        let sa: Vec<f64> = xa.par_iter().zip(self.x.par_iter()).map(|(a, b)| a - b).collect();
        self.last_step = self.step_measure(&sa, lin);
        (xa, Trial { cost, rho: trial.rho })
    }

    /// Evaluate the trial point against the decrease the linear model predicts.
    fn try_step(&mut self, xt: &[f64], step: &[f64]) -> Trial {
        timed!(self.clock.hessian, self.problem.hess_vec(&self.lin, step, &mut self.hv));
        let pred = -(dotp(&self.grad, step) + 0.5 * dotp(step, &self.hv));
        let cost = timed!(self.clock.evaluate, self.problem.cost(xt));
        let actual = self.costs.total - cost;
        let rho = if pred > 0.0 { actual / pred } else { -1.0 };
        Trial { cost, rho }
    }

    /// Diagnosis of a rejected step (trace level): along its direction `s`,
    /// the analytic slope `gᵀs` and Gauss–Newton curvature `sᵀJᵀJs` (in
    /// `hv`, from [`Self::try_step`]) against central differences of the
    /// cost at a hundredth of the step. A slope that disagrees means a wrong
    /// gradient; a curvature that disagrees, a model that does not hold over
    /// the step (or a wrong product).
    fn check_direction(&self, step: &[f64], trial: &Trial) {
        let h = 1e-2;
        // the cost along the step restricted to `range`
        let at = |sign: f64, range: std::ops::Range<usize>| -> Costs {
            let x: Vec<f64> = (0..step.len())
                .into_par_iter()
                .map(|v| {
                    let s = if range.contains(&v) { step[v] } else { 0.0 };
                    (self.x[v] + sign * h * s).clamp(self.lower[v], self.upper[v])
                })
                .collect();
            self.problem.cost_parts(&x)
        };
        let all = 0..step.len();
        let (cp, cm, c0) = (at(1.0, all.clone()), at(-1.0, all), self.problem.cost_parts(&self.x));
        let slope = dotp(&self.grad, step);
        let curvature = dotp(step, &self.hv);
        let diff = |f: fn(&Costs) -> f64| (f(&cp) - f(&cm)) / (2.0 * h);
        let moved = step.iter().filter(|&&s| s != 0.0).count();
        log::trace!(
            "rejected step ({moved} of {} variables moved): slope gᵀs {slope:.6e} vs difference {:.6e} \
             (data {:.6e}, smoothing {:.6e}, curvature {:.6e}); curvature sᵀJᵀJs {curvature:.6e} vs \
             difference {:.6e}; full step: predicted decrease {:.6e}, actual {:.6e}; cost here {:.6e} \
             (linearised {:.6e})",
            step.len(),
            diff(|c| c.total),
            diff(|c| c.data),
            diff(|c| c.smoothing),
            diff(|c| c.curvature),
            (cp.total - 2.0 * c0.total + cm.total) / (h * h),
            -(slope + 0.5 * curvature),
            self.costs.total - trial.cost,
            c0.total,
            self.costs.total
        );
        // by block: each field channel, then the globals
        let (nfc, n, ng) = self.problem.shape();
        let blocks = (0..nfc).map(|fi| (format!("field {fi}"), fi * n..(fi + 1) * n));
        for (name, range) in blocks.chain(std::iter::once(("globals".to_string(), nfc * n..nfc * n + ng))) {
            let part = dotp(&self.grad[range.clone()], &step[range.clone()]);
            let (p, m) = (at(1.0, range.clone()), at(-1.0, range.clone()));
            let big = range.clone().max_by(|&a, &b| {
                (self.grad[a] * step[a])
                    .abs()
                    .total_cmp(&(self.grad[b] * step[b]).abs())
            });
            log::trace!(
                "  {name}: slope {part:.6e} vs difference {:.6e} (data {:.6e}); largest gᵥsᵥ at {big:?}: {}",
                (p.total - m.total) / (2.0 * h),
                (p.data - m.data) / (2.0 * h),
                big.map_or(String::new(), |v| format!(
                    "g {:.3e}, s {:.3e}, x {:.6e} in [{:.3e}, {:.3e}]",
                    self.grad[v], step[v], self.x[v], self.lower[v], self.upper[v]
                ))
            );
        }
    }

    /// Where the stopping tests' measures are largest, and how they are
    /// distributed (debug level): the step and gradient measures of every
    /// variable with data, their largest value (variable, as channel and
    /// dof) and their 50th and 99th percentiles.
    fn log_measures(&self, step: &[f64], lin: &Linearisation) {
        let sigma = self.noise();
        if !(sigma > 0.0) {
            return;
        }
        let (nfc, ndof, _) = self.shape;
        let place = |v: usize| {
            if v < nfc * ndof {
                format!("channel {} dof {}", v / ndof, v % ndof)
            } else {
                format!("global {}", v - nfc * ndof)
            }
        };
        // a dof's values on every channel, `*` marking those on a bound
        let node = |v: usize| -> String {
            if v >= nfc * ndof {
                return String::new();
            }
            let k = v % ndof;
            let vals: Vec<String> = (0..nfc)
                .map(|c| {
                    let u = c * ndof + k;
                    let bound = self.x[u] <= self.lower[u] || self.x[u] >= self.upper[u];
                    format!("{:.4}{}", self.x[u], if bound { "*" } else { "" })
                })
                .collect();
            format!(
                " (values [{}], data weight {:.2e} of {:.2e})",
                vals.join(", "),
                self.weight[v],
                self.weight_total
            )
        };
        let summary = |m: Vec<(f64, usize)>| -> String {
            let mut vals: Vec<f64> = m.iter().map(|p| p.0).collect();
            if vals.is_empty() {
                return "none".into();
            }
            let (big, at) = m
                .iter()
                .fold((f64::MIN, 0), |a, &(x, v)| if x > a.0 { (x, v) } else { a });
            vals.sort_by(f64::total_cmp);
            let q = |p: f64| vals[((vals.len() - 1) as f64 * p).round() as usize];
            format!(
                "max {big:.2e} at {}{}, p50 {:.2e}, p99 {:.2e}",
                place(at),
                node(at),
                q(0.5),
                q(0.99)
            )
        };
        let bounded = (0..self.x.len())
            .filter(|&v| self.x[v] <= self.lower[v] || self.x[v] >= self.upper[v])
            .count();
        let with_data = |v: &usize| self.weight[*v] > 0.0 && lin.diag[*v] > 0.0;
        let steps = (0..step.len())
            .filter(with_data)
            .map(|v| {
                (
                    (lin.diag[v].max(lin.dfloor) / self.weight[v]).sqrt() * step[v].abs() / sigma,
                    v,
                )
            })
            .collect();
        let grads = (0..step.len())
            .filter(|v| with_data(v) && self.free[*v])
            .map(|v| (self.grad[v].abs() / ((lin.diag[v] * self.weight[v]).sqrt() * sigma), v))
            .collect();
        log::debug!(
            "  step measure {}; gradient measure {}; {bounded} of {} variables on a bound",
            summary(steps),
            summary(grads),
            self.x.len()
        );
    }

    /// Progress line (every `progress_every` trial steps).
    fn report(&self, trial: &Trial, gmax: f64, cg: usize) {
        let s = self.settings;
        let progress = s.progress_every > 0 && self.iterations.is_multiple_of(s.progress_every);
        if !progress && !log::log_enabled!(log::Level::Debug) {
            return;
        }
        let line = format!(
            "fit iteration {}/{}: cost {:.6e} -> {:.6e} ({}, gain ratio {:.3}), gradient {gmax:.2e}, CG {cg} \
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
            trial.rho,
            self.cg_total,
            s.cg_max_total,
            self.mu,
            self.clock.start.elapsed().as_secs_f64()
        );
        log::debug!("{line}");
        if progress {
            eprintln!("  {line}");
        }
    }

    /// Move to the trial point, relax the damping, and test for convergence
    /// (edge change, stall).
    fn accept(&mut self, xt: Vec<f64>, trial: &Trial) -> Option<Status> {
        self.x = xt;
        let (costs, grad, lin) = timed!(self.clock.evaluate, self.problem.linearise(&self.x));
        (self.costs, self.grad, self.lin) = (costs, grad, lin);
        self.mu *= (1.0 - (2.0 * trial.rho - 1.0).powi(3)).max(1.0 / 3.0);
        self.nu = 2.0;
        // on accepted steps only (a rejected step is short because of the
        // damping), and only near a stationary point: far from the solution
        // the residual is misfit rather than noise, and a slow crawl would
        // look negligible next to it
        if self.last_step <= self.settings.xtol && self.last_gmax <= self.settings.gtol.sqrt() {
            log::debug!(
                "solve stops: the step changed the predictions by {:.2e} of the noise at most (xtol {:.0e})",
                self.last_step,
                self.settings.xtol
            );
            return Some(Status::Xtol);
        }
        if self.settings.edge_tol > 0.0 && self.edges_settled() {
            log::debug!(
                "solve stops: the edges moved less than edge_tol {}",
                self.settings.edge_tol
            );
            self.edge_stop = true;
            return Some(Status::Ftol);
        }
        let stalled = self.stalled();
        if stalled {
            log::debug!(
                "solve stops: stalled (the last {STALL_WINDOW} steps each changed the predictions by less than {:.0e} of the noise)",
                STALL_FACTOR * self.settings.xtol
            );
        }
        stalled.then_some(Status::Ftol)
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

    /// The last `STALL_WINDOW` accepted steps each changed the predictions
    /// by less than `STALL_FACTOR × xtol` of the noise: happens when steps
    /// are cut by the bounds at many nodes (projected Newton).
    fn stalled(&mut self) -> bool {
        self.recent.push_back(self.last_step);
        if self.recent.len() > STALL_WINDOW {
            self.recent.pop_front();
        }
        self.recent.len() == STALL_WINDOW && self.recent.iter().all(|&s| s <= STALL_FACTOR * self.settings.xtol)
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
