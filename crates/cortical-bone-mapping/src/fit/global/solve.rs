//! The bounded, damped Gauss–Newton (Levenberg–Marquardt) iteration with
//! matrix-free preconditioned conjugate gradients.

use rayon::prelude::*;

use super::{Evaluation, Problem, Settings, SolveInfo, State};
use crate::fit::local::Status;
use crate::model::{M, NPARAM, T};
use crate::util::{cholesky_solve, det_sum};

#[derive(Default)]
struct Clock {
    evaluate: f64,
    hessian: f64,
    precond: f64,
    edge_stop: bool,
    start: Option<std::time::Instant>,
}

macro_rules! timed {
    ($acc:expr, $e:expr) => {{
        let t = std::time::Instant::now();
        let r = $e;
        $acc += t.elapsed().as_secs_f64();
        r
    }};
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

const STALL_WINDOW: usize = 10;
const STALL_DECREASE: f64 = 1e-4;

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

    let mut clock = Clock {
        start: Some(std::time::Instant::now()),
        ..Clock::default()
    };
    let mut ev = timed!(clock.evaluate, problem.evaluate(&lay, state, true));
    let mut mu = 1e-3;
    let mut nu = 2.0;
    let mut status = Status::MaxIterations;
    let mut iterations = 0;
    let mut cg_total = 0;
    let mut trial = state.clone();
    let mut free = vec![true; nv];
    let mut hv = vec![0.0; nv];
    // costs after the last accepted steps, for the stall test
    let mut recent: std::collections::VecDeque<f64> = std::collections::VecDeque::from([ev.cost]);
    let start = std::time::Instant::now();
    let edge_of = |s: &State| -> Vec<f64> { problem.node_params(s).iter().map(|p| p[M] - 0.5 * p[T]).collect() };
    let mut edges = if settings.edge_tol > 0.0 {
        edge_of(state)
    } else {
        Vec::new()
    };

    while iterations < settings.max_iterations {
        // Gauss–Newton diagonal for scaling, damping and the gradient test.
        let (fblocks, gblock) = timed!(clock.precond, problem.diag_blocks(&lay, &ev, &smooth_diag, &curv_diag));
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
            let held = v < off && problem.frozen.is_some_and(|fz| fz[v % ndof]);
            let blocked = held || (xv[v] <= lower[v] && g > 0.0) || (xv[v] >= upper[v] && g < 0.0);
            free[v] = !blocked;
            if !blocked && diag[v] > 0.0 && rnorm > 0.0 {
                gmax = gmax.max(g.abs() / (diag[v].sqrt() * rnorm));
            }
        }
        if gmax <= settings.gtol || !free.iter().any(|&f| f) {
            status = Status::Gtol;
            break;
        }
        // inexact Newton (forcing term min(0.1, sqrt(|g|)) on the scaled
        // gradient): loose solves far from the optimum, tight near it
        let eta = gmax.sqrt().min(0.1).max(settings.cg_tolerance);
        // Inner loop: try steps with increasing damping until one is accepted.
        // Every trial counts towards the iteration budget.
        let mut accepted = false;
        while !accepted {
            if iterations >= settings.max_iterations || cg_total >= settings.cg_max_total {
                unpack(&xv, state);
                return finish(iterations, cg_total, Status::MaxIterations, &ev, &clock);
            }
            iterations += 1;
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
            timed!(clock.precond, pre.apply(&r, &free, &mut z));
            let mut p = z.clone();
            let mut rz = dotp(&r, &z);
            let cg_before = cg_total;
            for _ in 0..settings.cg_max_iterations {
                if cg_total >= settings.cg_max_total {
                    break;
                }
                cg_total += 1;
                timed!(clock.hessian, problem.hess_vec(&lay, &ev, &p, &mut hv));
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
                if dotp(&r, &r).sqrt() <= eta * bnorm {
                    break;
                }
                timed!(clock.precond, pre.apply(&r, &free, &mut z));
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
                return finish(iterations, cg_total, status, &ev, &clock);
            }
            timed!(clock.hessian, problem.hess_vec(&lay, &ev, &step, &mut hv));
            let pred = -(dotp(&ev.grad, &step) + 0.5 * dotp(&step, &hv));
            unpack(&xt, &mut trial);
            let ct = timed!(clock.evaluate, problem.evaluate(&lay, &trial, false)).cost;
            let actual = ev.cost - ct;
            let rho = if pred > 0.0 { actual / pred } else { -1.0 };
            if settings.progress_every > 0 && iterations % settings.progress_every == 0 {
                eprintln!(
                    "  fit iteration {iterations}/{}: cost {:.6e} -> {ct:.6e} ({}), gradient {gmax:.2e}, CG {} \
                     ({cg_total}/{} total), damping {mu:.1e}, {:.1} s",
                    settings.max_iterations,
                    ev.cost,
                    if rho > 1e-4 { "accepted" } else { "rejected" },
                    cg_total - cg_before,
                    settings.cg_max_total,
                    start.elapsed().as_secs_f64()
                );
            }
            if rho > 1e-4 {
                accepted = true;
                let prev = ev.cost;
                xv = xt;
                unpack(&xv, state);
                ev = timed!(clock.evaluate, problem.evaluate(&lay, state, true));
                mu *= (1.0 - (2.0 * rho - 1.0).powi(3)).max(1.0 / 3.0);
                nu = 2.0;
                if actual <= settings.ftol * prev {
                    return finish(iterations, cg_total, Status::Ftol, &ev, &clock);
                }
                if settings.edge_tol > 0.0 {
                    let e = edge_of(state);
                    let (s2, cnt) = (0..e.len())
                        .filter(|&i| !problem.frozen.is_some_and(|fz| fz[i]))
                        .fold((0.0, 0usize), |(a, c), i| (a + (e[i] - edges[i]).powi(2), c + 1));
                    edges = e;
                    if cnt == 0 || (s2 / cnt as f64).sqrt() < settings.edge_tol {
                        clock.edge_stop = true;
                        return finish(iterations, cg_total, Status::Ftol, &ev, &clock);
                    }
                }
                // stall: the last STALL_WINDOW accepted steps together reduced
                // the cost by less than STALL_DECREASE (relative). Happens when
                // steps are cut by the bounds at many nodes (projected Newton).
                recent.push_back(ev.cost);
                if recent.len() > STALL_WINDOW {
                    let old = recent.pop_front().unwrap();
                    if old - ev.cost <= STALL_DECREASE * old {
                        return finish(iterations, cg_total, Status::Ftol, &ev, &clock);
                    }
                }
            } else {
                mu *= nu;
                nu *= 2.0;
                if mu > 1e16 {
                    return finish(iterations, cg_total, Status::Xtol, &ev, &clock);
                }
            }
        }
    }
    finish(iterations, cg_total, status, &ev, &clock)
}

fn finish(iterations: usize, cg: usize, status: Status, ev: &Evaluation, clock: &Clock) -> SolveInfo {
    SolveInfo {
        edge_stop: clock.edge_stop,
        time_evaluate: clock.evaluate,
        time_hessian: clock.hessian,
        time_precond: clock.precond,
        time_total: clock.start.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0),
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
