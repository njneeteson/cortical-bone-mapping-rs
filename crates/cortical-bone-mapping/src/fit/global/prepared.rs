//! The profile problem ([`Problem`]) as a [`LeastSquares`] problem for the
//! solver, and the entry points [`solve`], [`cost`] and [`gradient`].
//!
//! The solver works on one packed vector of variables: the field channels
//! one after the other (each with one value per dof), then the globals. A
//! [`State`] holds the same values by channel and by parameter; [`pack`] and
//! [`unpack`] convert between the two. The Gram diagonals of the penalties
//! (for the preconditioner) depend only on the operators, so they are
//! computed once per solve.

use super::solve::{minimise, Costs, LeastSquares};
use super::{Evaluation, Layout, Problem, Settings, SolveInfo, State};
use crate::model::{M, T};

/// Solve the problem, updating `state` in place.
pub fn solve(problem: &Problem, state: &mut State, settings: &Settings) -> SolveInfo {
    let prepared = Prepared::new(problem, state);
    let mut x = pack(&prepared.lay, state);
    let info = minimise(&prepared, &mut x, settings);
    unpack(&prepared.lay, &x, state);
    info
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

/// A [`Problem`] with its layout, the Gram diagonals of its penalties, and a
/// template state: the starting state, whose fixed scalars every evaluation
/// keeps.
struct Prepared<'p, 'a> {
    problem: &'p Problem<'a>,
    lay: Layout,
    smooth_diag: Option<Vec<f64>>,
    curv_diag: Option<Vec<f64>>,
    template: State,
}

impl<'p, 'a> Prepared<'p, 'a> {
    fn new(problem: &'p Problem<'a>, start: &State) -> Self {
        let lay = problem.layout();
        check_state(&lay, start);
        let template = start.clone();
        Prepared {
            problem,
            smooth_diag: problem.smoothing.as_ref().map(|s| s.op.gram_diag()),
            curv_diag: problem.curvature.map(|c| c.gram_diag()),
            lay,
            template,
        }
    }

    /// The state of the variables `x`.
    fn state(&self, x: &[f64]) -> State {
        let mut s = self.template.clone();
        unpack(&self.lay, x, &mut s);
        s
    }
}

impl LeastSquares for Prepared<'_, '_> {
    type Lin = Evaluation;

    fn shape(&self) -> (usize, usize, usize) {
        (self.lay.fields.len(), self.lay.ndof, self.lay.globals.len())
    }

    fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let nv = self.lay.nvars();
        let lower = (0..nv).map(|v| self.problem.lower[self.lay.param_of(v)]).collect();
        let upper = (0..nv).map(|v| self.problem.upper[self.lay.param_of(v)]).collect();
        (lower, upper)
    }

    fn held(&self, dof: usize) -> bool {
        self.problem.frozen.is_some_and(|fz| fz[dof])
    }

    fn cost(&self, x: &[f64]) -> f64 {
        self.problem.evaluate(&self.lay, &self.state(x), false).cost
    }

    fn linearise(&self, x: &[f64]) -> (Costs, Vec<f64>, Evaluation) {
        let mut ev = self.problem.evaluate(&self.lay, &self.state(x), true);
        let costs = Costs {
            total: ev.cost,
            data: ev.data_cost,
            smoothing: ev.smoothing_cost,
            curvature: ev.curvature_cost,
        };
        let grad = std::mem::take(&mut ev.grad);
        (costs, grad, ev)
    }

    fn diag_blocks(&self, lin: &Evaluation) -> (Vec<f64>, Vec<f64>) {
        self.problem
            .diag_blocks(&self.lay, lin, &self.smooth_diag, &self.curv_diag)
    }

    fn hess_vec(&self, lin: &Evaluation, v: &[f64], out: &mut [f64]) {
        self.problem.hess_vec(&self.lay, lin, v, out)
    }

    fn edges(&self, x: &[f64]) -> Vec<f64> {
        let params = self.problem.node_params(&self.state(x));
        params.iter().map(|p| p[M] - 0.5 * p[T]).collect()
    }
}

/// Panics unless the state has one value per field channel and dof.
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

/// The inverse of [`pack`] (the fixed scalars of `s` are kept).
fn unpack(lay: &Layout, x: &[f64], s: &mut State) {
    let (ndof, off) = (lay.ndof, lay.fields.len() * lay.ndof);
    for (fi, f) in s.field.iter_mut().enumerate() {
        f.copy_from_slice(&x[fi * ndof..(fi + 1) * ndof]);
    }
    for (gi, &c) in lay.globals.iter().enumerate() {
        s.scalars[c] = x[off + gi];
    }
}
