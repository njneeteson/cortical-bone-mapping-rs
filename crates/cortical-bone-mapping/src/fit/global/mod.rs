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
//! sparse matrix `A` (multigrid method). The global initialisation of our
//! paper (one model for all profiles; Neeteson & Boyd 2025) is the special case
//! with every parameter global.
//!
//! All three terms are sums of squared residuals, so the problem is solved by
//! a bounded, damped Gauss–Newton (Levenberg–Marquardt) iteration that uses
//! that structure ([`solve()`]). Each node's 5×5 Gauss–Newton block
//! `Bᵢ = Σⱼ wⱼ ∇f̂ᵢⱼ ∇f̂ᵢⱼᵀ` is formed once per outer iteration; the linear
//! system `(JᵀJ + μD) δ = −g` is then solved matrix-free by preconditioned
//! conjugate gradients (Hestenes & Stiefel 1952) with a block-Jacobi
//! preconditioner, so the Hessian is never assembled (`problem` forms the
//! products). Memory is ~25 doubles per node beyond the profiles, and every
//! step is parallel. This replaces the L-BFGS-B solver of our 2025 paper: a
//! first-order quasi-Newton method ignores the least-squares structure, and
//! typically needs hundreds of iterations where Gauss–Newton needs tens.
//!
//! # References
//!
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.
//! * M. R. Hestenes, E. Stiefel, *Methods of conjugate gradients for solving
//!   linear systems*, Journal of Research of the National Bureau of Standards
//!   49(6):409–436, 1952.
//! * J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer,
//!   2006 (Gauss–Newton and Levenberg–Marquardt: ch. 10; CG: ch. 5).

use crate::fit::local::Status;
use crate::fit::profiles::Profiles;
use crate::model::{Params, NPARAM};
use crate::operators::{CurvatureOperator, CurvatureState, SmoothingOperator, SparseRows};

pub(crate) mod penalty;
mod prepared;
mod problem;
mod solve;
#[cfg(test)]
mod tests;

pub use prepared::{cost, gradient, solve};
pub(crate) use solve::{minimise, Costs, LeastSquares};

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
    /// Nodes whose field values are held at their value in the state:
    /// boundary conditions for the smoothing and curvature penalties of
    /// their free neighbours. Only with one dof per node (`interp = None`).
    pub frozen: Option<&'a [bool]>,
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
    /// Maximum number of linear solves (accepted and rejected trial steps).
    pub max_iterations: usize,
    pub ftol: f64,
    pub gtol: f64,
    pub xtol: f64,
    /// Maximum CG iterations per linear solve.
    pub cg_max_iterations: usize,
    /// Relative CG residual reached near convergence. Solves stop at the
    /// looser forcing term `min(0.1, sqrt(g))` (`g` the scaled gradient)
    /// while far from it (inexact Newton).
    pub cg_tolerance: f64,
    /// Maximum CG iterations over the whole solve: bounds the work when the
    /// problem is badly posed (status `MaxIterations`).
    pub cg_max_total: usize,
    /// Print the solver's state to stderr every this many iterations
    /// (trial steps; 0 = never).
    pub progress_every: usize,
    /// Stop once an accepted step changes the fitted periosteal edges
    /// `m − t/2` of the free nodes by less than this, in RMS (physical
    /// units; 0 = never). For refinement, where the surface moves on anyway
    /// and only the edges matter.
    pub edge_tol: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            max_iterations: 100,
            ftol: 1e-6,
            gtol: 1e-6,
            xtol: 1e-10,
            cg_max_iterations: 200,
            cg_max_total: 2000,
            cg_tolerance: 1e-3,
            progress_every: 0,
            edge_tol: 0.0,
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
    /// Stopped on [`Settings::edge_tol`].
    pub edge_stop: bool,
    /// Wall time (s) in cost/gradient evaluations, Gauss–Newton
    /// products, the preconditioner (set-up and solves) and the rest.
    pub time_evaluate: f64,
    pub time_hessian: f64,
    pub time_precond: f64,
    pub time_total: f64,
}

struct Layout {
    fields: Vec<usize>,  // parameter indices with Role::Field
    globals: Vec<usize>, // parameter indices with Role::Global
    ndof: usize,
    n: usize,
}

impl Layout {
    /// Number of variables: one per field channel and dof, plus the globals.
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
