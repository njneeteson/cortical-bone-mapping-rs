//! The result of cortical bone mapping and the diagnostics of a fit.

use crate::fit::global::SolveInfo;
use crate::fit::local::Status;
use crate::fit::profiles::Profiles;
use ndarray::{Array3, Zip};

use crate::model::{Params, M, NPARAM, RHO_B, RHO_S, SIGMA, T};
use crate::options::Options;
use crate::surface::mesh::Mesh;
use crate::util::{add, scale, V3};
use crate::volume::Geometry;

/// Diagnostics of a fit.
#[derive(Clone, Debug, Default)]
pub struct FitInfo {
    pub method: String,
    /// Fraction of local fits that converged (local method), or 0/1.
    pub converged: f64,
    pub iterations: usize,
    pub cg_iterations: usize,
    pub cost: f64,
    pub data_cost: f64,
    pub smoothing_cost: f64,
    pub curvature_cost: f64,
    /// A global solve stopped on its iteration or CG budget.
    pub budget_exhausted: bool,
    /// A global solve stopped early because the fitted edges had settled
    /// ([`FitConstraints::edge_tol`](crate::fit::FitConstraints::edge_tol)).
    pub edge_stop: bool,
    /// Wall time per stage (seconds).
    pub timings: Vec<(String, f64)>,
}

/// The result of cortical bone mapping.
#[derive(Clone, Debug)]
pub struct CbmResult {
    /// The full smoothed outer surface.
    pub surface: Mesh,
    /// Indices (into `surface.vertices`) of the fitted nodes.
    pub vertex_ids: Vec<u32>,
    /// The fitted nodes and the faces between them (indices into `points`).
    pub points: Vec<V3>,
    pub faces: Vec<[u32; 3]>,
    /// Inward unit sampling directions.
    pub normals: Vec<V3>,
    /// `[m, t, σ, ρs, ρb]` per node.
    pub params: Vec<Params>,
    /// Weighted RMS residual per node.
    pub residual: Vec<f64>,
    /// The cortical density used (for [`RhoCMode::ProfileMax`](crate::options::RhoCMode::ProfileMax), the median
    /// over nodes).
    pub rho_c: f64,
    /// Per-node cortical densities ([`RhoCMode::ProfileMax`](crate::options::RhoCMode::ProfileMax) only).
    pub rho_c_node: Option<Vec<f64>>,
    /// The single model fitted to all profiles.
    pub global_params: Params,
    /// The PSF used or estimated, if any.
    pub psf: Option<[f64; 3]>,
    pub info: FitInfo,
    /// Sample positions and profiles (if `keep_profiles`).
    pub profiles: Option<(Vec<f64>, Profiles)>,
}

impl CbmResult {
    pub fn n(&self) -> usize {
        self.points.len()
    }
    pub fn param(&self, c: usize) -> Vec<f64> {
        self.params.iter().map(|p| p[c]).collect()
    }
    pub fn thickness(&self) -> Vec<f64> {
        self.param(T)
    }
    /// The cortical density at node `i`.
    pub fn rho_c_at(&self, i: usize) -> f64 {
        self.rho_c_node.as_ref().map_or(self.rho_c, |v| v[i])
    }
    /// Fitted surface `p + (m + side·t/2) n` (`side` −1 periosteal, +1 endosteal).
    fn offset_points(&self, side: f64) -> Vec<V3> {
        self.points
            .iter()
            .zip(&self.normals)
            .zip(&self.params)
            .map(|((&p, &n), q)| add(p, scale(n, q[M] + side * 0.5 * q[T])))
            .collect()
    }
    pub fn periosteal_points(&self) -> Vec<V3> {
        self.offset_points(-1.0)
    }
    pub fn endosteal_points(&self) -> Vec<V3> {
        self.offset_points(1.0)
    }
    /// The full surface with fitted vertices moved to the fitted periosteal
    /// (`side = -1`) or endosteal (`side = +1`) position; other vertices stay.
    pub fn displaced_surface(&self, side: f64) -> Mesh {
        let mut m = self.surface.clone();
        for (k, &v) in self.vertex_ids.iter().enumerate() {
            let q = &self.params[k];
            m.vertices[v as usize] = add(self.points[k], scale(self.normals[k], q[M] + side * 0.5 * q[T]));
        }
        m
    }
    /// Mean `(ρs, ρb)` over the nodes.
    pub fn mean_densities(&self) -> [f64; 2] {
        let n = self.n().max(1) as f64;
        [
            self.params.iter().map(|p| p[RHO_S]).sum::<f64>() / n,
            self.params.iter().map(|p| p[RHO_B]).sum::<f64>() / n,
        ]
    }
    /// Mean `σ` over the nodes.
    pub fn mean_sigma(&self) -> f64 {
        self.params.iter().map(|p| p[SIGMA]).sum::<f64>() / self.n().max(1) as f64
    }
    /// One line about the fit: nodes, solver work, and how the solve ended.
    pub fn summary(&self) -> String {
        let i = &self.info;
        format!(
            "{} nodes, fit {} it / {} CG{}",
            self.n(),
            i.iterations,
            i.cg_iterations,
            if i.budget_exhausted {
                " (budget exhausted)"
            } else if i.edge_stop {
                " (edges settled)"
            } else {
                ""
            }
        )
    }
    /// Write the fitted nodes as VTK PolyData with all parameters attached.
    pub fn write_vtk(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mesh = Mesh {
            vertices: self.points.clone(),
            faces: self.faces.clone(),
        };
        let cols: Vec<Vec<f64>> = (0..NPARAM).map(|c| self.param(c)).collect();
        let names = ["center", "thickness", "sigma", "rho_s", "rho_b"];
        let mut scalars: Vec<(&str, &[f64])> = names.iter().zip(&cols).map(|(n, c)| (*n, c.as_slice())).collect();
        scalars.push(("residual", &self.residual));
        mesh.write_vtk(path, &scalars, &[("normals", &self.normals)])
    }
}

/// Add `t` seconds to the stage `name` of a list of timings (appended if new).
pub(crate) fn add_timing(timings: &mut Vec<(String, f64)>, name: &str, t: f64) {
    match timings.iter_mut().find(|(n, _)| n == name) {
        Some(e) => e.1 += t,
        None => timings.push((name.to_string(), t)),
    }
}

impl FitInfo {
    /// Account for one global solve.
    pub(crate) fn add_solve(&mut self, si: &SolveInfo) {
        self.iterations += si.iterations;
        self.cg_iterations += si.cg_iterations;
        self.cost = si.cost;
        self.data_cost = si.data_cost;
        self.smoothing_cost = si.smoothing_cost;
        self.curvature_cost = si.curvature_cost;
        self.converged = if si.status == Status::MaxIterations { 0.0 } else { 1.0 };
        self.budget_exhausted |= si.status == Status::MaxIterations;
        self.edge_stop |= si.edge_stop;
        for (name, t) in [
            ("solver: evaluation", si.time_evaluate),
            ("solver: Gauss-Newton products", si.time_hessian),
            ("solver: preconditioner", si.time_precond),
            (
                "solver: other",
                si.time_total - si.time_evaluate - si.time_hessian - si.time_precond,
            ),
        ] {
            add_timing(&mut self.timings, name, t);
        }
    }
}

/// Masks `(periosteal, endosteal, cortical)` of the regions enclosed by two
/// surfaces (cortical = periosteal and not endosteal).
pub(crate) fn region_masks(
    periosteal: &Mesh,
    endosteal: &Mesh,
    shape: (usize, usize, usize),
    geom: &Geometry,
) -> [Array3<bool>; 3] {
    let p = periosteal.voxelize(shape, geom);
    let e = endosteal.voxelize(shape, geom);
    let mut c = p.clone();
    Zip::from(&mut c).and(&e).par_for_each(|c, &e| *c = *c && !e);
    [p, e, c]
}

/// Masks `(periosteal, endosteal, cortical)` from a fit: the regions enclosed
/// by the fitted periosteal and endosteal surfaces.
pub fn cortical_masks(result: &CbmResult, shape: (usize, usize, usize), options: &Options) -> [Array3<bool>; 3] {
    region_masks(
        &result.displaced_surface(-1.0),
        &result.displaced_surface(1.0),
        shape,
        &options.geometry,
    )
}
