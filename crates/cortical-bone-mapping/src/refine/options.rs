//! Options and results of refinement.

use ndarray::Array3;

use super::ScannerState;
use crate::result::CbmResult;
use crate::surface::mesh::Mesh;

/// Options of [`refine_segmentation`](super::refine_segmentation).
#[derive(Clone, Debug, PartialEq)]
pub struct RefineOptions {
    /// Maximum number of moves.
    pub max_iterations: usize,
    /// Under-relaxation factor `α` in (0, 1].
    pub relaxation: f64,
    /// Converged when a full fit moves the surface by less than this (RMS,
    /// physical units); `None` = a tenth of the smallest voxel spacing. The
    /// displacement cannot fall much below `α` × the noise of the fitted edge
    /// position, so very small tolerances are never reached on noisy data.
    pub tolerance: Option<f64>,
    /// Taubin iterations applied after each move (keeps the mesh regular;
    /// not used with `level_set`, where the mesh is extracted anew).
    pub smoothing_iterations: usize,
    /// Cap on the displacement of a vertex per iteration (physical units).
    pub max_step: Option<f64>,
    /// Keep the surface of every iteration (for visualisation).
    pub keep_history: bool,
    /// Rasterise the refined surfaces into periosteal, endosteal and
    /// cortical masks of the image's shape (the meshes are always returned).
    pub masks: bool,
    /// Print one line to stderr after every fit (displacement, thickness,
    /// fit iterations, time), and the time per stage at the end.
    pub verbose: bool,
    /// Prefix of the progress lines (e.g. the bone's name).
    pub label: String,
    /// Inexact inner fits: stop the global solve once its steps change the
    /// edges by less than this × the last displacement (0 = always solve
    /// fully).
    pub edge_ratio: f64,
    /// Stop as stalled when, for this many fits, the displacement has not
    /// fallen below 95% of its best and no further region has converged.
    pub stall_iterations: usize,
    /// Regional convergence test: the displacement RMS over the nodes
    /// within this distance of each node (`None` = 5 voxels) …
    pub region_radius: Option<f64>,
    /// … must be below this everywhere (`None` = 2 × the tolerance;
    /// `f64::INFINITY` disables the test). A narrow band of nodes that still
    /// moves (e.g. a surface zipping into a concavity) barely changes the RMS
    /// over the whole surface, but stands out regionally.
    pub region_tolerance: Option<f64>,
    /// Carry the surface as a level set and extract a new mesh every
    /// iteration (remeshing) instead of moving the same vertices (default:
    /// on, as on the clinical knee CT).
    pub level_set: bool,
    /// Regional convergence: freeze regions that have stopped moving
    /// (default: on).
    pub freeze: Option<FreezeOptions>,
    /// The scanner parameters (blur, `ρc`) are held through a refinement
    /// pass and then re-fitted on the refined surface; at most this many
    /// passes (1 = hold the first fit's values throughout; see the module
    /// `scanner`).
    pub scanner_passes: usize,
    /// Under-relaxation `β` in (0, 1] of the scanner parameters between
    /// passes: `p ← p + β (p_fit − p)`.
    pub scanner_relaxation: f64,
    /// Another pass follows only if the re-fit changes a scanner parameter
    /// by more than this (relative).
    pub scanner_tolerance: f64,
}

impl Default for RefineOptions {
    fn default() -> Self {
        RefineOptions {
            max_iterations: 20,
            relaxation: 0.5,
            tolerance: None,
            smoothing_iterations: 2,
            max_step: None,
            keep_history: false,
            masks: true,
            verbose: true,
            label: String::new(),
            edge_ratio: 0.1,
            stall_iterations: 5,
            region_radius: None,
            region_tolerance: None,
            level_set: true,
            freeze: Some(FreezeOptions::default()),
            scanner_passes: 3,
            scanner_relaxation: 0.5,
            scanner_tolerance: 0.01,
        }
    }
}

/// When a region counts as converged; see [`RefineOptions::freeze`].
#[derive(Clone, Debug, PartialEq)]
pub struct FreezeOptions {
    /// Number of consecutive iterations a node must be still.
    pub patience: usize,
    /// A node is still if it moved less than this (`None` = the refinement
    /// tolerance) …
    pub displacement: Option<f64>,
    /// … and its thickness changed by less than this (`None` = the
    /// refinement tolerance).
    pub thickness: Option<f64>,
    /// A node freezes only when every node within this distance is still
    /// (`None` = 3 voxels).
    pub radius: Option<f64>,
    /// Frozen nodes within this distance of a moving node are refitted as
    /// fixed values (`None` = the reach of the regularisation, `cutoff ·
    /// σ_reg`, or `radius` without it).
    pub halo: Option<f64>,
    /// A frozen node thaws once the moving nodes within `radius` of it have
    /// moved this far in total (the largest move nearby, summed over
    /// iterations; `None` = 1 voxel): the surface next to it is changing.
    pub thaw: Option<f64>,
}

impl Default for FreezeOptions {
    fn default() -> Self {
        FreezeOptions {
            patience: 3,
            displacement: None,
            thickness: None,
            radius: None,
            halo: None,
            thaw: None,
        }
    }
}

/// Statistics of one fit of the refinement and the move it implies.
#[derive(Clone, Copy, Debug)]
pub struct IterationStats {
    /// RMS and maximum displacement implied by the fit (over the fitted,
    /// not fixed, nodes).
    pub rms_displacement: f64,
    pub max_displacement: f64,
    pub mean_thickness: f64,
    pub mean_residual: f64,
    /// Surface nodes that may be fitted.
    pub nodes: usize,
    /// Of those, not frozen.
    pub moving_nodes: usize,
    /// A full solve on the whole surface (initial or verifying fit).
    pub full_fit: bool,
    /// The global solve converged (did not exhaust its budget).
    pub fit_converged: bool,
    /// Fitted nodes whose fit could not be trusted to locate the edge.
    pub untrusted: usize,
    /// Largest regional displacement RMS (see
    /// [`RefineOptions::region_radius`]).
    pub max_regional_rms: f64,
}

pub struct RefineResult {
    /// The final fit, on the refined surface.
    pub result: CbmResult,
    /// The refined periosteal surface (= `result.surface`).
    pub periosteal: Mesh,
    /// The fitted endosteal surface.
    pub endosteal: Mesh,
    /// The regions enclosed by the refined surfaces, at the image's shape
    /// (`None` unless [`RefineOptions::masks`]).
    pub periosteal_mask: Option<Array3<bool>>,
    pub endosteal_mask: Option<Array3<bool>>,
    /// `periosteal_mask & !endosteal_mask`.
    pub cortical_mask: Option<Array3<bool>>,
    /// One entry per fit: `history[k]` is the fit on `surfaces[k]`.
    pub history: Vec<IterationStats>,
    /// The input segmentation's surface and the surface after every move (if
    /// `keep_history`). With `level_set` every surface has its own faces.
    pub surfaces: Vec<Mesh>,
    /// A full solve moved the surface by less than the tolerance.
    pub converged: bool,
    /// Stopped because the surface stopped making progress (see `message`).
    pub stalled: bool,
    /// Why refinement stopped, with a diagnosis if it stalled.
    pub message: String,
    /// The scanner parameters held in each pass.
    pub scanner: Vec<ScannerState>,
    /// Wall time per stage, summed over all fits (seconds).
    pub timings: Vec<(String, f64)>,
}
