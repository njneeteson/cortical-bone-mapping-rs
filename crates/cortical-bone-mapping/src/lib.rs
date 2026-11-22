//! # Cortical bone mapping
//!
//! A Rust implementation of cortical bone mapping (CBM): model-based
//! measurement of cortical thickness from CT/HR-pQCT images, as developed by
//! Treece et al. (2010, 2012, 2015) and extended with global optimisation and
//! spatial regularisation by Neeteson & Boyd (2025, *Comput. Biol. Med.*,
//! doi:10.1016/j.compbiomed.2025.110636).
//!
//! Given a grey-scale image and a binary segmentation of the outer bone
//! surface, CBM samples a density profile along the inward normal at every
//! vertex of the surface and fits a blurred three-level step model
//! ([`model`]) whose parameters are the cortical mid-line position `m`, the
//! cortical thickness `t`, the blur `σ` and the soft-tissue and trabecular
//! densities `ρs`, `ρb` (the cortical density `ρc` is fixed).
//!
//! Methods ([`Method`]):
//!
//! * **local**: an independent fit per node (the original method);
//! * **multigrid**: `m, t, σ` on sparse control points interpolated to the
//!   nodes, refined coarse to fine (Neeteson & Boyd);
//! * **regularized**: `m, t, σ` per node with a spatial smoothness penalty
//!   (Neeteson & Boyd), optionally plus a **curvature penalty** on the fitted
//!   periosteal/endosteal surfaces ([`CurvatureMode`], extension).
//!
//! The blur can be free, fixed, given by a PSF, or estimated as a PSF from
//! the data ([`SigmaMode`], after Treece et al.). [`refine_segmentation`]
//! iteratively snaps a poor segmentation to the fitted periosteal surface
//! (extension).
//!
//! ```no_run
//! use cortical_bone_mapping::{run, Options, Method, Geometry};
//! # let image = ndarray::Array3::<i16>::zeros((10, 10, 10));
//! # let mask = ndarray::Array3::<u8>::zeros((10, 10, 10));
//! let opts = Options {
//!     geometry: Geometry::new([0.0607; 3]),
//!     method: Method::regularized(),
//!     ..Options::default()
//! };
//! let res = run(image.view(), mask.view(), None::<ndarray::ArrayView3<u8>>, &opts).unwrap();
//! let thickness = res.thickness();
//! ```

// `!(x > 0.0)` is used deliberately: unlike `x <= 0.0` it also rejects NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod bone_phantom;
pub mod global;
pub mod joint_phantom;
pub mod kdtree;
pub mod levelset;
pub mod local;
pub mod mesh;
pub mod model;
pub mod operators;
pub mod pipeline;
pub mod profiles;
pub mod psf;
pub mod refine;
pub mod segmentation;
pub mod surface;
pub mod synthetic;
pub mod util;
pub mod volume;

pub use mesh::Mesh;
pub use operators::CurvatureMode;
pub use pipeline::{
    run, CbmResult, CurvatureOptions, Error, FitInfo, Method, ModelOptions, Options, RhoCMode, SamplingOptions,
    SigmaMode, Site, SolverOptions, SurfaceOptions,
};
pub use profiles::NormalConstraint;
pub use psf::PsfModel;
pub use refine::{cortical_masks, refine_segmentation, FreezeOptions, IterationStats, RefineOptions, RefineResult};
pub use volume::{Geometry, Voxel};
