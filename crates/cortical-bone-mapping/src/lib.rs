//! # Cortical bone mapping
//!
//! A Rust implementation of cortical bone mapping (CBM): model-based
//! measurement of cortical thickness from CT/HR-pQCT images, as developed by
//! Treece et al. (2010, 2012, 2015) and extended with global optimisation and
//! spatial regularisation in our paper (Neeteson & Boyd 2025).
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
//! * **local**: an independent fit per node (Treece et al. 2010; the "local
//!   method" of Neeteson & Boyd 2025);
//! * **multigrid**: `m, t, σ` on sparse control points interpolated to the
//!   nodes, refined coarse to fine (Neeteson & Boyd 2025);
//! * **regularized**: `m, t, σ` per node with a spatial smoothness penalty
//!   (Neeteson & Boyd 2025), optionally plus a **curvature penalty** on the
//!   fitted periosteal/endosteal surfaces ([`CurvatureMode`], extension);
//! * **volumetric**: the model evaluated at every voxel of a band around the
//!   surface, with one PSF for the scan ([`VolumetricOptions`], extension).
//!
//! The blur can be free, fixed, given by a PSF, or estimated as a PSF from
//! the data ([`SigmaMode`], after Treece et al. 2012, 2015).
//! [`refine_segmentation`] iteratively snaps a poor segmentation to the fitted
//! periosteal surface (extension).
//!
//! The Python package (`cortical-bone-mapping-rs`) wraps this crate and has a
//! user guide.
//!
//! Every stage logs through the [`log`] facade (debug level: starts, sizes,
//! durations, solver iterations); see [`logging`].
//!
//! # Example
//!
//! A phantom shell with a known cortex, measured on its own mask and then
//! refined from a mask that lies 0.3 mm outside the bone:
//!
//! ```
//! use cortical_bone_mapping::phantoms::shell::ShellPhantom;
//! use cortical_bone_mapping::{
//!     refine_segmentation, run, Method, NormalConstraint, Options, RefineOptions, SamplingOptions,
//! };
//! # fn main() -> Result<(), cortical_bone_mapping::Error> {
//! let phantom = ShellPhantom { noise: 20.0, ..ShellPhantom::default() };
//! let (image, mask) = phantom.generate();
//! let opts = Options {
//!     geometry: phantom.geometry(),
//!     sampling: SamplingOptions {
//!         outside: 1.5,
//!         inside: 2.5,
//!         normal_constraint: NormalConstraint::InPlane(0), // normals in each slice's plane
//!         ..Default::default()
//!     },
//!     method: Method::regularized(),
//!     ..Options::default()
//! };
//! let res = run(image.view(), mask.view(), None::<ndarray::ArrayView3<u8>>, &opts)?;
//! let t = res.thickness();
//! let rmse = (res.points.iter().zip(&t)
//!     .map(|(&p, &t)| (t - phantom.true_thickness(p)).powi(2))
//!     .sum::<f64>() / t.len() as f64).sqrt();
//! assert!(rmse < 0.03); // mm
//!
//! let rough = ShellPhantom { mask_offset: 0.3, ..phantom.clone() }.generate().1;
//! let refine = RefineOptions { verbose: false, ..RefineOptions::default() };
//! let rr = refine_segmentation(image.view(), rough.view(), None::<ndarray::ArrayView3<u8>>, &opts, &refine)?;
//! assert!(rr.converged);
//! let cortex = &rr.cortical_mask; // boolean, the image's shape
//! # let _ = cortex;
//! # Ok(())
//! # }
//! ```
//!
//! # Where each part comes from
//!
//! Every module documents the equations it implements and either the
//! publication it follows or, for choices of our own, the reasoning behind
//! them, with its own list of references. In short:
//!
//! | part | source |
//! |---|---|
//! | profile model, residual weighting, profiles along normals ([`model`], [`fit::profiles`]) | Treece et al. 2010 |
//! | fixed / PSF blur, fixed cortical density ([`fit::psf`], [`fit::density`]) | Treece et al. 2012, 2015 |
//! | local Levenberg–Marquardt fit ([`fit::local`]) | Levenberg 1944, Marquardt 1963, Nielsen 1999, Moré 1978 |
//! | global initialisation, multigrid and regularised methods, `ω` ([`fit::multigrid`], [`fit::regularized`], [`operators`]) | Neeteson & Boyd 2025 |
//! | Gauss–Newton/LM with matrix-free PCG ([`fit::global`]) | Hestenes & Stiefel 1952, Dembo et al. 1982, Nocedal & Wright 2006 |
//! | surface nets, Taubin smoothing, cotangent Laplacian ([`surface`]) | Gibson 1998, Taubin 1995, Pinkall & Polthier 1993, Meyer et al. 2003 |
//! | point-cloud Laplacian ([`operators`]) | Belkin et al. 2009 |
//! | level set ([`surface::levelset`]) | Osher & Sethian 1988 |
//! | Otsu thresholds, Buie's segmentation, distance transforms ([`segmentation`]) | Otsu 1979, Buie et al. 2007, Felzenszwalb & Huttenlocher 2012 |
//! | curvature penalty, PSF estimation, `ρc` estimation, segmentation refinement, starting segmentations, phantoms | this crate (reasoning in each module) |
//!
//! # References
//!
//! * N. J. Neeteson, S. K. Boyd, *Advancing Cortical Bone Mapping with global
//!   optimization*, Computers in Biology and Medicine, 2025,
//!   doi:10.1016/j.compbiomed.2025.110636.
//! * G. M. Treece, A. H. Gee, P. M. Mayhew, K. E. S. Poole, *High resolution
//!   cortical bone thickness measurement from clinical CT data*, Medical Image
//!   Analysis 14(3):276–290, 2010, doi:10.1016/j.media.2010.01.003.
//! * G. M. Treece, K. E. S. Poole, A. H. Gee, *Imaging the femoral cortex:
//!   thickness, density and mass from clinical CT*, Medical Image Analysis
//!   16(5):952–965, 2012.
//! * G. M. Treece, A. H. Gee, *Independent measurement of femoral cortical
//!   thickness and cortical bone density using clinical CT*, Medical Image
//!   Analysis 20(1):249–264, 2015.
//! * G. M. Treece, A. H. Gee, *Cortical bone mapping: measurement and
//!   statistical analysis of localised skeletal changes*, Current Osteoporosis
//!   Reports 16(5):617–625, 2018, doi:10.1007/s11914-018-0475-3.
//! * H. R. Buie, G. M. Campbell, R. J. Klinck, J. A. MacNeil, S. K. Boyd,
//!   *Automatic segmentation of cortical and trabecular compartments based on
//!   a dual threshold technique for in vivo micro-CT bone analysis*, Bone
//!   41(4):505–515, 2007, doi:10.1016/j.bone.2007.07.007.
//!
//! The numerical and geometric references are listed in the modules that use
//! them.

// `!(x > 0.0)` is used deliberately: unlike `x <= 0.0` it also rejects NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod error;
pub mod fit;
pub mod kdtree;
pub mod logging;
pub mod model;
pub mod operators;
pub mod options;
pub mod phantoms;
pub mod pipeline;
pub mod refine;
pub mod result;
pub mod segmentation;
pub mod surface;
pub mod util;
pub mod volume;

pub use error::Error;
pub use fit::profiles::NormalConstraint;
pub use fit::psf::PsfModel;
pub use operators::{mesh_curvature, CurvatureMeasure, CurvatureMode, Distance};
pub use options::{
    AxialBlur, CurvatureOptions, LaplacianScale, Method, ModelOptions, Options, RhoCMode, SamplingOptions, SigmaMode,
    Site, SolverOptions, SurfaceOptions, VolumetricOptions,
};
pub use pipeline::run;
pub use refine::{
    cortical_masks, refine_segmentation, refine_segmentation_excluding, FreezeOptions, IterationStats, RefineOptions,
    RefineResult, ScannerState,
};
pub use result::{CbmResult, FitInfo, VolumetricInfo};
pub use surface::mesh::Mesh;
pub use volume::{Geometry, Voxel};
