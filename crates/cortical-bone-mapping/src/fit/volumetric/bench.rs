//! The volumetric fit's internal stages, exposed to the benchmarks
//! (`benches/`, feature `bench`). Not part of the API: it changes with the
//! internals it times.
//!
//! [`Setup`] holds what a fit builds once from the image and the mask (the
//! surface and its nodes); [`Setup::band`] builds the band; [`Setup::with_parts`]
//! prepares the problem at the field stage's start and hands its parts to a
//! closure, where a benchmark times them.

use ndarray::ArrayView3;

use super::band::Band;
use super::problem::{Correlation, Lin, Prepared, VolumetricProblem};
use super::{band_reach, band_spec, face_reach, field_variables, Fit, FitNodes, ImageEdge};
use crate::error::Error;
use crate::fit::global::LeastSquares;
use crate::fit::FitConstraints;
use crate::options::{Method, Options, VolumetricOptions};
use crate::pipeline::{candidate_nodes, prepare_surface};
use crate::result::FitInfo;
use crate::volume::Voxel;

/// The surface and nodes of a volumetric fit, built once.
pub struct Setup<'a, T: Voxel> {
    image: ArrayView3<'a, T>,
    options: &'a Options,
    v: &'a VolumetricOptions,
    surface: crate::surface::mesh::Mesh,
    nodes: FitNodes,
}

impl<'a, T: Voxel> Setup<'a, T> {
    /// The surface of `mask` and its fitted nodes, as `fit_surface` makes them.
    pub fn new<U: Voxel>(image: ArrayView3<'a, T>, mask: ArrayView3<U>, options: &'a Options) -> Result<Self, Error> {
        let Method::Volumetric(v) = &options.method else {
            return Err(Error::InvalidOption(
                "the volumetric benchmarks need the volumetric method".into(),
            ));
        };
        let (surface, border) = prepare_surface(&mask, options);
        let keep = candidate_nodes(&surface, &border, None::<&ArrayView3<u8>>, options);
        let (d0, d1, d2) = image.dim();
        let cut = ImageEdge::new([d0, d1, d2], &options.geometry);
        let nodes = FitNodes::new(&surface, &keep, options.sampling.normal_constraint, &cut)?;
        Ok(Setup {
            image,
            options,
            v,
            surface,
            nodes,
        })
    }

    /// Number of fitted nodes.
    pub fn nodes(&self) -> usize {
        self.nodes.ids.len()
    }

    /// Build the band.
    pub fn band(&self) -> BuiltBand {
        let reach = band_reach(&self.image, &self.nodes, self.options, self.v);
        let mut spec = band_spec(self.options, self.v, None);
        if self.v.adaptive_band.is_some() {
            spec.face_reach = Some(face_reach(&self.nodes.mesh, &reach));
        }
        BuiltBand(Band::build(
            &self.image,
            &self.surface,
            &self.nodes.fitted,
            self.nodes.mesh.faces.len(),
            &spec,
        ))
    }

    /// Prepare the field stage at its start (after the global stage) on
    /// `band` with cortical density `rho_c`, and run `f` on its parts.
    pub fn with_parts<R>(&self, band: &BuiltBand, rho_c: f64, f: impl FnOnce(&Parts) -> R) -> R {
        let band = &band.0;
        let c = FitConstraints {
            rho_c: Some(rho_c),
            ..FitConstraints::default()
        };
        let fit = Fit::new(band, &self.nodes, self.options, self.v, c, None);
        let mut info = FitInfo::default();
        let global = fit.global_stage(rho_c, &mut info);
        let starts = fit.node_starts(&global, None);
        let curv = fit.curvature(rho_c, &global);
        let problem = fit.problem(true, rho_c, &global, curv.as_ref());
        let prep = Prepared::new(&problem);
        let x = field_variables(&prep, &global, &starts);
        let (_, _, lin) = prep.linearise(&x);
        f(&Parts {
            problem: &problem,
            prep: &prep,
            x,
            lin,
        })
    }
}

/// A band built by [`Setup::band`].
pub struct BuiltBand(Band);

impl BuiltBand {
    /// Number of voxels.
    pub fn voxels(&self) -> usize {
        self.0.len()
    }
}

/// The field stage's problem at its start.
pub struct Parts<'p, 'a, 'b> {
    problem: &'p VolumetricProblem<'a>,
    prep: &'b Prepared<'p, 'a>,
    x: Vec<f64>,
    lin: Lin,
}

impl Parts<'_, '_, '_> {
    /// Number of variables (the length of a product's input and output).
    pub fn variables(&self) -> usize {
        self.x.len()
    }

    /// Number of band voxels.
    pub fn voxels(&self) -> usize {
        self.problem.band.len()
    }

    /// One linearisation: cost, gradient, face blocks (once per outer iteration).
    pub fn linearise(&self) -> f64 {
        self.prep.linearise(&self.x).0.total
    }

    /// The whole Gauss–Newton product `JᵀJ v` (once per CG iteration).
    pub fn product(&self, v: &[f64], out: &mut [f64]) {
        self.prep.hess_vec(&self.lin, v, out);
    }

    /// Its data-term part.
    pub fn data_product(&self, v: &[f64], out: &mut [f64]) {
        let p = self.prep;
        self.problem.data_product(&p.lay, &p.incident, &self.lin, v, out);
    }

    /// Its smoothing part.
    pub fn smoothing_product(&self, v: &[f64], out: &mut [f64]) {
        self.problem.smoothing_product(&self.prep.lay, v, out);
    }

    /// Its curvature part.
    pub fn curvature_product(&self, v: &[f64], out: &mut [f64]) {
        self.problem.curvature_product(&self.prep.lay, &self.lin, v, out);
    }

    /// The per-node moments of the standard errors, for noise with lag-one
    /// correlation `rho` per axis (the band's grid step 1).
    pub fn node_moments(&self, rho: [f64; 3]) -> usize {
        self.prep.node_moments(&self.x, &Correlation { rho, step: 1 }).w.len()
    }
}
