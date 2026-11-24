//! Parsing the flat option dict passed by the Python package.

use ::cortical_bone_mapping as cbm;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::*;

pub(crate) fn parse_options(d: &Bound<'_, PyDict>) -> PyResult<cbm::Options> {
    let method = parse_method(d)?;
    if curvature_requested(d)? && !matches!(method, cbm::Method::Regularized { .. }) {
        return Err(PyValueError::new_err(
            "curvature regularisation requires method='regularized'",
        ));
    }
    Ok(cbm::Options {
        geometry: cbm::Geometry {
            spacing: req(d, "spacing")?,
            origin: req(d, "origin")?,
        },
        surface: parse_surface(d)?,
        sampling: parse_sampling(d)?,
        model: parse_model(d)?,
        solver: parse_solver(d)?,
        method,
        init_from_global: req(d, "init_from_global")?,
        keep_profiles: req(d, "keep_profiles")?,
        threads: opt(d, "threads")?,
        neighbour_distance: parse_distance(d)?,
    })
}

/// How node neighbourhoods are measured; `None` = the method's default.
fn parse_distance(d: &Bound<'_, PyDict>) -> PyResult<Option<cbm::Distance>> {
    Ok(match opt::<String>(d, "neighbour_distance")?.as_deref() {
        None => None,
        Some("euclidean") => Some(cbm::Distance::Euclidean),
        Some("surface") => Some(cbm::Distance::Surface),
        Some(o) => return Err(unknown("neighbour_distance", o)),
    })
}

/// The error for an unknown value of a string option.
fn unknown(what: &str, value: &str) -> PyErr {
    PyValueError::new_err(format!("unknown {what} {value:?}"))
}

fn parse_surface(d: &Bound<'_, PyDict>) -> PyResult<cbm::SurfaceOptions> {
    Ok(cbm::SurfaceOptions {
        mask_smoothing: req(d, "mask_smoothing")?,
        taubin_iterations: req(d, "taubin_iterations")?,
        taubin_passband: req(d, "taubin_passband")?,
        taubin_lambda: req(d, "taubin_lambda")?,
        exclude_border: req(d, "exclude_border")?,
    })
}

fn parse_sampling(d: &Bound<'_, PyDict>) -> PyResult<cbm::SamplingOptions> {
    let normal_constraint = match req::<String>(d, "normal_constraint")?.as_str() {
        "none" => cbm::NormalConstraint::None,
        "in_plane" => cbm::NormalConstraint::InPlane(axis(d, "normal_axis")?),
        "along_axis" => cbm::NormalConstraint::AlongAxis(axis(d, "normal_axis")?),
        o => return Err(unknown("normal_constraint", o)),
    };
    Ok(cbm::SamplingOptions {
        outside: req(d, "outside")?,
        inside: req(d, "inside")?,
        resolution: opt(d, "resolution")?,
        smoothing: req(d, "profile_smoothing")?,
        normal_constraint,
        half_width_inside: req(d, "half_width_inside")?,
    })
}

fn parse_model(d: &Bound<'_, PyDict>) -> PyResult<cbm::ModelOptions> {
    let rho_c_mode = match req::<String>(d, "rho_c_mode")?.as_str() {
        "fixed" => cbm::RhoCMode::Fixed,
        "estimate" => cbm::RhoCMode::Estimate,
        "profile_max" => cbm::RhoCMode::ProfileMax,
        m => return Err(unknown("rho_c mode", m)),
    };
    Ok(cbm::ModelOptions {
        rho_c: req(d, "rho_c")?,
        rho_c_mode,
        residual_boost: req(d, "residual_boost")?,
        init: req(d, "init")?,
        lower: req(d, "lower")?,
        upper: req(d, "upper")?,
        sigma: parse_sigma(d)?,
        global: req(d, "global")?,
        density_smoothing: req(d, "density_smoothing")?,
    })
}

/// The blur: free, fixed, from a PSF, or a PSF estimated under a model.
fn parse_sigma(d: &Bound<'_, PyDict>) -> PyResult<cbm::SigmaMode> {
    Ok(match req::<String>(d, "sigma")?.as_str() {
        "free" => cbm::SigmaMode::Free,
        "fixed" => cbm::SigmaMode::Fixed(req(d, "sigma_value")?),
        "psf" => cbm::SigmaMode::Psf(req(d, "psf")?),
        "estimate_psf" => cbm::SigmaMode::EstimatePsf {
            model: match req::<String>(d, "psf_model")?.as_str() {
                "isotropic" => cbm::PsfModel::Isotropic,
                "axial" => cbm::PsfModel::Axial(axis(d, "psf_axis")?),
                "anisotropic" => cbm::PsfModel::Anisotropic,
                o => return Err(unknown("psf_model", o)),
            },
            min_ratio: req(d, "psf_min_ratio")?,
        },
        o => return Err(unknown("sigma mode", o)),
    })
}

fn parse_solver(d: &Bound<'_, PyDict>) -> PyResult<cbm::SolverOptions> {
    Ok(cbm::SolverOptions {
        max_iterations: req(d, "max_iterations")?,
        ftol: req(d, "ftol")?,
        gtol: req(d, "gtol")?,
        xtol: req(d, "xtol")?,
        cg_max_iterations: req(d, "cg_max_iterations")?,
        cg_tolerance: req(d, "cg_tolerance")?,
        cg_max_total: req(d, "cg_max_total")?,
        progress_every: req(d, "progress_every")?,
    })
}

fn parse_method(d: &Bound<'_, PyDict>) -> PyResult<cbm::Method> {
    Ok(match req::<String>(d, "method")?.as_str() {
        "local" => cbm::Method::Local,
        "multigrid" => cbm::Method::Multigrid {
            spacings: req(d, "multigrid_spacings")?,
            neighbours: req(d, "multigrid_neighbours")?,
            seed: req(d, "seed")?,
        },
        "regularized" => cbm::Method::Regularized {
            lambda: req(d, "reg_lambda")?,
            sigma: req(d, "reg_sigma")?,
            neighbours: req(d, "reg_neighbours")?,
            cutoff: req(d, "reg_cutoff")?,
            init_from_local: req(d, "init_from_local")?,
            curvature: parse_curvature(d, "curvature", "curvature_weight")?,
        },
        "volumetric" => cbm::Method::Volumetric(parse_volumetric(d)?),
        o => return Err(unknown("method", o)),
    })
}

/// The volumetric model's options.
fn parse_volumetric(d: &Bound<'_, PyDict>) -> PyResult<cbm::VolumetricOptions> {
    let sigma_z = match req::<String>(d, "vol_sigma_z_mode")?.as_str() {
        "fit" => cbm::AxialBlur::Fit,
        "tied" => cbm::AxialBlur::Tied,
        "fixed" => cbm::AxialBlur::Fixed(req(d, "vol_sigma_z")?),
        o => return Err(unknown("vol_sigma_z", o)),
    };
    Ok(cbm::VolumetricOptions {
        scanner_axis: req(d, "vol_scanner_axis")?,
        sigma_xy: opt(d, "vol_sigma_xy")?,
        sigma_z,
        lambda: req(d, "vol_lambda")?,
        sigma: req(d, "reg_sigma")?,
        neighbours: req(d, "reg_neighbours")?,
        cutoff: req(d, "reg_cutoff")?,
        global_densities: req(d, "vol_global_densities")?,
        global_soft_tissue: req(d, "vol_global_soft_tissue")?,
        adaptive_band: opt(d, "vol_adaptive_band")?,
        curvature: parse_curvature(d, "vol_curvature", "vol_curvature_weight")?,
        voxel_stride: req(d, "vol_voxel_stride")?,
    })
}

/// A curvature penalty (mode under `mode_key`, weight under `weight_key`;
/// the other settings are shared), if any.
fn parse_curvature(d: &Bound<'_, PyDict>, mode_key: &str, weight_key: &str) -> PyResult<Option<cbm::CurvatureOptions>> {
    let mode = match opt::<String>(d, mode_key)?.as_deref() {
        None | Some("none") => return Ok(None),
        Some("change") => cbm::CurvatureMode::Change,
        Some("variation") => cbm::CurvatureMode::Variation,
        Some("absolute") => cbm::CurvatureMode::Absolute {
            kappa: req(d, "curvature_kappa")?,
        },
        Some(o) => return Err(unknown("curvature mode", o)),
    };
    let (periosteal, endosteal) = match req::<String>(d, "curvature_surfaces")?.as_str() {
        "both" => (true, true),
        "periosteal" => (true, false),
        "endosteal" => (false, true),
        o => return Err(unknown("curvature_surfaces", o)),
    };
    let measure = match req::<String>(d, "curvature_measure")?.as_str() {
        "mean" => cbm::CurvatureMeasure::Mean,
        "total" => cbm::CurvatureMeasure::Total,
        o => return Err(unknown("curvature_measure", o)),
    };
    Ok(Some(cbm::CurvatureOptions {
        mode,
        measure,
        weight: req(d, weight_key)?,
        endosteal_weight: opt(d, "curvature_endosteal_weight")?,
        length: opt(d, "curvature_length")?,
        periosteal,
        endosteal,
        scale: match (
            opt::<f64>(d, "curvature_scale")?,
            req::<String>(d, "curvature_scale_mode")?.as_str(),
        ) {
            (_, "cotangent") => cbm::LaplacianScale::Cotangent,
            (Some(s), "physical") => cbm::LaplacianScale::Physical(s),
            (_, "spacing") => cbm::LaplacianScale::NodeSpacing(req(d, "curvature_scale_spacings")?),
            (_, o) => return Err(unknown("curvature_scale", o)),
        },
        neighbours: req(d, "curvature_neighbours")?,
    }))
}

pub(crate) fn curvature_requested(d: &Bound<'_, PyDict>) -> PyResult<bool> {
    Ok(!matches!(
        opt::<String>(d, "curvature")?.as_deref(),
        None | Some("none")
    ))
}

/// Refinement options from the dict of `RefineOptions._to_core()`.
pub(crate) fn parse_refine(refine: &Bound<'_, PyDict>) -> PyResult<cbm::RefineOptions> {
    Ok(cbm::RefineOptions {
        max_iterations: req(refine, "max_iterations")?,
        relaxation: req(refine, "relaxation")?,
        tolerance: opt(refine, "tolerance")?,
        smoothing_iterations: req(refine, "smoothing_iterations")?,
        max_step: opt(refine, "max_step")?,
        keep_history: req(refine, "keep_history")?,
        verbose: req(refine, "verbose")?,
        label: req(refine, "label")?,
        level_set: req(refine, "level_set")?,
        edge_ratio: req(refine, "edge_ratio")?,
        stall_iterations: req(refine, "stall_iterations")?,
        region_radius: opt(refine, "region_radius")?,
        region_tolerance: opt(refine, "region_tolerance")?,
        freeze: if req::<bool>(refine, "freeze")? {
            Some(cbm::FreezeOptions {
                patience: req(refine, "freeze_patience")?,
                displacement: opt(refine, "freeze_displacement")?,
                thickness: opt(refine, "freeze_thickness")?,
                radius: opt(refine, "freeze_radius")?,
                halo: opt(refine, "freeze_halo")?,
                thaw: opt(refine, "freeze_thaw")?,
            })
        } else {
            None
        },
        scanner_passes: req(refine, "scanner_passes")?,
        scanner_relaxation: req(refine, "scanner_relaxation")?,
        scanner_tolerance: req(refine, "scanner_tolerance")?,
    })
}
