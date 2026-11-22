//! Cortical bone mapping, refinement and profile fits.

use ::cortical_bone_mapping as cbm;
use cbm::model::Params;
use numpy::ndarray::{Array2, ArrayView3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::convert::*;
use crate::options::*;

pub(crate) fn result_dict<'py>(py: Python<'py>, r: &cbm::CbmResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("points", v3_array(py, &r.points))?;
    d.set_item("normals", v3_array(py, &r.normals))?;
    d.set_item("faces", faces_array(py, &r.faces))?;
    d.set_item("params", params_array(py, &r.params))?;
    d.set_item("residual", r.residual.clone().into_pyarray(py))?;
    d.set_item("vertex_ids", r.vertex_ids.clone().into_pyarray(py))?;
    d.set_item("surface_vertices", v3_array(py, &r.surface.vertices))?;
    d.set_item("surface_faces", faces_array(py, &r.surface.faces))?;
    d.set_item("global_params", r.global_params.to_vec())?;
    d.set_item("psf", r.psf.map(|p| p.to_vec()))?;
    d.set_item("rho_c", r.rho_c)?;
    if let Some(v) = &r.rho_c_node {
        d.set_item("rho_c_node", v.clone().into_pyarray(py))?;
    }
    let info = PyDict::new(py);
    info.set_item("method", &r.info.method)?;
    info.set_item("converged", r.info.converged)?;
    info.set_item("iterations", r.info.iterations)?;
    info.set_item("cg_iterations", r.info.cg_iterations)?;
    info.set_item("cost", r.info.cost)?;
    info.set_item("data_cost", r.info.data_cost)?;
    info.set_item("smoothing_cost", r.info.smoothing_cost)?;
    info.set_item("curvature_cost", r.info.curvature_cost)?;
    info.set_item("budget_exhausted", r.info.budget_exhausted)?;
    let t = PyDict::new(py);
    for (k, v) in &r.info.timings {
        t.set_item(k, v)?;
    }
    info.set_item("timings", t)?;
    d.set_item("info", info)?;
    if let Some((x, prof)) = &r.profiles {
        d.set_item("profile_x", x.clone().into_pyarray(py))?;
        let a = Array2::from_shape_vec((prof.n, prof.m), prof.values.clone()).unwrap();
        d.set_item("profiles", a.into_pyarray(py))?;
    }
    Ok(d)
}

/// Run cortical bone mapping; returns a dict of arrays.
#[pyfunction]
#[pyo3(signature = (image, mask, sub_mask, options))]
pub(crate) fn run<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    mask: PyReadonlyArray3<'py, u8>,
    sub_mask: Option<PyReadonlyArray3<'py, u8>>,
    options: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = parse_options(options)?;
    let m = mask.as_array();
    let s: Option<ArrayView3<'_, u8>> = sub_mask.as_ref().map(|s| s.as_array());
    let res = with_image!(image, |img| py.detach(|| cbm::run(img, m, s, &opts)).map_err(to_pyerr))?;
    result_dict(py, &res)
}

/// Refine a segmentation; returns a dict of arrays.
#[pyfunction]
#[pyo3(signature = (image, mask, sub_mask, options, refine, exclude=None))]
pub(crate) fn refine<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    mask: PyReadonlyArray3<'py, u8>,
    sub_mask: Option<PyReadonlyArray3<'py, u8>>,
    options: &Bound<'py, PyDict>,
    refine: &Bound<'py, PyDict>,
    exclude: Option<PyReadonlyArray3<'py, u8>>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = parse_options(options)?;
    let ro = parse_refine(refine)?;
    let m = mask.as_array();
    let s: Option<ArrayView3<'_, u8>> = sub_mask.as_ref().map(|s| s.as_array());
    let e: Option<ArrayView3<'_, u8>> = exclude.as_ref().map(|e| e.as_array());
    let r = with_image!(image, |img| py
        .detach(|| cbm::refine_segmentation_excluding(img, m, s, e, &opts, &ro))
        .map_err(to_pyerr))?;
    let d = PyDict::new(py);
    d.set_item("result", result_dict(py, &r.result)?)?;
    d.set_item("endosteal_vertices", v3_array(py, &r.endosteal.vertices))?;
    d.set_item("periosteal_mask", r.periosteal_mask.into_pyarray(py))?;
    d.set_item("endosteal_mask", r.endosteal_mask.into_pyarray(py))?;
    d.set_item("cortical_mask", r.cortical_mask.into_pyarray(py))?;
    d.set_item("history", history_list(py, &r.history)?)?;
    let surfs = PyList::empty(py);
    let sfaces = PyList::empty(py);
    for s in &r.surfaces {
        surfs.append(v3_array(py, &s.vertices))?;
        sfaces.append(faces_array(py, &s.faces))?;
    }
    d.set_item("surfaces", surfs)?;
    d.set_item("surfaces_faces", sfaces)?;
    d.set_item("converged", r.converged)?;
    d.set_item("stalled", r.stalled)?;
    d.set_item("message", &r.message)?;
    d.set_item("timings", r.timings.clone())?;
    Ok(d)
}

/// A refinement's per-fit statistics as a list of dicts.
fn history_list<'py>(py: Python<'py>, history: &[cbm::IterationStats]) -> PyResult<Bound<'py, PyList>> {
    let hist = PyList::empty(py);
    for h in history {
        let e = PyDict::new(py);
        e.set_item("rms_displacement", h.rms_displacement)?;
        e.set_item("max_displacement", h.max_displacement)?;
        e.set_item("mean_thickness", h.mean_thickness)?;
        e.set_item("mean_residual", h.mean_residual)?;
        e.set_item("nodes", h.nodes)?;
        e.set_item("moving_nodes", h.moving_nodes)?;
        e.set_item("full_fit", h.full_fit)?;
        e.set_item("fit_converged", h.fit_converged)?;
        e.set_item("untrusted", h.untrusted)?;
        e.set_item("max_regional_rms", h.max_regional_rms)?;
        hist.append(e)?;
    }
    Ok(hist)
}

/// Fit the profile model independently to each row of `profiles` (N × M)
/// sampled at `x` (M). Returns (params (N,5), cost (N,), iterations (N,),
/// converged (N,)).
#[pyfunction]
#[pyo3(signature = (profiles, x, init, rho_c, residual_boost, lower, upper, free, max_iterations, ftol, gtol, xtol))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fit_profiles<'py>(
    py: Python<'py>,
    profiles: PyReadonlyArray2<'py, f32>,
    x: Vec<f64>,
    init: PyReadonlyArray2<'py, f64>,
    rho_c: f64,
    residual_boost: f64,
    lower: Params,
    upper: Params,
    free: [bool; 5],
    max_iterations: usize,
    ftol: f64,
    gtol: f64,
    xtol: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, numpy::PyArray1<f64>>,
    Bound<'py, numpy::PyArray1<u32>>,
    Bound<'py, numpy::PyArray1<bool>>,
)> {
    let p = profiles.as_array();
    let (n, m) = p.dim();
    if m != x.len() {
        return Err(PyValueError::new_err("profiles must have one column per x"));
    }
    let ia = init.as_array();
    if ia.ncols() != 5 || (ia.nrows() != 1 && ia.nrows() != n) {
        return Err(PyValueError::new_err("init must be (5,) or (N, 5)"));
    }
    let inits: Vec<Params> = ia.rows().into_iter().map(|r| [r[0], r[1], r[2], r[3], r[4]]).collect();
    let values: Vec<f32> = p.iter().copied().collect();
    let prof = cbm::fit::profiles::Profiles {
        n,
        m,
        values,
        used: None,
    };
    let w = cbm::model::residual_weights(&x, residual_boost);
    let settings = cbm::fit::local::FitSettings {
        lower,
        upper,
        free,
        max_iterations,
        ftol,
        gtol,
        xtol,
    };
    let fits = py.detach(|| cbm::fit::local::fit_all(&prof, &x, &w, rho_c, &inits, &settings));
    let params: Vec<Params> = fits.iter().map(|f| f.params).collect();
    Ok((
        params_array(py, &params),
        fits.iter().map(|f| f.cost).collect::<Vec<_>>().into_pyarray(py),
        fits.iter().map(|f| f.iterations).collect::<Vec<_>>().into_pyarray(py),
        fits.iter()
            .map(|f| f.status.converged())
            .collect::<Vec<_>>()
            .into_pyarray(py),
    ))
}
