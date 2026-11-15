//! Python bindings for the `cortical-bone-mapping` crate.
//!
//! This is the private extension module `cortical_bone_mapping_rs._core`.
//! The public, documented API (option dataclasses, validation, result
//! classes) lives in `python/cortical_bone_mapping_rs/__init__.py`, which
//! passes a flat dict of options. Images of any common numeric dtype are
//! sampled in place without conversion; masks arrive as `uint8`.

use ::cortical_bone_mapping as cbm;
use cbm::model::Params;
use cbm::util::V3;
use numpy::ndarray::{Array2, ArrayView3};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn to_pyerr(e: cbm::Error) -> PyErr {
    match e {
        cbm::Error::ThreadPool(_) => PyRuntimeError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

fn req<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    match d.get_item(key)? {
        Some(v) => v.extract::<T>().map_err(|e| {
            let e: PyErr = e.into();
            PyValueError::new_err(format!("option {key}: {e}"))
        }),
        None => Err(PyValueError::new_err(format!("missing option {key}"))),
    }
}

fn opt<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract::<T>().map(Some).map_err(|e| {
            let e: PyErr = e.into();
            PyValueError::new_err(format!("option {key}: {e}"))
        }),
        _ => Ok(None),
    }
}

fn axis(d: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    let a: usize = req(d, key)?;
    if a > 2 {
        return Err(PyValueError::new_err(format!("{key} must be 0, 1 or 2")));
    }
    Ok(a)
}

fn parse_options(d: &Bound<'_, PyDict>) -> PyResult<cbm::Options> {
    let normal_constraint = match req::<String>(d, "normal_constraint")?.as_str() {
        "none" => cbm::NormalConstraint::None,
        "in_plane" => cbm::NormalConstraint::InPlane(axis(d, "normal_axis")?),
        "along_axis" => cbm::NormalConstraint::AlongAxis(axis(d, "normal_axis")?),
        o => return Err(PyValueError::new_err(format!("unknown normal_constraint {o:?}"))),
    };
    let psf_model = match req::<String>(d, "psf_model")?.as_str() {
        "isotropic" => cbm::PsfModel::Isotropic,
        "axial" => cbm::PsfModel::Axial(axis(d, "psf_axis")?),
        "anisotropic" => cbm::PsfModel::Anisotropic,
        o => return Err(PyValueError::new_err(format!("unknown psf_model {o:?}"))),
    };
    let sigma = match req::<String>(d, "sigma")?.as_str() {
        "free" => cbm::SigmaMode::Free,
        "fixed" => cbm::SigmaMode::Fixed(req(d, "sigma_value")?),
        "psf" => cbm::SigmaMode::Psf(req(d, "psf")?),
        "estimate_psf" => cbm::SigmaMode::EstimatePsf {
            model: psf_model,
            min_ratio: req(d, "psf_min_ratio")?,
        },
        o => return Err(PyValueError::new_err(format!("unknown sigma mode {o:?}"))),
    };
    let curvature = match opt::<String>(d, "curvature")?.as_deref() {
        None | Some("none") => None,
        Some(m) => {
            let mode = match m {
                "change" => cbm::CurvatureMode::Change,
                "excess" => cbm::CurvatureMode::Excess {
                    kappa: req(d, "curvature_kappa")?,
                },
                o => return Err(PyValueError::new_err(format!("unknown curvature mode {o:?}"))),
            };
            let (periosteal, endosteal) = match req::<String>(d, "curvature_surfaces")?.as_str() {
                "both" => (true, true),
                "periosteal" => (true, false),
                "endosteal" => (false, true),
                o => return Err(PyValueError::new_err(format!("unknown curvature_surfaces {o:?}"))),
            };
            Some(cbm::CurvatureOptions {
                mode,
                weight: req(d, "curvature_weight")?,
                length: opt(d, "curvature_length")?,
                periosteal,
                endosteal,
                scale: opt(d, "curvature_scale")?,
                neighbours: req(d, "curvature_neighbours")?,
            })
        }
    };
    let method = match req::<String>(d, "method")?.as_str() {
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
            curvature,
        },
        o => return Err(PyValueError::new_err(format!("unknown method {o:?}"))),
    };
    if curvature_requested(d)? && !matches!(method, cbm::Method::Regularized { .. }) {
        return Err(PyValueError::new_err(
            "curvature regularisation requires method='regularized'",
        ));
    }
    let threads: Option<usize> = opt(d, "threads")?;
    Ok(cbm::Options {
        geometry: cbm::Geometry {
            spacing: req(d, "spacing")?,
            origin: req(d, "origin")?,
        },
        surface: cbm::SurfaceOptions {
            mask_smoothing: req(d, "mask_smoothing")?,
            taubin_iterations: req(d, "taubin_iterations")?,
            taubin_passband: req(d, "taubin_passband")?,
            taubin_lambda: req(d, "taubin_lambda")?,
            exclude_border: req(d, "exclude_border")?,
        },
        sampling: cbm::SamplingOptions {
            outside: req(d, "outside")?,
            inside: req(d, "inside")?,
            resolution: opt(d, "resolution")?,
            smoothing: req(d, "profile_smoothing")?,
            normal_constraint,
        },
        model: cbm::ModelOptions {
            rho_c: req(d, "rho_c")?,
            residual_boost: req(d, "residual_boost")?,
            init: req(d, "init")?,
            lower: req(d, "lower")?,
            upper: req(d, "upper")?,
            sigma,
        },
        solver: cbm::SolverOptions {
            max_iterations: req(d, "max_iterations")?,
            ftol: req(d, "ftol")?,
            gtol: req(d, "gtol")?,
            xtol: req(d, "xtol")?,
            cg_max_iterations: req(d, "cg_max_iterations")?,
            cg_tolerance: req(d, "cg_tolerance")?,
        },
        method,
        init_from_global: req(d, "init_from_global")?,
        keep_profiles: req(d, "keep_profiles")?,
        threads,
    })
}

fn curvature_requested(d: &Bound<'_, PyDict>) -> PyResult<bool> {
    Ok(!matches!(
        opt::<String>(d, "curvature")?.as_deref(),
        None | Some("none")
    ))
}

fn v3_array<'py>(py: Python<'py>, v: &[V3]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = v.iter().flat_map(|p| p.iter().copied()).collect();
    Array2::from_shape_vec((v.len(), 3), flat).unwrap().into_pyarray(py)
}

fn faces_array<'py>(py: Python<'py>, f: &[[u32; 3]]) -> Bound<'py, PyArray2<u32>> {
    let flat: Vec<u32> = f.iter().flat_map(|p| p.iter().copied()).collect();
    Array2::from_shape_vec((f.len(), 3), flat).unwrap().into_pyarray(py)
}

fn params_array<'py>(py: Python<'py>, p: &[Params]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = p.iter().flat_map(|q| q.iter().copied()).collect();
    Array2::from_shape_vec((p.len(), 5), flat).unwrap().into_pyarray(py)
}

fn result_dict<'py>(py: Python<'py>, r: &cbm::CbmResult) -> PyResult<Bound<'py, PyDict>> {
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
    let info = PyDict::new(py);
    info.set_item("method", &r.info.method)?;
    info.set_item("converged", r.info.converged)?;
    info.set_item("iterations", r.info.iterations)?;
    info.set_item("cg_iterations", r.info.cg_iterations)?;
    info.set_item("cost", r.info.cost)?;
    info.set_item("data_cost", r.info.data_cost)?;
    info.set_item("smoothing_cost", r.info.smoothing_cost)?;
    info.set_item("curvature_cost", r.info.curvature_cost)?;
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

/// Dispatch over the image dtype without copying it.
macro_rules! with_image {
    ($img:expr, |$v:ident| $body:expr) => {{
        let img = $img;
        if let Ok(a) = img.extract::<PyReadonlyArray3<'_, f32>>() {
            let $v = a.as_array();
            $body
        } else if let Ok(a) = img.extract::<PyReadonlyArray3<'_, f64>>() {
            let $v = a.as_array();
            $body
        } else if let Ok(a) = img.extract::<PyReadonlyArray3<'_, i16>>() {
            let $v = a.as_array();
            $body
        } else if let Ok(a) = img.extract::<PyReadonlyArray3<'_, u16>>() {
            let $v = a.as_array();
            $body
        } else if let Ok(a) = img.extract::<PyReadonlyArray3<'_, u8>>() {
            let $v = a.as_array();
            $body
        } else if let Ok(a) = img.extract::<PyReadonlyArray3<'_, i8>>() {
            let $v = a.as_array();
            $body
        } else if let Ok(a) = img.extract::<PyReadonlyArray3<'_, i32>>() {
            let $v = a.as_array();
            $body
        } else if let Ok(a) = img.extract::<PyReadonlyArray3<'_, u32>>() {
            let $v = a.as_array();
            $body
        } else {
            Err(PyTypeError::new_err(
                "image must be a 3D array of float32, float64, int8/16/32 or uint8/16/32",
            ))
        }
    }};
}

/// Run cortical bone mapping; returns a dict of arrays.
#[pyfunction]
#[pyo3(signature = (image, mask, sub_mask, options))]
fn run<'py>(
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
#[pyo3(signature = (image, mask, sub_mask, options, refine))]
fn refine<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    mask: PyReadonlyArray3<'py, u8>,
    sub_mask: Option<PyReadonlyArray3<'py, u8>>,
    options: &Bound<'py, PyDict>,
    refine: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = parse_options(options)?;
    let ro = cbm::RefineOptions {
        max_iterations: req(refine, "max_iterations")?,
        relaxation: req(refine, "relaxation")?,
        tolerance: opt(refine, "tolerance")?,
        smoothing_iterations: req(refine, "smoothing_iterations")?,
        max_step: opt(refine, "max_step")?,
        keep_history: req(refine, "keep_history")?,
    };
    let m = mask.as_array();
    let s: Option<ArrayView3<'_, u8>> = sub_mask.as_ref().map(|s| s.as_array());
    let r = with_image!(image, |img| py
        .detach(|| cbm::refine_segmentation(img, m, s, &opts, &ro))
        .map_err(to_pyerr))?;
    let d = PyDict::new(py);
    d.set_item("result", result_dict(py, &r.result)?)?;
    d.set_item("endosteal_vertices", v3_array(py, &r.endosteal.vertices))?;
    d.set_item("periosteal_mask", r.periosteal_mask.into_pyarray(py))?;
    d.set_item("endosteal_mask", r.endosteal_mask.into_pyarray(py))?;
    d.set_item("cortical_mask", r.cortical_mask.into_pyarray(py))?;
    let hist = PyList::empty(py);
    for h in &r.history {
        let e = PyDict::new(py);
        e.set_item("rms_displacement", h.rms_displacement)?;
        e.set_item("max_displacement", h.max_displacement)?;
        e.set_item("mean_thickness", h.mean_thickness)?;
        e.set_item("mean_residual", h.mean_residual)?;
        hist.append(e)?;
    }
    d.set_item("history", hist)?;
    let surfs = PyList::empty(py);
    for s in &r.surfaces {
        surfs.append(v3_array(py, &s.vertices))?;
    }
    d.set_item("surfaces", surfs)?;
    d.set_item("converged", r.converged)?;
    Ok(d)
}

/// Rasterise a closed triangle mesh into a boolean mask.
#[pyfunction]
#[pyo3(signature = (vertices, faces, shape, spacing, origin))]
fn voxelize<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, u32>,
    shape: (usize, usize, usize),
    spacing: V3,
    origin: V3,
) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
    let mesh = mesh_from(vertices, faces)?;
    let g = cbm::Geometry { spacing, origin };
    let out = py.detach(|| mesh.voxelize(shape, &g));
    Ok(out.into_pyarray(py))
}

fn mesh_from(vertices: PyReadonlyArray2<'_, f64>, faces: PyReadonlyArray2<'_, u32>) -> PyResult<cbm::Mesh> {
    let v = vertices.as_array();
    let f = faces.as_array();
    if v.ncols() != 3 || f.ncols() != 3 {
        return Err(PyValueError::new_err("vertices and faces must have 3 columns"));
    }
    let n = v.nrows() as u32;
    if f.iter().any(|&i| i >= n) {
        return Err(PyValueError::new_err("face index out of range"));
    }
    Ok(cbm::Mesh {
        vertices: v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect(),
        faces: f.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect(),
    })
}

/// Write a mesh with point data as legacy binary VTK PolyData.
#[pyfunction]
#[pyo3(signature = (path, vertices, faces, scalars, vectors))]
fn write_vtk(
    path: std::path::PathBuf,
    vertices: PyReadonlyArray2<'_, f64>,
    faces: PyReadonlyArray2<'_, u32>,
    scalars: Vec<(String, Vec<f64>)>,
    vectors: Vec<(String, PyReadonlyArray2<'_, f64>)>,
) -> PyResult<()> {
    let mesh = mesh_from(vertices, faces)?;
    let n = mesh.vertices.len();
    for (name, s) in &scalars {
        if s.len() != n {
            return Err(PyValueError::new_err(format!(
                "scalar {name} must have one value per vertex"
            )));
        }
    }
    let vecs: Vec<(String, Vec<V3>)> = vectors
        .iter()
        .map(|(name, a)| {
            let a = a.as_array();
            if a.nrows() != n || a.ncols() != 3 {
                return Err(PyValueError::new_err(format!("vector {name} must be (n_vertices, 3)")));
            }
            Ok((name.clone(), a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect()))
        })
        .collect::<PyResult<_>>()?;
    let s: Vec<(&str, &[f64])> = scalars.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
    let v: Vec<(&str, &[V3])> = vecs.iter().map(|(k, v)| (k.as_str(), v.as_slice())).collect();
    mesh.write_vtk(&path, &s, &v)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Generate a synthetic shell phantom; returns (image float32, mask uint8).
#[pyfunction]
#[pyo3(signature = (spec))]
#[allow(clippy::type_complexity)]
fn phantom<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, numpy::PyArray3<f32>>, Bound<'py, numpy::PyArray3<u8>>)> {
    use cbm::synthetic::{Shape, ShellPhantom};
    let shape = match req::<String>(spec, "shape")?.as_str() {
        "cylinder" => Shape::Cylinder,
        "sphere" => Shape::Sphere,
        o => return Err(PyValueError::new_err(format!("unknown phantom shape {o:?}"))),
    };
    let ph = ShellPhantom {
        dims: req(spec, "dims")?,
        spacing: req(spec, "spacing")?,
        shape,
        radius: req(spec, "radius")?,
        radius_bump: req(spec, "radius_bump")?,
        radius_lobes: req(spec, "radius_lobes")?,
        thickness_mean: req(spec, "thickness_mean")?,
        thickness_amplitude: req(spec, "thickness_amplitude")?,
        thickness_lobes: req(spec, "thickness_lobes")?,
        rho_s: req(spec, "rho_s")?,
        rho_c: req(spec, "rho_c")?,
        rho_b: req(spec, "rho_b")?,
        blur: req(spec, "blur")?,
        noise: req(spec, "noise")?,
        seed: req(spec, "seed")?,
        supersample: req(spec, "supersample")?,
        mask_offset: req(spec, "mask_offset")?,
        mask_bump: req(spec, "mask_bump")?,
        mask_lobes: req(spec, "mask_lobes")?,
    };
    let (d0, d1, d2) = ph.dims;
    if d0 < 2 || d1 < 2 || d2 < 2 || ph.spacing.iter().any(|s| s.is_nan() || *s <= 0.0) {
        return Err(PyValueError::new_err("phantom dims must be >= 2 and spacing positive"));
    }
    let (img, mask) = py.detach(|| ph.generate());
    Ok((img.into_pyarray(py), mask.into_pyarray(py)))
}

/// Fit the profile model independently to each row of `profiles` (N × M)
/// sampled at `x` (M). Returns (params (N,5), cost (N,), iterations (N,),
/// converged (N,)).
#[pyfunction]
#[pyo3(signature = (profiles, x, init, rho_c, residual_boost, lower, upper, free, max_iterations, ftol, gtol, xtol))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn fit_profiles<'py>(
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
    let prof = cbm::profiles::Profiles { n, m, values };
    let w = cbm::model::residual_weights(&x, residual_boost);
    let settings = cbm::local::FitSettings {
        lower,
        upper,
        free,
        max_iterations,
        ftol,
        gtol,
        xtol,
    };
    let fits = py.detach(|| cbm::local::fit_all(&prof, &x, &w, rho_c, &inits, &settings));
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

fn bone_phantom_from(spec: &Bound<'_, PyDict>) -> PyResult<cbm::bone_phantom::BonePhantom> {
    let ph = cbm::bone_phantom::BonePhantom {
        dims: req(spec, "dims")?,
        spacing: req(spec, "spacing")?,
        radius: req(spec, "radius")?,
        shape_harmonics: req(spec, "shape_harmonics")?,
        shape_amplitude: req(spec, "shape_amplitude")?,
        axial_wavelength: req(spec, "axial_wavelength")?,
        center_wander: req(spec, "center_wander")?,
        thickness_mean: req(spec, "thickness_mean")?,
        thickness_harmonics: req(spec, "thickness_harmonics")?,
        thickness_variation: req(spec, "thickness_variation")?,
        thickness_min: req(spec, "thickness_min")?,
        bvtv: req(spec, "bvtv")?,
        trabecular_wavelength: req(spec, "trabecular_wavelength")?,
        porosity: req(spec, "porosity")?,
        pore_wavelength: req(spec, "pore_wavelength")?,
        endo_porosity: req(spec, "endo_porosity")?,
        endo_bvtv: req(spec, "endo_bvtv")?,
        transition_width: req(spec, "transition_width")?,
        rho_soft: req(spec, "rho_soft")?,
        rho_bone: req(spec, "rho_bone")?,
        rho_marrow: req(spec, "rho_marrow")?,
        blur: req(spec, "blur")?,
        noise: req(spec, "noise")?,
        seed: req(spec, "seed")?,
        supersample: req(spec, "supersample")?,
    };
    let (d0, d1, d2) = ph.dims;
    if d0 < 2 || d1 < 2 || d2 < 2 || ph.spacing.iter().any(|s| s.is_nan() || *s <= 0.0) {
        return Err(PyValueError::new_err("phantom dims must be >= 2 and spacing positive"));
    }
    Ok(ph)
}

/// Generate a realistic bone phantom: (image, periosteal mask, endosteal mask).
#[pyfunction]
#[pyo3(signature = (spec))]
#[allow(clippy::type_complexity)]
fn bone_phantom<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyDict>,
) -> PyResult<(
    Bound<'py, numpy::PyArray3<f32>>,
    Bound<'py, numpy::PyArray3<bool>>,
    Bound<'py, numpy::PyArray3<bool>>,
)> {
    let ph = bone_phantom_from(spec)?;
    let (img, p, e) = py.detach(|| ph.generate());
    Ok((img.into_pyarray(py), p.into_pyarray(py), e.into_pyarray(py)))
}

/// True (periosteal, endosteal) crossings along lines p + x n, x in [x0, x1].
#[pyfunction]
#[pyo3(signature = (spec, points, normals, x0, x1))]
fn bone_phantom_edges<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyDict>,
    points: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    x0: f64,
    x1: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ph = bone_phantom_from(spec)?;
    let p = points.as_array();
    let n = normals.as_array();
    if p.ncols() != 3 || n.dim() != p.dim() {
        return Err(PyValueError::new_err("points and normals must both be (N, 3)"));
    }
    let pv: Vec<V3> = p.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let nv: Vec<V3> = n.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let e = py.detach(|| ph.edges_along(&pv, &nv, x0, x1));
    let flat: Vec<f64> = e.iter().flat_map(|(a, b)| [*a, *b]).collect();
    Ok(Array2::from_shape_vec((e.len(), 2), flat).unwrap().into_pyarray(py))
}

/// Buie et al. (2007) dual-threshold segmentation: (periosteal, cortical, trabecular).
#[pyfunction]
#[pyo3(signature = (image, opts))]
#[allow(clippy::type_complexity)]
fn buie<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    opts: &Bound<'py, PyDict>,
) -> PyResult<(
    Bound<'py, numpy::PyArray3<bool>>,
    Bound<'py, numpy::PyArray3<bool>>,
    Bound<'py, numpy::PyArray3<bool>>,
)> {
    let o = cbm::segmentation::BuieOptions {
        periosteal_threshold: req(opts, "periosteal_threshold")?,
        endosteal_threshold: req(opts, "endosteal_threshold")?,
        median_kernel: req(opts, "median_kernel")?,
        periosteal_kernel: req(opts, "periosteal_kernel")?,
        endosteal_kernel: req(opts, "endosteal_kernel")?,
        gaussian_sigma: req(opts, "gaussian_sigma")?,
        gaussian_support: req(opts, "gaussian_support")?,
        smoothed_threshold: req(opts, "smoothed_threshold")?,
        connectivity_2d: req(opts, "connectivity_2d")?,
        largest_bone_only: req(opts, "largest_bone_only")?,
    };
    let r = with_image!(image, |img| Ok::<_, PyErr>(
        py.detach(|| cbm::segmentation::buie(img, &o))
    ))?;
    Ok((
        r.periosteal.into_pyarray(py),
        r.cortical.into_pyarray(py),
        r.trabecular.into_pyarray(py),
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(refine, m)?)?;
    m.add_function(wrap_pyfunction!(voxelize, m)?)?;
    m.add_function(wrap_pyfunction!(write_vtk, m)?)?;
    m.add_function(wrap_pyfunction!(phantom, m)?)?;
    m.add_function(wrap_pyfunction!(fit_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(bone_phantom, m)?)?;
    m.add_function(wrap_pyfunction!(bone_phantom_edges, m)?)?;
    m.add_function(wrap_pyfunction!(buie, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
