//! Phantoms with exact ground truth.

use ::cortical_bone_mapping as cbm;
use cbm::util::V3;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::*;

/// Generate a synthetic shell phantom; returns (image float32, mask uint8).
#[pyfunction]
#[pyo3(signature = (spec))]
#[allow(clippy::type_complexity)]
pub(crate) fn phantom<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyDict>,
) -> PyResult<(Bound<'py, numpy::PyArray3<f32>>, Bound<'py, numpy::PyArray3<u8>>)> {
    use cbm::phantoms::shell::{Shape, ShellPhantom};
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

pub(crate) fn bone_phantom_from(spec: &Bound<'_, PyDict>) -> PyResult<cbm::phantoms::bone::BonePhantom> {
    let ph = cbm::phantoms::bone::BonePhantom {
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
        neighbour: opt::<(f64, f64, f64, f64)>(spec, "neighbour")?.map(|(radius, thickness, gap, angle)| {
            cbm::phantoms::bone::NeighbourBone {
                radius,
                thickness,
                gap,
                angle,
            }
        }),
        breaks: opt::<Vec<(f64, f64, f64, f64)>>(spec, "breaks")?
            .unwrap_or_default()
            .into_iter()
            .map(|(angle, width, z_center, length)| cbm::phantoms::bone::CorticalBreak {
                angle,
                width,
                z_center,
                length,
            })
            .collect(),
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
pub(crate) fn bone_phantom<'py>(
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
pub(crate) fn bone_phantom_edges<'py>(
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

/// Feature labels of a bone phantom: 1 = neighbouring bone, 2 = cortical break.
#[pyfunction]
#[pyo3(signature = (spec))]
pub(crate) fn bone_phantom_labels<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
    let ph = bone_phantom_from(spec)?;
    Ok(py.detach(|| ph.labels()).into_pyarray(py))
}

pub(crate) fn joint_phantom_from(spec: &Bound<'_, PyDict>) -> PyResult<cbm::phantoms::joint::JointPhantom> {
    let ph = cbm::phantoms::joint::JointPhantom {
        dims: req(spec, "dims")?,
        spacing: req(spec, "spacing")?,
        radius: req(spec, "radius")?,
        femur_scale: req(spec, "femur_scale")?,
        shape_harmonics: req(spec, "shape_harmonics")?,
        shape_amplitude: req(spec, "shape_amplitude")?,
        joint_level: req(spec, "joint_level")?,
        joint_space: req(spec, "joint_space")?,
        contact_offset: req(spec, "contact_offset")?,
        tibia_curvature: req(spec, "tibia_curvature")?,
        femur_curvature: req(spec, "femur_curvature")?,
        bump_amplitude: req(spec, "bump_amplitude")?,
        bump_wavelength: req(spec, "bump_wavelength")?,
        plate_thickness: req(spec, "plate_thickness")?,
        plate_variation: req(spec, "plate_variation")?,
        plate_min: req(spec, "plate_min")?,
        shell_thickness: req(spec, "shell_thickness")?,
        rim_rounding: req(spec, "rim_rounding")?,
        bvtv: req(spec, "bvtv")?,
        endo_bvtv: req(spec, "endo_bvtv")?,
        transition_width: req(spec, "transition_width")?,
        trabecular_wavelength: req(spec, "trabecular_wavelength")?,
        porosity: req(spec, "porosity")?,
        pore_wavelength: req(spec, "pore_wavelength")?,
        rho_soft: req(spec, "rho_soft")?,
        rho_cartilage: req(spec, "rho_cartilage")?,
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

/// Generate a knee-joint phantom: (image, periosteal labels, endosteal labels).
#[pyfunction]
#[pyo3(signature = (spec))]
#[allow(clippy::type_complexity)]
pub(crate) fn joint_phantom<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyDict>,
) -> PyResult<(
    Bound<'py, numpy::PyArray3<f32>>,
    Bound<'py, numpy::PyArray3<u8>>,
    Bound<'py, numpy::PyArray3<u8>>,
)> {
    let ph = joint_phantom_from(spec)?;
    let (img, p, e) = py.detach(|| ph.generate());
    Ok((img.into_pyarray(py), p.into_pyarray(py), e.into_pyarray(py)))
}

/// True (periosteal, endosteal, bone) crossings along lines p + x n.
#[pyfunction]
#[pyo3(signature = (spec, points, normals, x0, x1))]
pub(crate) fn joint_phantom_edges<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyDict>,
    points: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    x0: f64,
    x1: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let ph = joint_phantom_from(spec)?;
    let p = points.as_array();
    let n = normals.as_array();
    if p.ncols() != 3 || n.dim() != p.dim() {
        return Err(PyValueError::new_err("points and normals must both be (N, 3)"));
    }
    let pv: Vec<V3> = p.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let nv: Vec<V3> = n.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let e = py.detach(|| ph.edges_along(&pv, &nv, x0, x1));
    let flat: Vec<f64> = e.iter().flat_map(|(a, b, c)| [*a, *b, *c as f64]).collect();
    Ok(Array2::from_shape_vec((e.len(), 3), flat).unwrap().into_pyarray(py))
}
