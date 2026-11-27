//! Mesh rasterisation and VTK output.

use ::cortical_bone_mapping as cbm;
use cbm::util::V3;
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::convert::*;

/// Rasterise a closed triangle mesh into a boolean mask.
#[pyfunction]
#[pyo3(signature = (vertices, faces, shape, spacing, origin))]
pub(crate) fn voxelize<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, u32>,
    shape: (usize, usize, usize),
    spacing: V3,
    origin: V3,
) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
    crate::logs::refresh();
    let mesh = mesh_from(vertices, faces)?;
    let g = cbm::Geometry { spacing, origin };
    let out = py.detach(|| mesh.voxelize(shape, &g));
    Ok(out.into_pyarray(py))
}

/// The curvature measure at every vertex of a mesh (see
/// `cbm::mesh_curvature`).
#[pyfunction]
#[pyo3(signature = (vertices, faces, measure, scale, neighbours, distance))]
pub(crate) fn mesh_curvature<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, u32>,
    measure: &str,
    scale: Option<f64>,
    neighbours: usize,
    distance: &str,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    crate::logs::refresh();
    let mesh = mesh_from(vertices, faces)?;
    let measure = match measure {
        "mean" => cbm::CurvatureMeasure::Mean,
        "total" => cbm::CurvatureMeasure::Total,
        o => return Err(PyValueError::new_err(format!("unknown measure {o:?}"))),
    };
    let distance = match distance {
        "euclidean" => cbm::Distance::Euclidean,
        "surface" => cbm::Distance::Surface,
        o => return Err(PyValueError::new_err(format!("unknown distance {o:?}"))),
    };
    if scale.is_some_and(|s| s.is_nan() || s <= 0.0) || neighbours < 3 {
        return Err(PyValueError::new_err("scale must be positive and neighbours >= 3"));
    }
    let out = py.detach(|| cbm::mesh_curvature(&mesh, measure, scale, neighbours, distance));
    Ok(out.into_pyarray(py))
}

/// Write a mesh with point data as legacy binary VTK PolyData.
#[pyfunction]
#[pyo3(signature = (path, vertices, faces, scalars, vectors))]
pub(crate) fn write_vtk(
    path: std::path::PathBuf,
    vertices: PyReadonlyArray2<'_, f64>,
    faces: PyReadonlyArray2<'_, u32>,
    scalars: Vec<(String, Vec<f64>)>,
    vectors: Vec<(String, PyReadonlyArray2<'_, f64>)>,
) -> PyResult<()> {
    crate::logs::refresh();
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
