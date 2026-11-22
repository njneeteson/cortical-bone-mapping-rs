//! Conversions between Python and the crate: errors, option-dict access,
//! arrays, and dispatch over the image dtype.

use ::cortical_bone_mapping as cbm;
use cbm::model::Params;
use cbm::util::V3;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) fn to_pyerr(e: cbm::Error) -> PyErr {
    match e {
        cbm::Error::ThreadPool(_) => PyRuntimeError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

pub(crate) fn req<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    match d.get_item(key)? {
        Some(v) => v.extract::<T>().map_err(|e| {
            let e: PyErr = e.into();
            PyValueError::new_err(format!("option {key}: {e}"))
        }),
        None => Err(PyValueError::new_err(format!("missing option {key}"))),
    }
}

pub(crate) fn opt<'py, T: FromPyObjectOwned<'py>>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Option<T>> {
    match d.get_item(key)? {
        Some(v) if !v.is_none() => v.extract::<T>().map(Some).map_err(|e| {
            let e: PyErr = e.into();
            PyValueError::new_err(format!("option {key}: {e}"))
        }),
        _ => Ok(None),
    }
}

pub(crate) fn axis(d: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    let a: usize = req(d, key)?;
    if a > 2 {
        return Err(PyValueError::new_err(format!("{key} must be 0, 1 or 2")));
    }
    Ok(a)
}

pub(crate) fn v3_array<'py>(py: Python<'py>, v: &[V3]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = v.iter().flat_map(|p| p.iter().copied()).collect();
    Array2::from_shape_vec((v.len(), 3), flat).unwrap().into_pyarray(py)
}

pub(crate) fn faces_array<'py>(py: Python<'py>, f: &[[u32; 3]]) -> Bound<'py, PyArray2<u32>> {
    let flat: Vec<u32> = f.iter().flat_map(|p| p.iter().copied()).collect();
    Array2::from_shape_vec((f.len(), 3), flat).unwrap().into_pyarray(py)
}

pub(crate) fn params_array<'py>(py: Python<'py>, p: &[Params]) -> Bound<'py, PyArray2<f64>> {
    let flat: Vec<f64> = p.iter().flat_map(|q| q.iter().copied()).collect();
    Array2::from_shape_vec((p.len(), 5), flat).unwrap().into_pyarray(py)
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

pub(crate) use with_image;
pub(crate) fn mesh_from(vertices: PyReadonlyArray2<'_, f64>, faces: PyReadonlyArray2<'_, u32>) -> PyResult<cbm::Mesh> {
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
