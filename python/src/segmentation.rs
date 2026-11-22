//! Segmentation: Otsu thresholds, bones, starts, hulls and Buie's method.

use ::cortical_bone_mapping as cbm;
use numpy::{IntoPyArray, PyReadonlyArray3};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::*;

/// Buie et al. (2007) dual-threshold segmentation: (periosteal, cortical, trabecular).
#[pyfunction]
#[pyo3(signature = (image, opts))]
#[allow(clippy::type_complexity)]
pub(crate) fn buie<'py>(
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

/// Threshold + median + largest component + per-slice convex hull.
#[pyfunction]
#[pyo3(signature = (image, threshold, median_kernel, largest_component))]
pub(crate) fn hull_segmentation<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    threshold: f64,
    median_kernel: [usize; 3],
    largest_component: bool,
) -> PyResult<Bound<'py, numpy::PyArray3<bool>>> {
    let o = cbm::segmentation::HullOptions {
        threshold,
        median_kernel,
        largest_component,
    };
    let r = with_image!(image, |img| Ok::<_, PyErr>(
        py.detach(|| cbm::segmentation::hull_segmentation(img, &o))
    ))?;
    Ok(r.into_pyarray(py))
}

/// Histogram thresholds by (multi-level) Otsu's method.
#[pyfunction]
#[pyo3(signature = (image, floor, classes, bins))]
pub(crate) fn otsu_thresholds(
    image: &Bound<'_, PyAny>,
    floor: Option<f64>,
    classes: usize,
    bins: usize,
) -> PyResult<Vec<f64>> {
    if !(classes == 2 || classes == 3) {
        return Err(PyValueError::new_err("classes must be 2 or 3"));
    }
    if bins < 4 {
        return Err(PyValueError::new_err("bins must be at least 4"));
    }
    let o = cbm::segmentation::OtsuOptions { floor, classes, bins };
    let py = image.py();
    with_image!(image, |img| Ok::<_, PyErr>(
        py.detach(|| cbm::segmentation::otsu_thresholds(img, &o))
    ))
}

/// Threshold, median, connected components → bones labelled by size.
#[pyfunction]
#[pyo3(signature = (image, threshold, median_kernel, min_size, max_bones))]
pub(crate) fn label_bones<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    threshold: f64,
    median_kernel: [usize; 3],
    min_size: usize,
    max_bones: Option<usize>,
) -> PyResult<(Bound<'py, numpy::PyArray3<u16>>, Vec<usize>)> {
    let o = cbm::segmentation::BoneLabelOptions {
        threshold,
        median_kernel,
        min_size,
        max_bones,
    };
    let r = with_image!(image, |img| Ok::<_, PyErr>(
        py.detach(|| cbm::segmentation::label_bones(img, &o))
    ))?;
    Ok((r.labels.into_pyarray(py), r.sizes))
}

/// Hysteresis: grow labelled bones into connected voxels >= low_threshold.
#[pyfunction]
#[pyo3(signature = (image, labels, low_threshold, median_kernel))]
pub(crate) fn grow_bones<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    labels: PyReadonlyArray3<'py, u16>,
    low_threshold: f64,
    median_kernel: [usize; 3],
) -> PyResult<Bound<'py, numpy::PyArray3<u16>>> {
    let l = labels.as_array();
    let r =
        with_image!(image, |img| Ok::<_, PyErr>(py.detach(|| {
            cbm::segmentation::grow_bones(img, l, low_threshold, median_kernel)
        })))?;
    Ok(r.into_pyarray(py))
}

/// Per-bone per-slice convex hulls (or closings), kept apart.
#[pyfunction]
#[pyo3(signature = (labels, spacing, max_distance, closing_radius))]
pub(crate) fn bone_hulls<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray3<'py, u16>,
    spacing: [f64; 3],
    max_distance: Option<f64>,
    closing_radius: Option<f64>,
) -> PyResult<Bound<'py, numpy::PyArray3<u16>>> {
    let l = labels.as_array();
    let r = py.detach(|| cbm::segmentation::bone_hulls(l, spacing, max_distance, closing_radius));
    Ok(r.into_pyarray(py))
}
