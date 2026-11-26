//! Python bindings for the `cortical-bone-mapping` crate.
//!
//! This is the private extension module `cortical_bone_mapping_rs._core`.
//! The public, documented API (option dataclasses, validation, result
//! classes) lives in the Python package `python/cortical_bone_mapping_rs/`,
//! which passes a flat dict of options. Images of any common numeric dtype
//! are sampled in place without conversion; masks arrive as `uint8`.
//! The crate's log records go to Python's `logging` ([`logs`]).

mod convert;
mod fitting;
mod logs;
mod mesh;
mod options;
mod phantoms;
mod segmentation;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    logs::install(m.py())?;
    m.add_function(wrap_pyfunction!(segmentation::otsu_thresholds, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::label_bones, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::bone_hulls, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::grow_bones, m)?)?;
    m.add_function(wrap_pyfunction!(fitting::run, m)?)?;
    m.add_function(wrap_pyfunction!(fitting::refine, m)?)?;
    m.add_function(wrap_pyfunction!(fitting::mask_surface, m)?)?;
    m.add_function(wrap_pyfunction!(mesh::voxelize, m)?)?;
    m.add_function(wrap_pyfunction!(mesh::write_vtk, m)?)?;
    m.add_function(wrap_pyfunction!(mesh::mesh_curvature, m)?)?;
    m.add_function(wrap_pyfunction!(phantoms::phantom, m)?)?;
    m.add_function(wrap_pyfunction!(fitting::fit_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phantoms::bone_phantom, m)?)?;
    m.add_function(wrap_pyfunction!(phantoms::bone_phantom_edges, m)?)?;
    m.add_function(wrap_pyfunction!(phantoms::bone_phantom_labels, m)?)?;
    m.add_function(wrap_pyfunction!(phantoms::femur_phantom, m)?)?;
    m.add_function(wrap_pyfunction!(phantoms::femur_phantom_edges, m)?)?;
    m.add_function(wrap_pyfunction!(phantoms::site_phantom, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::buie, m)?)?;
    m.add_function(wrap_pyfunction!(segmentation::hull_segmentation, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
