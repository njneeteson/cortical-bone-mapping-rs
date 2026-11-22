//! Surfaces: triangle meshes, their extraction from masks and level sets
//! (surface nets), rasterisation, VTK output, the cotangent Laplacian, and
//! the level-set embedding used by refinement.

pub mod cotan;
pub mod levelset;
pub mod mesh;
pub mod nets;
pub mod voxelize;
pub mod vtk;
