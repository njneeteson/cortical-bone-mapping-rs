//! Phantoms with exact ground truth, for tests, examples and experiments.
//!
//! * [`shell`]: smooth closed cortical shells with known thickness and blur.
//! * [`bone`]: realistic long-bone segments (porous cortex, trabeculae,
//!   optionally a neighbouring bone and cortical breaks).
//! * [`joint`]: the contact region of a knee.
//! * [`femur`]: a proximal femur at clinical CT scale, with the acetabulum.
//! * [`sites`]: the anatomical sites at their scanners' resolution, with
//!   parameters from the literature.

pub mod bone;
pub mod femur;
pub mod joint;
mod render;
pub mod shell;
pub mod sites;
