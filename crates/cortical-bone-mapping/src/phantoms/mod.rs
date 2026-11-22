//! Phantoms with exact ground truth, for tests, examples and experiments.
//!
//! * [`shell`]: smooth closed cortical shells with known thickness and blur.
//! * [`bone`]: realistic long-bone segments (porous cortex, trabeculae,
//!   optionally a neighbouring bone and cortical breaks).
//! * [`joint`]: the contact region of a knee.

pub mod bone;
pub mod joint;
mod render;
pub mod shell;
