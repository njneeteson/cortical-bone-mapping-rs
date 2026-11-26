//! Logging.
//!
//! The crate logs through the [`log`] facade, so nothing is printed unless
//! the application installs a logger (e.g. `env_logger` in Rust; the Python
//! package forwards the records to Python's `logging`, logger
//! `cortical_bone_mapping`). Levels:
//!
//! | level | what |
//! |---|---|
//! | `info` | one line per top-level call (fit, refinement, segmentation step) with its outcome |
//! | `debug` | every stage, when it starts and when it ends (sizes, duration), every solver iteration, every refinement move, the options in use |
//! | `trace` | details within a stage: conjugate-gradient runs, per-region decisions |
//! | `warn` | something went wrong but the call carries on (a fit that did not converge, a fallback) |
//!
//! Targets are module paths (`cortical_bone_mapping::fit::volumetric`, ...),
//! so one part of the crate can be followed alone. Nothing is logged inside
//! per-node or per-voxel loops: a disabled record costs one level check.
//!
//! A stage logs its start, so that the log of a run that stalls shows where
//! it is:
//!
//! ```ignore
//! let s = stage!("band");
//! let band = Band::build(...);
//! s.done(format_args!("{} voxels", band.len()));
//! ```

use std::fmt;
use std::time::Instant;

/// A timed stage of a computation (see the module documentation); made with
/// [`stage!`](crate::stage), which logs under the caller's module.
#[must_use = "a stage logs its end and duration with `done`"]
pub struct Stage {
    target: &'static str,
    name: &'static str,
    start: Instant,
}

impl Stage {
    /// Start stage `name`, logged at debug level under `target`.
    pub fn new(target: &'static str, name: &'static str) -> Self {
        log::debug!(target: target, "{name}: start");
        Stage {
            target,
            name,
            start: Instant::now(),
        }
    }

    /// Seconds since the stage started.
    pub fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// End the stage, logging `detail` and its duration; returns the
    /// duration in seconds.
    pub fn done(self, detail: fmt::Arguments) -> f64 {
        let t = self.elapsed();
        log::debug!(target: self.target, "{}: done in {t:.3} s; {detail}", self.name);
        t
    }

    /// End the stage with nothing more to report than its duration.
    pub fn end(self) -> f64 {
        let t = self.elapsed();
        log::debug!(target: self.target, "{}: done in {t:.3} s", self.name);
        t
    }
}

/// Start a [`Stage`] logged under the calling module.
#[macro_export]
#[doc(hidden)]
macro_rules! stage {
    ($name:expr) => {
        $crate::logging::Stage::new(module_path!(), $name)
    };
}
