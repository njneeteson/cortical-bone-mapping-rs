//! The crate's log records, forwarded to Python's `logging` (logger
//! `cortical_bone_mapping` and its children, one per module).
//!
//! `pyo3-log` caches each Python logger and its level the first time a
//! module logs, so that a disabled record never takes the GIL. The cache is
//! cleared at the start of every call into the extension ([`refresh`]): a
//! level set in Python takes effect at the next call.

use std::sync::OnceLock;

use log::LevelFilter;
use pyo3::prelude::*;
use pyo3_log::{Caching, Logger, ResetHandle};

static HANDLE: OnceLock<ResetHandle> = OnceLock::new();

/// Install the bridge (once, when the module is imported).
pub(crate) fn install(py: Python<'_>) -> PyResult<()> {
    let logger = Logger::new(py, Caching::LoggersAndLevels)?.filter(LevelFilter::Trace);
    if let Ok(handle) = logger.install() {
        let _ = HANDLE.set(handle);
    }
    Ok(())
}

/// Forget the cached Python loggers and levels.
pub(crate) fn refresh() {
    if let Some(h) = HANDLE.get() {
        h.reset();
    }
}
