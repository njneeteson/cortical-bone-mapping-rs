//! Errors of the pipeline.

/// Errors from the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    EmptyMask,
    ShapeMismatch(String),
    NoNodes,
    InvalidOption(String),
    ThreadPool(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::EmptyMask => write!(f, "the mask is empty"),
            Error::ShapeMismatch(s) => write!(f, "shape mismatch: {s}"),
            Error::NoNodes => write!(f, "no surface nodes left to fit (check sub-mask and normal constraint)"),
            Error::InvalidOption(s) => write!(f, "invalid option: {s}"),
            Error::ThreadPool(s) => write!(f, "could not build thread pool: {s}"),
        }
    }
}
impl std::error::Error for Error {}
