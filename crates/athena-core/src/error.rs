use core::fmt;

/// Failure to execute a solve.
#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub enum SolveError<E> {
    /// A vector or operator dimension did not match the system dimension.
    DimensionMismatch {
        /// Operation whose dimension contract failed.
        context: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// A backend allocation or arithmetic operation failed.
    Backend(E),
}

impl<E: fmt::Display> fmt::Display for SolveError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "{context} dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::Backend(error) => write!(formatter, "solver backend failed: {error}"),
        }
    }
}

impl<E: core::error::Error> core::error::Error for SolveError<E> {}
