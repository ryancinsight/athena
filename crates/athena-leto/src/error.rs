use core::fmt;

/// Failure in a Leto-backed Athena operation.
#[non_exhaustive]
#[derive(Debug)]
pub enum LetoBackendError {
    /// A delegated Leto operation failed.
    Leto(leto::LetoError),
    /// Athena received a non-contiguous vector view where its vector contract
    /// requires dense rank-one storage.
    NonContiguousVector,
    /// Operands had different logical lengths.
    LengthMismatch {
        /// First operand length.
        left: usize,
        /// Second operand length.
        right: usize,
    },
    /// A square operator was required.
    NonSquareOperator {
        /// Matrix row count.
        rows: usize,
        /// Matrix column count.
        columns: usize,
    },
    /// A diagonal or pivot entry required by a preconditioner was zero.
    SingularDiagonal {
        /// Index of the zero entry.
        index: usize,
    },
    /// A row carried no stored diagonal entry, so the sparsity pattern cannot
    /// support a triangular solve.
    MissingDiagonal {
        /// Row lacking a diagonal entry.
        row: usize,
    },
    /// A relaxation factor fell outside the convergent open interval `(0, 2)`.
    InvalidRelaxation,
}

impl From<leto::LetoError> for LetoBackendError {
    fn from(error: leto::LetoError) -> Self {
        Self::Leto(error)
    }
}

impl fmt::Display for LetoBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leto(error) => write!(formatter, "Leto operation failed: {error}"),
            Self::NonContiguousVector => {
                formatter.write_str("Athena Leto vectors must be contiguous")
            }
            Self::LengthMismatch { left, right } => {
                write!(formatter, "vector length mismatch: {left} != {right}")
            }
            Self::NonSquareOperator { rows, columns } => {
                write!(formatter, "operator must be square: got {rows} x {columns}")
            }
            Self::SingularDiagonal { index } => {
                write!(formatter, "diagonal is zero at index {index}")
            }
            Self::MissingDiagonal { row } => {
                write!(formatter, "row {row} has no stored diagonal entry")
            }
            Self::InvalidRelaxation => {
                formatter.write_str("relaxation factor must lie in the open interval (0, 2)")
            }
        }
    }
}

impl std::error::Error for LetoBackendError {}
