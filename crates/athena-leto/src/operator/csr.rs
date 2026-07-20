use athena_core::{KrylovBackend, LinearOperator};
use eunomia::RealField;
use leto_ops::{CsrMatrix, RealScalar, spmv_into};

use crate::{LetoBackend, LetoBackendError};

/// Square CSR operator backed by Leto's canonical sparse matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrOperator<T> {
    matrix: CsrMatrix<T>,
    dimension: usize,
}

impl<T: RealScalar + RealField> CsrOperator<T> {
    /// Validate and construct a square sparse operator.
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::NonSquareOperator`] for a rectangular
    /// matrix.
    pub fn new(matrix: CsrMatrix<T>) -> Result<Self, LetoBackendError> {
        let (rows, columns) = matrix.shape();
        if rows != columns {
            return Err(LetoBackendError::NonSquareOperator { rows, columns });
        }
        Ok(Self {
            matrix,
            dimension: rows,
        })
    }

    /// Borrow the canonical Leto CSR matrix.
    #[must_use]
    pub const fn matrix(&self) -> &CsrMatrix<T> {
        &self.matrix
    }

    /// Consume the operator and return the canonical Leto CSR matrix.
    #[must_use]
    pub fn into_matrix(self) -> CsrMatrix<T> {
        self.matrix
    }
}

impl<T: RealScalar + RealField> LinearOperator<LetoBackend<T>> for CsrOperator<T> {
    #[inline]
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(
        &self,
        _backend: &LetoBackend<T>,
        input: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        spmv_into(&self.matrix, &input, output).map_err(Into::into)
    }
}
