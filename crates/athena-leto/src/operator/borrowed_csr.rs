use athena_core::{KrylovBackend, LinearOperator};
use eunomia::RealField;
use leto_ops::{CsrMatrix, RealScalar, spmv_into};

use crate::{LetoBackend, LetoBackendError};

/// Square CSR operator that borrows a caller-owned matrix.
///
/// [`super::CsrOperator`] takes ownership, which forces a full sparse copy when
/// the caller keeps the matrix — a per-solve `O(nnz)` clone in a solver chain
/// that tries several preconditioners against the same system. This borrows
/// instead, so construction and every application are allocation-free.
#[derive(Clone, Debug, PartialEq)]
pub struct BorrowedCsrOperator<'a, T> {
    matrix: &'a CsrMatrix<T>,
    dimension: usize,
}

impl<'a, T: RealScalar + RealField> BorrowedCsrOperator<'a, T> {
    /// Borrow a square sparse matrix.
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::NonSquareOperator`] for a rectangular
    /// matrix.
    pub fn new(matrix: &'a CsrMatrix<T>) -> Result<Self, LetoBackendError> {
        let (rows, columns) = matrix.shape();
        if rows != columns {
            return Err(LetoBackendError::NonSquareOperator { rows, columns });
        }
        Ok(Self {
            matrix,
            dimension: rows,
        })
    }

    /// Borrow the underlying matrix.
    #[must_use]
    pub const fn matrix(&self) -> &'a CsrMatrix<T> {
        self.matrix
    }
}

impl<T: RealScalar + RealField> LinearOperator<LetoBackend<T>> for BorrowedCsrOperator<'_, T> {
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
        spmv_into(self.matrix, &input, output)?;
        Ok(())
    }
}
