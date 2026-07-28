use athena_core::{KrylovBackend, RectangularOperator};
use eunomia::{NumericElement, RealField};
use leto_ops::{CsrMatrix, RealScalar, spmv_into};

use crate::{LetoBackend, LetoBackendError};

/// Rectangular CSR operator supporting both `A x` and `Aᵀ y`.
///
/// Kept separate from the square [`super::CsrOperator`]: least-squares
/// recurrences need the adjoint and genuinely different operand lengths, and
/// a square operator should not be forced to carry a transpose it never uses.
#[derive(Clone, Debug, PartialEq)]
pub struct RectangularCsrOperator<T> {
    matrix: CsrMatrix<T>,
    rows: usize,
    columns: usize,
}

impl<T: RealScalar + RealField> RectangularCsrOperator<T> {
    /// Wrap a CSR matrix of any shape.
    #[must_use]
    pub fn new(matrix: CsrMatrix<T>) -> Self {
        let (rows, columns) = matrix.shape();
        Self {
            matrix,
            rows,
            columns,
        }
    }

    /// Borrow the canonical Leto CSR matrix.
    #[must_use]
    pub const fn matrix(&self) -> &CsrMatrix<T> {
        &self.matrix
    }
}

impl<T: RealScalar + RealField> RectangularOperator<LetoBackend<T>> for RectangularCsrOperator<T> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn columns(&self) -> usize {
        self.columns
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
        spmv_into(&self.matrix, &input, output)?;
        Ok(())
    }

    /// `y = Aᵀ·x`, accumulated by scattering each stored row into `y`.
    ///
    /// Transposing the matrix first would cost an `O(nnz)` allocation and a
    /// full rebuild per application; the scatter reads the same CSR arrays the
    /// forward product does, which matters because LSQR applies the adjoint
    /// once per iteration.
    fn apply_transpose(
        &self,
        _backend: &LetoBackend<T>,
        input: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let input = input
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        if input.len() != self.rows {
            return Err(LetoBackendError::LengthMismatch {
                left: input.len(),
                right: self.rows,
            });
        }
        if output.len() != self.columns {
            return Err(LetoBackendError::LengthMismatch {
                left: output.len(),
                right: self.columns,
            });
        }

        output.fill(<T as NumericElement>::ZERO);
        let row_ptr = self.matrix.row_ptr();
        let col_indices = self.matrix.col_indices();
        let values = self.matrix.values();
        for (row, &scale) in input.iter().enumerate() {
            for entry in row_ptr[row]..row_ptr[row + 1] {
                output[col_indices[entry]] += values[entry] * scale;
            }
        }
        Ok(())
    }
}
