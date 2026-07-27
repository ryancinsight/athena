use athena_core::{KrylovBackend, Preconditioner};
use eunomia::{NumericElement, RealField};
use leto_ops::{CsrMatrix, RealScalar};

use super::triangular::{DiagonalIndex, back_substitute, forward_substitute};
use crate::{LetoBackend, LetoBackendError};

/// Incomplete LU factorization with zero fill-in.
///
/// The factors reuse the sparsity pattern of the source matrix exactly: any
/// update that would land outside that pattern is discarded, which is what
/// makes the factorization incomplete and keeps its storage equal to the
/// matrix. `L` carries an implicit unit diagonal and `U` the stored one, so
/// both factors pack into one value array alongside the original indices.
///
/// # Reference
///
/// Saad (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed.,
/// §10.3.2, IKJ variant.
#[derive(Clone, Debug)]
pub struct IncompleteLu<T> {
    values: Vec<T>,
    col_indices: Vec<usize>,
    row_ptr: Vec<usize>,
    diagonal: DiagonalIndex,
    dimension: usize,
}

impl<T: RealScalar + RealField> IncompleteLu<T> {
    /// Factor a square CSR matrix in its own pattern.
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::NonSquareOperator`] for a rectangular
    /// matrix, [`LetoBackendError::MissingDiagonal`] when a row stores no
    /// diagonal entry, and [`LetoBackendError::SingularDiagonal`] when a pivot
    /// vanishes during elimination.
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Result<Self, LetoBackendError> {
        let diagonal = DiagonalIndex::new(matrix)?;
        let (dimension, _) = matrix.shape();
        let mut values = matrix.values().to_vec();
        let col_indices = matrix.col_indices().to_vec();
        let row_ptr = matrix.row_ptr().to_vec();

        for row in 1..dimension {
            let row_span = row_ptr[row]..row_ptr[row + 1];
            let row_diagonal = diagonal.position(row);
            // Strict lower entries of this row, in ascending column order,
            // which the CSR contract guarantees.
            for lower in row_span.start..row_diagonal {
                let pivot_row = col_indices[lower];
                let pivot_value = values[diagonal.position(pivot_row)];
                if pivot_value == <T as NumericElement>::ZERO {
                    return Err(LetoBackendError::SingularDiagonal { index: pivot_row });
                }
                let multiplier = values[lower] / pivot_value;
                values[lower] = multiplier;
                if multiplier == <T as NumericElement>::ZERO {
                    continue;
                }

                // Subtract the multiple of the pivot row, keeping only entries
                // the source pattern already carries. Both rows are sorted, so
                // one merge pass locates the shared columns.
                let mut target = lower + 1;
                let pivot_upper = diagonal.position(pivot_row) + 1..row_ptr[pivot_row + 1];
                for source in pivot_upper {
                    let column = col_indices[source];
                    while target < row_span.end && col_indices[target] < column {
                        target += 1;
                    }
                    if target == row_span.end {
                        break;
                    }
                    if col_indices[target] == column {
                        let update = multiplier * values[source];
                        values[target] -= update;
                    }
                }
            }
            if values[row_diagonal] == <T as NumericElement>::ZERO {
                return Err(LetoBackendError::SingularDiagonal { index: row });
            }
        }

        Ok(Self {
            values,
            col_indices,
            row_ptr,
            diagonal,
            dimension,
        })
    }

    /// Dimension of the factored system.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

impl<T: RealScalar + RealField> Preconditioner<LetoBackend<T>> for IncompleteLu<T> {
    fn apply(
        &self,
        _backend: &LetoBackend<T>,
        residual: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let residual = residual
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        if residual.len() != output.len() {
            return Err(LetoBackendError::LengthMismatch {
                left: residual.len(),
                right: output.len(),
            });
        }
        if residual.len() != self.dimension {
            return Err(LetoBackendError::LengthMismatch {
                left: residual.len(),
                right: self.dimension,
            });
        }

        output.copy_from_slice(residual);
        forward_substitute(
            &self.values,
            &self.col_indices,
            &self.row_ptr,
            &self.diagonal,
            output,
            true,
            <T as NumericElement>::ONE,
        )?;
        back_substitute(
            &self.values,
            &self.col_indices,
            &self.row_ptr,
            &self.diagonal,
            output,
        )
    }
}
