use eunomia::{NumericElement, RealField};
use leto_ops::{CsrMatrix, RealScalar};

use crate::LetoBackendError;

/// Position of each row's diagonal entry within a CSR value array.
///
/// CSR rows carry strictly increasing column indices, so the diagonal is
/// located once at construction and every later triangular sweep splits a row
/// into its strict lower part `[start, diagonal)` and upper part
/// `(diagonal, end)` by slicing rather than searching.
#[derive(Clone, Debug)]
pub(super) struct DiagonalIndex {
    positions: Vec<usize>,
}

impl DiagonalIndex {
    /// Locate every diagonal entry, requiring the matrix to be square and each
    /// row to store one.
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::NonSquareOperator`] or
    /// [`LetoBackendError::MissingDiagonal`].
    pub(super) fn new<T: RealScalar>(matrix: &CsrMatrix<T>) -> Result<Self, LetoBackendError> {
        let (rows, columns) = matrix.shape();
        if rows != columns {
            return Err(LetoBackendError::NonSquareOperator { rows, columns });
        }
        let row_ptr = matrix.row_ptr();
        let col_indices = matrix.col_indices();
        let mut positions = Vec::with_capacity(rows);
        for row in 0..rows {
            let span = row_ptr[row]..row_ptr[row + 1];
            let offset = col_indices[span.clone()]
                .binary_search(&row)
                .map_err(|_| LetoBackendError::MissingDiagonal { row })?;
            positions.push(span.start + offset);
        }
        Ok(Self { positions })
    }

    pub(super) fn position(&self, row: usize) -> usize {
        self.positions[row]
    }
}

/// Solve `L y = rhs` in place, where `L` is the strict lower triangle of the
/// stored pattern with the supplied diagonal.
///
/// `unit_diagonal` selects between an implicit unit diagonal, as an incomplete
/// LU factor carries, and the stored diagonal scaled by `diagonal_scale`.
pub(super) fn forward_substitute<T: RealScalar + RealField>(
    values: &[T],
    col_indices: &[usize],
    row_ptr: &[usize],
    diagonal: &DiagonalIndex,
    vector: &mut [T],
    unit_diagonal: bool,
    diagonal_scale: T,
) -> Result<(), LetoBackendError> {
    for row in 0..vector.len() {
        let start = row_ptr[row];
        let pivot = diagonal.position(row);
        let mut accumulated = vector[row];
        for entry in start..pivot {
            accumulated -= values[entry] * vector[col_indices[entry]];
        }
        if unit_diagonal {
            vector[row] = accumulated;
        } else {
            let pivot_value = values[pivot] * diagonal_scale;
            if pivot_value == <T as NumericElement>::ZERO {
                return Err(LetoBackendError::SingularDiagonal { index: row });
            }
            vector[row] = accumulated / pivot_value;
        }
    }
    Ok(())
}

/// Solve `U z = rhs` in place, where `U` is the upper triangle of the stored
/// pattern including its diagonal.
pub(super) fn back_substitute<T: RealScalar + RealField>(
    values: &[T],
    col_indices: &[usize],
    row_ptr: &[usize],
    diagonal: &DiagonalIndex,
    vector: &mut [T],
) -> Result<(), LetoBackendError> {
    for row in (0..vector.len()).rev() {
        let end = row_ptr[row + 1];
        let pivot = diagonal.position(row);
        let mut accumulated = vector[row];
        for entry in pivot + 1..end {
            accumulated -= values[entry] * vector[col_indices[entry]];
        }
        let pivot_value = values[pivot];
        if pivot_value == <T as NumericElement>::ZERO {
            return Err(LetoBackendError::SingularDiagonal { index: row });
        }
        vector[row] = accumulated / pivot_value;
    }
    Ok(())
}
