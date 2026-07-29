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
    /// Value-array position of each row's diagonal, or `None` where the
    /// sparsity pattern stores no diagonal entry for that row.
    positions: Vec<Option<usize>>,
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
        let index = Self::new_optional(matrix)?;
        for (row, position) in index.positions.iter().enumerate() {
            if position.is_none() {
                return Err(LetoBackendError::MissingDiagonal { row });
            }
        }
        Ok(index)
    }

    /// Locate diagonal entries, tolerating rows that store none.
    ///
    /// A row without a stored diagonal carries no coupling to itself in the
    /// pattern. Callers that can treat such a row as an identity row use this;
    /// callers whose factorisation needs every pivot use [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::NonSquareOperator`] for a rectangular
    /// matrix.
    pub(super) fn new_optional<T: RealScalar>(
        matrix: &CsrMatrix<T>,
    ) -> Result<Self, LetoBackendError> {
        let (rows, columns) = matrix.shape();
        if rows != columns {
            return Err(LetoBackendError::NonSquareOperator { rows, columns });
        }
        let row_ptr = matrix.row_ptr();
        let col_indices = matrix.col_indices();
        let mut positions = Vec::with_capacity(rows);
        for row in 0..rows {
            let span = row_ptr[row]..row_ptr[row + 1];
            let offset = col_indices[span.clone()].binary_search(&row).ok();
            positions.push(offset.map(|offset| span.start + offset));
        }
        Ok(Self { positions })
    }

    pub(super) fn position(&self, row: usize) -> Option<usize> {
        self.positions[row]
    }

    /// Diagonal position for a row a required-diagonal index was built for.
    pub(super) fn required_position(&self, row: usize) -> usize {
        self.positions[row].expect("invariant: required diagonal index stores every row")
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
        // Without a stored diagonal the row has no self-coupling in the
        // pattern; its strict lower part ends where the row does, and the
        // implied pivot is one.
        let Some(pivot) = diagonal.position(row) else {
            let mut accumulated = vector[row];
            for entry in start..row_ptr[row + 1] {
                if col_indices[entry] < row {
                    accumulated -= values[entry] * vector[col_indices[entry]];
                }
            }
            vector[row] = accumulated;
            continue;
        };
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
        let pivot = diagonal.required_position(row);
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
