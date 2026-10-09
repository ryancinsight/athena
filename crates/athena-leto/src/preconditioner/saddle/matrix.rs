//! Sparse-matrix surgery for saddle-point preconditioning.
//!
//! Extracting blocks, reading diagonals, and regularising structurally empty
//! preconditioner rows are shared by every saddle-point family member, so the
//! helpers live here once.

use eunomia::{FloatElement, NumericElement, RealField};
use leto::{LetoError, Result};
use leto_ops::{CsrMatrix, RealScalar, Scalar as LetoScalar};

/// Add provider-pivot-scale diagonal entries to structurally empty or
/// numerically zero preconditioner rows.
///
/// Component extraction can leave an isolated velocity or pressure row even
/// when the full saddle system is valid. A direct factorization must reject
/// that singular block; the block preconditioner instead represents the row by
/// the provider's pivot-scale identity, which preserves a bounded solve while
/// leaving the assembled operator unchanged.
pub(super) fn stabilize_preconditioner_diagonal<T>(
    matrix: &CsrMatrix<T>,
    pivot_tolerance: f64,
) -> Result<CsrMatrix<T>>
where
    T: RealField + Copy + FloatElement + RealScalar,
{
    let rows = matrix.nrows();
    let pivot_scale = <T as FloatElement>::from_f64(pivot_tolerance);
    let mut row_ptr = Vec::with_capacity(rows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);

    for row_index in 0..rows {
        let row = matrix.row(row_index);
        let mut diagonal = <T as NumericElement>::ZERO;
        let mut row_scale = <T as NumericElement>::ZERO;
        let mut entries: Vec<(usize, T)> = Vec::with_capacity(row.nnz() + 1);
        for (&column, &value) in row.col_indices().iter().zip(row.values()) {
            let magnitude = NumericElement::abs(value);
            if magnitude > row_scale {
                row_scale = magnitude;
            }
            if column == row_index {
                diagonal += value;
            }
            entries.push((column, value));
        }

        if NumericElement::abs(diagonal) <= diagonal_epsilon() {
            let mut regularization = row_scale * pivot_scale;
            if regularization < diagonal_epsilon() {
                regularization = diagonal_epsilon();
            }
            entries.push((row_index, regularization));
            entries.sort_unstable_by_key(|&(column, _)| column);
        }

        for (column, value) in entries {
            col_indices.push(column);
            values.push(value);
        }
        row_ptr.push(values.len());
    }

    CsrMatrix::from_parts(values, col_indices, row_ptr, rows, matrix.ncols())
}

/// Extract diagonal element from a CSR row, zero when structurally absent.
pub(super) fn get_diagonal<T: RealField + Copy + RealScalar>(
    matrix: &CsrMatrix<T>,
    row: usize,
) -> T {
    let row_range = matrix.row(row);
    for (col_idx, &value) in row_range.col_indices().iter().zip(row_range.values()) {
        if *col_idx == row {
            return value;
        }
    }
    <T as NumericElement>::ZERO
}

/// Copy the sub-block of `matrix` spanning `rows` and the `columns`-wide
/// column range starting at `column_offset` into its own CSR matrix, with
/// row and column indices relative to the block.
///
/// Column order within each row is inherited from `matrix`, so the block keeps
/// the strictly increasing column invariant of CSR.
pub(super) fn extract_block<T: RealScalar>(
    matrix: &CsrMatrix<T>,
    rows: std::ops::Range<usize>,
    column_offset: usize,
    columns: usize,
) -> Result<CsrMatrix<T>> {
    let block_rows = rows.len();
    let mut row_ptr = Vec::with_capacity(block_rows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in rows {
        let source = matrix.row(row);
        for (&column, &value) in source.col_indices().iter().zip(source.values()) {
            if let Some(local) = column
                .checked_sub(column_offset)
                .filter(|&index| index < columns)
            {
                col_indices.push(local);
                values.push(value);
            }
        }
        row_ptr.push(values.len());
    }
    CsrMatrix::from_parts(values, col_indices, row_ptr, block_rows, columns)
}

/// The scale below which a diagonal counts as absent.
///
/// The saddle systems this family serves assemble entries whose smallest
/// physical scale is O(1); a double-precision diagonal below `1e-14` is
/// structural absence or cancellation, not data.
pub(super) fn diagonal_epsilon<T: FloatElement>() -> T {
    <T as FloatElement>::from_f64(1e-14)
}

/// Convert a count to the scalar type.
#[inline]
pub(super) fn from_usize<T: FloatElement>(value: usize) -> T {
    let value_u64 = u64::try_from(value).expect("invariant: usize fits in u64");
    <T as FloatElement>::from_f64(<u64 as NumericElement>::to_f64(value_u64))
}

/// A row-major triple collector for small CSR assembly inside this family.
///
/// Rows are built independently; within a row the columns must arrive in
/// strictly increasing order without duplicates — the CSR contract the
/// callers produce by scanning a sorted source row — and [`Self::build`]
/// enforces it.
pub(super) struct RowBuilder<T> {
    columns: usize,
    rows: usize,
    entries: Vec<(usize, usize, T)>,
}

impl<T: RealField + Copy + LetoScalar> RowBuilder<T> {
    /// Create a builder for a `rows × columns` matrix.
    pub(super) fn new(rows: usize, columns: usize) -> Self {
        Self {
            columns,
            rows,
            entries: Vec::new(),
        }
    }

    /// Append an entry. Arrival order is free; [`Self::build`] sorts by
    /// `(row, column)` and rejects duplicates.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] when the row or column is out of
    /// range.
    pub(super) fn add_entry(&mut self, row: usize, column: usize, value: T) -> Result<()> {
        if row >= self.rows {
            return Err(LetoError::InvalidInput(format!(
                "row index {row} out of range"
            )));
        }
        if column >= self.columns {
            return Err(LetoError::InvalidInput(format!(
                "column index {column} out of range"
            )));
        }
        self.entries.push((row, column, value));
        Ok(())
    }

    /// Build the CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns [`LetoError::InvalidInput`] for a duplicate `(row, column)`
    /// pair, and the CSR validation error of the assembled parts otherwise.
    pub(super) fn build(self) -> Result<CsrMatrix<T>> {
        let mut sorted = self.entries;
        sorted.sort_unstable_by_key(|&(row, column, _)| (row, column));
        let mut row_ptr = Vec::with_capacity(self.rows + 1);
        let mut col_indices = Vec::with_capacity(sorted.len());
        let mut values = Vec::with_capacity(sorted.len());
        row_ptr.push(0);
        for pair in sorted.windows(2) {
            if pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1 {
                return Err(LetoError::InvalidInput(format!(
                    "duplicate column {} in row {}",
                    pair[0].1, pair[0].0
                )));
            }
        }
        for (row, column, value) in sorted.iter().copied() {
            while row_ptr.len() <= row {
                row_ptr.push(values.len());
            }
            col_indices.push(column);
            values.push(value);
        }
        while row_ptr.len() <= self.rows {
            row_ptr.push(values.len());
        }
        CsrMatrix::from_parts(values, col_indices, row_ptr, self.rows, self.columns)
    }
}
