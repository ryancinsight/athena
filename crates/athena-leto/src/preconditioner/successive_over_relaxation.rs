use athena_core::{KrylovBackend, Preconditioner};
use eunomia::{NumericElement, RealField};
use leto_ops::{CsrMatrix, RealScalar};

use super::triangular::{DiagonalIndex, forward_substitute};
use crate::{LetoBackend, LetoBackendError};

/// Successive over-relaxation preconditioner.
///
/// Applies the inverse of the lower-triangular splitting factor
/// `D/omega + L`, which reduces to Gauss-Seidel at `omega = 1`. The apply is a
/// single forward sweep over the stored lower triangle, so it costs one pass
/// over the nonzeros and allocates nothing.
///
/// The factor is not symmetric, so this preconditioner suits GMRES and
/// `BiCGSTAB` rather than CG, whose contract requires a symmetric positive
/// definite preconditioner.
#[derive(Clone, Debug)]
pub struct SuccessiveOverRelaxation<T> {
    values: Vec<T>,
    col_indices: Vec<usize>,
    row_ptr: Vec<usize>,
    diagonal: DiagonalIndex,
    inverse_relaxation: T,
    dimension: usize,
}

impl<T: RealScalar + RealField> SuccessiveOverRelaxation<T> {
    /// Build from a square CSR matrix and a relaxation factor.
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::InvalidRelaxation`] unless `relaxation`
    /// lies in the open interval `(0, 2)`, the classical convergence range for
    /// the underlying splitting. Indexing the diagonal additionally returns
    /// [`LetoBackendError::NonSquareOperator`] for a rectangular matrix and
    /// [`LetoBackendError::MissingDiagonal`] when a row stores no diagonal
    /// entry.
    pub fn from_csr(matrix: &CsrMatrix<T>, relaxation: T) -> Result<Self, LetoBackendError> {
        Self::build(matrix, relaxation, false)
    }

    /// Build while treating a row with no stored diagonal as an identity row.
    ///
    /// A sparsity pattern may omit the diagonal of a row that carries no
    /// self-coupling — an inactive degree of freedom, or a constraint row
    /// eliminated during assembly. [`Self::from_csr`] rejects those, because a
    /// splitting factor with a structurally absent pivot is undefined. This
    /// constructor instead gives such a row an implied unit pivot, leaving it
    /// unchanged by the sweep.
    ///
    /// The distinction is deliberate rather than defaulted: a caller that does
    /// not know its assembly omits diagonals should hear about it.
    ///
    /// # Errors
    ///
    /// As [`Self::from_csr`], minus the missing-diagonal case.
    pub fn from_csr_with_identity_rows(
        matrix: &CsrMatrix<T>,
        relaxation: T,
    ) -> Result<Self, LetoBackendError> {
        Self::build(matrix, relaxation, true)
    }

    fn build(
        matrix: &CsrMatrix<T>,
        relaxation: T,
        identity_rows: bool,
    ) -> Result<Self, LetoBackendError> {
        let zero = <T as NumericElement>::ZERO;
        let two = <T as NumericElement>::ONE + <T as NumericElement>::ONE;
        if !relaxation.is_finite() || relaxation <= zero || relaxation >= two {
            return Err(LetoBackendError::InvalidRelaxation);
        }
        let diagonal = if identity_rows {
            DiagonalIndex::new_optional(matrix)?
        } else {
            DiagonalIndex::new(matrix)?
        };
        let (dimension, _) = matrix.shape();
        for row in 0..dimension {
            if let Some(position) = diagonal.position(row)
                && matrix.values()[position] == zero
            {
                return Err(LetoBackendError::SingularDiagonal { index: row });
            }
        }
        Ok(Self {
            values: matrix.values().to_vec(),
            col_indices: matrix.col_indices().to_vec(),
            row_ptr: matrix.row_ptr().to_vec(),
            diagonal,
            inverse_relaxation: <T as NumericElement>::ONE / relaxation,
            dimension,
        })
    }

    /// Dimension of the preconditioned system.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

impl<T: RealScalar + RealField> Preconditioner<LetoBackend<T>> for SuccessiveOverRelaxation<T> {
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
            false,
            self.inverse_relaxation,
        )
    }
}
