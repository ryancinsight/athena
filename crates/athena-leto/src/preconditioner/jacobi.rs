use athena_core::{KrylovBackend, Preconditioner};
use eunomia::{NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

use crate::{LetoBackend, LetoBackendError};

/// Jacobi preconditioner storing the inverse matrix diagonal.
#[derive(Clone, Debug)]
pub struct Jacobi<T> {
    inverse_diagonal: Array1<T>,
}

impl<T: RealScalar + RealField> Jacobi<T> {
    /// Build the inverse diagonal from a square CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns a shape error for a rectangular matrix or
    /// [`LetoBackendError::SingularDiagonal`] for a zero diagonal entry.
    pub fn from_csr(matrix: &CsrMatrix<T>) -> Result<Self, LetoBackendError> {
        let (rows, columns) = matrix.shape();
        if rows != columns {
            return Err(LetoBackendError::NonSquareOperator { rows, columns });
        }
        let mut diagonal = matrix.diagonal();
        for (index, value) in diagonal.iter_mut().enumerate() {
            if *value == <T as NumericElement>::ZERO {
                return Err(LetoBackendError::SingularDiagonal { index });
            }
            *value = <T as NumericElement>::ONE / *value;
        }
        Ok(Self {
            inverse_diagonal: Array1::from_shape_vec([rows], diagonal)?,
        })
    }

    /// Borrow the inverse diagonal.
    #[must_use]
    pub const fn inverse_diagonal(&self) -> &Array1<T> {
        &self.inverse_diagonal
    }
}

impl<T: RealScalar + RealField> Preconditioner<LetoBackend<T>> for Jacobi<T> {
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
        let inverse = self
            .inverse_diagonal
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        if residual.len() != output.len() {
            return Err(LetoBackendError::LengthMismatch {
                left: residual.len(),
                right: output.len(),
            });
        }
        if residual.len() != inverse.len() {
            return Err(LetoBackendError::LengthMismatch {
                left: residual.len(),
                right: inverse.len(),
            });
        }
        for ((target, &source), &scaling) in
            output.iter_mut().zip(residual.iter()).zip(inverse.iter())
        {
            *target = scaling * source;
        }
        Ok(())
    }
}
