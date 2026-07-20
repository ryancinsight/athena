use athena_core::{KrylovBackend, LinearOperator};
use eunomia::RealField;
use leto::{Array, CowStorage, Layout};
use leto_ops::{RealScalar, matvec};

use crate::{LetoBackend, LetoBackendError};

/// Dense square operator that borrows row-major coefficients without copying.
///
/// Leto's [`CowStorage`] records the ownership boundary. Applying the operator
/// only reads the borrowed coefficients, so construction and every application
/// remain zero-copy.
pub struct BorrowedDenseOperator<'a, T: Clone> {
    matrix: Array<T, CowStorage<'a, T>, 2>,
    dimension: usize,
}

impl<'a, T: RealScalar + RealField> BorrowedDenseOperator<'a, T> {
    /// Borrow `dimension × dimension` row-major coefficients.
    ///
    /// # Errors
    ///
    /// Returns a Leto layout error when the element count does not match the
    /// requested square shape.
    pub fn new(dimension: usize, coefficients: &'a [T]) -> Result<Self, LetoBackendError> {
        let layout = Layout::c_contiguous([dimension, dimension])?;
        let matrix = Array::new(layout, CowStorage::borrowed(coefficients))?;
        Ok(Self { matrix, dimension })
    }

    /// Return whether the matrix still borrows its caller-owned coefficients.
    #[must_use]
    pub const fn is_borrowed(&self) -> bool {
        self.matrix.storage().is_borrowed()
    }
}

impl<T: RealScalar + RealField> LinearOperator<LetoBackend<T>> for BorrowedDenseOperator<'_, T> {
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
        matvec(&self.matrix.view(), &input, &mut output).map_err(Into::into)
    }
}
