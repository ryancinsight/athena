use crate::KrylovBackend;

/// Matrix-free linear operator `y = A x`.
///
/// Implementations write into caller-owned output and must not retain either
/// borrowed view after returning.
pub trait LinearOperator<B: KrylovBackend> {
    /// Square operator dimension.
    fn dimension(&self) -> usize;

    /// Apply the operator.
    ///
    /// # Errors
    ///
    /// Returns the backend-specific application failure.
    fn apply(
        &self,
        backend: &B,
        input: B::View<'_>,
        output: B::ViewMut<'_>,
    ) -> Result<(), B::Error>;
}
