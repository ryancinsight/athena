use crate::KrylovBackend;

/// Matrix-free rectangular operator supporting both `A x` and `Aᵀ y`.
///
/// Least-squares recurrences need the adjoint as well as the forward product,
/// and they need the two to have genuinely different lengths, so this is a
/// separate contract from [`super::LinearOperator`] rather than an extension of
/// it: a square operator does not necessarily expose a transpose, and requiring
/// one would burden every consumer that never needs it.
///
/// Implementations write into caller-owned output and must not retain either
/// borrowed view after returning.
pub trait RectangularOperator<B: KrylovBackend> {
    /// Row count: the length of `A x` and of the right-hand side.
    fn rows(&self) -> usize;

    /// Column count: the length of `x` and of `Aᵀ y`.
    fn columns(&self) -> usize;

    /// Apply `output = A · input`, with `input` of length [`Self::columns`] and
    /// `output` of length [`Self::rows`].
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

    /// Apply `output = Aᵀ · input`, with `input` of length [`Self::rows`] and
    /// `output` of length [`Self::columns`].
    ///
    /// # Errors
    ///
    /// Returns the backend-specific application failure.
    fn apply_transpose(
        &self,
        backend: &B,
        input: B::View<'_>,
        output: B::ViewMut<'_>,
    ) -> Result<(), B::Error>;
}
