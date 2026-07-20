use crate::KrylovBackend;

/// Preconditioner solve `z = M⁻¹r`.
///
/// Individual algorithms define whether this operation enters on the left or
/// right of the transformed operator.
pub trait Preconditioner<B: KrylovBackend> {
    /// Apply the preconditioner into caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns the backend-specific application failure.
    fn apply(
        &self,
        backend: &B,
        residual: B::View<'_>,
        output: B::ViewMut<'_>,
    ) -> Result<(), B::Error>;
}
