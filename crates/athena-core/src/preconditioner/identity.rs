use crate::{KrylovBackend, Preconditioner};

/// Zero-sized identity preconditioner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Identity;

impl<B: KrylovBackend> Preconditioner<B> for Identity {
    #[inline]
    fn apply(
        &self,
        backend: &B,
        residual: B::View<'_>,
        output: B::ViewMut<'_>,
    ) -> Result<(), B::Error> {
        backend.copy(residual, output)
    }
}
