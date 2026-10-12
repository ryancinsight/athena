use crate::KrylovBackend;

/// Reusable Chebyshev vector workspace: three length-`n` buffers are the
/// whole footprint.
///
/// The recurrence is three-term in the step and needs one residual, one
/// step, and one operator image; no basis is stored and no reductions
/// beyond the residual norm are required.
pub struct ChebyshevWorkspace<B: KrylovBackend> {
    pub(super) residual: B::Vector,
    pub(super) step: B::Vector,
    pub(super) image: B::Vector,
    pub(super) residual_norm: B::PreparedNorm,
    len: usize,
}

impl<B: KrylovBackend> ChebyshevWorkspace<B> {
    /// Allocate a workspace for `len` unknowns.
    ///
    /// # Errors
    ///
    /// Returns the first backend allocation or reduction-preparation failure.
    pub fn new(backend: &B, len: usize) -> Result<Self, B::Error> {
        let residual = backend.allocate(len)?;
        let step = backend.allocate(len)?;
        let image = backend.allocate(len)?;
        let residual_norm = backend.prepare_norm_l2(backend.view(&residual))?;
        Ok(Self {
            residual,
            step,
            image,
            residual_norm,
            len,
        })
    }

    /// Workspace vector length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Return whether this is a zero-length workspace.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}
