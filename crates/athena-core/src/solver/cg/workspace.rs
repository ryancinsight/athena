use crate::KrylovBackend;

/// Reusable preconditioned-CG vector workspace.
///
/// Construction performs all vector allocations. Reusing the workspace keeps
/// Athena's solver recurrence allocation-free; individual backends remain
/// responsible for any provider-local dispatch scratch.
pub struct CgWorkspace<B: KrylovBackend> {
    pub(super) residual: B::Vector,
    pub(super) preconditioned_residual: B::Vector,
    pub(super) direction: B::Vector,
    pub(super) image: B::Vector,
    len: usize,
}

impl<B: KrylovBackend> CgWorkspace<B> {
    /// Allocate a workspace for `len` unknowns.
    ///
    /// # Errors
    ///
    /// Returns the first backend allocation failure.
    pub fn new(backend: &B, len: usize) -> Result<Self, B::Error> {
        Ok(Self {
            residual: backend.allocate(len)?,
            preconditioned_residual: backend.allocate(len)?,
            direction: backend.allocate(len)?,
            image: backend.allocate(len)?,
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
