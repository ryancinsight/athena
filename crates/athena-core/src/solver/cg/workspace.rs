use crate::KrylovBackend;

/// Reusable preconditioned-CG vector workspace.
///
/// Construction performs all vector allocations and prepares the reductions
/// bound to those vectors. Reusing the workspace keeps Athena's recurrence
/// allocation-free and lets accelerator backends reuse dispatch resources.
pub struct CgWorkspace<B: KrylovBackend> {
    pub(super) residual: B::Vector,
    pub(super) preconditioned_residual: B::Vector,
    pub(super) direction: B::Vector,
    pub(super) image: B::Vector,
    pub(super) residual_norm: B::PreparedNorm,
    pub(super) residual_preconditioned_dot: B::PreparedDot,
    pub(super) direction_image_dot: B::PreparedDot,
    len: usize,
}

impl<B: KrylovBackend> CgWorkspace<B> {
    /// Allocate a workspace for `len` unknowns.
    ///
    /// # Errors
    ///
    /// Returns the first backend allocation or reduction-preparation failure.
    pub fn new(backend: &B, len: usize) -> Result<Self, B::Error> {
        let residual = backend.allocate(len)?;
        let preconditioned_residual = backend.allocate(len)?;
        let direction = backend.allocate(len)?;
        let image = backend.allocate(len)?;
        let residual_norm = backend.prepare_norm_l2(backend.view(&residual))?;
        let residual_preconditioned_dot = backend.prepare_dot(
            backend.view(&residual),
            backend.view(&preconditioned_residual),
        )?;
        let direction_image_dot =
            backend.prepare_dot(backend.view(&direction), backend.view(&image))?;
        Ok(Self {
            residual,
            preconditioned_residual,
            direction,
            image,
            residual_norm,
            residual_preconditioned_dot,
            direction_image_dot,
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
