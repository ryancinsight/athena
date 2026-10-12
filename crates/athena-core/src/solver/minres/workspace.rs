use crate::KrylovBackend;

/// Reusable MINRES vector workspace.
///
/// Construction performs all vector allocations and prepares the reductions
/// bound to those buffers. Reusing the workspace keeps the recurrence
/// allocation-free: the Lanczos basis enters the solution through the
/// two-term update-vector recurrence, so seven length-`n` buffers are the
/// whole footprint — no `n`-by-`m` basis is ever stored.
pub struct MinresWorkspace<B: KrylovBackend> {
    /// Seed residual `r₀ = b − A·x₀`, normalised into the first Lanczos
    /// direction. The buffer is not needed after step 1; it is retained as
    /// the dedicated first direction because the update vectors `w₁`, `w₂`,
    /// `w₃` rotate through their roles and must not alias it.
    pub(super) seed_residual: B::Vector,
    /// Current Lanczos direction `vⱼ`.
    pub(super) direction: B::Vector,
    /// Previous Lanczos direction `vⱼ₋₁`; zero before the first step.
    pub(super) previous_direction: B::Vector,
    /// Operator image `A·vⱼ`, reused in place as the three-term Lanczos
    /// accumulator `w` and then as the next direction after normalisation.
    pub(super) image: B::Vector,
    /// Solution update vectors `wⱼ`, `wⱼ₋₁`, `wⱼ₋₂`.
    pub(super) update: B::Vector,
    pub(super) update_previous: B::Vector,
    pub(super) update_previous_2: B::Vector,
    pub(super) seed_norm: B::PreparedNorm,
    pub(super) image_norm: B::PreparedNorm,
    pub(super) direction_image_dot: B::PreparedDot,
    len: usize,
}

impl<B: KrylovBackend> MinresWorkspace<B> {
    /// Allocate a workspace for `len` unknowns.
    ///
    /// # Errors
    ///
    /// Returns the first backend allocation or reduction-preparation failure.
    pub fn new(backend: &B, len: usize) -> Result<Self, B::Error> {
        let seed_residual = backend.allocate(len)?;
        let direction = backend.allocate(len)?;
        let previous_direction = backend.allocate(len)?;
        let image = backend.allocate(len)?;
        let update = backend.allocate(len)?;
        let update_previous = backend.allocate(len)?;
        let update_previous_2 = backend.allocate(len)?;
        let seed_norm = backend.prepare_norm_l2(backend.view(&seed_residual))?;
        let image_norm = backend.prepare_norm_l2(backend.view(&image))?;
        let direction_image_dot =
            backend.prepare_dot(backend.view(&direction), backend.view(&image))?;
        Ok(Self {
            seed_residual,
            direction,
            previous_direction,
            image,
            update,
            update_previous,
            update_previous_2,
            seed_norm,
            image_norm,
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
