use crate::KrylovBackend;

/// Reusable right-preconditioned `BiCGSTAB` vector workspace.
///
/// Construction performs all vector allocations and prepares the reductions
/// bound to those vectors, so the recurrence itself is allocation-free and
/// accelerator backends reuse dispatch resources across solves.
///
/// The recurrence carries seven vectors rather than the eight of the textbook
/// statement: the intermediate `s = r - alpha v` is formed in place in
/// `residual`, which then becomes `r = s - omega t` by a second update.
pub struct BiCgStabWorkspace<B: KrylovBackend> {
    /// Recursive residual `r`, and the intermediate `s` within a step.
    pub(super) residual: B::Vector,
    /// Fixed shadow residual `r̂₀`.
    pub(super) shadow: B::Vector,
    /// Search direction `p`.
    pub(super) direction: B::Vector,
    /// Preconditioned direction `M⁻¹p`.
    pub(super) preconditioned_direction: B::Vector,
    /// Preconditioned intermediate `M⁻¹s`.
    pub(super) preconditioned_residual: B::Vector,
    /// Operator image `v = A·M⁻¹p`.
    pub(super) image: B::Vector,
    /// Operator image `t = A·M⁻¹s`.
    pub(super) stabilizer: B::Vector,
    pub(super) residual_norm: B::PreparedNorm,
    pub(super) shadow_residual_dot: B::PreparedDot,
    pub(super) shadow_image_dot: B::PreparedDot,
    pub(super) stabilizer_residual_dot: B::PreparedDot,
    pub(super) stabilizer_norm_dot: B::PreparedDot,
    len: usize,
}

impl<B: KrylovBackend> BiCgStabWorkspace<B> {
    /// Allocate a workspace for `len` unknowns.
    ///
    /// # Errors
    ///
    /// Returns the first backend allocation or reduction-preparation failure.
    pub fn new(backend: &B, len: usize) -> Result<Self, B::Error> {
        let residual = backend.allocate(len)?;
        let shadow = backend.allocate(len)?;
        let direction = backend.allocate(len)?;
        let preconditioned_direction = backend.allocate(len)?;
        let preconditioned_residual = backend.allocate(len)?;
        let image = backend.allocate(len)?;
        let stabilizer = backend.allocate(len)?;
        let residual_norm = backend.prepare_norm_l2(backend.view(&residual))?;
        let shadow_residual_dot =
            backend.prepare_dot(backend.view(&shadow), backend.view(&residual))?;
        let shadow_image_dot = backend.prepare_dot(backend.view(&shadow), backend.view(&image))?;
        let stabilizer_residual_dot =
            backend.prepare_dot(backend.view(&stabilizer), backend.view(&residual))?;
        let stabilizer_norm_dot =
            backend.prepare_dot(backend.view(&stabilizer), backend.view(&stabilizer))?;
        Ok(Self {
            residual,
            shadow,
            direction,
            preconditioned_direction,
            preconditioned_residual,
            image,
            stabilizer,
            residual_norm,
            shadow_residual_dot,
            shadow_image_dot,
            stabilizer_residual_dot,
            stabilizer_norm_dot,
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
