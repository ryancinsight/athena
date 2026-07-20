/// Terminal solver condition.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// The initial guess already met the convergence policy.
    InitialResidual,
    /// An iterative residual met the convergence policy.
    Converged,
    /// The configured iteration budget was exhausted.
    MaxIterations,
    /// A Krylov denominator was numerically zero.
    Breakdown,
    /// CG encountered non-positive curvature, violating its SPD contract.
    NonPositiveCurvature,
    /// A scalar residual or recurrence coefficient became non-finite.
    NonFinite,
}

impl Termination {
    /// Return whether this condition represents convergence.
    #[must_use]
    pub const fn converged(self) -> bool {
        matches!(self, Self::InitialResidual | Self::Converged)
    }
}

/// Allocation-free solver outcome.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveReport<T> {
    /// Terminal condition.
    pub termination: Termination,
    /// Completed Krylov iterations.
    pub iterations: usize,
    /// Operator applications.
    pub operator_applications: usize,
    /// Preconditioner applications.
    pub preconditioner_applications: usize,
    /// Initial Euclidean residual norm.
    pub initial_residual_norm: T,
    /// Final checked Euclidean residual norm.
    pub final_residual_norm: T,
    /// Effective convergence threshold.
    pub threshold: T,
}

impl<T> SolveReport<T> {
    pub(crate) const fn new(
        termination: Termination,
        iterations: usize,
        operator_applications: usize,
        preconditioner_applications: usize,
        initial_residual_norm: T,
        final_residual_norm: T,
        threshold: T,
    ) -> Self {
        Self {
            termination,
            iterations,
            operator_applications,
            preconditioner_applications,
            initial_residual_norm,
            final_residual_norm,
            threshold,
        }
    }

    /// Return whether the solve converged.
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.termination.converged()
    }
}
