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
    /// The normal-equation residual met the tolerance. An inconsistent
    /// least-squares system keeps a residual bounded away from zero, so this
    /// is the only criterion that can report its optimum.
    NormalEquations,
    /// A restart cycle stopped reducing the residual by more than the
    /// accuracy of the residual's own evaluation.
    ///
    /// Restarted GMRES(m) minimises over a subspace that contains the zero
    /// correction, so its residual cannot grow in exact arithmetic and a
    /// cycle that extracts no reduction seeds the next cycle with the same
    /// residual and reproduces the same non-reduction. Continuing therefore
    /// spends the remaining budget without approaching the solution. The
    /// scale of "no reduction" is
    /// [`residual_noise_floor`](crate::residual_noise_floor), not a tuned
    /// tolerance. Saad and Schultz (1986), *GMRES: A generalized minimal
    /// residual algorithm for solving nonsymmetric linear systems*, SIAM J.
    /// Sci. Stat. Comput. 7(3), 856-869, §3.2 records this stall for the
    /// restarted method.
    Stagnated,
    /// The residual exceeded its initial value by more than the accuracy of
    /// the residual's own evaluation.
    ///
    /// The same minimisation property bounds every cycle's residual by the
    /// initial one, and each cycle re-forms `b - Ax` explicitly rather than
    /// by recurrence, so the evaluation error does not accumulate across
    /// cycles. Exceeding the initial residual by more than one evaluation's
    /// worth of noise therefore means the recurrence has lost the property,
    /// and the iterate is worse than the initial guess.
    Diverged,
}

impl Termination {
    /// Return whether this condition represents convergence.
    #[must_use]
    pub const fn converged(self) -> bool {
        matches!(
            self,
            Self::InitialResidual | Self::Converged | Self::NormalEquations
        )
    }
}

/// Allocation-free solver outcome.
///
/// A solve that exhausted its budget, stagnated, or broke down still returns
/// this report rather than an error: which terminal condition was reached is
/// domain information the caller branches on, not a contract failure of the
/// call. Discarding the report is the way that distinction gets lost, so the
/// type is `#[must_use]` — an ignored report is an unexamined
/// [`Termination`], which is exactly the silently accepted partial result.
#[must_use]
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
