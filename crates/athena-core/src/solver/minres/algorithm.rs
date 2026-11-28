use core::marker::PhantomData;

use crate::{
    ConvergencePolicy, IterationObserver, KrylovBackend, LinearOperator, MinresWorkspace,
    NoObserver, SolveReport,
};

use super::execution::{Execution, Stage};
use crate::SolveError;

/// Zero-sized MINRES algorithm marker for symmetric (possibly indefinite)
/// operators.
///
/// MINRES minimises the residual norm over the Krylov space
/// `span{r₀, A·r₀, …, A^(j−1)·r₀}` built by the symmetric Lanczos
/// three-term recurrence, so it requires only `A = Aᵀ` — no positive
/// curvature, unlike [`crate::Cg`], which is why it is the method for
/// saddle-point and Helmholtz-type systems. The projected least-squares
/// problem is solved through an incremental Givens QR of the Lanczos
/// tridiagonal, and the solution enters through a two-term update-vector
/// recurrence: seven length-`n` buffers are the whole footprint, with no
/// stored basis.
///
/// # Preconditioning
///
/// Deliberately absent: a preconditioner that preserves MINRES' theory must
/// enter as a symmetric positive-definite factorisation `M = L·Lᵀ`, which is
/// a different seam from [`crate::Preconditioner`]'s apply. The
/// factorisation seam is deferred until a consumer needs it.
///
/// # Termination
///
/// The Lanczos residual estimate `|φ̄|` is exact in exact arithmetic (the
/// Givens rotations are norm-preserving) but drifts from `‖b − A x‖` in
/// floating point. Each step whose estimate meets the threshold therefore
/// recomputes `b − A x` with one extra operator application, and the report
/// carries that recomputed norm. [`crate::Termination::Converged`] means the
/// recomputed norm meets the threshold. When it does not, the run continues:
/// a later step can still meet the threshold, and a budget that ends first
/// reports [`crate::Termination::MaxIterations`] with the last recomputed
/// norm, as [`crate::Chebyshev`] does. The run reports
/// [`crate::Termination::Breakdown`] when the recurrence loses rank before the
/// policy is met — a zero next Lanczos coefficient, or a zero QR diagonal. A
/// zero Lanczos coefficient closes the Krylov space, so the run ends there
/// even when the recomputed norm is above the threshold. On a consistent
/// system the threshold then lies below the accuracy rounding leaves; a
/// singular space whose zero QR diagonal rounded to nonzero would end here
/// too (hypothesis, not witnessed on the host backend). The report says that
/// the recurrence ended, where a budget that ends first says `MaxIterations`.
///
/// Before accepting an initial or iterative convergence result, MINRES checks
/// every component of the candidate solution. A NaN or infinity changes the
/// termination to [`crate::Termination::NonFinite`]. The residual field
/// retains the value that triggered the check. Intermediate iterations
/// use the scalar recurrence, so a budget or breakdown exit can still
/// contain a non-finite iterate.
///
/// # Reference
///
/// Paige & Saunders (1975). *Solution of sparse indefinite systems of
/// linear equations and least squares problems.* SIAM Journal on Numerical
/// Analysis 12(4), 617–629.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Minres<B>(PhantomData<fn() -> B>);

impl<B: KrylovBackend> Minres<B> {
    /// Solve `A x = b` into caller-owned `solution`.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure. Numerical termination such as
    /// breakdown is returned value-semantically in [`SolveReport`].
    pub fn solve_into<O>(
        backend: &B,
        operator: &O,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut MinresWorkspace<B>,
        policy: ConvergencePolicy<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: LinearOperator<B>,
    {
        Self::solve_with_observer(
            backend,
            operator,
            right_hand_side,
            solution,
            workspace,
            policy,
            &mut NoObserver,
        )
    }

    /// Solve while reporting configured residual checks to `observer`.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure.
    pub fn solve_with_observer<O, Obs>(
        backend: &B,
        operator: &O,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut MinresWorkspace<B>,
        policy: ConvergencePolicy<B::Scalar>,
        observer: &mut Obs,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: LinearOperator<B>,
        Obs: IterationObserver<B::Scalar>,
    {
        let mut execution = Execution {
            backend,
            operator,
            right_hand_side,
            solution,
            workspace,
            policy,
            observer,
        };
        execution.validate_dimensions()?;
        let mut report = match execution.seed()? {
            Stage::Complete(report) => report,
            Stage::Continue(state) => execution.iterate(state)?,
        };
        if report.converged() && !execution.solution_is_finite()? {
            report.termination = crate::Termination::NonFinite;
        }
        Ok(report)
    }
}
