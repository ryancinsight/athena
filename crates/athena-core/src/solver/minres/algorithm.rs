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
/// The Lanczos residual estimate is exact in exact arithmetic (the Givens
/// rotations are norm-preserving), and the run reports [`crate::Termination::Breakdown`]
/// when the recurrence loses rank before the policy is met — a zero next
/// Lanczos coefficient, or a zero QR diagonal.
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
            solution,
            workspace,
            policy,
            observer,
        };
        execution.validate_dimensions(right_hand_side)?;
        let state = match execution.seed(right_hand_side)? {
            Stage::Complete(report) => return Ok(report),
            Stage::Continue(state) => state,
        };
        execution.iterate(state)
    }
}
