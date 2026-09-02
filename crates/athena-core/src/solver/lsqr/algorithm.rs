use core::marker::PhantomData;

use eunomia::NumericElement;

use crate::{
    ConvergencePolicy, IterationObserver, KrylovBackend, LsqrWorkspace, NoObserver,
    RectangularOperator, SolveError, SolveReport,
};

use super::execution::Execution;

/// Zero-sized LSQR algorithm marker.
///
/// Solves `min ‖A·x − b‖₂` for a rectangular operator by Golub-Kahan
/// bidiagonalisation, which is analytically equivalent to conjugate gradients
/// on the normal equations while never forming `AᵀA`. Forming that product
/// squares the condition number; the bidiagonalisation does not, which is the
/// whole reason to prefer LSQR over CG-on-normal-equations.
///
/// # Termination
///
/// A consistent system is detected on the residual itself. An inconsistent
/// one — the genuine least-squares case — has a residual that never reaches
/// zero, so the criterion there is the normal-equation residual `‖Aᵀr‖`
/// relative to `‖r‖`, reported as [`crate::Termination::NormalEquations`]. Testing
/// only the residual would run such a solve to the iteration cap despite it
/// having found the exact minimiser.
///
/// # Reference
///
/// Paige & Saunders (1982). *LSQR: An algorithm for sparse linear equations
/// and sparse least squares.* ACM TOMS 8(1), 43–71.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Lsqr<B>(PhantomData<fn() -> B>);

impl<B: KrylovBackend> Lsqr<B> {
    /// Solve `min ‖A·x − b‖₂` into caller-owned `solution` with no Tikhonov
    /// damping.
    ///
    /// Equivalent to `solve_damped_into` with `damping = 0`; the parameterless
    /// form is the common case and stays ergonomic at the call site.
    ///
    /// `solution` is both the initial guess and the output; the recurrence
    /// solves for a correction to it, so a warm start is honoured.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure. Numerical termination is
    /// returned value-semantically in [`SolveReport`].
    pub fn solve_into<O>(
        backend: &B,
        operator: &O,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut LsqrWorkspace<B>,
        policy: ConvergencePolicy<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: RectangularOperator<B>,
    {
        Self::solve_damped_with_observer(
            backend,
            operator,
            right_hand_side,
            solution,
            workspace,
            policy,
            <B::Scalar as NumericElement>::ZERO,
            &mut NoObserver,
        )
    }

    /// Solve while reporting configured residual checks to `observer`, with no
    /// Tikhonov damping.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure.
    pub fn solve_with_observer<O, Obs>(
        backend: &B,
        operator: &O,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut LsqrWorkspace<B>,
        policy: ConvergencePolicy<B::Scalar>,
        observer: &mut Obs,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: RectangularOperator<B>,
        Obs: IterationObserver<B::Scalar>,
    {
        Self::solve_damped_with_observer(
            backend,
            operator,
            right_hand_side,
            solution,
            workspace,
            policy,
            <B::Scalar as NumericElement>::ZERO,
            observer,
        )
    }

    /// Solve `min ‖A·x − b‖₂ + λ·‖x‖₂` (Tikhonov-regularised least squares)
    /// into caller-owned `solution`.
    ///
    /// `λ ≥ 0` is the regularisation weight. `λ = 0` recovers the unregularised
    /// least-squares problem; positive `λ` stabilises the iterate against
    /// measurement noise and small singular values, at the cost of a bias
    /// toward zero. The damped problem is solved exactly as
    /// `min ‖ [A; λI]·x − [b; 0] ‖₂`, so the recurrence is the same
    /// bidiagonalisation with one extra `λ²` term in the diagonal update at
    /// every step (Paige & Saunders 1982, §4, eqn 4.4).
    ///
    /// # Panics
    ///
    /// `damping` must be finite and non-negative; callers are expected to
    /// validate. The algorithm does not panic on these — it propagates a
    /// [`crate::Termination::NonFinite`] or runs as if `λ = 0` — but the test suite
    /// asserts the input discipline.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure. Numerical termination is
    /// returned value-semantically in [`SolveReport`].
    pub fn solve_damped_into<O>(
        backend: &B,
        operator: &O,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut LsqrWorkspace<B>,
        policy: ConvergencePolicy<B::Scalar>,
        damping: B::Scalar,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: RectangularOperator<B>,
    {
        Self::solve_damped_with_observer(
            backend,
            operator,
            right_hand_side,
            solution,
            workspace,
            policy,
            damping,
            &mut NoObserver,
        )
    }

    /// Solve the damped least-squares problem while reporting configured
    /// residual checks to `observer`.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "the solver boundary keeps backend, caller-owned buffers, policy, damping, and observer explicit"
    )]
    pub fn solve_damped_with_observer<O, Obs>(
        backend: &B,
        operator: &O,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut LsqrWorkspace<B>,
        policy: ConvergencePolicy<B::Scalar>,
        damping: B::Scalar,
        observer: &mut Obs,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: RectangularOperator<B>,
        Obs: IterationObserver<B::Scalar>,
    {
        Execution {
            backend,
            operator,
            right_hand_side,
            solution,
            workspace,
            policy,
            observer,
            damping,
        }
        .run()
    }
}
