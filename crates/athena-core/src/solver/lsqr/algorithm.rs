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

    /// Solve `min ‖A·x − b‖₂² + λ²·‖x − x₀‖₂²` (Tikhonov-regularised least
    /// squares) into caller-owned `solution`, where `x₀` is the initial
    /// content of `solution`.
    ///
    /// `λ ≥ 0` is the regularisation weight. `λ = 0` recovers the unregularised
    /// least-squares problem; positive `λ` stabilises the iterate against
    /// measurement noise and small singular values, at the cost of a bias
    /// toward `x₀`. The minimiser satisfies `(AᵀA + λ²I)·(x − x₀) = Aᵀ(b − A·x₀)`.
    ///
    /// The damped problem is the ordinary least-squares problem for the
    /// augmented system `[A; λI]·x ≈ [b; 0]` (Paige & Saunders 1982, §2.1,
    /// eq. 2.4). Every step of the bidiagonal recurrence applies two plane
    /// rotations to the projected problem `[B_k; λI]·y ≈ [β₁e₁; 0]` (eq.
    /// 2.7). The first eliminates the `λ` row against `ρ̄`:
    /// `ρ̂ = √(ρ̄² + λ²)`, `c₁ = ρ̄/ρ̂`, `s₁ = λ/ρ̂`, `ψ = s₁φ̄`, `φ̄ ← c₁φ̄`.
    /// The second eliminates the subdiagonal `β`: `ρ = √(ρ̂² + β²)`,
    /// `c = ρ̂/ρ`, `s = β/ρ`, `θ = sα`, `ρ̄ ← −cα`, `φ = cφ̄`, `φ̄ ← sφ̄`. The
    /// recurrence is the one in `SciPy` 1.18.0
    /// `scipy/sparse/linalg/_isolve/lsqr.py:443-463`.
    ///
    /// For `λ > 0` the reported residual norms are those of the augmented
    /// system, `‖[b − A·x; λ·(x − x₀)]‖₂ = √(φ̄² + Σψ²)`, not `‖b − A·x‖₂`.
    ///
    /// A non-finite `damping` ends the solve with
    /// [`crate::Termination::Breakdown`]; a negative one acts as its absolute
    /// value because only `λ²` enters the minimiser.
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
