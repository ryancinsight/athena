use core::marker::PhantomData;

use crate::{
    ChebyshevWorkspace, ConvergencePolicy, KrylovBackend, LinearOperator, NoObserver, SolveError,
    SolveReport,
};

use super::bounds::SpectralBounds;
use super::execution::{Execution, Stage};

/// Zero-sized Chebyshev semi-iteration marker for symmetric positive-definite
/// operators.
///
/// Damped Richardson steps with Chebyshev-polynomial coefficients damp every
/// error component uniformly over the supplied spectral interval
/// `[λ_min, λ_max]`: after `k` steps the error in the operator's own norm is
/// at most `1/T_k(μ)` of its start, with
/// `μ = (λ_max + λ_min)/(λ_max − λ_min)` and `T_k` the Chebyshev polynomial —
/// the optimal uniform bound a `k`-term stationary recurrence can achieve
/// without inner products. Unlike CG the recurrence is inner-product-free,
/// which is the property that keeps it synchronisation-free on parallel
/// hardware.
///
/// The step recurrence is `s_k = τ_k·ω·r_k + (τ_k − 1)·s_{k−1}` with
/// `ω = 2/(λ_max + λ_min)`, `τ_1 = 1`, and `τ_{k+1}` tracked through the
/// normalised ratio recurrence `v_{k+1} = 2 − η²/v_k`, `τ_{k+1} = 2/v_{k+1}`,
/// where `η = 1/μ` and `v_1 = 1`. Its coefficients stay bounded without
/// forming either `T_k(μ)` or `μ`. For equal bounds, `η = 0`, `τ_k = 1`,
/// and `ω = 1/λ`: the singleton-spectrum Richardson limit solves `λ I`
/// in one step in exact arithmetic.
///
/// # Termination
///
/// The residual is recomputed every iteration, so the convergence check sees
/// the true value, not a recurrence estimate. The bounds' validation lives in
/// [`SpectralBounds::new`]; the solve assumes a checked interval.
///
/// # Reference
///
/// The polynomial construction is the classical Chebyshev acceleration
/// (Golub & Van Loan, *Matrix Computations*, §11.3): the error propagates as
/// `p_k(A)` with `p_k(t) = T_k(φ(t))/T_k(μ)`, `φ` the affine map of
/// `[λ_min, λ_max]` onto `[−1, 1]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Chebyshev<B>(PhantomData<fn() -> B>);

impl<B: KrylovBackend> Chebyshev<B> {
    /// Solve `A x = b` into caller-owned `solution`.
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
        workspace: &mut ChebyshevWorkspace<B>,
        policy: ConvergencePolicy<B::Scalar>,
        bounds: SpectralBounds<B::Scalar>,
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
            bounds,
            &mut NoObserver,
        )
    }

    /// Solve while reporting each residual check to `observer`.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "the solver boundary keeps backend, policy, spectral bounds, caller-owned buffers, and observer explicit"
    )]
    pub fn solve_with_observer<O, Obs>(
        backend: &B,
        operator: &O,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut ChebyshevWorkspace<B>,
        policy: ConvergencePolicy<B::Scalar>,
        bounds: SpectralBounds<B::Scalar>,
        observer: &mut Obs,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: LinearOperator<B>,
        Obs: crate::IterationObserver<B::Scalar>,
    {
        let mut execution = Execution {
            backend,
            operator,
            right_hand_side,
            solution,
            workspace,
            policy,
            bounds,
            observer,
        };
        execution.validate_dimensions()?;
        match execution.seed()? {
            Stage::Complete(report) => Ok(report),
            Stage::Continue(state) => execution.iterate(state),
        }
    }
}
