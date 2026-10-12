use core::marker::PhantomData;

use crate::{
    ConvergencePolicy, GmresWorkspace, IterationObserver, KrylovBackend, LinearOperator,
    NoObserver, Preconditioner, SolveError, SolveReport,
};

use super::cycle::Stage;
use super::execution::Execution;

/// Zero-sized restarted right-preconditioned GMRES algorithm marker.
///
/// The solver is the *flexible* variant: every preconditioned direction
/// `zⱼ = Mⱼ·vⱼ` is stored in its own basis, the Arnoldi orthogonalises
/// `A·zⱼ` against the Krylov basis alone, and the solution correction is
/// `Z·y` — so the preconditioner may return a *different* mapping on every
/// application (an inner iteration, or a per-step adaptive scheme) without
/// leaving the method's theory. A constant preconditioner is the special
/// case; the flexible property is pinned by the
/// `fgmres_contract` oracle at `f32` and `f64`.
///
/// # Reference
///
/// Saad, Y. (1993). *A flexible inner-outer preconditioned GMRES algorithm.*
/// SIAM Journal on Scientific Computing 14(2), 461–469.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Gmres<B, const RESTART: usize>(PhantomData<fn() -> B>);

impl<B: KrylovBackend, const RESTART: usize> Gmres<B, RESTART> {
    /// Solve `A x = b` into caller-owned `solution`.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure. Numerical breakdown and
    /// non-finite recurrence state are returned value-semantically in
    /// [`SolveReport`].
    pub fn solve_into<O, P>(
        backend: &B,
        operator: &O,
        preconditioner: &P,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut GmresWorkspace<B, RESTART>,
        policy: ConvergencePolicy<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: LinearOperator<B>,
        P: Preconditioner<B>,
    {
        Self::solve_with_observer(
            backend,
            operator,
            preconditioner,
            right_hand_side,
            solution,
            workspace,
            policy,
            &mut NoObserver,
        )
    }

    /// Solve while reporting checked residuals to `observer`.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "the solver boundary keeps backend, policies, caller-owned buffers, and observer explicit"
    )]
    pub fn solve_with_observer<O, P, Obs>(
        backend: &B,
        operator: &O,
        preconditioner: &P,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut GmresWorkspace<B, RESTART>,
        policy: ConvergencePolicy<B::Scalar>,
        observer: &mut Obs,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: LinearOperator<B>,
        P: Preconditioner<B>,
        Obs: IterationObserver<B::Scalar>,
    {
        let mut execution = Execution {
            backend,
            operator,
            preconditioner,
            right_hand_side,
            solution,
            workspace,
            policy,
            observer,
        };
        execution.validate_dimensions()?;
        let initial = execution.initialize()?;
        match initial {
            Stage::Continue(state) => execution.iterate(state),
            Stage::Complete(report) => Ok(report),
        }
    }
}
