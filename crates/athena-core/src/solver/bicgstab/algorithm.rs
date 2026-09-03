use core::marker::PhantomData;

use crate::{
    BiCgStabWorkspace, ConvergencePolicy, IterationObserver, KrylovBackend, LinearOperator,
    NoObserver, Preconditioner, SolveError, SolveReport,
};

use super::execution::{Execution, Stage};

/// Zero-sized right-preconditioned `BiCGSTAB` algorithm marker.
///
/// Solves general nonsymmetric systems without the restart parameter and
/// `O(n·m)` basis storage GMRES requires, at the cost of a non-monotone
/// residual and two operator applications per iteration.
///
/// # Preconditioning side
///
/// The preconditioner enters on the right, `A·M⁻¹y = b` with `x = M⁻¹y`, so
/// the recurrence residual is the residual of the original system and the
/// convergence policy applies to it directly. Left preconditioning would
/// instead measure `‖M⁻¹(b − A·x)‖`, which differs from the true residual by
/// up to `κ(M)`.
///
/// # Reference
///
/// van der Vorst (1992). *Bi-CGSTAB: a fast and smoothly converging variant of
/// Bi-CG for the solution of nonsymmetric linear systems.* SIAM J. Sci. Stat.
/// Comput. 13(2), 631–644.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BiCgStab<B>(PhantomData<fn() -> B>);

impl<B: KrylovBackend> BiCgStab<B> {
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
        workspace: &mut BiCgStabWorkspace<B>,
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

    /// Solve while reporting configured residual checks to `observer`.
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
        workspace: &mut BiCgStabWorkspace<B>,
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
        match execution.initialize()? {
            Stage::Continue(state) => execution.iterate(state),
            Stage::Complete(report) => Ok(report),
        }
    }
}
