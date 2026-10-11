//! The chain's tier bodies: the ladder sequence members and the per-tier execution engine.

use super::direct::solve_direct_with_dense_retry;
use super::state::{CachedAmg, LinearSolverChain, LinearSolverState};
use athena_core::{
    BiCgStab, BiCgStabWorkspace, Identity, LinearOperator, Preconditioner, SolveReport, Termination,
};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};
use std::fmt::Debug;

use crate::{
    AMGConfig, AlgebraicMultigrid, BlockDiagonalPreconditioner, BorrowedCsrOperator,
    ComponentBlockPreconditioner, IncompleteLu, KrylovWorkspace, LetoBackend, LetoBackendError,
    SimplePreconditioner,
};

impl<T: RealField + Copy + FloatElement + RealScalar + Debug> LinearSolverChain<T> {
    /// The warm path's AMG tier with cached hierarchy reuse, gated on the
    /// exact CSR pattern; the cache is refreshed from the solve regardless
    /// of its outcome so the next warm solve reuses the freshest operators.
    pub(super) fn amg_cache_tier(
        &self,
        matrix: &CsrMatrix<T>,
        right_hand_side: &Array1<T>,
        solution: &mut Array1<T>,
        restart: usize,
        state: &mut LinearSolverState<T>,
        reset: impl Fn(&mut Array1<T>),
    ) -> bool {
        let cached_hierarchy = state
            .amg
            .take()
            .filter(|cached| cached.matches(matrix))
            .map(|cached| cached.hierarchy);
        let amg = match cached_hierarchy {
            Some(hierarchy) => {
                match AlgebraicMultigrid::with_hierarchy(matrix, AMGConfig::default(), hierarchy) {
                    Ok(amg) => {
                        tracing::debug!("LinearSolverChain(warm): reused AMG hierarchy");
                        Ok(amg)
                    }
                    Err(error) => {
                        tracing::warn!(
                            "LinearSolverChain(warm): cached AMG hierarchy failed ({error}); rebuilding"
                        );
                        AlgebraicMultigrid::new(matrix, AMGConfig::default())
                    }
                }
            }
            None => AlgebraicMultigrid::new(matrix, AMGConfig::default()),
        };
        match amg {
            Ok(amg) => {
                state.amg = Some(CachedAmg {
                    hierarchy: amg.get_hierarchy(),
                    row_ptr: matrix.row_ptr().to_vec(),
                    col_indices: matrix.col_indices().to_vec(),
                });
                self.tier(
                    "LinearSolverChain(warm): GMRES+AMG",
                    matrix,
                    right_hand_side,
                    solution,
                    restart,
                    Ok(amg),
                    reset,
                )
            }
            Err(error) => {
                tracing::debug!("LinearSolverChain(warm): AMG setup skipped/failed ({error})");
                false
            }
        }
    }

    /// The warm path's remaining tiers: SIMPLE, component-block (cached
    /// symbolic factors), block diagonal, plain, ILU — uniform through
    /// [`Self::tier`] with the warm iterate restored on every miss.
    #[expect(
        clippy::too_many_arguments,
        reason = "the warm tier boundary keeps system, buffers, block split, cache, and guess explicit"
    )]
    pub(super) fn warm_stateless_tiers(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &Array1<T>,
        x: &mut Array1<T>,
        n_velocity_dof: usize,
        n_pressure_dof: usize,
        restart: usize,
        state: &mut LinearSolverState<T>,
        initial_guess: Option<&Array1<T>>,
    ) -> bool {
        let reset = |x: &mut Array1<T>| {
            if let Some(guess) = initial_guess {
                for row in 0..guess.shape()[0] {
                    x[row] = guess[row];
                }
            } else {
                x.fill(<T as NumericElement>::ZERO);
            }
        };
        let simple = SimplePreconditioner::new(matrix, n_velocity_dof, n_pressure_dof);
        if self.tier(
            "LinearSolverChain(warm): GMRES+SIMPLE",
            matrix,
            rhs,
            x,
            restart,
            simple,
            reset,
        ) {
            return true;
        }
        let component = ComponentBlockPreconditioner::new_with_cache(
            matrix,
            n_velocity_dof,
            n_pressure_dof,
            &mut state.component,
        );
        if self.tier(
            "LinearSolverChain(warm): GMRES+ComponentBlock",
            matrix,
            rhs,
            x,
            restart,
            component,
            reset,
        ) {
            return true;
        }
        let block_built =
            BlockDiagonalPreconditioner::new(matrix, n_velocity_dof, n_pressure_dof).is_ok();
        if self.tier(
            "LinearSolverChain(warm): GMRES+BlockDiag",
            matrix,
            rhs,
            x,
            restart,
            BlockDiagonalPreconditioner::new(matrix, n_velocity_dof, n_pressure_dof),
            reset,
        ) {
            return true;
        }
        if !block_built
            && self.tier(
                "LinearSolverChain(warm): GMRES unpreconditioned",
                matrix,
                rhs,
                x,
                restart,
                Ok(Identity),
                reset,
            )
        {
            return true;
        }
        let ilu = IncompleteLu::from_csr(matrix);
        if self.tier(
            "LinearSolverChain(warm): GMRES+ILU",
            matrix,
            rhs,
            x,
            restart,
            ilu,
            reset,
        ) {
            return true;
        }
        false
    }

    /// The direct sparse-LU tier, exact for systems under the threshold.
    pub(super) fn direct_tier(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &Array1<T>,
        solution: &mut Array1<T>,
        context: &str,
    ) -> bool {
        if rhs.shape()[0] >= self.direct_threshold {
            return false;
        }
        match solve_direct_with_dense_retry(matrix, rhs, self.direct_threshold) {
            Ok(x_direct) => {
                tracing::debug!("{context}: direct LU succeeded (n={})", rhs.shape()[0]);
                *solution = x_direct;
                true
            }
            Err(error) => {
                tracing::warn!("{context}: direct LU failed ({error}); trying iterative");
                false
            }
        }
    }

    /// The terminal `BiCGSTAB` tier: unlike the tiers above this has nowhere to
    /// fall through to, so a non-convergence fails the chain.
    pub(super) fn last_resort(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &Array1<T>,
        x: &mut Array1<T>,
        context: &str,
    ) -> Result<(), LetoBackendError> {
        if self.bicgstab_tier(context, matrix, rhs, x).is_none() {
            return Err(LetoBackendError::Leto(leto::LetoError::InvalidInput(
                format!("{context}: all solver tiers failed"),
            )));
        }
        tracing::debug!("{context}: BiCGSTAB (last resort) converged");
        Ok(())
    }

    /// Run one preconditioned-GMRES tier: construct, solve, reset on miss.
    ///
    /// A setup failure or a non-converged solve logs the outcome, applies
    /// `reset`, and returns `Ok(false)` so the chain falls through — every
    /// fallback trigger is surfaced in the log, never silently skipped.
    #[expect(
        clippy::too_many_arguments,
        reason = "the tier boundary keeps the system, buffers, setup, and reset explicit"
    )]
    pub(super) fn tier<P>(
        &self,
        context: &str,
        matrix: &CsrMatrix<T>,
        right_hand_side: &Array1<T>,
        solution: &mut Array1<T>,
        restart: usize,
        setup: Result<P, LetoBackendError>,
        reset: impl Fn(&mut Array1<T>),
    ) -> bool
    where
        P: Preconditioner<LetoBackend<T>>,
    {
        let preconditioner = match setup {
            Ok(preconditioner) => preconditioner,
            Err(error) => {
                tracing::debug!("{context}: setup skipped ({error})");
                return false;
            }
        };
        if let Some(report) = self.gmres_tier(
            context,
            matrix,
            right_hand_side,
            &preconditioner,
            solution,
            restart,
        ) {
            tracing::debug!("{context} converged in {} iters", report.iterations);
            true
        } else {
            reset(solution);
            false
        }
    }

    /// Run one restarted-GMRES tier over the canonical workspace ladder.
    ///
    /// A tier that converges returns the report; a tier that stalls, breaks
    /// down, or hits an infrastructure failure logs the outcome and returns
    /// `Ok(None)` so the chain falls through to the next tier.
    pub(super) fn gmres_tier<P>(
        &self,
        context: &str,
        matrix: &CsrMatrix<T>,
        right_hand_side: &Array1<T>,
        preconditioner: &P,
        solution: &mut Array1<T>,
        restart: usize,
    ) -> Option<SolveReport<T>>
    where
        P: Preconditioner<LetoBackend<T>>,
    {
        let dimension = matrix.nrows();
        let mut workspace = match KrylovWorkspace::new(restart, dimension) {
            Ok(workspace) => workspace,
            Err(error) => {
                tracing::warn!("{context}: workspace allocation failed ({error})");
                return None;
            }
        };
        let operator = match BorrowedCsrOperator::new(matrix) {
            Ok(operator) => operator,
            Err(error) => {
                tracing::warn!("{context}: operator construction failed ({error})");
                return None;
            }
        };
        let report = match workspace.solve(
            &operator,
            preconditioner,
            right_hand_side,
            solution,
            self.policy,
        ) {
            Ok(report) => report,
            Err(error) => {
                tracing::warn!("{context}: {error}");
                return None;
            }
        };
        match report.termination {
            Termination::Converged
            | Termination::InitialResidual
            | Termination::NormalEquations => Some(report),
            other => {
                tracing::warn!(
                    iterations = report.iterations,
                    initial_residual = ?report.initial_residual_norm,
                    final_residual = ?report.final_residual_norm,
                    "{context}: did not converge ({other:?})"
                );
                None
            }
        }
    }

    /// Run one `BiCGSTAB` tier. Same contract as [`Self::gmres_tier`].
    pub(super) fn bicgstab_tier(
        &self,
        context: &str,
        matrix: &CsrMatrix<T>,
        right_hand_side: &Array1<T>,
        solution: &mut Array1<T>,
    ) -> Option<SolveReport<T>> {
        let backend = LetoBackend::<T>::default();
        let operator = match BorrowedCsrOperator::new(matrix) {
            Ok(operator) => operator,
            Err(error) => {
                tracing::warn!("{context}: operator construction failed ({error})");
                return None;
            }
        };
        let mut workspace =
            match BiCgStabWorkspace::new(&backend, LinearOperator::dimension(&operator)) {
                Ok(workspace) => workspace,
                Err(error) => {
                    tracing::warn!("{context}: workspace allocation failed ({error})");
                    return None;
                }
            };
        match BiCgStab::<LetoBackend<T>>::solve_into(
            &backend,
            &operator,
            &Identity,
            right_hand_side,
            solution,
            &mut workspace,
            self.policy,
        ) {
            Ok(report) => match report.termination {
                Termination::Converged
                | Termination::InitialResidual
                | Termination::NormalEquations => Some(report),
                other => {
                    tracing::warn!(
                        iterations = report.iterations,
                        "{context}: did not converge ({other:?})"
                    );
                    None
                }
            },
            Err(error) => {
                tracing::warn!("{context}: {error}");
                None
            }
        }
    }
}
