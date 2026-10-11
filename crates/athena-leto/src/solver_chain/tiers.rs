//! The chain's tier methods: the two solve sequences and their per-tier
//! helpers over the canonical workspace ladder.

use super::direct::solve_direct_with_dense_retry;
use super::state::{CachedAmg, LinearSolverChain, LinearSolverState};
use athena_core::{
    BiCgStab, BiCgStabWorkspace, ConvergencePolicy, Identity, LinearOperator, Preconditioner,
    SolveReport, Termination,
};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};
use std::fmt::Debug;

use crate::{
    AMGConfig, AlgebraicMultigrid, BlockDiagonalPreconditioner, BorrowedCsrOperator,
    ComponentBlockPreconditioner, IncompleteLu, KrylovWorkspace, LetoBackend, LetoBackendError,
    SimplePreconditioner, SuccessiveOverRelaxation,
};

impl<T: RealField + Copy + FloatElement + RealScalar + Debug> LinearSolverChain<T> {
    /// Create a new solver chain with the given convergence policy.
    ///
    /// Defaults: `direct_threshold` = 2,048; `krylov_restart` = 100.
    #[must_use]
    pub fn new(policy: ConvergencePolicy<T>) -> Self {
        Self {
            policy,
            direct_threshold: 2_048,
            krylov_restart: 100,
        }
    }

    /// Override the DOF count threshold below which direct LU is used.
    #[must_use]
    pub const fn with_direct_threshold(mut self, threshold: usize) -> Self {
        self.direct_threshold = threshold;
        self
    }

    /// Override the GMRES restart parameter.
    #[must_use]
    pub const fn with_krylov_restart(mut self, restart: usize) -> Self {
        self.krylov_restart = restart;
        self
    }

    /// Solve `A·x = b` from a zero iterate using the tiered fallback.
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::Leto`] naming the chain when every tier
    /// fails; each tier's own outcome is surfaced in the log.
    pub fn solve(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &Array1<T>,
        n_velocity_dof: usize,
    ) -> Result<Array1<T>, LetoBackendError> {
        let n_total_dof = rhs.shape()[0];
        let n_pressure_dof = n_total_dof.saturating_sub(n_velocity_dof);
        let mut x = Array1::zeros([n_total_dof]);

        if self.direct_tier(matrix, rhs, &mut x, "LinearSolverChain") {
            return Ok(x);
        }

        let restart = std::cmp::min(self.krylov_restart, n_total_dof.max(1));
        let zero = |x: &mut Array1<T>| x.fill(<T as NumericElement>::ZERO);

        let component = ComponentBlockPreconditioner::new(matrix, n_velocity_dof, n_pressure_dof);
        if self.tier(
            "LinearSolverChain: GMRES+ComponentBlock",
            matrix,
            rhs,
            &mut x,
            restart,
            component,
            zero,
        ) {
            return Ok(x);
        }
        let simple = SimplePreconditioner::new(matrix, n_velocity_dof, n_pressure_dof);
        if self.tier(
            "LinearSolverChain: GMRES+SIMPLE",
            matrix,
            rhs,
            &mut x,
            restart,
            simple,
            zero,
        ) {
            return Ok(x);
        }
        let amg = AlgebraicMultigrid::new(matrix, AMGConfig::default());
        if self.tier(
            "LinearSolverChain: GMRES+AMG",
            matrix,
            rhs,
            &mut x,
            restart,
            amg,
            zero,
        ) {
            return Ok(x);
        }
        let block_built =
            BlockDiagonalPreconditioner::new(matrix, n_velocity_dof, n_pressure_dof).is_ok();
        if self.tier(
            "LinearSolverChain: GMRES+BlockDiag",
            matrix,
            rhs,
            &mut x,
            restart,
            BlockDiagonalPreconditioner::new(matrix, n_velocity_dof, n_pressure_dof),
            zero,
        ) {
            return Ok(x);
        }
        // Skip the unpreconditioned tier when a block preconditioner was
        // built but stagnated — for saddle-point systems the plain solve is
        // strictly worse.
        if !block_built
            && self.tier(
                "LinearSolverChain: GMRES unpreconditioned",
                matrix,
                rhs,
                &mut x,
                restart,
                Ok(Identity),
                zero,
            )
        {
            return Ok(x);
        }
        let ilu = IncompleteLu::from_csr(matrix);
        if self.tier(
            "LinearSolverChain: GMRES+ILU",
            matrix,
            rhs,
            &mut x,
            restart,
            ilu,
            zero,
        ) {
            return Ok(x);
        }
        self.last_resort(matrix, rhs, &mut x, "LinearSolverChain")?;
        Ok(x)
    }

    /// Solve with a warm start and reusable AMG/component state.
    ///
    /// `state` may be retained across Picard or continuation solves. AMG
    /// transfer operators are reused only when the complete CSR sparsity
    /// pattern matches; changing mesh connectivity rebuilds the hierarchy.
    ///
    /// # Errors
    ///
    /// See [`Self::solve`].
    pub fn solve_with_guess_state(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &Array1<T>,
        n_velocity_dof: usize,
        initial_guess: Option<&Array1<T>>,
        state: &mut LinearSolverState<T>,
    ) -> Result<Array1<T>, LetoBackendError> {
        let n_total_dof = rhs.shape()[0];
        let n_pressure_dof = n_total_dof.saturating_sub(n_velocity_dof);
        let mut x = initial_guess
            .cloned()
            .unwrap_or_else(|| Array1::zeros([n_total_dof]));

        if self.direct_tier(matrix, rhs, &mut x, "LinearSolverChain(warm)") {
            return Ok(x);
        }

        let restart = std::cmp::min(self.krylov_restart, n_total_dof.max(1));
        let reset = |x: &mut Array1<T>| match initial_guess {
            Some(guess) => x
                .iter_mut()
                .zip(guess.iter())
                .for_each(|(target, &source)| *target = source),
            None => x.fill(<T as NumericElement>::ZERO),
        };

        let sor = SuccessiveOverRelaxation::from_csr_with_identity_rows(
            matrix,
            <T as FloatElement>::from_f64(1.0),
        );
        if self.tier(
            "LinearSolverChain(warm): GMRES+SOR",
            matrix,
            rhs,
            &mut x,
            restart,
            sor,
            reset,
        ) {
            return Ok(x);
        }

        if self.amg_cache_tier(matrix, rhs, &mut x, restart, state, reset) {
            return Ok(x);
        }

        if self.warm_stateless_tiers(
            matrix,
            rhs,
            &mut x,
            n_velocity_dof,
            n_pressure_dof,
            restart,
            state,
            initial_guess,
        ) {
            return Ok(x);
        }
        self.last_resort(matrix, rhs, &mut x, "LinearSolverChain(warm)")?;
        Ok(x)
    }

    /// The warm path's AMG tier with cached hierarchy reuse, gated on the
    /// exact CSR pattern; the cache is refreshed from the solve regardless
    /// of its outcome so the next warm solve reuses the freshest operators.
    fn amg_cache_tier(
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
    fn warm_stateless_tiers(
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
    fn direct_tier(
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
    fn last_resort(
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
    fn tier<P>(
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
    fn gmres_tier<P>(
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
    fn bicgstab_tier(
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
