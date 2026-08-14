use core::marker::PhantomData;
use eunomia::NumericElement;

use crate::{
    ConvergencePolicy, GmresWorkspace, IterationObserver, IterationState, KrylovBackend,
    LinearOperator, NoObserver, Preconditioner, SolveError, SolveReport, Termination,
    residual_noise_floor,
};

use super::{
    super::dimension::validate_dimension,
    cycle::{ArnoldiOutcome, CycleOutcome, CycleProgress, GmresState, Stage},
    rotation::{ScalarFailure, back_substitute, givens},
};

type ExecutionResult<B, T> = Result<T, SolveError<<B as KrylovBackend>::Error>>;

/// Zero-sized restarted right-preconditioned GMRES algorithm marker.
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

struct Execution<'a, B, O, P, Obs, const RESTART: usize>
where
    B: KrylovBackend,
{
    backend: &'a B,
    operator: &'a O,
    preconditioner: &'a P,
    right_hand_side: &'a B::Vector,
    solution: &'a mut B::Vector,
    workspace: &'a mut GmresWorkspace<B, RESTART>,
    policy: ConvergencePolicy<B::Scalar>,
    observer: &'a mut Obs,
}

impl<B, O, P, Obs, const RESTART: usize> Execution<'_, B, O, P, Obs, RESTART>
where
    B: KrylovBackend,
    O: LinearOperator<B>,
    P: Preconditioner<B>,
    Obs: IterationObserver<B::Scalar>,
{
    fn validate_dimensions(&self) -> Result<(), SolveError<B::Error>> {
        let dimension = self.operator.dimension();
        validate_dimension(
            "right-hand side",
            dimension,
            self.backend.vector_len(self.right_hand_side),
        )?;
        validate_dimension(
            "solution",
            dimension,
            self.backend.vector_len(self.solution),
        )?;
        validate_dimension("GMRES workspace", dimension, self.workspace.len())
    }

    fn initialize(&mut self) -> ExecutionResult<B, Stage<GmresState<B::Scalar>, B::Scalar>> {
        self.backend
            .copy(
                self.backend.view(self.right_hand_side),
                self.backend.view_mut(&mut self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        let right_hand_side_norm = self
            .backend
            .norm_l2_prepared(
                &self.workspace.residual_norm,
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        let threshold = self.policy.threshold(right_hand_side_norm);
        let initial_residual = self.recompute_residual()?;
        let termination = if !right_hand_side_norm.is_finite()
            || !threshold.is_finite()
            || !initial_residual.is_finite()
        {
            Some(Termination::NonFinite)
        } else if initial_residual <= threshold {
            Some(Termination::InitialResidual)
        } else {
            None
        };
        if let Some(termination) = termination {
            return Ok(Stage::Complete(SolveReport::new(
                termination,
                0,
                1,
                0,
                initial_residual,
                initial_residual,
                threshold,
            )));
        }

        Ok(Stage::Continue(GmresState {
            initial_residual,
            last_residual: initial_residual,
            threshold,
            residual_noise: residual_noise_floor(self.workspace.len(), right_hand_side_norm),
            iterations: 0,
            operator_applications: 1,
            preconditioner_applications: 0,
        }))
    }

    fn iterate(
        &mut self,
        mut state: GmresState<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>> {
        while state.iterations < self.policy.max_iterations() {
            let cycle_entry_residual = state.last_residual;
            self.prepare_cycle(state.last_residual)?;
            let remaining = self.policy.max_iterations() - state.iterations;
            let cycle_limit = core::cmp::min(RESTART, remaining);
            let cycle = match self.run_cycle(cycle_limit, &mut state)? {
                CycleProgress::Ready(cycle) => cycle,
                CycleProgress::Terminated(termination) => {
                    return Ok(state.report(termination));
                }
            };

            if let Some(failure) = self.update_solution(cycle.vectors_used)? {
                return Ok(state.report(failure.termination()));
            }
            state.last_residual = self.recompute_residual()?;
            state.operator_applications += 1;
            if !self.policy.should_check(state.iterations)
                && (state.last_residual <= state.threshold || cycle.happy_breakdown)
            {
                self.observer.observe(IterationState::new(
                    state.iterations,
                    state.last_residual,
                    state.threshold,
                ));
            }

            if !state.last_residual.is_finite() {
                return Ok(state.report(Termination::NonFinite));
            }
            if state.last_residual <= state.threshold {
                return Ok(state.report(Termination::Converged));
            }
            if cycle.happy_breakdown {
                return Ok(state.report(Termination::Breakdown));
            }
            if let Some(termination) = Self::progress_failure(&state, cycle_entry_residual) {
                return Ok(state.report(termination));
            }
        }
        Ok(state.report(Termination::MaxIterations))
    }

    /// Classify a completed, unconverged restart cycle by the progress it made.
    ///
    /// Runs after the convergence, breakdown, and non-finite tests, so it sees
    /// only cycles that ended with work still to do. Both comparisons are
    /// against [`GmresState::residual_noise`], the accuracy of one recomputed
    /// residual: a difference smaller than that is a property of the
    /// evaluation rather than of the iteration, and no threshold here is
    /// tuned.
    ///
    /// The cycle, not the iteration, is the unit. Restarted GMRES only re-forms
    /// the true residual at a restart, and the residual is monotone
    /// non-increasing within a cycle by construction, so an inner iteration
    /// carries no independent progress signal.
    fn progress_failure(
        state: &GmresState<B::Scalar>,
        cycle_entry_residual: B::Scalar,
    ) -> Option<Termination> {
        if state.last_residual > state.initial_residual + state.residual_noise {
            return Some(Termination::Diverged);
        }
        if cycle_entry_residual - state.last_residual <= state.residual_noise {
            return Some(Termination::Stagnated);
        }
        None
    }

    fn prepare_cycle(&mut self, residual_norm: B::Scalar) -> Result<(), SolveError<B::Error>> {
        self.workspace.reset_cycle();
        self.workspace.transformed_residual[0] = residual_norm;
        self.backend
            .copy(
                self.backend.view(&self.workspace.residual),
                self.backend.block_view_mut(&mut self.workspace.basis, 0),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .scale(
                self.backend.block_view_mut(&mut self.workspace.basis, 0),
                B::Scalar::ONE / residual_norm,
            )
            .map_err(SolveError::Backend)
    }

    fn run_cycle(
        &mut self,
        cycle_limit: usize,
        state: &mut GmresState<B::Scalar>,
    ) -> ExecutionResult<B, CycleProgress> {
        let mut outcome = CycleOutcome {
            vectors_used: 0,
            happy_breakdown: false,
        };
        for column in 0..cycle_limit {
            let happy_breakdown = match self.arnoldi_step(column, state)? {
                ArnoldiOutcome::Ready { happy_breakdown } => happy_breakdown,
                ArnoldiOutcome::Failed(failure) => {
                    return Ok(CycleProgress::Terminated(failure.termination()));
                }
            };
            state.iterations += 1;
            outcome.vectors_used = column + 1;
            outcome.happy_breakdown = happy_breakdown;
            let estimate = self.workspace.transformed_residual[column + 1].abs();
            if !estimate.is_finite() {
                return Ok(CycleProgress::Terminated(Termination::NonFinite));
            }
            if self.policy.should_check(state.iterations) {
                self.observer.observe(IterationState::new(
                    state.iterations,
                    estimate,
                    state.threshold,
                ));
            }
            if estimate <= state.threshold
                || happy_breakdown
                || state.iterations == self.policy.max_iterations()
            {
                break;
            }
        }
        Ok(CycleProgress::Ready(outcome))
    }

    fn arnoldi_step(
        &mut self,
        column: usize,
        state: &mut GmresState<B::Scalar>,
    ) -> ExecutionResult<B, ArnoldiOutcome> {
        self.preconditioner
            .apply(
                self.backend,
                self.backend.block_view(&self.workspace.basis, column),
                self.backend
                    .block_view_mut(&mut self.workspace.preconditioned_basis, column),
            )
            .map_err(SolveError::Backend)?;
        state.preconditioner_applications += 1;
        self.operator
            .apply(
                self.backend,
                self.backend
                    .block_view(&self.workspace.preconditioned_basis, column),
                self.backend.view_mut(&mut self.workspace.work),
            )
            .map_err(SolveError::Backend)?;
        state.operator_applications += 1;

        let outcome = self.orthogonalize(column)?;
        let ArnoldiOutcome::Ready { happy_breakdown } = outcome else {
            return Ok(outcome);
        };
        if let Err(failure) = self.apply_rotations(column) {
            return Ok(ArnoldiOutcome::Failed(failure));
        }
        Ok(ArnoldiOutcome::Ready { happy_breakdown })
    }

    fn orthogonalize(&mut self, column: usize) -> ExecutionResult<B, ArnoldiOutcome> {
        for row in 0..=column {
            let coefficient = self
                .backend
                .dot_prepared(
                    &self.workspace.work_basis_dot[row],
                    self.backend.view(&self.workspace.work),
                    self.backend.block_view(&self.workspace.basis, row),
                )
                .map_err(SolveError::Backend)?;
            let index = GmresWorkspace::<B, RESTART>::hessenberg_index(row, column);
            self.workspace.hessenberg[index] = coefficient;
            if !coefficient.is_finite() {
                return Ok(ArnoldiOutcome::Failed(ScalarFailure::NonFinite));
            }
            self.backend
                .axpy(
                    self.backend.view_mut(&mut self.workspace.work),
                    self.backend.block_view(&self.workspace.basis, row),
                    -coefficient,
                )
                .map_err(SolveError::Backend)?;
        }

        let next_norm = self
            .backend
            .norm_l2_prepared(
                &self.workspace.work_norm,
                self.backend.view(&self.workspace.work),
            )
            .map_err(SolveError::Backend)?;
        let next_index = GmresWorkspace::<B, RESTART>::hessenberg_index(column + 1, column);
        self.workspace.hessenberg[next_index] = next_norm;
        if !next_norm.is_finite() {
            return Ok(ArnoldiOutcome::Failed(ScalarFailure::NonFinite));
        }
        if next_norm == B::Scalar::ZERO {
            return Ok(ArnoldiOutcome::Ready {
                happy_breakdown: true,
            });
        }

        self.backend
            .copy(
                self.backend.view(&self.workspace.work),
                self.backend
                    .block_view_mut(&mut self.workspace.basis, column + 1),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .scale(
                self.backend
                    .block_view_mut(&mut self.workspace.basis, column + 1),
                B::Scalar::ONE / next_norm,
            )
            .map_err(SolveError::Backend)?;
        Ok(ArnoldiOutcome::Ready {
            happy_breakdown: false,
        })
    }

    fn apply_rotations(&mut self, column: usize) -> Result<(), ScalarFailure> {
        for row in 0..column {
            let upper_index = GmresWorkspace::<B, RESTART>::hessenberg_index(row, column);
            let lower_index = GmresWorkspace::<B, RESTART>::hessenberg_index(row + 1, column);
            let upper = self.workspace.hessenberg[upper_index];
            let lower = self.workspace.hessenberg[lower_index];
            let cosine = self.workspace.cosine[row];
            let sine = self.workspace.sine[row];
            self.workspace.hessenberg[upper_index] = cosine * upper + sine * lower;
            self.workspace.hessenberg[lower_index] = -sine * upper + cosine * lower;
        }

        let diagonal = GmresWorkspace::<B, RESTART>::hessenberg_index(column, column);
        let subdiagonal = GmresWorkspace::<B, RESTART>::hessenberg_index(column + 1, column);
        let (cosine, sine) = givens(
            self.workspace.hessenberg[diagonal],
            self.workspace.hessenberg[subdiagonal],
        )?;
        self.workspace.cosine[column] = cosine;
        self.workspace.sine[column] = sine;
        let upper = self.workspace.hessenberg[diagonal];
        let lower = self.workspace.hessenberg[subdiagonal];
        self.workspace.hessenberg[diagonal] = cosine * upper + sine * lower;
        self.workspace.hessenberg[subdiagonal] = B::Scalar::ZERO;

        let transformed = self.workspace.transformed_residual[column];
        self.workspace.transformed_residual[column] = cosine * transformed;
        self.workspace.transformed_residual[column + 1] = -sine * transformed;
        Ok(())
    }

    fn update_solution(&mut self, count: usize) -> ExecutionResult<B, Option<ScalarFailure>> {
        if let Err(failure) = back_substitute::<B::Scalar, RESTART>(
            &self.workspace.hessenberg,
            &self.workspace.transformed_residual,
            &mut self.workspace.coefficients,
            count,
        ) {
            return Ok(Some(failure));
        }
        for index in 0..count {
            self.backend
                .axpy(
                    self.backend.view_mut(self.solution),
                    self.backend
                        .block_view(&self.workspace.preconditioned_basis, index),
                    self.workspace.coefficients[index],
                )
                .map_err(SolveError::Backend)?;
        }
        Ok(None)
    }

    fn recompute_residual(&mut self) -> ExecutionResult<B, B::Scalar> {
        self.operator
            .apply(
                self.backend,
                self.backend.view(self.solution),
                self.backend.view_mut(&mut self.workspace.work),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .residual(
                self.backend.view(self.right_hand_side),
                self.backend.view(&self.workspace.work),
                self.backend.view_mut(&mut self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .norm_l2_prepared(
                &self.workspace.residual_norm,
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)
    }
}
