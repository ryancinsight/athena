use core::marker::PhantomData;
use eunomia::NumericElement;

use crate::{
    CgWorkspace, ConvergencePolicy, IterationObserver, IterationState, KrylovBackend,
    LinearOperator, NoObserver, Preconditioner, SolveError, SolveReport, Termination,
};

use super::super::dimension::validate_dimension;

type ExecutionResult<B, T> = Result<T, SolveError<<B as KrylovBackend>::Error>>;

/// Zero-sized preconditioned conjugate-gradient algorithm marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cg<B>(PhantomData<fn() -> B>);

impl<B: KrylovBackend> Cg<B> {
    /// Solve `A x = b` into caller-owned `solution`.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure. Numerical termination such as
    /// non-positive curvature is returned value-semantically in
    /// [`SolveReport`].
    pub fn solve_into<O, P>(
        backend: &B,
        operator: &O,
        preconditioner: &P,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut CgWorkspace<B>,
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
        workspace: &mut CgWorkspace<B>,
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
            solution,
            workspace,
            policy,
            observer,
        };
        execution.validate_dimensions(right_hand_side)?;

        let residual = match execution.initialize_residual(right_hand_side)? {
            Stage::Continue(residual) => residual,
            Stage::Complete(report) => return Ok(report),
        };
        let state = match execution.initialize_direction(residual)? {
            Stage::Continue(state) => state,
            Stage::Complete(report) => return Ok(report),
        };
        execution.iterate(state)
    }
}

struct Execution<'a, B, O, P, Obs>
where
    B: KrylovBackend,
{
    backend: &'a B,
    operator: &'a O,
    preconditioner: &'a P,
    solution: &'a mut B::Vector,
    workspace: &'a mut CgWorkspace<B>,
    policy: ConvergencePolicy<B::Scalar>,
    observer: &'a mut Obs,
}

impl<B, O, P, Obs> Execution<'_, B, O, P, Obs>
where
    B: KrylovBackend,
    O: LinearOperator<B>,
    P: Preconditioner<B>,
    Obs: IterationObserver<B::Scalar>,
{
    fn validate_dimensions(&self, right_hand_side: &B::Vector) -> Result<(), SolveError<B::Error>> {
        let dimension = self.operator.dimension();
        validate_dimension(
            "right-hand side",
            dimension,
            self.backend.vector_len(right_hand_side),
        )?;
        validate_dimension(
            "solution",
            dimension,
            self.backend.vector_len(self.solution),
        )?;
        validate_dimension("CG workspace", dimension, self.workspace.len())
    }

    fn initialize_residual(
        &mut self,
        right_hand_side: &B::Vector,
    ) -> ExecutionResult<B, Stage<ResidualState<B::Scalar>, B::Scalar>> {
        self.backend
            .copy(
                self.backend.view(right_hand_side),
                self.backend.view_mut(&mut self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        let rhs_norm = self
            .backend
            .norm_l2_prepared(
                &self.workspace.residual_norm,
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        let threshold = self.policy.threshold(rhs_norm);

        self.operator
            .apply(
                self.backend,
                self.backend.view(self.solution),
                self.backend.view_mut(&mut self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .residual(
                self.backend.view(right_hand_side),
                self.backend.view(&self.workspace.image),
                self.backend.view_mut(&mut self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;

        let initial_residual = self
            .backend
            .norm_l2_prepared(
                &self.workspace.residual_norm,
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        let termination = if !initial_residual.is_finite() {
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

        Ok(Stage::Continue(ResidualState {
            initial_residual,
            threshold,
        }))
    }

    fn initialize_direction(
        &mut self,
        residual: ResidualState<B::Scalar>,
    ) -> ExecutionResult<B, Stage<CgState<B::Scalar>, B::Scalar>> {
        self.preconditioner
            .apply(
                self.backend,
                self.backend.view(&self.workspace.residual),
                self.backend
                    .view_mut(&mut self.workspace.preconditioned_residual),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .copy(
                self.backend.view(&self.workspace.preconditioned_residual),
                self.backend.view_mut(&mut self.workspace.direction),
            )
            .map_err(SolveError::Backend)?;
        let rho = self
            .backend
            .dot_prepared(
                &self.workspace.residual_preconditioned_dot,
                self.backend.view(&self.workspace.residual),
                self.backend.view(&self.workspace.preconditioned_residual),
            )
            .map_err(SolveError::Backend)?;

        if let Some(termination) = curvature_termination(rho) {
            return Ok(Stage::Complete(SolveReport::new(
                termination,
                0,
                1,
                1,
                residual.initial_residual,
                residual.initial_residual,
                residual.threshold,
            )));
        }

        Ok(Stage::Continue(CgState {
            initial_residual: residual.initial_residual,
            last_residual: residual.initial_residual,
            threshold: residual.threshold,
            rho,
            operator_applications: 1,
            preconditioner_applications: 1,
        }))
    }

    fn iterate(
        &mut self,
        mut state: CgState<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>> {
        for iteration in 1..=self.policy.max_iterations() {
            if let Some(report) = self.step(iteration, &mut state)? {
                return Ok(report);
            }
        }
        Ok(state.report(Termination::MaxIterations, self.policy.max_iterations()))
    }

    fn step(
        &mut self,
        iteration: usize,
        state: &mut CgState<B::Scalar>,
    ) -> Result<Option<SolveReport<B::Scalar>>, SolveError<B::Error>> {
        self.operator
            .apply(
                self.backend,
                self.backend.view(&self.workspace.direction),
                self.backend.view_mut(&mut self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        state.operator_applications += 1;

        let denominator = self
            .backend
            .dot_prepared(
                &self.workspace.direction_image_dot,
                self.backend.view(&self.workspace.direction),
                self.backend.view(&self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        if let Some(termination) = curvature_termination(denominator) {
            return Ok(Some(state.report(termination, iteration - 1)));
        }

        let alpha = state.rho / denominator;
        self.backend
            .fused_cg_update(
                self.backend.view_mut(self.solution),
                self.backend.view(&self.workspace.direction),
                self.backend.view_mut(&mut self.workspace.residual),
                self.backend.view(&self.workspace.image),
                alpha,
            )
            .map_err(SolveError::Backend)?;

        if let Some(report) = self.check_residual(iteration, state)? {
            return Ok(Some(report));
        }
        self.prepare_direction(iteration, state)
    }

    fn check_residual(
        &mut self,
        iteration: usize,
        state: &mut CgState<B::Scalar>,
    ) -> Result<Option<SolveReport<B::Scalar>>, SolveError<B::Error>> {
        if !self.policy.should_check(iteration) {
            return Ok(None);
        }
        state.last_residual = self
            .backend
            .norm_l2_prepared(
                &self.workspace.residual_norm,
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        self.observer.observe(IterationState::new(
            iteration,
            state.last_residual,
            state.threshold,
        ));
        let termination = if !state.last_residual.is_finite() {
            Some(Termination::NonFinite)
        } else if state.last_residual <= state.threshold {
            Some(Termination::Converged)
        } else if iteration == self.policy.max_iterations() {
            Some(Termination::MaxIterations)
        } else {
            None
        };
        Ok(termination.map(|reason| state.report(reason, iteration)))
    }

    fn prepare_direction(
        &mut self,
        iteration: usize,
        state: &mut CgState<B::Scalar>,
    ) -> Result<Option<SolveReport<B::Scalar>>, SolveError<B::Error>> {
        self.preconditioner
            .apply(
                self.backend,
                self.backend.view(&self.workspace.residual),
                self.backend
                    .view_mut(&mut self.workspace.preconditioned_residual),
            )
            .map_err(SolveError::Backend)?;
        state.preconditioner_applications += 1;
        let next_rho = self
            .backend
            .dot_prepared(
                &self.workspace.residual_preconditioned_dot,
                self.backend.view(&self.workspace.residual),
                self.backend.view(&self.workspace.preconditioned_residual),
            )
            .map_err(SolveError::Backend)?;
        if let Some(termination) = curvature_termination(next_rho) {
            return Ok(Some(state.report(termination, iteration)));
        }

        let beta = next_rho / state.rho;
        self.backend
            .combine_direction(
                self.backend.view_mut(&mut self.workspace.direction),
                self.backend.view(&self.workspace.preconditioned_residual),
                beta,
            )
            .map_err(SolveError::Backend)?;
        state.rho = next_rho;
        Ok(None)
    }
}

enum Stage<S, T> {
    Continue(S),
    Complete(SolveReport<T>),
}

#[derive(Clone, Copy)]
struct ResidualState<T> {
    initial_residual: T,
    threshold: T,
}

struct CgState<T> {
    initial_residual: T,
    last_residual: T,
    threshold: T,
    rho: T,
    operator_applications: usize,
    preconditioner_applications: usize,
}

impl<T: Copy> CgState<T> {
    const fn report(&self, termination: Termination, iterations: usize) -> SolveReport<T> {
        SolveReport::new(
            termination,
            iterations,
            self.operator_applications,
            self.preconditioner_applications,
            self.initial_residual,
            self.last_residual,
            self.threshold,
        )
    }
}

fn curvature_termination<T: NumericElement>(value: T) -> Option<Termination> {
    if !value.is_finite() {
        Some(Termination::NonFinite)
    } else if value < T::ZERO {
        Some(Termination::NonPositiveCurvature)
    } else if value == T::ZERO {
        Some(Termination::Breakdown)
    } else {
        None
    }
}
