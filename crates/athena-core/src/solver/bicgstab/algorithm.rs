use core::marker::PhantomData;
use eunomia::NumericElement;

use crate::{
    BiCgStabWorkspace, ConvergencePolicy, IterationObserver, IterationState, KrylovBackend,
    LinearOperator, NoObserver, Preconditioner, SolveError, SolveReport, Termination,
};

use super::super::dimension::validate_dimension;

type ExecutionResult<B, T> = Result<T, SolveError<<B as KrylovBackend>::Error>>;

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

struct Execution<'a, B, O, P, Obs>
where
    B: KrylovBackend,
{
    backend: &'a B,
    operator: &'a O,
    preconditioner: &'a P,
    right_hand_side: &'a B::Vector,
    solution: &'a mut B::Vector,
    workspace: &'a mut BiCgStabWorkspace<B>,
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
        validate_dimension("BiCGSTAB workspace", dimension, self.workspace.len())
    }

    fn initialize(&mut self) -> ExecutionResult<B, Stage<BiCgStabState<B::Scalar>, B::Scalar>> {
        let rhs_norm = self
            .backend
            .norm_l2(self.backend.view(self.right_hand_side))
            .map_err(SolveError::Backend)?;
        let threshold = self.policy.threshold(rhs_norm);
        let initial_residual = self.recompute_residual()?;

        let termination =
            if !rhs_norm.is_finite() || !threshold.is_finite() || !initial_residual.is_finite() {
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

        // The shadow residual is fixed at r̂₀ = r₀; the recurrence only ever
        // reads it, so it is never refreshed.
        self.backend
            .copy(
                self.backend.view(&self.workspace.residual),
                self.backend.view_mut(&mut self.workspace.shadow),
            )
            .map_err(SolveError::Backend)?;

        Ok(Stage::Continue(BiCgStabState {
            initial_residual,
            last_residual: initial_residual,
            threshold,
            rho: <B::Scalar as NumericElement>::ONE,
            alpha: <B::Scalar as NumericElement>::ONE,
            omega: <B::Scalar as NumericElement>::ONE,
            iterations: 0,
            operator_applications: 1,
            preconditioner_applications: 0,
        }))
    }

    fn iterate(
        &mut self,
        mut state: BiCgStabState<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>> {
        while state.iterations < self.policy.max_iterations() {
            match self.step(&mut state)? {
                Step::Continue => {}
                Step::Complete(report) => return Ok(report),
            }
        }
        Ok(state.report(Termination::MaxIterations))
    }

    /// One iteration: direction update, half step on `s`, stabilizer step.
    fn step(
        &mut self,
        state: &mut BiCgStabState<B::Scalar>,
    ) -> ExecutionResult<B, Step<B::Scalar>> {
        let zero = <B::Scalar as NumericElement>::ZERO;

        let rho = self
            .backend
            .dot_prepared(
                &self.workspace.shadow_residual_dot,
                self.backend.view(&self.workspace.shadow),
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        if !rho.is_finite() {
            return Ok(Step::Complete(state.report(Termination::NonFinite)));
        }
        // A vanishing shadow product means the shadow space has degenerated
        // and the recurrence can produce no further direction.
        if rho == zero {
            return Ok(Step::Complete(state.report(Termination::Breakdown)));
        }
        if let Some(termination) = self.update_direction(state, rho)? {
            return Ok(Step::Complete(state.report(termination)));
        }
        state.rho = rho;

        self.form_image(state)?;
        if let Some(termination) = self.compute_alpha(state, rho)? {
            return Ok(Step::Complete(state.report(termination)));
        }

        // The half step forms s in place in `residual`.
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.residual),
                self.backend.view(&self.workspace.image),
                -state.alpha,
            )
            .map_err(SolveError::Backend)?;
        state.iterations += 1;

        let half_step_norm = self
            .backend
            .norm_l2_prepared(
                &self.workspace.residual_norm,
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        if !half_step_norm.is_finite() {
            return Ok(Step::Complete(state.report(Termination::NonFinite)));
        }

        // Early half-step exit: the half-step update already solves the system
        // when s has collapsed, and forming the stabilizer coefficient from a
        // zero s would divide by zero.
        if half_step_norm <= state.threshold {
            self.backend
                .axpy(
                    self.backend.view_mut(self.solution),
                    self.backend.view(&self.workspace.preconditioned_direction),
                    state.alpha,
                )
                .map_err(SolveError::Backend)?;
            return Ok(self
                .confirm_convergence(state)?
                .map_or(Step::Continue, Step::Complete));
        }

        self.form_stabilizer(state)?;
        if let Some(termination) = self.compute_omega(state)? {
            return Ok(Step::Complete(state.report(termination)));
        }
        self.apply_updates(state)?;

        let recursive_residual = self
            .backend
            .norm_l2_prepared(
                &self.workspace.residual_norm,
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        if !recursive_residual.is_finite() {
            return Ok(Step::Complete(state.report(Termination::NonFinite)));
        }
        state.last_residual = recursive_residual;
        if self.policy.should_check(state.iterations) {
            self.observer.observe(IterationState::new(
                state.iterations,
                recursive_residual,
                state.threshold,
            ));
        }
        if recursive_residual <= state.threshold {
            return Ok(self
                .confirm_convergence(state)?
                .map_or(Step::Continue, Step::Complete));
        }
        // A vanishing stabilizer coefficient leaves r unchanged by the
        // stabilizer step, so the next direction update would divide by zero.
        if state.omega == zero {
            return Ok(Step::Complete(state.report(Termination::Breakdown)));
        }
        Ok(Step::Continue)
    }

    /// Set the direction to the residual on the first step, else advance the
    /// direction recurrence in place.
    fn update_direction(
        &mut self,
        state: &BiCgStabState<B::Scalar>,
        rho: B::Scalar,
    ) -> ExecutionResult<B, Option<Termination>> {
        if state.iterations == 0 {
            self.backend
                .copy(
                    self.backend.view(&self.workspace.residual),
                    self.backend.view_mut(&mut self.workspace.direction),
                )
                .map_err(SolveError::Backend)?;
            return Ok(None);
        }
        let beta = (rho / state.rho) * (state.alpha / state.omega);
        if !beta.is_finite() {
            return Ok(Some(Termination::NonFinite));
        }
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.direction),
                self.backend.view(&self.workspace.image),
                -state.omega,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .scale(self.backend.view_mut(&mut self.workspace.direction), beta)
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.direction),
                self.backend.view(&self.workspace.residual),
                <B::Scalar as NumericElement>::ONE,
            )
            .map_err(SolveError::Backend)?;
        Ok(None)
    }

    /// Apply the preconditioner and operator to the direction.
    fn form_image(&mut self, state: &mut BiCgStabState<B::Scalar>) -> ExecutionResult<B, ()> {
        self.preconditioner
            .apply(
                self.backend,
                self.backend.view(&self.workspace.direction),
                self.backend
                    .view_mut(&mut self.workspace.preconditioned_direction),
            )
            .map_err(SolveError::Backend)?;
        state.preconditioner_applications += 1;
        self.operator
            .apply(
                self.backend,
                self.backend.view(&self.workspace.preconditioned_direction),
                self.backend.view_mut(&mut self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        state.operator_applications += 1;
        Ok(())
    }

    /// Form the half-step coefficient from the shadow projection.
    fn compute_alpha(
        &mut self,
        state: &mut BiCgStabState<B::Scalar>,
        rho: B::Scalar,
    ) -> ExecutionResult<B, Option<Termination>> {
        let shadow_image = self
            .backend
            .dot_prepared(
                &self.workspace.shadow_image_dot,
                self.backend.view(&self.workspace.shadow),
                self.backend.view(&self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        if !shadow_image.is_finite() {
            return Ok(Some(Termination::NonFinite));
        }
        if shadow_image == <B::Scalar as NumericElement>::ZERO {
            return Ok(Some(Termination::Breakdown));
        }
        state.alpha = rho / shadow_image;
        if state.alpha.is_finite() {
            Ok(None)
        } else {
            Ok(Some(Termination::NonFinite))
        }
    }

    /// Apply the preconditioner and operator to the half step.
    fn form_stabilizer(&mut self, state: &mut BiCgStabState<B::Scalar>) -> ExecutionResult<B, ()> {
        self.preconditioner
            .apply(
                self.backend,
                self.backend.view(&self.workspace.residual),
                self.backend
                    .view_mut(&mut self.workspace.preconditioned_residual),
            )
            .map_err(SolveError::Backend)?;
        state.preconditioner_applications += 1;
        self.operator
            .apply(
                self.backend,
                self.backend.view(&self.workspace.preconditioned_residual),
                self.backend.view_mut(&mut self.workspace.stabilizer),
            )
            .map_err(SolveError::Backend)?;
        state.operator_applications += 1;
        Ok(())
    }

    /// Form the stabilizing coefficient that minimises the half-step residual.
    fn compute_omega(
        &mut self,
        state: &mut BiCgStabState<B::Scalar>,
    ) -> ExecutionResult<B, Option<Termination>> {
        let stabilizer_residual = self
            .backend
            .dot_prepared(
                &self.workspace.stabilizer_residual_dot,
                self.backend.view(&self.workspace.stabilizer),
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        let stabilizer_square = self
            .backend
            .dot_prepared(
                &self.workspace.stabilizer_norm_dot,
                self.backend.view(&self.workspace.stabilizer),
                self.backend.view(&self.workspace.stabilizer),
            )
            .map_err(SolveError::Backend)?;
        if !stabilizer_residual.is_finite() || !stabilizer_square.is_finite() {
            return Ok(Some(Termination::NonFinite));
        }
        if stabilizer_square == <B::Scalar as NumericElement>::ZERO {
            return Ok(Some(Termination::Breakdown));
        }
        state.omega = stabilizer_residual / stabilizer_square;
        if state.omega.is_finite() {
            Ok(None)
        } else {
            Ok(Some(Termination::NonFinite))
        }
    }

    /// Advance the solution by both step components and the residual by the
    /// stabilizer step.
    fn apply_updates(&mut self, state: &BiCgStabState<B::Scalar>) -> ExecutionResult<B, ()> {
        self.backend
            .axpy(
                self.backend.view_mut(self.solution),
                self.backend.view(&self.workspace.preconditioned_direction),
                state.alpha,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(self.solution),
                self.backend.view(&self.workspace.preconditioned_residual),
                state.omega,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.residual),
                self.backend.view(&self.workspace.stabilizer),
                -state.omega,
            )
            .map_err(SolveError::Backend)?;
        Ok(())
    }

    /// Recompute `b − A·x` and accept convergence only if the true residual
    /// meets the threshold.
    ///
    /// `BiCGSTAB` propagates `r` by a recurrence rather than recomputing it, so
    /// the recursive residual drifts from `b − A·x` under rounding — sharply
    /// so after a small `ω`. Reporting convergence from the recurrence alone
    /// would return an unsolved system. On rejection the recomputed residual
    /// replaces the recursive one, which restarts the recurrence from a
    /// consistent state.
    fn confirm_convergence(
        &mut self,
        state: &mut BiCgStabState<B::Scalar>,
    ) -> ExecutionResult<B, Option<SolveReport<B::Scalar>>> {
        let true_residual = self.recompute_residual()?;
        state.operator_applications += 1;
        state.last_residual = true_residual;
        if !true_residual.is_finite() {
            return Ok(Some(state.report(Termination::NonFinite)));
        }
        if true_residual <= state.threshold {
            return Ok(Some(state.report(Termination::Converged)));
        }
        // The shadow residual stays fixed at r̂₀ across the reset, as the
        // recurrence requires.
        Ok(None)
    }

    fn recompute_residual(&mut self) -> ExecutionResult<B, B::Scalar> {
        self.operator
            .apply(
                self.backend,
                self.backend.view(self.solution),
                self.backend.view_mut(&mut self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .residual(
                self.backend.view(self.right_hand_side),
                self.backend.view(&self.workspace.image),
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

enum Step<T> {
    Continue,
    Complete(SolveReport<T>),
}

enum Stage<S, T> {
    Continue(S),
    Complete(SolveReport<T>),
}

struct BiCgStabState<T> {
    initial_residual: T,
    last_residual: T,
    threshold: T,
    rho: T,
    alpha: T,
    omega: T,
    iterations: usize,
    operator_applications: usize,
    preconditioner_applications: usize,
}

impl<T: Copy> BiCgStabState<T> {
    const fn report(&self, termination: Termination) -> SolveReport<T> {
        SolveReport::new(
            termination,
            self.iterations,
            self.operator_applications,
            self.preconditioner_applications,
            self.initial_residual,
            self.last_residual,
            self.threshold,
        )
    }
}
