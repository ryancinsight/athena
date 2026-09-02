//! Golub–Kahan bidiagonalisation driver behind [`super::Lsqr`]: dimension
//! validation, the seeding of both Krylov directions, and the per-iteration
//! step with its termination tests. The public entry points in `algorithm`
//! only assemble an [`Execution`] and run it.

use eunomia::NumericElement;

use crate::{
    ConvergencePolicy, IterationObserver, IterationState, KrylovBackend, LsqrWorkspace,
    RectangularOperator, SolveError, SolveReport, Termination,
};

use super::super::dimension::validate_dimension;

type ExecutionResult<B, T> = Result<T, SolveError<<B as KrylovBackend>::Error>>;

pub(super) struct Execution<'a, B, O, Obs>
where
    B: KrylovBackend,
{
    pub(super) backend: &'a B,
    pub(super) operator: &'a O,
    pub(super) right_hand_side: &'a B::Vector,
    pub(super) solution: &'a mut B::Vector,
    pub(super) workspace: &'a mut LsqrWorkspace<B>,
    pub(super) policy: ConvergencePolicy<B::Scalar>,
    pub(super) observer: &'a mut Obs,
    /// Tikhonov regularisation weight. `λ ≥ 0`; `0` recovers the unregularised
    /// least-squares problem. Encoded as a single scalar, not a buffer, so it
    /// adds no workspace.
    pub(super) damping: B::Scalar,
}

impl<B, O, Obs> Execution<'_, B, O, Obs>
where
    B: KrylovBackend,
    O: RectangularOperator<B>,
    Obs: IterationObserver<B::Scalar>,
{
    pub(super) fn run(mut self) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>> {
        self.validate_dimensions()?;
        match self.initialize()? {
            Stage::Continue(state) => self.iterate(state),
            Stage::Complete(report) => Ok(report),
        }
    }

    fn validate_dimensions(&self) -> Result<(), SolveError<B::Error>> {
        let rows = self.operator.rows();
        let columns = self.operator.columns();
        validate_dimension(
            "right-hand side",
            rows,
            self.backend.vector_len(self.right_hand_side),
        )?;
        validate_dimension("solution", columns, self.backend.vector_len(self.solution))?;
        validate_dimension("LSQR workspace rows", rows, self.workspace.rows())?;
        validate_dimension("LSQR workspace columns", columns, self.workspace.columns())
    }

    /// `u = b − A·x₀`, normalised; then `v = Aᵀu`, normalised; `w = v`.
    fn initialize(&mut self) -> ExecutionResult<B, Stage<LsqrState<B::Scalar>, B::Scalar>> {
        let rhs_norm = self
            .backend
            .norm_l2(self.backend.view(self.right_hand_side))
            .map_err(SolveError::Backend)?;
        let threshold = self.policy.threshold(rhs_norm);
        let beta = self.seed_left()?;

        if !beta.is_finite() || !threshold.is_finite() {
            return Ok(Stage::Complete(initial_report(
                Termination::NonFinite,
                1,
                beta,
                threshold,
            )));
        }
        if beta <= threshold {
            return Ok(Stage::Complete(initial_report(
                Termination::InitialResidual,
                1,
                beta,
                threshold,
            )));
        }

        self.backend
            .scale(
                self.backend.view_mut(&mut self.workspace.left),
                <B::Scalar as NumericElement>::ONE / beta,
            )
            .map_err(SolveError::Backend)?;
        let alpha = self.seed_right()?;
        if !alpha.is_finite() {
            return Ok(Stage::Complete(initial_report(
                Termination::NonFinite,
                2,
                beta,
                threshold,
            )));
        }
        // `Aᵀr = 0` at a non-zero residual is the exact least-squares optimum:
        // the residual already lies in the null space of `Aᵀ`.
        if alpha == <B::Scalar as NumericElement>::ZERO {
            return Ok(Stage::Complete(initial_report(
                Termination::NormalEquations,
                2,
                beta,
                threshold,
            )));
        }

        self.backend
            .scale(
                self.backend.view_mut(&mut self.workspace.right),
                <B::Scalar as NumericElement>::ONE / alpha,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .copy(
                self.backend.view(&self.workspace.right),
                self.backend.view_mut(&mut self.workspace.direction),
            )
            .map_err(SolveError::Backend)?;

        // The initial `rho_bar` carries the diagonal entry from the previous
        // step. For step 1 there is no previous step; the undamped recurrence
        // initialises `rho_bar = alpha`. Damping enters only via the `+λ²`
        // in the Givens rotation at every step (see `step`); the initial
        // `rho_bar` is unchanged, because the rotation's `ρ` is the
        // accumulated diagonal element of the augmented bidiagonal factor and
        // the very first rotation's diagonal is just `α`. This matches the
        // reference Hansen 1998 damped-LSQR recurrence.
        Ok(Stage::Continue(LsqrState {
            initial_residual: beta,
            residual: beta,
            threshold,
            alpha,
            phi_bar: beta,
            rho_bar: alpha,
            iterations: 0,
            operator_applications: 2,
        }))
    }

    /// `u ← b − A·x₀`, returning its norm before normalisation.
    fn seed_left(&mut self) -> ExecutionResult<B, B::Scalar> {
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
                self.backend.view_mut(&mut self.workspace.left),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .norm_l2_prepared(
                &self.workspace.left_norm,
                self.backend.view(&self.workspace.left),
            )
            .map_err(SolveError::Backend)
    }

    /// `v ← Aᵀ·u`, returning its norm before normalisation.
    fn seed_right(&mut self) -> ExecutionResult<B, B::Scalar> {
        self.operator
            .apply_transpose(
                self.backend,
                self.backend.view(&self.workspace.left),
                self.backend.view_mut(&mut self.workspace.right),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .norm_l2_prepared(
                &self.workspace.right_norm,
                self.backend.view(&self.workspace.right),
            )
            .map_err(SolveError::Backend)
    }

    fn iterate(
        &mut self,
        mut state: LsqrState<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>> {
        while state.iterations < self.policy.max_iterations() {
            match self.step(&mut state)? {
                Step::Continue => {}
                Step::Complete(report) => return Ok(report),
            }
        }
        Ok(state.report(Termination::MaxIterations))
    }

    fn step(&mut self, state: &mut LsqrState<B::Scalar>) -> ExecutionResult<B, Step<B::Scalar>> {
        let one = <B::Scalar as NumericElement>::ONE;
        let zero = <B::Scalar as NumericElement>::ZERO;
        let damping_sq = self.damping * self.damping;

        let Some(beta) = self.advance_left(state)? else {
            return Ok(Step::Complete(state.report(Termination::NonFinite)));
        };
        let Some(alpha) = self.advance_right(state, beta)? else {
            return Ok(Step::Complete(state.report(Termination::NonFinite)));
        };
        state.iterations += 1;

        // Plane rotation eliminating the subdiagonal of the bidiagonal factor.
        // The unregularised recurrence carries `ρ = √(ρ_bar² + β²)`. For
        // Tikhonov damping, the augmented system `[A; λI]·x ≈ [b; 0]`
        // contributes an extra `λ²` to the diagonal at every step (Paige &
        // Saunders 1982, eqn 4.4): the rotation's `ρ` is therefore
        // `√(ρ_bar² + β² + λ²)`. The trigonometric identities that follow
        // are unchanged because `cosine`, `sine`, `θ`, `φ` only depend on
        // `ρ` and `ρ_bar`.
        let rho = (state.rho_bar * state.rho_bar + beta * beta + damping_sq).sqrt();
        if !rho.is_finite() || rho == zero {
            return Ok(Step::Complete(state.report(Termination::Breakdown)));
        }
        let cosine = state.rho_bar / rho;
        let sine = beta / rho;
        let theta = sine * alpha;
        state.rho_bar = -cosine * alpha;
        let phi = cosine * state.phi_bar;
        state.phi_bar = sine * state.phi_bar;

        // x += (phi/rho) w
        self.backend
            .axpy(
                self.backend.view_mut(self.solution),
                self.backend.view(&self.workspace.direction),
                phi / rho,
            )
            .map_err(SolveError::Backend)?;
        // w = v - (theta/rho) w
        self.backend
            .scale(
                self.backend.view_mut(&mut self.workspace.direction),
                -(theta / rho),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.direction),
                self.backend.view(&self.workspace.right),
                one,
            )
            .map_err(SolveError::Backend)?;

        // `phi_bar` is the residual norm of the current iterate, and
        // `phi_bar * alpha * |cosine|` is the norm of `Aᵀr`, both available
        // from the rotation without extra operator applications.
        state.residual = state.phi_bar.abs();
        let normal_residual = state.phi_bar.abs() * alpha * cosine.abs();
        if !state.residual.is_finite() || !normal_residual.is_finite() {
            return Ok(Step::Complete(state.report(Termination::NonFinite)));
        }
        state.alpha = alpha;

        if self.policy.should_check(state.iterations) {
            self.observer.observe(IterationState::new(
                state.iterations,
                state.residual,
                state.threshold,
            ));
        }
        if state.residual <= state.threshold {
            return Ok(Step::Complete(state.report(Termination::Converged)));
        }
        // Scale-free normal-equation test: an inconsistent system keeps a
        // residual bounded away from zero, so only this criterion can report
        // its optimum.
        if state.residual > zero
            && normal_residual <= self.policy.relative_tolerance() * state.residual
        {
            return Ok(Step::Complete(state.report(Termination::NormalEquations)));
        }
        Ok(Step::Continue)
    }

    /// `βu ← A·v − αu`, returning the new `β` after normalising `u`.
    fn advance_left(
        &mut self,
        state: &LsqrState<B::Scalar>,
    ) -> ExecutionResult<B, Option<B::Scalar>> {
        let one = <B::Scalar as NumericElement>::ONE;
        self.operator
            .apply(
                self.backend,
                self.backend.view(&self.workspace.right),
                self.backend.view_mut(&mut self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .scale(
                self.backend.view_mut(&mut self.workspace.left),
                -state.alpha,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.left),
                self.backend.view(&self.workspace.image),
                one,
            )
            .map_err(SolveError::Backend)?;
        let beta = self
            .backend
            .norm_l2_prepared(
                &self.workspace.left_norm,
                self.backend.view(&self.workspace.left),
            )
            .map_err(SolveError::Backend)?;
        if !beta.is_finite() {
            return Ok(None);
        }
        if beta > <B::Scalar as NumericElement>::ZERO {
            self.backend
                .scale(self.backend.view_mut(&mut self.workspace.left), one / beta)
                .map_err(SolveError::Backend)?;
        }
        Ok(Some(beta))
    }

    /// `αv ← Aᵀ·u − βv`, returning the new `α` after normalising `v`.
    fn advance_right(
        &mut self,
        state: &mut LsqrState<B::Scalar>,
        beta: B::Scalar,
    ) -> ExecutionResult<B, Option<B::Scalar>> {
        let one = <B::Scalar as NumericElement>::ONE;
        self.operator
            .apply_transpose(
                self.backend,
                self.backend.view(&self.workspace.left),
                self.backend.view_mut(&mut self.workspace.adjoint_image),
            )
            .map_err(SolveError::Backend)?;
        state.operator_applications += 2;
        self.backend
            .scale(self.backend.view_mut(&mut self.workspace.right), -beta)
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.right),
                self.backend.view(&self.workspace.adjoint_image),
                one,
            )
            .map_err(SolveError::Backend)?;
        let alpha = self
            .backend
            .norm_l2_prepared(
                &self.workspace.right_norm,
                self.backend.view(&self.workspace.right),
            )
            .map_err(SolveError::Backend)?;
        if !alpha.is_finite() {
            return Ok(None);
        }
        if alpha > <B::Scalar as NumericElement>::ZERO {
            self.backend
                .scale(
                    self.backend.view_mut(&mut self.workspace.right),
                    one / alpha,
                )
                .map_err(SolveError::Backend)?;
        }
        Ok(Some(alpha))
    }
}

/// Report for a solve that terminated before the first iteration.
fn initial_report<T: Copy>(
    termination: Termination,
    operator_applications: usize,
    residual: T,
    threshold: T,
) -> SolveReport<T> {
    SolveReport::new(
        termination,
        0,
        operator_applications,
        0,
        residual,
        residual,
        threshold,
    )
}

enum Step<T> {
    Continue,
    Complete(SolveReport<T>),
}

enum Stage<S, T> {
    Continue(S),
    Complete(SolveReport<T>),
}

struct LsqrState<T> {
    initial_residual: T,
    residual: T,
    threshold: T,
    alpha: T,
    phi_bar: T,
    rho_bar: T,
    iterations: usize,
    operator_applications: usize,
}

impl<T: Copy> LsqrState<T> {
    const fn report(&self, termination: Termination) -> SolveReport<T> {
        SolveReport::new(
            termination,
            self.iterations,
            self.operator_applications,
            0,
            self.initial_residual,
            self.residual,
            self.threshold,
        )
    }
}
