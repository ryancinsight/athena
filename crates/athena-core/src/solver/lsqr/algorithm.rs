use core::marker::PhantomData;
use eunomia::NumericElement;

use crate::{
    ConvergencePolicy, IterationObserver, IterationState, KrylovBackend, LsqrWorkspace, NoObserver,
    RectangularOperator, SolveError, SolveReport, Termination,
};

use super::super::dimension::validate_dimension;

type ExecutionResult<B, T> = Result<T, SolveError<<B as KrylovBackend>::Error>>;

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
/// relative to `‖r‖`, reported as [`Termination::NormalEquations`]. Testing
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

    /// Solve `min ‖A·x − b‖₂ + λ·‖x‖₂` (Tikhonov-regularised least squares)
    /// into caller-owned `solution`.
    ///
    /// `λ ≥ 0` is the regularisation weight. `λ = 0` recovers the unregularised
    /// least-squares problem; positive `λ` stabilises the iterate against
    /// measurement noise and small singular values, at the cost of a bias
    /// toward zero. The damped problem is solved exactly as
    /// `min ‖ [A; λI]·x − [b; 0] ‖₂`, so the recurrence is the same
    /// bidiagonalisation with one extra `λ²` term in the diagonal update at
    /// every step (Paige & Saunders 1982, §4, eqn 4.4).
    ///
    /// # Panics
    ///
    /// `damping` must be finite and non-negative; callers are expected to
    /// validate. The algorithm does not panic on these — it propagates a
    /// [`Termination::NonFinite`] or runs as if `λ = 0` — but the test suite
    /// asserts the input discipline.
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
        let mut execution = Execution {
            backend,
            operator,
            right_hand_side,
            solution,
            workspace,
            policy,
            observer,
            damping,
        };
        execution.validate_dimensions()?;
        match execution.initialize()? {
            Stage::Continue(state) => execution.iterate(state),
            Stage::Complete(report) => Ok(report),
        }
    }
}

struct Execution<'a, B, O, Obs>
where
    B: KrylovBackend,
{
    backend: &'a B,
    operator: &'a O,
    right_hand_side: &'a B::Vector,
    solution: &'a mut B::Vector,
    workspace: &'a mut LsqrWorkspace<B>,
    policy: ConvergencePolicy<B::Scalar>,
    observer: &'a mut Obs,
    /// Tikhonov regularisation weight. `λ ≥ 0`; `0` recovers the unregularised
    /// least-squares problem. Encoded as a single scalar, not a buffer, so it
    /// adds no workspace.
    damping: B::Scalar,
}

impl<B, O, Obs> Execution<'_, B, O, Obs>
where
    B: KrylovBackend,
    O: RectangularOperator<B>,
    Obs: IterationObserver<B::Scalar>,
{
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
