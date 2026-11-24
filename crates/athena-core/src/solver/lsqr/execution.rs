//! Golub–Kahan bidiagonalisation driver behind [`super::Lsqr`]: dimension
//! validation, the seeding of both Krylov directions, and the per-iteration
//! step with its termination tests. The public entry points in `algorithm`
//! only assemble an [`Execution`] and run it.

use eunomia::NumericElement;

use crate::{
    ConvergencePolicy, IterationObserver, IterationState, KrylovBackend, LsqrWorkspace,
    RectangularOperator, SolveError, SolveReport, Termination,
};

use super::super::dimension::validate_dimensions;
use super::super::rotation::givens_with_radius;

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
        validate_dimensions(&[
            (
                "right-hand side",
                rows,
                self.backend.vector_len(self.right_hand_side),
            ),
            ("solution", columns, self.backend.vector_len(self.solution)),
            ("LSQR workspace rows", rows, self.workspace.rows()),
            ("LSQR workspace columns", columns, self.workspace.columns()),
        ])
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

        // `rho_bar` and `phi_bar` start at `alpha` and `beta`, the first
        // entries of the bidiagonal factor and of `||r0|| e1`. Damping acts
        // only through the extra rotation at the head of every `step`, which
        // also accumulates the damping residual `psi`; nothing is
        // pre-rotated here.
        Ok(Stage::Continue(LsqrState {
            initial_residual: beta,
            residual: beta,
            threshold,
            alpha,
            phi_bar: beta,
            rho_bar: alpha,
            damping_residual_sq: <B::Scalar as NumericElement>::ZERO,
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

        let Some(beta) = self.advance_left(state)? else {
            return Ok(Step::Complete(state.report(Termination::NonFinite)));
        };
        let Some(alpha) = self.advance_right(state, beta)? else {
            return Ok(Step::Complete(state.report(Termination::NonFinite)));
        };
        state.iterations += 1;

        // Paige & Saunders 1982, §2.1 eqs 2.4 and 2.7: the damped problem is
        // the least-squares problem for `[B_k; λI]·y ≈ [β₁e₁; 0]`. Each step
        // first eliminates the `λ` row against `ρ̄` with one plane rotation,
        // `ρ̂ = √(ρ̄² + λ²)`, `c₁ = ρ̄/ρ̂`, `s₁ = λ/ρ̂`, which turns `φ̄` into
        // `c₁φ̄` and leaves `ψ = s₁φ̄` as a component of the augmented
        // residual that no later step changes. A second rotation then
        // eliminates `β` against `ρ̂`. A single rotation with
        // `√(ρ̄² + β² + λ²)` mixes the `λ` row into the `β` row and is not
        // an orthogonal elimination of either. Recurrence as in SciPy
        // 1.18.0 `scipy/sparse/linalg/_isolve/lsqr.py:443-463`.
        let rho_hat = if self.damping == zero {
            state.rho_bar
        } else {
            let Ok((damping_cosine, damping_sine, rho_hat)) =
                givens_with_radius(state.rho_bar, self.damping)
            else {
                return Ok(Step::Complete(state.report(Termination::Breakdown)));
            };
            let psi = damping_sine * state.phi_bar;
            state.phi_bar = damping_cosine * state.phi_bar;
            state.damping_residual_sq += psi * psi;
            rho_hat
        };

        // Second rotation: eliminate the subdiagonal `β` against `ρ̂`.
        let Ok((cosine, sine, rho)) = givens_with_radius(rho_hat, beta) else {
            return Ok(Step::Complete(state.report(Termination::Breakdown)));
        };
        if rho == zero {
            return Ok(Step::Complete(state.report(Termination::Breakdown)));
        }
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

        // The augmented residual `[b − A·x; −λ·x]` has norm
        // `√(φ̄² + Σψ²)` (equal to `|φ̄|` undamped), and
        // `|φ̄|·α·|cosine|` is the norm of `Āᵀr̄ = Aᵀr − λ²x`, both
        // available from the rotations without extra operator applications
        // (SciPy `lsqr.py:496-499`: `rnorm`, `arnorm = α|τ|`).
        state.residual = (state.phi_bar * state.phi_bar + state.damping_residual_sq).sqrt();
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
    /// `Σψ²`: squared norm of the augmented-residual components that the
    /// damping rotations have already fixed.
    damping_residual_sq: T,
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
