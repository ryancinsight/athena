//! The Chebyshev recurrence machinery: the three-term step with
//! overflow-safe Chebyshev coefficients, and the residual-driven loop.

use eunomia::NumericElement;

use crate::{
    ConvergencePolicy, IterationObserver, IterationState, KrylovBackend, LinearOperator,
    SolveError, SolveReport, Termination,
};

use super::bounds::SpectralBounds;
use super::workspace::ChebyshevWorkspace;

use super::super::dimension::validate_dimensions;

type ExecutionResult<B, T> = Result<T, SolveError<<B as KrylovBackend>::Error>>;

/// Scalar recurrence state.
pub(super) struct ChebyshevState<T> {
    initial_residual: T,
    threshold: T,
    last_residual: T,
    /// Inverse affine centre `η = 1/μ`, in `[0, 1)`. A singleton
    /// interval has `η = 0`, so the same recurrence becomes Richardson.
    inverse_mu_squared: T,
    /// The base step scale `ω = 2/(λ_max + λ_min)`.
    omega: T,
    /// Normalised ratio `v_k = w_k/μ`, updated by
    /// `v_{k+1} = 2 − η²/v_k`. It stays in `[1, 2]` even when μ
    /// is unrepresentable for a narrow interval.
    ratio: T,
    /// The current step coefficient `τ_k = 2/v_k` for k ≥ 2;
    /// the first step uses `τ_1 = 1`.
    tau: T,
    iterations: usize,
    operator_applications: usize,
}

impl<T: Copy> ChebyshevState<T> {
    const fn report(&self, termination: Termination) -> SolveReport<T> {
        SolveReport::new(
            termination,
            self.iterations,
            self.operator_applications,
            0,
            self.initial_residual,
            self.last_residual,
            self.threshold,
        )
    }
}

pub(super) enum Stage<T> {
    Continue(ChebyshevState<T>),
    Complete(SolveReport<T>),
}

pub(super) struct Execution<'a, B, O, Obs>
where
    B: KrylovBackend,
{
    pub(super) backend: &'a B,
    pub(super) operator: &'a O,
    pub(super) right_hand_side: &'a B::Vector,
    pub(super) solution: &'a mut B::Vector,
    pub(super) workspace: &'a mut ChebyshevWorkspace<B>,
    pub(super) policy: ConvergencePolicy<B::Scalar>,
    pub(super) bounds: SpectralBounds<B::Scalar>,
    pub(super) observer: &'a mut Obs,
}

impl<B, O, Obs> Execution<'_, B, O, Obs>
where
    B: KrylovBackend,
    O: LinearOperator<B>,
    Obs: IterationObserver<B::Scalar>,
{
    pub(super) fn validate_dimensions(&self) -> Result<(), SolveError<B::Error>> {
        let dimension = self.operator.dimension();
        validate_dimensions(&[
            (
                "right-hand side",
                dimension,
                self.backend.vector_len(self.right_hand_side),
            ),
            (
                "solution",
                dimension,
                self.backend.vector_len(self.solution),
            ),
            ("Chebyshev workspace", dimension, self.workspace.len()),
        ])
    }

    /// Seed the state: `r₀ ← b − A·x₀` with the Chebyshev coefficients from
    /// the spectral bounds. A residual already within policy completes the
    /// solve without iterating.
    pub(super) fn seed(&mut self) -> ExecutionResult<B, Stage<B::Scalar>> {
        self.backend
            .copy(
                self.backend.view(self.right_hand_side),
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
        let initial_residual = self
            .backend
            .norm_l2_prepared(
                &self.workspace.residual_norm,
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        let threshold = self.policy.threshold(rhs_norm);
        if !initial_residual.is_finite() || !threshold.is_finite() {
            return Ok(Stage::Complete(SolveReport::new(
                Termination::NonFinite,
                0,
                1,
                0,
                initial_residual,
                initial_residual,
                threshold,
            )));
        }
        if initial_residual <= threshold {
            return Ok(Stage::Complete(SolveReport::new(
                Termination::InitialResidual,
                0,
                1,
                0,
                initial_residual,
                initial_residual,
                threshold,
            )));
        }

        let minimum = self.bounds.minimum();
        let maximum = self.bounds.maximum();
        let one = <B::Scalar as NumericElement>::ONE;
        // Scale by the upper bound before adding/subtracting. This avoids
        // overflow in λ_max + λ_min and division by a zero interval width.
        let relative_minimum = minimum / maximum;
        let inverse_mu = (one - relative_minimum) / (one + relative_minimum);
        Ok(Stage::Continue(ChebyshevState {
            initial_residual,
            threshold,
            last_residual: initial_residual,
            inverse_mu_squared: inverse_mu * inverse_mu,
            omega: ((one + one) / (one + relative_minimum)) / maximum,
            // v_1 = w_1/μ = 1, including the singleton limit μ → ∞.
            ratio: one,
            tau: <B::Scalar as NumericElement>::ONE,
            iterations: 0,
            operator_applications: 1,
        }))
    }

    pub(super) fn iterate(
        &mut self,
        mut state: ChebyshevState<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>> {
        while state.iterations < self.policy.max_iterations() {
            if let Some(report) = self.step(&mut state)? {
                return Ok(report);
            }
        }
        Ok(state.report(Termination::MaxIterations))
    }

    fn step(
        &mut self,
        state: &mut ChebyshevState<B::Scalar>,
    ) -> ExecutionResult<B, Option<SolveReport<B::Scalar>>> {
        let one = <B::Scalar as NumericElement>::ONE;

        // `s_k = τ_k·ω·r_k + (τ_k − 1)·s_{k−1}`; the first step has no
        // predecessor, and `τ_1 = 1` makes the correction term vanish.
        self.backend
            .scale(
                self.backend.view_mut(&mut self.workspace.step),
                state.tau - one,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.step),
                self.backend.view(&self.workspace.residual),
                state.tau * state.omega,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(self.solution),
                self.backend.view(&self.workspace.step),
                one,
            )
            .map_err(SolveError::Backend)?;
        state.iterations += 1;

        // Recompute the residual: the convergence decision always sees the
        // true value, never a recurrence estimate.
        self.operator
            .apply(
                self.backend,
                self.backend.view(self.solution),
                self.backend.view_mut(&mut self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        state.operator_applications += 1;
        self.backend
            .residual(
                self.backend.view(self.right_hand_side),
                self.backend.view(&self.workspace.image),
                self.backend.view_mut(&mut self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        state.last_residual = self
            .backend
            .norm_l2_prepared(
                &self.workspace.residual_norm,
                self.backend.view(&self.workspace.residual),
            )
            .map_err(SolveError::Backend)?;
        if !state.last_residual.is_finite() {
            return Ok(Some(state.report(Termination::NonFinite)));
        }
        if self.policy.should_check(state.iterations) {
            self.observer.observe(IterationState::new(
                state.iterations,
                state.last_residual,
                state.threshold,
            ));
        }
        if state.last_residual <= state.threshold {
            return Ok(Some(state.report(Termination::Converged)));
        }
        if state.iterations == self.policy.max_iterations() {
            return Ok(Some(state.report(Termination::MaxIterations)));
        }

        // Only advance after checking the updated iterate. With η=0 the
        // coefficients remain τ=1, ω=1/λ, the exact singleton-spectrum
        // Richardson method. No infinity or inf/inf is formed.
        let two = one + one;
        state.ratio = two - state.inverse_mu_squared / state.ratio;
        if !state.ratio.is_finite() || state.ratio <= <B::Scalar as NumericElement>::ZERO {
            return Ok(Some(state.report(Termination::NonFinite)));
        }
        state.tau = two / state.ratio;
        if !state.tau.is_finite() {
            return Ok(Some(state.report(Termination::NonFinite)));
        }
        Ok(None)
    }
}
