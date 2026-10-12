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
    /// The affine map's centre `μ = (λ_max + λ_min)/(λ_max − λ_min)`;
    /// the Chebyshev arguments live above one, so the damping bound after
    /// `k` steps is `1/T_k(μ)`.
    mu: T,
    /// The base step scale `ω = 2/(λ_max + λ_min)`.
    omega: T,
    /// Running ratio `w_k = T_k(μ)/T_{k−1}(μ)`, updated by
    /// `w_{k+1} = 2μ − 1/w_k`. It stays in `[1, 2μ]`, so it never
    /// overflows the way the exponentially growing `T_k(μ)` would.
    ratio: T,
    /// The current step coefficient `τ_k = 2μ/w_k`, with `τ_1 = 1`.
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
        let threshold = self.policy.threshold(initial_residual);
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
        Ok(Stage::Continue(ChebyshevState {
            initial_residual,
            threshold,
            last_residual: initial_residual,
            mu: (maximum + minimum) / (maximum - minimum),
            omega: (<B::Scalar as NumericElement>::ONE + <B::Scalar as NumericElement>::ONE)
                / (maximum + minimum),
            // w_1 = T_1(μ)/T_0(μ) = μ.
            ratio: (maximum + minimum) / (maximum - minimum),
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

        // Advance the Chebyshev coefficients for the next step:
        // `w_{k+1} = 2μ − 1/w_k`, then `τ_{k+1} = 2μ/w_{k+1}`.
        let two_mu = state.mu + state.mu;
        state.ratio = two_mu - one / state.ratio;
        if !state.ratio.is_finite() || state.ratio <= <B::Scalar as NumericElement>::ZERO {
            return Ok(Some(state.report(Termination::NonFinite)));
        }
        state.tau = two_mu / state.ratio;
        if !state.tau.is_finite() {
            return Ok(Some(state.report(Termination::NonFinite)));
        }

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
        Ok(None)
    }
}
