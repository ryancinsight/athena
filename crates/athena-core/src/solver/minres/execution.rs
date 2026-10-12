//! The MINRES recurrence machinery: Lanczos scalars, the step driver,
//! and the update-vector solution path.

use eunomia::{NumericElement, RealField};

use crate::{
    ConvergencePolicy, IterationObserver, IterationState, KrylovBackend, LinearOperator,
    SolveError, SolveReport, Termination,
};

use super::super::dimension::validate_dimensions;
use super::super::rotation::{ScalarFailure, givens};
use super::MinresWorkspace;

pub(super) type ExecutionResult<B, T> = Result<T, SolveError<<B as KrylovBackend>::Error>>;

pub(super) enum Stage<S, T> {
    Continue(S),
    Complete(SolveReport<T>),
}

/// Lanczos and rotation scalars carried between steps.
pub(super) struct MinresState<T> {
    initial_residual: T,
    threshold: T,
    /// `|φ̄ⱼ₊₁|`, maintained so the const report needs no method calls.
    last_residual: T,
    /// `βⱼ`: the Lanczos coefficient connecting `vⱼ₋₁` and `vⱼ`; produced by
    /// the previous step, consumed as the new tridiagonal column's
    /// subdiagonal.
    beta: T,
    /// Residual-norm carrier `φ̄ⱼ`: the rotated right-hand side's pending
    /// entry. `|φ̄ⱼ₊₁|` is the residual norm of the current iterate.
    phi_bar: T,
    /// Previous two Givens rotations, as `(cosine, sine)` pairs. Both start
    /// as the identity because the first column has no subdiagonal: the
    /// first band entries vanish under the identity, and the first modified
    /// diagonal is `α₁` directly.
    cosine_1: T,
    sine_1: T,
    cosine_2: T,
    sine_2: T,
    iterations: usize,
    operator_applications: usize,
}

impl<T: Copy> MinresState<T> {
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

pub(super) struct Execution<'a, B, O, Obs>
where
    B: KrylovBackend,
{
    pub(super) backend: &'a B,
    pub(super) operator: &'a O,
    pub(super) solution: &'a mut B::Vector,
    pub(super) workspace: &'a mut MinresWorkspace<B>,
    pub(super) policy: ConvergencePolicy<B::Scalar>,
    pub(super) observer: &'a mut Obs,
}

impl<B, O, Obs> Execution<'_, B, O, Obs>
where
    B: KrylovBackend,
    O: LinearOperator<B>,
    Obs: IterationObserver<B::Scalar>,
{
    pub(super) fn validate_dimensions(
        &self,
        right_hand_side: &B::Vector,
    ) -> Result<(), SolveError<B::Error>> {
        let dimension = self.operator.dimension();
        validate_dimensions(&[
            (
                "right-hand side",
                dimension,
                self.backend.vector_len(right_hand_side),
            ),
            (
                "solution",
                dimension,
                self.backend.vector_len(self.solution),
            ),
            ("MINRES workspace", dimension, self.workspace.len()),
        ])
    }

    /// Seed the recurrence: `v₁ = (b − A·x₀)/‖b − A·x₀‖`, the norm as the
    /// first Lanczos coefficient and the seed of the residual carrier.
    pub(super) fn seed(
        &mut self,
        right_hand_side: &B::Vector,
    ) -> ExecutionResult<B, Stage<MinresState<B::Scalar>, B::Scalar>> {
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
                self.backend.view_mut(&mut self.workspace.seed_residual),
            )
            .map_err(SolveError::Backend)?;
        let initial_residual = self
            .backend
            .norm_l2_prepared(
                &self.workspace.seed_norm,
                self.backend.view(&self.workspace.seed_residual),
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

        self.backend
            .copy(
                self.backend.view(&self.workspace.seed_residual),
                self.backend.view_mut(&mut self.workspace.direction),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .scale(
                self.backend.view_mut(&mut self.workspace.direction),
                <B::Scalar as NumericElement>::ONE / initial_residual,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .scale(
                self.backend
                    .view_mut(&mut self.workspace.previous_direction),
                <B::Scalar as NumericElement>::ZERO,
            )
            .map_err(SolveError::Backend)?;

        Ok(Stage::Continue(MinresState {
            initial_residual,
            threshold,
            last_residual: initial_residual,
            beta: initial_residual,
            phi_bar: initial_residual,
            cosine_1: <B::Scalar as NumericElement>::ONE,
            sine_1: <B::Scalar as NumericElement>::ZERO,
            cosine_2: <B::Scalar as NumericElement>::ONE,
            sine_2: <B::Scalar as NumericElement>::ZERO,
            iterations: 0,
            operator_applications: 1,
        }))
    }

    pub(super) fn iterate(
        &mut self,
        mut state: MinresState<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>> {
        while state.iterations < self.policy.max_iterations() {
            match self.step(&mut state)? {
                Step::Continue => {}
                Step::Complete(report) => return Ok(report),
            }
        }
        Ok(state.report(Termination::MaxIterations))
    }

    fn step(&mut self, state: &mut MinresState<B::Scalar>) -> ExecutionResult<B, Step<B::Scalar>> {
        let (alpha, beta_next) = self.advance_lanczos(state)?;

        // Apply the two previous rotations to the new tridiagonal column
        // `(βⱼ, αⱼ, βⱼ₊₁)`: row j−2 picks up the band-2 entry `εⱼ`, row j−1
        // the superdiagonal `λⱼ`, row j the modified diagonal `ᾱⱼ`.
        let epsilon = state.sine_2 * state.beta;
        let lambda = state.cosine_1 * state.cosine_2 * state.beta + state.sine_1 * alpha;
        let diagonal = -state.sine_1 * state.cosine_2 * state.beta + state.cosine_1 * alpha;

        let (cosine, sine) = match givens(diagonal, beta_next) {
            Ok(rotation) => rotation,
            Err(ScalarFailure::NonFinite) => {
                return Ok(Step::Complete(state.report(Termination::NonFinite)));
            }
            Err(ScalarFailure::Breakdown) => {
                return Ok(Step::Complete(state.report(Termination::Breakdown)));
            }
        };
        let Ok(radius) = givens_radius(diagonal, cosine) else {
            return Ok(Step::Complete(state.report(Termination::Breakdown)));
        };

        // Rotate the right-hand side: `φⱼ` enters the solution update, and
        // `φ̄ⱼ₊₁ = −sⱼ·φ̄ⱼ` carries the residual norm of the new iterate.
        let phi = cosine * state.phi_bar;
        let phi_bar_next = -sine * state.phi_bar;

        self.apply_update(lambda, epsilon, radius, phi)?;

        // Lanczos shift; a zero `βⱼ₊₁` ends the basis at an invariant
        // subspace.
        self.backend
            .copy(
                self.backend.view(&self.workspace.direction),
                self.backend
                    .view_mut(&mut self.workspace.previous_direction),
            )
            .map_err(SolveError::Backend)?;
        if beta_next != <B::Scalar as NumericElement>::ZERO {
            self.backend
                .scale(
                    self.backend.view_mut(&mut self.workspace.image),
                    <B::Scalar as NumericElement>::ONE / beta_next,
                )
                .map_err(SolveError::Backend)?;
            self.backend
                .copy(
                    self.backend.view(&self.workspace.image),
                    self.backend.view_mut(&mut self.workspace.direction),
                )
                .map_err(SolveError::Backend)?;
        }

        // Rotation shift: the current rotation becomes the previous one.
        state.cosine_2 = state.cosine_1;
        state.sine_2 = state.sine_1;
        state.cosine_1 = cosine;
        state.sine_1 = sine;
        state.beta = beta_next;
        state.phi_bar = phi_bar_next;
        state.last_residual = phi_bar_next.abs();
        if !state.last_residual.is_finite() {
            return Ok(Step::Complete(state.report(Termination::NonFinite)));
        }
        if self.policy.should_check(state.iterations) {
            self.observer.observe(IterationState::new(
                state.iterations,
                state.last_residual,
                state.threshold,
            ));
        }
        if state.last_residual <= state.threshold {
            return Ok(Step::Complete(state.report(Termination::Converged)));
        }
        if beta_next == <B::Scalar as NumericElement>::ZERO {
            // The Krylov space closed on an invariant subspace while the
            // residual is still above policy: the system is inconsistent over
            // the reachable space, or the operator is singular there.
            return Ok(Step::Complete(state.report(Termination::Breakdown)));
        }
        Ok(Step::Continue)
    }

    /// Three-term Lanczos: `w ← A·vⱼ − βⱼ·vⱼ₋₁`, then `w ← w − αⱼ·vⱼ`,
    /// leaving `‖w‖ = βⱼ₊₁` with `vⱼ₊₁ = w/βⱼ₊₁` formable from the image
    /// buffer. The step count advances here so every termination this step
    /// reports counts the step that found it.
    fn advance_lanczos(
        &mut self,
        state: &mut MinresState<B::Scalar>,
    ) -> ExecutionResult<B, (B::Scalar, B::Scalar)> {
        self.operator
            .apply(
                self.backend,
                self.backend.view(&self.workspace.direction),
                self.backend.view_mut(&mut self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        state.operator_applications += 1;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.image),
                self.backend.view(&self.workspace.previous_direction),
                -state.beta,
            )
            .map_err(SolveError::Backend)?;
        let alpha = self
            .backend
            .dot_prepared(
                &self.workspace.direction_image_dot,
                self.backend.view(&self.workspace.direction),
                self.backend.view(&self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.image),
                self.backend.view(&self.workspace.direction),
                -alpha,
            )
            .map_err(SolveError::Backend)?;
        let beta_next = self
            .backend
            .norm_l2_prepared(
                &self.workspace.image_norm,
                self.backend.view(&self.workspace.image),
            )
            .map_err(SolveError::Backend)?;
        state.iterations += 1;
        Ok((alpha, beta_next))
    }

    /// Solution update: `wⱼ = (vⱼ − λⱼ·wⱼ₋₁ − εⱼ·wⱼ₋₂)/ρⱼ`, then
    /// `xⱼ = xⱼ₋₁ + φⱼ·wⱼ`. The band corrections of the QR factor are
    /// absorbed into the update-vector recurrence, so no basis is stored and
    /// the triangular system is never re-solved. The update vectors rotate
    /// their roles for the next step.
    fn apply_update(
        &mut self,
        lambda: B::Scalar,
        epsilon: B::Scalar,
        radius: B::Scalar,
        phi: B::Scalar,
    ) -> Result<(), SolveError<B::Error>> {
        self.backend
            .copy(
                self.backend.view(&self.workspace.direction),
                self.backend.view_mut(&mut self.workspace.update),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.update),
                self.backend.view(&self.workspace.update_previous),
                -lambda,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(&mut self.workspace.update),
                self.backend.view(&self.workspace.update_previous_2),
                -epsilon,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .scale(
                self.backend.view_mut(&mut self.workspace.update),
                <B::Scalar as NumericElement>::ONE / radius,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(self.solution),
                self.backend.view(&self.workspace.update),
                phi,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .copy(
                self.backend.view(&self.workspace.update_previous),
                self.backend.view_mut(&mut self.workspace.update_previous_2),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .copy(
                self.backend.view(&self.workspace.update),
                self.backend.view_mut(&mut self.workspace.update_previous),
            )
            .map_err(SolveError::Backend)
    }
}

enum Step<T> {
    Continue,
    Complete(SolveReport<T>),
}

/// Recover the rotation radius from its `(cosine, sine)` pair.
///
/// [`givens`] returns only the trigonometric pair; the radius is the divisor
/// the update recurrence needs. `diagonal = cosine · radius` holds by
/// construction, with the pair built overflow-safely; a zero cosine means a
/// zero diagonal, which the caller reports as breakdown before dividing.
fn givens_radius<T: RealField>(diagonal: T, cosine: T) -> Result<T, ScalarFailure> {
    if !cosine.is_finite() || cosine == T::ZERO {
        return Err(ScalarFailure::Breakdown);
    }
    let radius = diagonal / cosine;
    if !radius.is_finite() || radius <= T::ZERO {
        return Err(ScalarFailure::Breakdown);
    }
    Ok(radius)
}
