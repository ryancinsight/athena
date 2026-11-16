//! Recurrence state carried between conjugate-gradient iterations.

use eunomia::NumericElement;

use crate::{SolveReport, Termination};

pub(super) enum Stage<S, T> {
    Continue(S),
    Complete(SolveReport<T>),
}

#[derive(Clone, Copy)]
pub(super) struct ResidualState<T> {
    pub(super) initial_residual: T,
    pub(super) threshold: T,
}

pub(super) struct CgState<T> {
    pub(super) initial_residual: T,
    pub(super) last_residual: T,
    pub(super) last_true_residual_iteration: usize,
    pub(super) threshold: T,
    pub(super) rho: T,
    pub(super) operator_applications: usize,
    pub(super) preconditioner_applications: usize,
}

impl<T: Copy> CgState<T> {
    pub(super) const fn report(
        &self,
        termination: Termination,
        iterations: usize,
    ) -> SolveReport<T> {
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

pub(super) fn curvature_termination<T: NumericElement>(value: T) -> Option<Termination> {
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
