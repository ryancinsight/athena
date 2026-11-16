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
    /// Recurrence residual norm at the latest check; `last_residual` holds
    /// the measured `‖b - A x‖` whenever one was taken.
    pub(super) recurrence_residual: T,
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

impl<T: NumericElement> CgState<T> {
    /// The recurrence residual met the threshold, and `‖b - A x‖` exceeds it
    /// by more than the threshold: the gap alone is above the threshold.
    pub(super) fn stalled(&self) -> bool {
        self.recurrence_residual <= self.threshold
            && self.last_residual - self.recurrence_residual > self.threshold
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

#[cfg(test)]
mod tests {
    use eunomia::{NumericElement, RealField};

    use super::CgState;

    /// A state with threshold one after a measurement of `b - A x`.
    fn measured<T: RealField>(recurrence: T, true_residual: T) -> CgState<T> {
        let one = <T as NumericElement>::ONE;
        CgState {
            initial_residual: one + one + one + one,
            last_residual: true_residual,
            recurrence_residual: recurrence,
            last_true_residual_iteration: 1,
            threshold: one,
            rho: one,
            operator_applications: 3,
            preconditioner_applications: 1,
        }
    }

    fn classifies_the_gap_against_the_threshold<T: RealField>() {
        let one = <T as NumericElement>::ONE;
        let half = one / (one + one);
        // Dyadic values make every difference exact in both formats.
        // Converged recurrence, gap bound 3/2 above the threshold: stall.
        assert!(measured(half, one + one).stalled());
        // Gap bound exactly the threshold, then below it: the gap may be
        // below the threshold, so iteration continues.
        assert!(!measured(half, one + half).stalled());
        assert!(!measured(half, one + half / (one + one)).stalled());
        // Gap bound 5/2, but the recurrence has not met the threshold.
        assert!(!measured(one + half, one + one + one + one).stalled());
    }

    #[test]
    fn a_stall_needs_a_converged_recurrence_and_a_gap_above_the_threshold() {
        classifies_the_gap_against_the_threshold::<f32>();
        classifies_the_gap_against_the_threshold::<f64>();
    }
}
