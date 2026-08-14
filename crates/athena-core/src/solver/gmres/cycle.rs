use crate::{SolveReport, Termination};

use super::rotation::ScalarFailure;

/// Outcome of the pre-iteration phase: either running state or a finished
/// report, so `initialize` can complete a solve without entering the loop.
pub(super) enum Stage<S, T> {
    Continue(S),
    Complete(SolveReport<T>),
}

/// What one restart cycle produced for the solution update.
pub(super) struct CycleOutcome {
    /// Arnoldi vectors the cycle built, and so the width of the least-squares
    /// problem the update solves.
    pub(super) vectors_used: usize,
    /// The Krylov space became invariant, so the cycle's solution is exact
    /// within it and no further vector can be built.
    pub(super) happy_breakdown: bool,
}

pub(super) enum CycleProgress {
    Ready(CycleOutcome),
    Terminated(Termination),
}

pub(super) enum ArnoldiOutcome {
    Ready { happy_breakdown: bool },
    Failed(ScalarFailure),
}

/// Scalar state carried across restart cycles.
///
/// Separate from [`SolveReport`] because a report is the terminal value while
/// this accumulates toward one; `report` is the single conversion point, so
/// every termination path emits the same counters.
pub(super) struct GmresState<T> {
    pub(super) initial_residual: T,
    pub(super) last_residual: T,
    pub(super) threshold: T,
    pub(super) iterations: usize,
    pub(super) operator_applications: usize,
    pub(super) preconditioner_applications: usize,
}

impl<T: Copy> GmresState<T> {
    pub(super) const fn report(&self, termination: Termination) -> SolveReport<T> {
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

impl ScalarFailure {
    pub(super) const fn termination(self) -> Termination {
        match self {
            Self::Breakdown => Termination::Breakdown,
            Self::NonFinite => Termination::NonFinite,
        }
    }
}
