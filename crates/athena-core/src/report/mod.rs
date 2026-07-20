//! Allocation-free iteration and solve reports.

mod iteration;
mod solve;

pub use iteration::{IterationObserver, IterationState, NoObserver};
pub use solve::{SolveReport, Termination};
