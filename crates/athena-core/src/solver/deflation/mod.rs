//! Two-level deflated CG over an orthonormal spectral basis.

mod algorithm;
mod basis;
mod execution;
mod lu;

pub use algorithm::{DeflatedCg, DeflatedOperator};
pub use basis::{DeflationSubspace, DeflationSubspaceError};
