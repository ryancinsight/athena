//! CPU preconditioners.

mod incomplete_lu;
mod jacobi;
mod saddle;
mod successive_over_relaxation;
mod triangular;

pub use incomplete_lu::IncompleteLu;
pub use jacobi::Jacobi;
pub use saddle::{
    BlockDiagonalPreconditioner, ComponentBlockPattern, ComponentBlockPreconditioner,
    SimplePreconditioner,
};
pub use successive_over_relaxation::SuccessiveOverRelaxation;
