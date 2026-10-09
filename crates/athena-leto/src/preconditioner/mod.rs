//! CPU preconditioners.

mod incomplete_lu;
mod jacobi;
mod multigrid;
mod saddle;
mod successive_over_relaxation;
mod triangular;

pub use incomplete_lu::IncompleteLu;
pub use jacobi::Jacobi;
pub use multigrid::{
    AMGConfig, AMGHierarchy, AMGStatistics, ChebyshevSmoother, CoarseningStrategy, CycleType,
    GaussSeidelSmoother, InterpolationStrategy, JacobiSmoother, MultigridLevel, SORSmoother,
    Smoother, SmootherType, SymmetricGaussSeidelSmoother,
};
pub use saddle::{
    BlockDiagonalPreconditioner, ComponentBlockPattern, ComponentBlockPreconditioner,
    SimplePreconditioner,
};
pub use successive_over_relaxation::SuccessiveOverRelaxation;
