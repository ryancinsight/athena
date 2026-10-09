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
    AMGConfig, AMGHierarchy, AMGStatistics, AlgebraicDistances, ChebyshevSmoother,
    CoarseningQuality, CoarseningResult, CoarseningStrategy, CycleType, GaussSeidelSmoother,
    InterpolationStrategy, JacobiSmoother, MultigridLevel, SORSmoother, Smoother, SmootherType,
    SymmetricGaussSeidelSmoother, aggregation_coarsening, analyze_coarsening_quality,
    falgout_coarsening, hmis_coarsening, hybrid_coarsening, pmis_coarsening,
    ruge_stueben_coarsening,
};
pub use saddle::{
    BlockDiagonalPreconditioner, ComponentBlockPattern, ComponentBlockPreconditioner,
    SimplePreconditioner,
};
pub use successive_over_relaxation::SuccessiveOverRelaxation;
