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
    InterpolationQuality, InterpolationStrategy, JacobiSmoother, MultigridLevel,
    RestrictionQuality, SORSmoother, Smoother, SmootherType, SymmetricGaussSeidelSmoother,
    aggregation_coarsening, analyze_coarsening_quality, create_classical_interpolation,
    create_direct_interpolation, create_full_weighting_restriction,
    create_half_weighting_restriction, create_injection_restriction,
    create_restriction_from_interpolation, create_standard_interpolation, falgout_coarsening,
    hmis_coarsening, hybrid_coarsening, pmis_coarsening, restrict_matrix, restrict_vector,
    ruge_stueben_coarsening, validate_interpolation_operator, validate_restriction_operator,
};
pub use saddle::{
    BlockDiagonalPreconditioner, ComponentBlockPattern, ComponentBlockPreconditioner,
    SimplePreconditioner,
};
pub use successive_over_relaxation::SuccessiveOverRelaxation;
