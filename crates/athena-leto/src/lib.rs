//! Leto-backed CPU execution for Athena.
//!
//! The backend maps Athena vectors to Leto arrays and generic associated views
//! to Leto's zero-copy array views. Operator and preconditioner implementations
//! reuse Leto storage and kernels; solver policy and recurrence remain owned by
//! `athena-core`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// Leto backend implementation.
pub mod backend;
/// Backend error vocabulary.
pub mod error;
/// Reusable restarted-GMRES workspaces: the runtime restart bridge.
pub mod krylov;
/// Leto-backed linear operators.
pub mod operator;
/// Leto-backed preconditioners.
pub mod preconditioner;

pub use backend::{LetoBackend, LetoVectorBlock};
pub use error::LetoBackendError;
pub use krylov::KrylovWorkspace;
pub use operator::{
    BorrowedCsrOperator, BorrowedDenseOperator, CsrOperator, RectangularCsrOperator,
};
pub use preconditioner::{
    AMGConfig, AMGHierarchy, AMGStatistics, BlockDiagonalPreconditioner, ChebyshevSmoother,
    CoarseningStrategy, ComponentBlockPattern, ComponentBlockPreconditioner, CycleType,
    GaussSeidelSmoother, IncompleteLu, InterpolationStrategy, Jacobi, JacobiSmoother,
    MultigridLevel, SORSmoother, SimplePreconditioner, Smoother, SmootherType,
    SuccessiveOverRelaxation, SymmetricGaussSeidelSmoother,
};
