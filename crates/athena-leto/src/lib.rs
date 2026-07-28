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
/// Leto-backed linear operators.
pub mod operator;
/// Leto-backed preconditioners.
pub mod preconditioner;

pub use backend::LetoBackend;
pub use error::LetoBackendError;
pub use operator::{BorrowedDenseOperator, CsrOperator, RectangularCsrOperator};
pub use preconditioner::{IncompleteLu, Jacobi, SuccessiveOverRelaxation};
