//! Device-neutral Hephaestus accelerator backend for Athena.
//!
//! One backend serves every Hephaestus device. Athena binds to the
//! [`DenseVectorOps`] seam rather than to a device API, so a solver runs on any
//! backend implementing that seam without a crate, a feature, or a recurrence
//! per device.
//!
//! [`DenseVectorOps`]: hephaestus_core::DenseVectorOps

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod backend;
mod operator;
pub mod preconditioner;

pub use backend::HephaestusBackend;
pub use operator::CsrOperator;
pub use preconditioner::{Jacobi, inverse_diagonal_from_csr};
