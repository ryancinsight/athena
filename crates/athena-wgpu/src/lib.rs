//! Hephaestus-backed WGPU execution for Athena.
//!
//! Solver vectors stay in typed Hephaestus device buffers. Athena authors only
//! the solver-specific fused vector recurrences; sparse storage, `SpMV`,
//! reductions, allocation, transfer, and dispatch remain Hephaestus-owned.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// WGPU backend implementation.
pub mod backend;
/// GPU-resident linear operators.
pub mod operator;

pub use backend::WgpuBackend;
pub use operator::WgpuCsrOperator;
