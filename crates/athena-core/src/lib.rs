//! Backend-neutral solver law for Athena.
//!
//! This crate owns convergence policy, operator and preconditioner contracts,
//! reusable Krylov workspaces, and solver recurrences. Storage and arithmetic
//! are supplied by a [`KrylovBackend`], so the same recurrence monomorphizes
//! over Leto host arrays and Hephaestus device buffers without dynamic
//! dispatch.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

extern crate alloc;

/// Backend contracts.
pub mod backend;
/// Convergence policy.
pub mod convergence;
/// Solver error vocabulary.
pub mod error;
/// Linear-operator contracts.
pub mod operator;
/// Preconditioner contracts and policies.
pub mod preconditioner;
/// Allocation-free iteration and solve reports.
pub mod report;
/// Iterative solver implementations.
pub mod solver;

pub use backend::KrylovBackend;
pub use convergence::{ConvergencePolicy, InvalidConvergencePolicy};
pub use error::SolveError;
pub use operator::LinearOperator;
pub use preconditioner::{Identity, Preconditioner};
pub use report::{IterationObserver, IterationState, NoObserver, SolveReport, Termination};
pub use solver::{Cg, CgWorkspace, Gmres, GmresWorkspace};
