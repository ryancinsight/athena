//! Saddle-point block preconditioners over the Leto backend.
//!
//! These are the saddle-point families `CFDrs` carried locally before ADR 0062
//! moved them to their owner: the block-diagonal Schur approximation
//! (Murphy–Golub–Wathen), the SIMPLE pressure-linked correction (Patankar),
//! and the component-block sparse-LU composite for component-major 3D
//! systems. All implement Athena's [`Preconditioner`] seam over borrowed
//! views, so a Krylov solve accepts them directly.
//!
//! [`Preconditioner`]: athena_core::Preconditioner

mod component;
mod diagonal;
mod matrix;
mod simple;

#[cfg(test)]
mod tests;

pub use component::{ComponentBlockPattern, ComponentBlockPreconditioner};
pub use diagonal::BlockDiagonalPreconditioner;
pub use simple::SimplePreconditioner;
