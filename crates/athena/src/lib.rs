//! Athena iterative linear solvers.
//!
//! `athena-core` owns backend-neutral solver law. Enable `cpu` for the Leto
//! implementation and `wgpu` for the Hephaestus implementation.
//!
//! # Examples
//!
//! ```
//! use athena::ConvergencePolicy;
//!
//! let policy = ConvergencePolicy::new(1.0e-12_f64, 1.0e-10, 100)?;
//! assert_eq!(policy.max_iterations(), 100);
//! # Ok::<(), athena::InvalidConvergencePolicy>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub use athena_core::*;

/// Leto CPU implementation.
#[cfg(feature = "cpu")]
pub use athena_leto as cpu;

/// Hephaestus WGPU implementation.
#[cfg(feature = "wgpu")]
pub use athena_wgpu as wgpu;
