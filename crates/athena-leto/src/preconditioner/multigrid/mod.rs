//! Algebraic and geometric multigrid over the Leto backend.
//!
//! The multigrid family `CFDrs` carried locally before ADR 0062 moved it to its
//! owner. [`config`] holds the strategy vocabulary, [`level`] the
//! level/hierarchy representation, and [`smoothers`] the sweep kernels; the
//! coarsening, transfer, cycle, and solver modules land with them in the
//! campaign's later increments.
//!
//! # Theorem — Two-Grid Convergence
//!
//! For a symmetric positive definite operator `A_h` with smoothing iteration
//! `M_h` satisfying `‖I − M_h A‖ ≤ η < 1` and approximation property
//! `‖A_h − P_h A_{2h} R_h‖ ≤ C h^α`, the two-grid method converges with factor
//!
//! ```text
//! ρ₂ ≤ η + C(1−η)² / (1 + C(1−η))
//! ```
//!
//! **Proof sketch.** The two-grid error propagation is
//! `E = S^ν₂ (I − P A_c⁻¹ R A) S^ν₁` where `S = I − M⁻¹A`; the smoothing
//! property bounds `‖A S^ν‖ ≤ C_S / ν` (Hackbusch 1985), the approximation
//! property gives `‖(I − P A_c⁻¹ R) v‖² ≤ C_A ⟨A v, v⟩`, and combining via
//! Cauchy-Schwarz yields the stated bound.
//!
//! # References
//!
//! - Hackbusch, W. (1985). *Multi-Grid Methods and Applications.* Springer.
//! - Bramble, J. H. (1993). *Multigrid Methods.* Pitman Research Notes.
//! - Stüben, K. (2001). "A review of algebraic multigrid." *JCAM* 128:281–309.

mod coarsening;
mod config;
mod level;
mod scalars;
mod smoothers;

pub use coarsening::{
    AlgebraicDistances, CoarseningQuality, CoarseningResult, aggregation_coarsening,
    analyze_coarsening_quality, falgout_coarsening, hmis_coarsening, hybrid_coarsening,
    pmis_coarsening, ruge_stueben_coarsening,
};
pub use config::{AMGConfig, CoarseningStrategy, CycleType, InterpolationStrategy, SmootherType};
pub use level::{AMGHierarchy, AMGStatistics, MultigridLevel, Smoother};
pub use smoothers::{
    ChebyshevSmoother, GaussSeidelSmoother, JacobiSmoother, SORSmoother,
    SymmetricGaussSeidelSmoother,
};
