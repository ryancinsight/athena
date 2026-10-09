//! Multigrid configuration vocabulary.
//!
//! The strategy enums and the level-counting knobs the multigrid family reads;
//! every value is decided once at `AMGConfig` construction and read by the
//! coarsening, interpolation, cycle, and smoother modules.

/// Configuration for the algebraic multigrid preconditioner.
#[derive(Debug, Clone)]
pub struct AMGConfig {
    /// Maximum number of levels in the hierarchy.
    pub max_levels: usize,
    /// Minimum size for the coarsest level.
    pub min_coarse_size: usize,
    /// Coarsening strategy to use.
    pub coarsening_strategy: CoarseningStrategy,
    /// Interpolation strategy.
    pub interpolation_strategy: InterpolationStrategy,
    /// Cycle type (V-cycle, W-cycle, F-cycle).
    pub cycle_type: CycleType,
    /// Smoother type for pre- and post-smoothing.
    pub smoother_type: SmootherType,
    /// Number of pre-smoothing iterations.
    pub pre_smooth_iterations: usize,
    /// Number of post-smoothing iterations.
    pub post_smooth_iterations: usize,
    /// Relaxation parameter for smoothers.
    pub relaxation_factor: f64,
    /// Chebyshev semi-iteration degree; the spectral bounds are estimated
    /// from each level's matrix by
    /// [`ChebyshevSmoother::estimate_eigenvalues`](super::smoothers::ChebyshevSmoother::estimate_eigenvalues)
    /// (Gershgorin). The default matches the degree the smoother tests
    /// validate.
    pub chebyshev_degree: usize,
    /// Strength threshold for coarsening.
    pub strength_threshold: f64,
    /// Maximum number of interpolation points.
    pub max_interpolation_points: usize,
}

impl Default for AMGConfig {
    fn default() -> Self {
        Self {
            max_levels: 10,
            min_coarse_size: 50,
            coarsening_strategy: CoarseningStrategy::RugeStueben,
            interpolation_strategy: InterpolationStrategy::Classical,
            cycle_type: CycleType::VCycle,
            smoother_type: SmootherType::GaussSeidel,
            pre_smooth_iterations: 2,
            post_smooth_iterations: 2,
            relaxation_factor: 1.0,
            chebyshev_degree: 3,
            strength_threshold: 0.25,
            max_interpolation_points: 4,
        }
    }
}

/// Coarsening strategy for AMG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoarseningStrategy {
    /// Ruge-Stüben algorithm (classical AMG).
    RugeStueben,
    /// Aggregation-based coarsening.
    Aggregation,
    /// Hybrid approach combining both methods.
    Hybrid,
    /// Falgout coarsening (CLJP method).
    Falgout,
    /// PMIS (Parallel Modified Independent Set).
    PMIS,
    /// HMIS (Hybrid Modified Independent Set).
    HMIS,
}

/// Interpolation strategy for AMG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationStrategy {
    /// Classical interpolation (Ruge-Stüben).
    Classical,
    /// Direct interpolation.
    Direct,
    /// Standard interpolation.
    Standard,
}

/// Smoother type for multigrid levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmootherType {
    /// Gauss-Seidel relaxation.
    GaussSeidel,
    /// Symmetric Gauss-Seidel.
    SymmetricGaussSeidel,
    /// Jacobi relaxation.
    Jacobi,
    /// SOR (Successive Over-Relaxation).
    SOR,
    /// Chebyshev polynomial smoother.
    Chebyshev,
}

/// Multigrid cycle type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleType {
    /// V-cycle: efficient, most commonly used.
    VCycle,
    /// W-cycle: more work per cycle, sometimes better convergence.
    WCycle,
    /// F-cycle: full multigrid, theoretically optimal.
    FCycle,
}
