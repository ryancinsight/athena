//! Multigrid level and hierarchy representation.
//!
//! A level pairs its system matrix with the transfer operators that reach its
//! neighbours and the smoother that damps its high-frequency error; the
//! hierarchy is the cached transfer pairing the cycles rewalk.

use eunomia::RealField;
use leto_ops::{CsrMatrix, RealScalar};

use super::smoothers::{
    ChebyshevSmoother, GaussSeidelSmoother, JacobiSmoother, SORSmoother,
    SymmetricGaussSeidelSmoother,
};

/// Smoother carried by each multigrid level: one enum variant per closed-set
/// smoother, dispatched by an exhaustive match on every pre- and
/// post-smoothing sweep of every cycle. The variant carries its concrete
/// smoother's parameters, so the level holds no vtable and clones by value.
#[derive(Clone)]
pub enum Smoother<T: RealField + Copy> {
    /// Gauss-Seidel relaxation.
    GaussSeidel(GaussSeidelSmoother<T>),
    /// Symmetric Gauss-Seidel relaxation.
    SymmetricGaussSeidel(SymmetricGaussSeidelSmoother<T>),
    /// Jacobi relaxation.
    Jacobi(JacobiSmoother<T>),
    /// Successive over-relaxation.
    Sor(SORSmoother<T>),
    /// Chebyshev semi-iteration.
    Chebyshev(ChebyshevSmoother<T>),
}

impl<T> Smoother<T>
where
    T: RealField + Copy + eunomia::FloatElement + RealScalar,
{
    /// Apply the carried smoother to the system `Ax = b` for `iterations`
    /// sweeps. The match is exhaustive over the closed smoother set: adding a
    /// variant is a compile error until this dispatch names it.
    pub fn apply(
        &self,
        matrix: &CsrMatrix<T>,
        x: &mut leto::Array1<T>,
        b: &leto::Array1<T>,
        iterations: usize,
    ) {
        match self {
            Self::GaussSeidel(smoother) => smoother.apply(matrix, x, b, iterations),
            Self::SymmetricGaussSeidel(smoother) => smoother.apply(matrix, x, b, iterations),
            Self::Jacobi(smoother) => smoother.apply(matrix, x, b, iterations),
            Self::Sor(smoother) => smoother.apply(matrix, x, b, iterations),
            Self::Chebyshev(smoother) => smoother.apply(matrix, x, b, iterations),
        }
    }
}

/// Multigrid level representation.
#[derive(Clone)]
pub struct MultigridLevel<T: RealField + Copy + RealScalar> {
    /// System matrix for this level.
    pub matrix: CsrMatrix<T>,
    /// Restriction operator from fine to coarse.
    pub restriction: Option<CsrMatrix<T>>,
    /// Interpolation operator from coarse to fine.
    pub interpolation: Option<CsrMatrix<T>>,
    /// Smoother for this level.
    pub smoother: Smoother<T>,
}

/// A cached AMG hierarchy containing transfer operators.
#[derive(Clone)]
pub struct AMGHierarchy<T: RealField + Copy + RealScalar> {
    /// Transfer operators for each level: (restriction, interpolation).
    pub operators: Vec<(CsrMatrix<T>, CsrMatrix<T>)>,
}

impl<T: RealField + Copy + RealScalar> AMGHierarchy<T> {
    /// Create a new hierarchy from existing levels.
    pub fn from_levels(levels: &[MultigridLevel<T>]) -> Self {
        let operators = levels
            .iter()
            .filter_map(|level| {
                if let (Some(restriction), Some(interpolation)) =
                    (&level.restriction, &level.interpolation)
                {
                    Some((restriction.clone(), interpolation.clone()))
                } else {
                    None
                }
            })
            .collect();

        Self { operators }
    }
}

/// AMG statistics and diagnostics.
#[derive(Debug, Clone)]
pub struct AMGStatistics {
    /// Number of levels in hierarchy.
    pub num_levels: usize,
    /// Size of each level.
    pub level_sizes: Vec<usize>,
    /// Operator complexity (total nonzeros / fine nonzeros).
    pub operator_complexity: f64,
    /// Grid complexity (total variables / fine variables).
    pub grid_complexity: f64,
    /// Average convergence factor per cycle.
    pub convergence_factor: f64,
    /// Setup time in seconds.
    pub setup_time: f64,
    /// Average solve time per cycle in seconds.
    pub solve_time_per_cycle: f64,
}

impl Default for AMGStatistics {
    fn default() -> Self {
        Self {
            num_levels: 0,
            level_sizes: Vec::new(),
            operator_complexity: 1.0,
            grid_complexity: 1.0,
            convergence_factor: 1.0,
            setup_time: 0.0,
            solve_time_per_cycle: 0.0,
        }
    }
}
