//! The chain's state: the tiered solver, its reusable solve state, and
//! the cached hierarchy holder.

use athena_core::ConvergencePolicy;
use eunomia::{FloatElement, RealField};
use leto_ops::{CsrMatrix, RealScalar};
use std::fmt::Debug;

use crate::{AMGHierarchy, ComponentBlockPattern};

/// Tiered linear solver fallback chain for saddle-point systems arising from
/// mixed FEM discretizations of incompressible flow.
///
/// The chain attempts solvers in priority order, returning the first
/// successful solution; the convergence policy is Athena's own vocabulary, so
/// callers translating a domain-specific configuration convert once at this
/// boundary.
pub struct LinearSolverChain<T: RealField + Copy + FloatElement + RealScalar + Debug> {
    /// Convergence policy (tolerances, budget) for all iterative solvers.
    pub(super) policy: ConvergencePolicy<T>,
    /// DOF count below which the direct LU solver is preferred.
    ///
    /// Default: 2,048. Direct LU is exact but has `O(n^1.5)` cost; for
    /// large systems iterative solvers are more efficient.
    pub(super) direct_threshold: usize,
    /// GMRES restart parameter m (maximum Krylov subspace dimension).
    ///
    /// Default: 100. Larger values reduce restart overhead but increase
    /// memory `O(n·m)`; for saddle-point systems `m ∈ [50, 200]` is typical.
    pub(super) krylov_restart: usize,
}

/// Reusable state for successive solves with the same sparse matrix pattern.
///
/// The state caches only AMG transfer operators and component-block symbolic
/// factors. Matrix values are rebuilt for every solve, while exact CSR row
/// and column arrays gate reuse across mesh or boundary-topology changes.
pub struct LinearSolverState<T: RealField + Copy + FloatElement + RealScalar + Debug> {
    pub(super) amg: Option<CachedAmg<T>>,
    pub(super) component: Option<ComponentBlockPattern>,
}

pub(super) struct CachedAmg<T: RealField + Copy + FloatElement + RealScalar + Debug> {
    pub(super) hierarchy: AMGHierarchy<T>,
    pub(super) row_ptr: Vec<usize>,
    pub(super) col_indices: Vec<usize>,
}

impl<T: RealField + Copy + FloatElement + RealScalar + Debug> Default for LinearSolverState<T> {
    fn default() -> Self {
        Self {
            amg: None,
            component: None,
        }
    }
}

impl<T: RealField + Copy + FloatElement + RealScalar + Debug> CachedAmg<T> {
    pub(super) fn matches(&self, matrix: &CsrMatrix<T>) -> bool {
        self.row_ptr == matrix.row_ptr() && self.col_indices == matrix.col_indices()
    }
}
