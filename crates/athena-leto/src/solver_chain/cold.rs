//! The chain's cold path: the zero-iterate solve sequence and the
//! constructor/configurator surface.

use super::state::LinearSolverChain;
use athena_core::{ConvergencePolicy, Identity};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};
use std::fmt::Debug;

use crate::{
    AMGConfig, AlgebraicMultigrid, BlockDiagonalPreconditioner, ComponentBlockPreconditioner,
    IncompleteLu, LetoBackendError, SimplePreconditioner,
};

impl<T: RealField + Copy + FloatElement + RealScalar + Debug> LinearSolverChain<T> {
    /// Create a new solver chain with the given convergence policy.
    ///
    /// Defaults: `direct_threshold` = 2,048; `krylov_restart` = 100.
    #[must_use]
    pub fn new(policy: ConvergencePolicy<T>) -> Self {
        Self {
            policy,
            direct_threshold: 2_048,
            krylov_restart: 100,
        }
    }

    /// Override the DOF count threshold below which direct LU is used.
    #[must_use]
    pub const fn with_direct_threshold(mut self, threshold: usize) -> Self {
        self.direct_threshold = threshold;
        self
    }

    /// Override the GMRES restart parameter.
    #[must_use]
    pub const fn with_krylov_restart(mut self, restart: usize) -> Self {
        self.krylov_restart = restart;
        self
    }

    /// Solve `A·x = b` from a zero iterate using the tiered fallback.
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::Leto`] naming the chain when every tier
    /// fails; each tier's own outcome is surfaced in the log.
    pub fn solve(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &Array1<T>,
        n_velocity_dof: usize,
    ) -> Result<Array1<T>, LetoBackendError> {
        let n_total_dof = rhs.shape()[0];
        let n_pressure_dof = n_total_dof.saturating_sub(n_velocity_dof);
        let mut x = Array1::zeros([n_total_dof]);

        if self.direct_tier(matrix, rhs, &mut x, "LinearSolverChain") {
            return Ok(x);
        }

        let restart = std::cmp::min(self.krylov_restart, n_total_dof.max(1));
        let zero = |x: &mut Array1<T>| x.fill(<T as NumericElement>::ZERO);

        let component = ComponentBlockPreconditioner::new(matrix, n_velocity_dof, n_pressure_dof);
        if self.tier(
            "LinearSolverChain: GMRES+ComponentBlock",
            matrix,
            rhs,
            &mut x,
            restart,
            component,
            zero,
        ) {
            return Ok(x);
        }
        let simple = SimplePreconditioner::new(matrix, n_velocity_dof, n_pressure_dof);
        if self.tier(
            "LinearSolverChain: GMRES+SIMPLE",
            matrix,
            rhs,
            &mut x,
            restart,
            simple,
            zero,
        ) {
            return Ok(x);
        }
        let amg = AlgebraicMultigrid::new(matrix, AMGConfig::default());
        if self.tier(
            "LinearSolverChain: GMRES+AMG",
            matrix,
            rhs,
            &mut x,
            restart,
            amg,
            zero,
        ) {
            return Ok(x);
        }
        let block_built =
            BlockDiagonalPreconditioner::new(matrix, n_velocity_dof, n_pressure_dof).is_ok();
        if self.tier(
            "LinearSolverChain: GMRES+BlockDiag",
            matrix,
            rhs,
            &mut x,
            restart,
            BlockDiagonalPreconditioner::new(matrix, n_velocity_dof, n_pressure_dof),
            zero,
        ) {
            return Ok(x);
        }
        // Skip the unpreconditioned tier when a block preconditioner was
        // built but stagnated — for saddle-point systems the plain solve is
        // strictly worse.
        if !block_built
            && self.tier(
                "LinearSolverChain: GMRES unpreconditioned",
                matrix,
                rhs,
                &mut x,
                restart,
                Ok(Identity),
                zero,
            )
        {
            return Ok(x);
        }
        let ilu = IncompleteLu::from_csr(matrix);
        if self.tier(
            "LinearSolverChain: GMRES+ILU",
            matrix,
            rhs,
            &mut x,
            restart,
            ilu,
            zero,
        ) {
            return Ok(x);
        }
        self.last_resort(matrix, rhs, &mut x, "LinearSolverChain")?;
        Ok(x)
    }
}
