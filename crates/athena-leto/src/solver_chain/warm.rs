//! The warm path: guess-started solve with reusable AMG/component state.

use super::state::{LinearSolverChain, LinearSolverState};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};
use std::fmt::Debug;

use crate::{LetoBackendError, SuccessiveOverRelaxation};

impl<T: RealField + Copy + FloatElement + RealScalar + Debug> LinearSolverChain<T> {
    /// Solve with a warm start and reusable AMG/component state.
    ///
    /// `state` may be retained across Picard or continuation solves. AMG
    /// transfer operators are reused only when the complete CSR sparsity
    /// pattern matches; changing mesh connectivity rebuilds the hierarchy.
    ///
    /// # Errors
    ///
    /// See [`Self::solve`].
    pub fn solve_with_guess_state(
        &self,
        matrix: &CsrMatrix<T>,
        rhs: &Array1<T>,
        n_velocity_dof: usize,
        initial_guess: Option<&Array1<T>>,
        state: &mut LinearSolverState<T>,
    ) -> Result<Array1<T>, LetoBackendError> {
        let n_total_dof = rhs.shape()[0];
        let n_pressure_dof = n_total_dof.saturating_sub(n_velocity_dof);
        let mut x = initial_guess
            .cloned()
            .unwrap_or_else(|| Array1::zeros([n_total_dof]));

        if self.direct_tier(matrix, rhs, &mut x, "LinearSolverChain(warm)") {
            return Ok(x);
        }

        let restart = std::cmp::min(self.krylov_restart, n_total_dof.max(1));
        let reset = |x: &mut Array1<T>| match initial_guess {
            Some(guess) => x
                .iter_mut()
                .zip(guess.iter())
                .for_each(|(target, &source)| *target = source),
            None => x.fill(<T as NumericElement>::ZERO),
        };

        let sor = SuccessiveOverRelaxation::from_csr_with_identity_rows(
            matrix,
            <T as FloatElement>::from_f64(1.0),
        );
        if self.tier(
            "LinearSolverChain(warm): GMRES+SOR",
            matrix,
            rhs,
            &mut x,
            restart,
            sor,
            reset,
        ) {
            return Ok(x);
        }

        if self.amg_cache_tier(matrix, rhs, &mut x, restart, state, reset) {
            return Ok(x);
        }

        if self.warm_stateless_tiers(
            matrix,
            rhs,
            &mut x,
            n_velocity_dof,
            n_pressure_dof,
            restart,
            state,
            initial_guess,
        ) {
            return Ok(x);
        }
        self.last_resort(matrix, rhs, &mut x, "LinearSolverChain(warm)")?;
        Ok(x)
    }
}
