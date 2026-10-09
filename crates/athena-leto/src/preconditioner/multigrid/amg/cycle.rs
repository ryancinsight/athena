//! The V-cycle recurrence and the Athena seam over it.

use athena_core::{KrylovBackend, Preconditioner};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::RealScalar;

use super::state::AlgebraicMultigrid;
use super::state::sparse_apply_into;
use crate::{LetoBackend, LetoBackendError};

/// Per-level V-cycle scratch: the applied image, the residual, and the coarse
/// pair, reused across applications so a cycle allocates nothing.
pub(super) struct AMGLevelWorkspace<T: RealField + Copy> {
    applied: Array1<T>,
    residual: Array1<T>,
    coarse_residual: Array1<T>,
    coarse_solution: Array1<T>,
    correction: Array1<T>,
}

impl<T: RealField + Copy + RealScalar> AMGLevelWorkspace<T> {
    fn new(size: usize, coarse_size: usize) -> Self {
        Self {
            applied: Array1::zeros([size]),
            residual: Array1::zeros([size]),
            coarse_residual: Array1::zeros([coarse_size]),
            coarse_solution: Array1::zeros([coarse_size]),
            correction: Array1::zeros([size]),
        }
    }
}

impl<T: RealField + Copy + FloatElement + RealScalar> AlgebraicMultigrid<T> {
    /// Perform a single V-cycle from `level_idx` down to the coarsest level.
    fn v_cycle(
        &self,
        level_idx: usize,
        b: &Array1<T>,
        x: &mut Array1<T>,
        workspace: &mut [AMGLevelWorkspace<T>],
    ) {
        let current_level = &self.levels[level_idx];
        let (current_workspace, remaining_workspace) = workspace
            .split_first_mut()
            .expect("invariant: AMG workspace matches hierarchy levels");

        // 1. Pre-smoothing
        current_level.smoother.apply(
            &current_level.matrix,
            x,
            b,
            self.config.pre_smooth_iterations,
        );

        if level_idx < self.levels.len() - 1 {
            // 2. Compute residual: r = b - A*x
            sparse_apply_into(&current_level.matrix, x, &mut current_workspace.applied)
                .expect("invariant: AMG SpMV inputs are valid");
            let residual = &mut current_workspace.residual;
            for (residual_entry, (rhs, applied)) in residual
                .iter_mut()
                .zip(b.iter().zip(current_workspace.applied.iter()))
            {
                *residual_entry = *rhs - *applied;
            }

            // 3. Restriction: r_coarse = R * r
            sparse_apply_into(
                current_level
                    .restriction
                    .as_ref()
                    .expect("invariant: non-coarsest AMG level has restriction"),
                residual,
                &mut current_workspace.coarse_residual,
            )
            .expect("invariant: AMG restriction dimensions are valid");

            // 4. Recursive call to coarse level
            current_workspace
                .coarse_solution
                .fill(<T as NumericElement>::ZERO);
            self.v_cycle(
                level_idx + 1,
                &current_workspace.coarse_residual,
                &mut current_workspace.coarse_solution,
                remaining_workspace,
            );

            // 5. Interpolation: e = P * e_coarse
            sparse_apply_into(
                current_level
                    .interpolation
                    .as_ref()
                    .expect("invariant: non-coarsest AMG level has interpolation"),
                &current_workspace.coarse_solution,
                &mut current_workspace.correction,
            )
            .expect("invariant: AMG interpolation dimensions are valid");

            // 6. Correction: x = x + e
            for (solution, correction) in x.iter_mut().zip(current_workspace.correction.iter()) {
                *solution += *correction;
            }

            // 7. Post-smoothing
            current_level.smoother.apply(
                &current_level.matrix,
                x,
                b,
                self.config.post_smooth_iterations,
            );
        } else {
            // Coarsest level solve
            current_level
                .smoother
                .apply(&current_level.matrix, x, b, 10);
        }
    }

    /// Apply one multigrid cycle as a preconditioner: `z = M^-1 r`.
    ///
    /// This is the concrete cycle. The Athena [`Preconditioner`] impl below is
    /// the seam that exposes it to a Krylov solve; both share this one body.
    ///
    /// # Errors
    ///
    /// Returns [`leto::LetoError::InvalidInput`] when `r` and `z` disagree in
    /// shape, or when the cached level workspace lock is poisoned.
    ///
    /// # Panics
    ///
    /// Panics never: the cycle's internal failures are invariant-guarded
    /// (workspace matches the assembled hierarchy by construction).
    ///
    /// Returns [`leto::LetoError::InvalidInput`] when `r` and `z` disagree in
    /// shape, or when the cached level workspace lock is poisoned.
    pub fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> Result<(), leto::LetoError> {
        if r.shape() != z.shape() {
            return Err(leto::LetoError::InvalidInput(format!(
                "AMG preconditioner vector length mismatch: {} != {}",
                r.shape()[0],
                z.shape()[0]
            )));
        }
        let mut workspace_guard = self.workspace.lock().map_err(|_| {
            leto::LetoError::InvalidInput("AMG workspace lock poisoned".to_string())
        })?;
        let needs_workspace = workspace_guard.as_ref().is_none_or(|workspace| {
            workspace.len() != self.levels.len()
                || workspace.iter().zip(&self.levels).enumerate().any(
                    |(level_idx, (workspace, level))| {
                        let coarse_size = self
                            .levels
                            .get(level_idx + 1)
                            .map_or(0, |coarse_level| coarse_level.matrix.nrows());
                        workspace.applied.shape() != [level.matrix.nrows()]
                            || workspace.coarse_residual.shape() != [coarse_size]
                    },
                )
        });
        if needs_workspace {
            *workspace_guard = Some(
                self.levels
                    .iter()
                    .enumerate()
                    .map(|(level_idx, level)| {
                        let coarse_size = self
                            .levels
                            .get(level_idx + 1)
                            .map_or(0, |coarse_level| coarse_level.matrix.nrows());
                        AMGLevelWorkspace::new(level.matrix.nrows(), coarse_size)
                    })
                    .collect(),
            );
        }
        let workspace = workspace_guard
            .as_mut()
            .expect("invariant: AMG workspace initialized above");
        z.fill(<T as NumericElement>::ZERO);
        self.v_cycle(0, r, z, workspace);

        Ok(())
    }
}

impl<T: RealField + Copy + FloatElement + RealScalar> Preconditioner<LetoBackend<T>>
    for AlgebraicMultigrid<T>
{
    fn apply(
        &self,
        _backend: &LetoBackend<T>,
        residual: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let length = residual.shape()[0];
        if output.shape()[0] != length {
            return Err(LetoBackendError::LengthMismatch {
                left: length,
                right: output.shape()[0],
            });
        }
        let mut guard = self.athena_boundary.lock().map_err(|_| {
            LetoBackendError::Leto(leto::LetoError::InvalidInput(
                "AMG boundary buffers poisoned".to_string(),
            ))
        })?;
        if guard.as_ref().is_none_or(|(r, _)| r.shape()[0] != length) {
            *guard = Some((Array1::zeros([length]), Array1::zeros([length])));
        }
        let (scratch_residual, scratch_output) = guard
            .as_mut()
            .expect("invariant: AMG boundary buffers installed above");

        for index in 0..length {
            scratch_residual[index] = residual[index];
        }
        self.apply_to(scratch_residual, scratch_output)
            .map_err(LetoBackendError::Leto)?;
        for index in 0..length {
            output[index] = scratch_output[index];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::state::AlgebraicMultigrid;
    use crate::AMGConfig;
    use crate::LetoBackend;
    use athena_core::Preconditioner;
    use eunomia::{FloatElement, NumericElement};
    use leto::Array1;
    use leto_ops::CsrMatrix;

    /// A diagonally dominant SPD tridiagonal fixture.
    macro_rules! tridiagonal {
        ($t:ty, $n:expr) => {{
            let mut row_ptr = Vec::with_capacity($n + 1);
            let mut col_indices = Vec::new();
            let mut values = Vec::new();
            row_ptr.push(0);
            for i in 0..$n {
                if i > 0 {
                    col_indices.push(i - 1);
                    values.push(<$t as FloatElement>::from_f64(-1.0));
                }
                col_indices.push(i);
                values.push(<$t as FloatElement>::from_f64(4.0));
                if i + 1 < $n {
                    col_indices.push(i + 1);
                    values.push(<$t as FloatElement>::from_f64(-1.0));
                }
                row_ptr.push(col_indices.len());
            }
            CsrMatrix::from_parts(values, col_indices, row_ptr, $n, $n)
                .expect("invariant: tridiagonal fixture CSR structure is valid")
        }};
    }

    /// The cycle is linear: a zero right-hand side from a zero iterate
    /// produces exactly zero.
    #[test]
    fn zero_rhs_yields_zero() {
        for dimension in [4usize, 16] {
            let matrix = tridiagonal!(f64, dimension);
            let amg = AlgebraicMultigrid::new(&matrix, AMGConfig::default())
                .expect("invariant: tridiagonal SPD system assembles");
            let backend = LetoBackend::<f64>::default();
            let zero = Array1::zeros([dimension]);
            let mut out = Array1::zeros([dimension]);
            Preconditioner::apply(&amg, &backend, zero.view(), out.view_mut())
                .expect("invariant: shapes agree");
            for &value in &out {
                assert_eq!(value.to_bits(), 0.0_f64.to_bits());
            }
        }
    }

    /// On a diagonal system the Gauss-Seidel smoother solves exactly in one
    /// sweep, so one cycle reproduces the exact inverse to the rounding of a
    /// division: the recorded smoother fixtures' property at both precisions.
    macro_rules! diagonal_case {
        ($t:ty) => {{
            let dimension = 4;
            let values = [
                <$t as FloatElement>::from_f64(2.0),
                <$t as FloatElement>::from_f64(4.0),
                <$t as FloatElement>::from_f64(8.0),
                <$t as FloatElement>::from_f64(16.0),
            ];
            let matrix = CsrMatrix::from_parts(
                values.to_vec(),
                vec![0usize, 1, 2, 3],
                vec![0usize, 1, 2, 3, 4],
                dimension,
                dimension,
            )
            .expect("invariant: diagonal fixture CSR structure is valid");
            let amg = AlgebraicMultigrid::new(&matrix, AMGConfig::default())
                .expect("invariant: diagonal system assembles");
            let backend = LetoBackend::<$t>::default();
            let rhs =
                Array1::from_shape_vec([dimension], values.iter().copied().collect::<Vec<_>>())
                    .expect("invariant: rhs shape matches");
            let mut out = Array1::zeros([dimension]);
            Preconditioner::apply(&amg, &backend, rhs.view(), out.view_mut())
                .expect("invariant: shapes agree");
            // z_i = r_i / d_i = 1 exactly in floating point (dyadic inputs).
            for &value in &out {
                assert_eq!(
                    value.to_bits(),
                    <$t as NumericElement>::ONE.to_bits(),
                    "diagonal solve must be exact"
                );
            }
        }};
    }

    #[test]
    fn diagonal_system_solve_is_exact() {
        diagonal_case!(f32);
        diagonal_case!(f64);
    }
}
