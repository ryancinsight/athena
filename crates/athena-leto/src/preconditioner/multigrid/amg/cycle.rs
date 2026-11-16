//! The V-cycle recurrence and the Athena seam over it.

use athena_core::{KrylovBackend, Preconditioner};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::RealScalar;

use super::state::AlgebraicMultigrid;
use std::ops::{Index, IndexMut};

use super::super::vector::{Vector, sparse_apply_into};
use crate::{LetoBackend, LetoBackendError};

/// Per-level V-cycle scratch: the applied image, the residual, and the coarse
/// pair, reused across applications. All buffers have dense element storage.
pub(super) struct AMGLevelWorkspace<T: RealField + Copy> {
    applied: Vec<T>,
    residual: Vec<T>,
    coarse_residual: Vec<T>,
    coarse_solution: Vec<T>,
    correction: Vec<T>,
}

impl<T: RealField + Copy + RealScalar> AMGLevelWorkspace<T> {
    fn new(size: usize, coarse_size: usize) -> Self {
        Self {
            applied: vec![<T as NumericElement>::ZERO; size],
            residual: vec![<T as NumericElement>::ZERO; size],
            coarse_residual: vec![<T as NumericElement>::ZERO; coarse_size],
            coarse_solution: vec![<T as NumericElement>::ZERO; coarse_size],
            correction: vec![<T as NumericElement>::ZERO; size],
        }
    }
}

impl<T: RealField + Copy + FloatElement + RealScalar> AlgebraicMultigrid<T> {
    /// Perform a single V-cycle from `level_idx` down to the coarsest level.
    fn v_cycle(
        &self,
        level_idx: usize,
        b: &(impl Index<usize, Output = T> + ?Sized),
        x: &mut (impl Vector<T> + IndexMut<usize> + ?Sized),
        workspace: &mut [AMGLevelWorkspace<T>],
    ) {
        let current_level = &self.levels[level_idx];
        let (current_workspace, remaining_workspace) = workspace
            .split_first_mut()
            .expect("invariant: AMG workspace matches hierarchy levels");

        // 1. Pre-smoothing
        current_level.smoother.sweep(
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
            for (index, (entry, applied)) in residual
                .iter_mut()
                .zip(&current_workspace.applied)
                .enumerate()
            {
                *entry = b[index] - *applied;
            }

            // 3. Restriction: r_coarse = R * r
            sparse_apply_into(
                current_level
                    .restriction
                    .as_ref()
                    .expect("invariant: non-coarsest AMG level has restriction"),
                current_workspace.residual.as_slice(),
                &mut current_workspace.coarse_residual,
            )
            .expect("invariant: AMG restriction dimensions are valid");

            // 4. Recursive call to coarse level
            current_workspace
                .coarse_solution
                .fill(<T as NumericElement>::ZERO);
            self.v_cycle(
                level_idx + 1,
                current_workspace.coarse_residual.as_slice(),
                current_workspace.coarse_solution.as_mut_slice(),
                remaining_workspace,
            );

            // 5. Interpolation: e = P * e_coarse
            sparse_apply_into(
                current_level
                    .interpolation
                    .as_ref()
                    .expect("invariant: non-coarsest AMG level has interpolation"),
                current_workspace.coarse_solution.as_slice(),
                &mut current_workspace.correction,
            )
            .expect("invariant: AMG interpolation dimensions are valid");

            // 6. Correction: x = x + e
            for (index, correction) in current_workspace.correction.iter().enumerate() {
                x[index] += *correction;
            }

            // 7. Post-smoothing
            current_level.smoother.sweep(
                &current_level.matrix,
                x,
                b,
                self.config.post_smooth_iterations,
            );
        } else {
            // Coarsest level solve
            current_level
                .smoother
                .sweep(&current_level.matrix, x, b, 10);
        }
    }

    /// Apply one multigrid cycle as a preconditioner: `z = M^-1 r`.
    ///
    /// # Errors
    ///
    /// Returns [`leto::LetoError::InvalidInput`] if a vector length differs
    /// from the fine system size or the level workspace lock is poisoned.
    pub fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> Result<(), leto::LetoError> {
        self.cycle(r.view(), z.view_mut())
    }

    fn cycle(
        &self,
        r: leto::ArrayView1<'_, T>,
        mut z: leto::ArrayViewMut1<'_, T>,
    ) -> leto::Result<()> {
        let size = self.levels[0].matrix.nrows();
        let [residual] = r.shape();
        let [output] = z.shape();
        if residual != size || output != size {
            return Err(leto::LetoError::InvalidInput(format!(
                "AMG system size {size} differs from residual {residual} or output {output}"
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
                        workspace.applied.len() != level.matrix.nrows()
                            || workspace.coarse_residual.len() != coarse_size
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
        let zero = <T as NumericElement>::ZERO;
        if let (Some(residual), Some(output)) = (r.as_slice(), z.as_mut_slice()) {
            output.fill(zero);
            self.v_cycle(0, residual, output, workspace);
        } else {
            for index in 0..size {
                z[index] = zero;
            }
            self.v_cycle(0, &r, &mut z, workspace);
        }

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
        output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let length = residual.shape()[0];
        if output.shape()[0] != length {
            return Err(LetoBackendError::LengthMismatch {
                left: length,
                right: output.shape()[0],
            });
        }
        self.cycle(residual, output).map_err(LetoBackendError::Leto)
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

    #[test]
    fn borrowed_cycle_preserves_previous_output_bits() {
        // Captured from the owned cycle at d07c9ddec854cfe9f72add038d69c093f94968f7.
        let expected = include_bytes!("reference/v_cycle.bits").as_chunks::<8>().0;
        let dimension = expected.len();
        let matrix = tridiagonal!(f64, dimension);
        let amg = AlgebraicMultigrid::new(&matrix, AMGConfig::default())
            .expect("invariant: tridiagonal SPD system assembles");
        assert_eq!(amg.levels.len(), 3);
        let backend = LetoBackend::<f64>::default();

        for stride in [1usize, 2] {
            let extent = dimension * stride + 1;
            let mut rhs = Array1::zeros([extent]);
            let mut output = Array1::zeros([extent]);
            output.fill(-1.0);
            for index in 0..dimension {
                rhs[1 + index * stride] =
                    (f64::from(u32::try_from(index).expect("invariant: fixture index fits")) + 1.0)
                        * 0.5;
            }
            let range = [(
                1,
                extent,
                isize::try_from(stride).expect("invariant: stride fits"),
            )];
            Preconditioner::apply(
                &amg,
                &backend,
                rhs.slice(&range).expect("invariant: residual range fits"),
                output
                    .slice_mut(&range)
                    .expect("invariant: output range fits"),
            )
            .expect("invariant: both views match the system");
            for (index, entry) in output.iter().enumerate() {
                let bits = if index > 0 && (index - 1).is_multiple_of(stride) {
                    u64::from_le_bytes(expected[(index - 1) / stride])
                } else {
                    (-1.0_f64).to_bits()
                };
                assert_eq!(
                    entry.to_bits(),
                    bits,
                    "stride {stride}, storage index {index}"
                );
            }
        }
        let mut invalid = Array1::from_shape_vec([1], vec![3.0])
            .expect("invariant: backing length matches shape");
        let residual = invalid.clone();
        assert!(matches!(
            Preconditioner::apply(&amg, &backend, residual.view(), invalid.view_mut()),
            Err(crate::LetoBackendError::Leto(
                leto::LetoError::InvalidInput(_)
            ))
        ));
        assert_eq!(invalid[0].to_bits(), 3.0_f64.to_bits());
    }
}
