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
///
/// The buffers are plain contiguous `Vec`s, not `Array1`s: the cycle and the
/// smoothers walk them as slices, so a rank-one layout wrapper would only be
/// rebuilt on every access.
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
    ///
    /// Runs entirely on borrowed slices: the caller's residual and output are
    /// passed straight through, so the seam copies nothing in or out.
    fn v_cycle(
        &self,
        level_idx: usize,
        b: &[T],
        x: &mut [T],
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
                &current_workspace.residual,
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
    /// # Errors
    ///
    /// Returns [`leto::LetoError::InvalidInput`] when `r` and `z` disagree in
    /// shape, or when the cached level workspace lock is poisoned.
    ///
    /// # Panics
    ///
    /// Panics never: the cycle's internal failures are invariant-guarded
    /// (workspace matches the assembled hierarchy by construction).
    pub fn apply_to(&self, r: &Array1<T>, z: &mut Array1<T>) -> Result<(), leto::LetoError> {
        let residual = r.as_slice().ok_or_else(|| non_contiguous("residual"))?;
        let output = z.as_slice_mut().ok_or_else(|| non_contiguous("output"))?;
        self.cycle(residual, output)
    }

    /// Apply one cycle over borrowed slices: `z = M^-1 r`.
    ///
    /// This is the cycle proper. [`Self::apply_to`] and the Athena
    /// [`Preconditioner`] seam both delegate here, so every caller — an owned
    /// array, a view, or a plain buffer — runs the same body with no copy in
    /// or out.
    fn cycle(&self, r: &[T], z: &mut [T]) -> Result<(), leto::LetoError> {
        if r.len() != z.len() {
            return Err(leto::LetoError::InvalidInput(format!(
                "AMG preconditioner vector length mismatch: {} != {}",
                r.len(),
                z.len()
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
        z.fill(<T as NumericElement>::ZERO);
        self.v_cycle(0, r, z, workspace);

        Ok(())
    }
}

/// The typed error for a vector the cycle cannot address as one slice.
fn non_contiguous(role: &str) -> leto::LetoError {
    leto::LetoError::InvalidInput(format!(
        "AMG {role} vector must be contiguous to run the cycle in place"
    ))
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
        let residual_slice = residual
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output_slice = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        self.cycle(residual_slice, output_slice)
            .map_err(LetoBackendError::Leto)
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

    /// The Athena seam and the public owned entry point return the same bytes.
    ///
    /// Both delegate to one slice-based cycle, so the seam is a
    /// view-to-slice hand-off and nothing else: no copy can change a result,
    /// and the shared body is the one the deletion of the boundary buffers
    /// left behind.
    ///
    /// The `digest` is the pre-change cycle's own output over this fixture at
    /// origin/main `bd349c5e` (measured before the slice migration): the
    /// arithmetic order is unchanged, so the value must not move.
    #[test]
    fn seam_matches_owned_cycle_bitwise() {
        let dimension = 128;
        let matrix = tridiagonal!(f64, dimension);
        let amg = AlgebraicMultigrid::new(&matrix, AMGConfig::default())
            .expect("invariant: tridiagonal SPD system assembles");
        assert_eq!(
            amg.levels.len(),
            3,
            "invariant: fixture must exercise a multi-level cycle"
        );

        let values: Vec<f64> = (0..dimension)
            .map(|index| {
                let index = u32::try_from(index).expect("invariant: fixture index fits in u32");
                (f64::from(index) + 1.0) * 0.5
            })
            .collect();
        let rhs = Array1::from_shape_vec([dimension], values).expect("rhs shape matches");

        let mut through_seam = Array1::zeros([dimension]);
        Preconditioner::apply(
            &amg,
            &LetoBackend::<f64>::default(),
            rhs.view(),
            through_seam.view_mut(),
        )
        .expect("invariant: shapes agree");

        let mut through_owned = Array1::zeros([dimension]);
        amg.apply_to(&rhs, &mut through_owned)
            .expect("invariant: shapes agree");

        for (index, (seam, owned)) in through_seam.iter().zip(through_owned.iter()).enumerate() {
            assert_eq!(
                seam.to_bits(),
                owned.to_bits(),
                "seam and owned cycle disagree at index {index}"
            );
        }

        let digest = through_owned
            .iter()
            .fold(0u64, |acc, value| acc.wrapping_add(value.to_bits()));
        assert_eq!(
            digest, 1_457_391_189_608_935_786,
            "cycle must reproduce the pre-change output bitwise"
        );
    }
}
