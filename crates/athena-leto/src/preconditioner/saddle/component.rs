//! Component-block sparse-LU preconditioner for a three-dimensional saddle
//! system.
//!
//! The velocity block in `CFDrs` is component-major. When the assembled operator
//! has no cross-component velocity entries, its momentum block is the direct
//! sum of three scalar CSR operators. This preconditioner factors each scalar
//! block once with the provider-owned Leto sparse LU implementation and
//! combines those solves with the independently assembled SIMPLE pressure
//! correction.
//!
//! The constructor rejects cross-component entries. Applying only diagonal
//! component blocks to a coupled operator would change the preconditioner
//! contract while appearing to succeed, so such systems fall through to the
//! general saddle-point tiers.

use athena_core::{KrylovBackend, Preconditioner};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, LetoError};
use leto_ops::{
    CscMatrix, CsrMatrix, OwnedNumericLu, RealScalar, SparseLuSolver, SymbolicLu, factor_symbolic,
};

use super::diagonal::{DiagonalPreconditioner, lumped_pressure_preconditioner};
use super::matrix::{RowBuilder, diagonal_epsilon, stabilize_preconditioner_diagonal};
use super::simple::SimplePreconditioner;
use crate::{LetoBackend, LetoBackendError};

/// Number of velocity components in the three-dimensional Taylor–Hood system.
pub(crate) const VELOCITY_COMPONENTS: usize = 3;

/// Component-block sparse-LU preconditioner for a three-dimensional saddle
/// system.
pub struct ComponentBlockPreconditioner<T: RealField + FloatElement + RealScalar> {
    momentum_blocks: Vec<OwnedNumericLu<T>>,
    pressure_block: PressureBlock<T>,
    simple: SimplePreconditioner<T>,
    component_size: usize,
    n_velocity: usize,
    n_pressure: usize,
}

enum PressureBlock<T: RealField + FloatElement + RealScalar> {
    LumpedDiagonal(DiagonalPreconditioner<T>),
}

/// Cached provider symbolic factors for an unchanged reduced FEM topology.
#[derive(Debug, Clone)]
pub struct ComponentBlockPattern {
    component_size: usize,
    n_velocity: usize,
    n_pressure: usize,
    row_ptr: Vec<usize>,
    col_indices: Vec<usize>,
    symbols: Vec<SymbolicLu>,
}

impl ComponentBlockPattern {
    /// Whether the cached pattern matches this matrix and block split.
    #[must_use]
    pub fn matches<T: RealField + Copy + RealScalar>(
        &self,
        matrix: &CsrMatrix<T>,
        n_velocity: usize,
        n_pressure: usize,
    ) -> bool {
        self.component_size * VELOCITY_COMPONENTS == n_velocity
            && self.n_velocity == n_velocity
            && self.n_pressure == n_pressure
            && self.row_ptr == matrix.row_ptr()
            && self.col_indices == matrix.col_indices()
    }
}

impl<T> ComponentBlockPreconditioner<T>
where
    T: RealField + FloatElement + Copy + RealScalar,
{
    /// Factor the independent velocity component blocks of `matrix`.
    ///
    /// # Errors
    ///
    /// Returns an error when the matrix dimensions are inconsistent, when a
    /// nonzero cross-component velocity entry is present, or when a component
    /// block cannot be factored by the provider sparse LU implementation.
    pub fn new(
        matrix: &CsrMatrix<T>,
        n_velocity: usize,
        n_pressure: usize,
    ) -> Result<Self, LetoBackendError> {
        Self::new_with_symbols(matrix, n_velocity, n_pressure, None)
            .map(|(preconditioner, _)| preconditioner)
    }

    /// Construct while reusing symbolic factorization for an unchanged mesh.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn new_with_cache(
        matrix: &CsrMatrix<T>,
        n_velocity: usize,
        n_pressure: usize,
        cache: &mut Option<ComponentBlockPattern>,
    ) -> Result<Self, LetoBackendError> {
        let cache_matches = cache
            .as_ref()
            .is_some_and(|pattern| pattern.matches(matrix, n_velocity, n_pressure));
        let (preconditioner, symbols) = {
            let cached_symbols = cache
                .as_ref()
                .filter(|_| cache_matches)
                .map(|pattern| pattern.symbols.as_slice());
            Self::new_with_symbols(matrix, n_velocity, n_pressure, cached_symbols)?
        };
        if !cache_matches {
            *cache = Some(ComponentBlockPattern {
                component_size: n_velocity / VELOCITY_COMPONENTS,
                n_velocity,
                n_pressure,
                row_ptr: matrix.row_ptr().to_vec(),
                col_indices: matrix.col_indices().to_vec(),
                symbols,
            });
        }
        Ok(preconditioner)
    }

    fn new_with_symbols(
        matrix: &CsrMatrix<T>,
        n_velocity: usize,
        n_pressure: usize,
        cached_symbols: Option<&[SymbolicLu]>,
    ) -> Result<(Self, Vec<SymbolicLu>), LetoBackendError> {
        if matrix.nrows() != n_velocity + n_pressure {
            return Err(LetoBackendError::LengthMismatch {
                left: matrix.nrows(),
                right: n_velocity + n_pressure,
            });
        }
        if n_velocity == 0 || !n_velocity.is_multiple_of(VELOCITY_COMPONENTS) {
            return Err(LetoError::InvalidInput(format!(
                "velocity DOF count {n_velocity} is not divisible by {VELOCITY_COMPONENTS}"
            ))
            .into());
        }

        let component_size = n_velocity / VELOCITY_COMPONENTS;
        let mut momentum_blocks = Vec::with_capacity(VELOCITY_COMPONENTS);
        let mut symbols = Vec::with_capacity(VELOCITY_COMPONENTS);

        for component in 0..VELOCITY_COMPONENTS {
            let offset = component * component_size;
            let mut block = RowBuilder::new(component_size, component_size);
            for local_row in 0..component_size {
                let global_row = offset + local_row;
                let row = matrix.row(global_row);
                for (&global_col, &value) in row.col_indices().iter().zip(row.values()) {
                    if global_col >= n_velocity {
                        continue;
                    }
                    let column_component = global_col / component_size;
                    if column_component != component {
                        if NumericElement::abs(value) > diagonal_epsilon() {
                            return Err(LetoError::InvalidInput(format!(
                                "velocity component coupling ({global_row}, {global_col}) \
                                 is not supported by component-block preconditioning"
                            ))
                            .into());
                        }
                        continue;
                    }
                    block.add_entry(local_row, global_col - offset, value)?;
                }
            }

            let solver = SparseLuSolver {
                max_size: component_size,
                ..SparseLuSolver::default()
            };
            let block = stabilize_preconditioner_diagonal(&block.build()?, solver.pivot_tolerance)?;
            let symbol = cached_symbols
                .and_then(|symbols| symbols.get(component))
                .cloned()
                .unwrap_or_else(|| factor_symbolic(&CscMatrix::from_csr(&block.as_view())));
            let factor = solver.factor_sparse_with_symbolic(&block, &symbol)?;
            symbols.push(symbol);
            momentum_blocks.push(factor);
        }

        let simple = SimplePreconditioner::new(matrix, n_velocity, n_pressure)?;
        let mut pressure_matrix = RowBuilder::new(n_pressure, n_pressure);
        for pressure_row in 0..n_pressure {
            let row = matrix.row(n_velocity + pressure_row);
            for (&column, &value) in row.col_indices().iter().zip(row.values()) {
                if let Some(pressure_column) = column
                    .checked_sub(n_velocity)
                    .filter(|&index| index < n_pressure)
                {
                    pressure_matrix.add_entry(pressure_row, pressure_column, value)?;
                }
            }
            let divergence = simple.divergence().row(pressure_row);
            for (&velocity, &divergence_value) in
                divergence.col_indices().iter().zip(divergence.values())
            {
                let momentum_inverse = simple.momentum_inv().diag_inv()[velocity];
                let gradient = simple.gradient().row(velocity);
                for (&pressure_column, &gradient_value) in
                    gradient.col_indices().iter().zip(gradient.values())
                {
                    pressure_matrix.add_entry(
                        pressure_row,
                        pressure_column,
                        -divergence_value * momentum_inverse * gradient_value,
                    )?;
                }
            }
        }
        let pressure_matrix = stabilize_preconditioner_diagonal(
            &pressure_matrix.build()?,
            SparseLuSolver::default().pivot_tolerance,
        )?;

        let pressure_block =
            PressureBlock::LumpedDiagonal(lumped_pressure_preconditioner(&pressure_matrix));

        Ok((
            Self {
                momentum_blocks,
                pressure_block,
                simple,
                component_size,
                n_velocity,
                n_pressure,
            },
            symbols,
        ))
    }
}

impl<T> Preconditioner<LetoBackend<T>> for ComponentBlockPreconditioner<T>
where
    T: RealField + FloatElement + Copy + RealScalar,
{
    /// Apply component sparse-LU solves directly to Athena's borrowed vectors.
    fn apply(
        &self,
        _backend: &LetoBackend<T>,
        residual: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let expected = self.n_velocity + self.n_pressure;
        if residual.shape()[0] != expected {
            return Err(LetoBackendError::LengthMismatch {
                left: residual.shape()[0],
                right: expected,
            });
        }
        if output.shape()[0] != expected {
            return Err(LetoBackendError::LengthMismatch {
                left: output.shape()[0],
                right: expected,
            });
        }

        let mut block_rhs = Array1::zeros([self.component_size]);
        let mut block_solution = Array1::zeros([self.component_size]);
        for (component, factor) in self.momentum_blocks.iter().enumerate() {
            let offset = component * self.component_size;
            for local in 0..self.component_size {
                block_rhs[local] = residual[offset + local];
            }
            factor.solve_into(&block_rhs.view(), &mut block_solution.view_mut())?;
            for local in 0..self.component_size {
                output[offset + local] = block_solution[local];
            }
        }

        let mut pressure_rhs = Array1::zeros([self.n_pressure]);
        for pressure_index in 0..self.n_pressure {
            pressure_rhs[pressure_index] = residual[self.n_velocity + pressure_index];
            let mut divergence_u = <T as NumericElement>::ZERO;
            let row = self.simple.divergence().row(pressure_index);
            for (&velocity, &value) in row.col_indices().iter().zip(row.values()) {
                divergence_u += value * output[velocity];
            }
            pressure_rhs[pressure_index] -= divergence_u;
        }

        let pressure = match &self.pressure_block {
            PressureBlock::LumpedDiagonal(preconditioner) => {
                let mut pressure = Array1::zeros([self.n_pressure]);
                for index in 0..self.n_pressure {
                    pressure[index] = pressure_rhs[index] * preconditioner.diag_inv()[index];
                }
                pressure
            }
        };

        for pressure_index in 0..self.n_pressure {
            output[self.n_velocity + pressure_index] = pressure[pressure_index];
        }

        // Velocity correction through the exact factored component blocks:
        // u = u* - A^-1 (G p), the sparse-LU counterpart of the momentum
        // prediction above.
        for (component, factor) in self.momentum_blocks.iter().enumerate() {
            let offset = component * self.component_size;
            block_rhs.fill(<T as NumericElement>::ZERO);
            for local in 0..self.component_size {
                let row = self.simple.gradient().row(offset + local);
                for (&pressure_index, &value) in row.col_indices().iter().zip(row.values()) {
                    block_rhs[local] += value * output[self.n_velocity + pressure_index];
                }
            }
            factor.solve_into(&block_rhs.view(), &mut block_solution.view_mut())?;
            for local in 0..self.component_size {
                output[offset + local] -= block_solution[local];
            }
        }
        Ok(())
    }
}
