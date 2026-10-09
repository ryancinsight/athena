//! SIMPLE preconditioner (Semi-Implicit Method for Pressure-Linked
//! Equations).
//!
//! More sophisticated than block diagonal: it uses the coupling between `u`
//! and `p`.
//!
//! # Algorithm
//!
//! For
//! `[[A, G], [D, C]] · [u, p] = [f, g]`:
//!
//! 1. Solve `A u* = f` (momentum prediction via diagonal approximation)
//! 2. Solve `S p = g − D u*` (pressure Schur equation)
//! 3. Correct `u = u* − diag(A)⁻¹ G p`
//!
//! where `S ≈ D diag(A)⁻¹ G` (Schur complement).
//!
//! The diagonal Schur approximation is a block-structured heuristic. Its
//! effectiveness depends on the momentum block, coupling signs,
//! stabilisation, and pressure nullspace; convergence is verified by the
//! enclosing Krylov solver rather than guaranteed by this preconditioner
//! alone.
//!
//! # Reference
//!
//! Patankar, S. V. (1980): *Numerical Heat Transfer and Fluid Flow*.

use athena_core::{KrylovBackend, Preconditioner};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

use super::diagonal::DiagonalPreconditioner;
use super::matrix::{diagonal_epsilon, extract_block, get_diagonal};
use crate::{LetoBackend, LetoBackendError};

/// SIMPLE preconditioner with momentum-pressure coupling.
pub struct SimplePreconditioner<T: RealField + FloatElement> {
    /// Momentum block preconditioner (diag(A)⁻¹)
    momentum_inv: DiagonalPreconditioner<T>,
    /// Inverse diagonal of the Schur complement approximation
    schur_diag_inv: Array1<T>,
    /// Divergence block `D` (pressure rows, velocity columns), used for the
    /// `D u*` product.
    divergence: CsrMatrix<T>,
    /// Gradient block `G` (velocity rows, pressure columns), used for the
    /// `G p` correction. Stored by velocity row so the correction gathers
    /// into each velocity entry instead of scattering from pressure columns.
    gradient: CsrMatrix<T>,
    n_velocity: usize,
    n_pressure: usize,
}

impl<T: RealField + FloatElement + Copy + RealScalar> SimplePreconditioner<T> {
    /// Create a SIMPLE preconditioner.
    ///
    /// Extracts the `A`, `D`, and `G` sub-blocks from the full saddle-point
    /// matrix and builds the diagonal Schur complement
    /// `diag(C − D diag(A)⁻¹ G)`.
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::LengthMismatch`] when the matrix order is
    /// not `n_velocity + n_pressure`, and the leto block-extraction error
    /// otherwise.
    pub fn new(
        matrix: &CsrMatrix<T>,
        n_velocity: usize,
        n_pressure: usize,
    ) -> Result<Self, LetoBackendError> {
        if matrix.nrows() != n_velocity + n_pressure {
            return Err(LetoBackendError::LengthMismatch {
                left: matrix.nrows(),
                right: n_velocity + n_pressure,
            });
        }
        let eps = diagonal_epsilon();

        // Momentum diagonal and its inverse.
        let mut momentum_diag = Array1::zeros([n_velocity]);
        for index in 0..n_velocity {
            momentum_diag[index] = get_diagonal(matrix, index);
        }
        let mut momentum_diag_inv = Array1::zeros([n_velocity]);
        for index in 0..n_velocity {
            let d = momentum_diag[index];
            momentum_diag_inv[index] = if NumericElement::abs(d) > eps {
                <T as NumericElement>::ONE / d
            } else {
                <T as NumericElement>::ONE
            };
        }
        let momentum_inv = DiagonalPreconditioner::from_inverse_diagonal(momentum_diag_inv);

        // Extract the coupling blocks independently. The continuity row may
        // be scaled by the formulation, so `G` is not reconstructed from `D`.
        let divergence = extract_block(matrix, n_velocity..n_velocity + n_pressure, 0, n_velocity)?;
        let gradient = extract_block(matrix, 0..n_velocity, n_velocity, n_pressure)?;

        // Compute diag(C − D diag(A)⁻¹ G) for the actual assembled coupling
        // signs. This remains valid when the continuity row is normalized and
        // retains the pressure stabilisation block C.
        let mut schur_diag_inv = Array1::zeros([n_pressure]);
        for index in 0..n_pressure {
            let mut s_ii = get_diagonal(matrix, n_velocity + index);
            let row = divergence.row(index);
            for (&velocity_index, &divergence_value) in row.col_indices().iter().zip(row.values()) {
                if let Some(gradient_value) = gradient.get(velocity_index, index) {
                    s_ii -=
                        divergence_value * momentum_inv.diag_inv()[velocity_index] * gradient_value;
                }
            }
            schur_diag_inv[index] = if NumericElement::abs(s_ii) > eps {
                <T as NumericElement>::ONE / s_ii
            } else {
                <T as NumericElement>::ONE
            };
        }

        Ok(Self {
            momentum_inv,
            schur_diag_inv,
            divergence,
            gradient,
            n_velocity,
            n_pressure,
        })
    }

    /// The velocity DOF count.
    #[must_use]
    pub const fn n_velocity(&self) -> usize {
        self.n_velocity
    }

    /// The pressure DOF count.
    #[must_use]
    pub const fn n_pressure(&self) -> usize {
        self.n_pressure
    }

    /// The divergence block `D` (pressure rows, velocity columns).
    #[must_use]
    pub(crate) const fn divergence(&self) -> &CsrMatrix<T> {
        &self.divergence
    }

    /// The gradient block `G` (velocity rows, pressure columns).
    #[must_use]
    pub(crate) const fn gradient(&self) -> &CsrMatrix<T> {
        &self.gradient
    }

    /// The momentum block's inverse diagonal.
    #[must_use]
    pub(crate) const fn momentum_inv(&self) -> &DiagonalPreconditioner<T> {
        &self.momentum_inv
    }
}

impl<T> Preconditioner<LetoBackend<T>> for SimplePreconditioner<T>
where
    T: RealField + FloatElement + Copy + RealScalar,
{
    /// Apply SIMPLE directly to Athena's borrowed vectors.
    ///
    /// The velocity part of `output` first stores the diagonal momentum
    /// prediction. Pressure is then formed from the borrowed divergence rows,
    /// and the same output buffer receives the pressure correction. This
    /// preserves the SIMPLE recurrence without allocating an intermediate
    /// vector on the Krylov hot path.
    fn apply(
        &self,
        _backend: &LetoBackend<T>,
        residual: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let expected = self.n_velocity + self.n_pressure;
        let residual = residual
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        if residual.len() != expected {
            return Err(LetoBackendError::LengthMismatch {
                left: residual.len(),
                right: expected,
            });
        }
        if output.len() != expected {
            return Err(LetoBackendError::LengthMismatch {
                left: output.len(),
                right: expected,
            });
        }

        for index in 0..self.n_velocity {
            output[index] = residual[index] * self.momentum_inv.diag_inv()[index];
        }
        for pressure_index in 0..self.n_pressure {
            let mut pressure_rhs = residual[self.n_velocity + pressure_index];
            let row = self.divergence.row(pressure_index);
            for (&velocity_index, &divergence_value) in row.col_indices().iter().zip(row.values()) {
                pressure_rhs -= divergence_value * output[velocity_index];
            }
            let pressure_offset = self.n_velocity + pressure_index;
            output[pressure_offset] = pressure_rhs * self.schur_diag_inv[pressure_index];
        }
        for velocity_index in 0..self.n_velocity {
            let row = self.gradient.row(velocity_index);
            for (&pressure_index, &gradient_value) in row.col_indices().iter().zip(row.values()) {
                let pressure = output[self.n_velocity + pressure_index];
                output[velocity_index] -=
                    self.momentum_inv.diag_inv()[velocity_index] * gradient_value * pressure;
            }
        }
        Ok(())
    }
}
