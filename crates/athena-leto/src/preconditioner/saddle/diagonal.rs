//! Diagonal and block-diagonal preconditioners for saddle-point systems.
//!
//! [`DiagonalPreconditioner`] is this family's tolerant inverse-diagonal
//! storage: a zero or absent diagonal entry maps to identity rather than an
//! error, because saddle systems legitimately carry such rows before
//! stabilisation. It is deliberately not the public [`Jacobi`](super::Jacobi),
//! whose contract rejects zero diagonals; the saddle family needs the
//! per-index inverse array and the tolerant fallback, and nothing outside this
//! module consumes it as a standalone preconditioner.

use athena_core::{KrylovBackend, Preconditioner};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

use super::matrix::{diagonal_epsilon, from_usize, get_diagonal};
use crate::{LetoBackend, LetoBackendError};

/// Simple diagonal preconditioner (Jacobi with a tolerant fallback).
pub(crate) struct DiagonalPreconditioner<T: RealField> {
    /// Inverse of diagonal entries
    diag_inv: Array1<T>,
}

impl<T: RealField + FloatElement + RealScalar> DiagonalPreconditioner<T> {
    /// Create a diagonal preconditioner from the matrix diagonal.
    #[cfg(test)]
    ///
    /// A zero or absent diagonal entry yields identity for that index: the
    /// saddle systems this serves can carry structurally empty rows that
    /// [`stabilize_preconditioner_diagonal`](super::matrix::stabilize_preconditioner_diagonal)
    /// handles at the assembled-matrix level.
    pub(super) fn new(matrix: &CsrMatrix<T>) -> Self {
        let rows = matrix.nrows();
        let mut diag_inv = Array1::zeros([rows]);
        for index in 0..rows {
            let d = get_diagonal(matrix, index);
            diag_inv[index] = if NumericElement::abs(d) > diagonal_epsilon() {
                <T as NumericElement>::ONE / d
            } else {
                <T as NumericElement>::ONE
            };
        }
        Self { diag_inv }
    }

    /// The inverse-diagonal entries.
    pub(super) const fn diag_inv(&self) -> &Array1<T> {
        &self.diag_inv
    }

    /// Wrap a precomputed inverse-diagonal array.
    pub(super) fn from_inverse_diagonal(diag_inv: Array1<T>) -> Self {
        Self { diag_inv }
    }
}

/// Build a pressure preconditioner from signed row-sum lumping of a Schur
/// approximation. Preserving the sign is required for the indefinite saddle
/// system; if row entries cancel, the raw diagonal supplies the signed scale.
pub(super) fn lumped_pressure_preconditioner<T: RealField + Copy + RealScalar>(
    matrix: &CsrMatrix<T>,
) -> DiagonalPreconditioner<T> {
    let rows = matrix.nrows();
    let mut diag_inv = Array1::zeros([rows]);
    for row_index in 0..rows {
        let row_sum = matrix
            .row(row_index)
            .values()
            .iter()
            .copied()
            .fold(<T as NumericElement>::ZERO, |sum, value| sum + value);
        let scale = if NumericElement::abs(row_sum) > diagonal_epsilon() {
            row_sum
        } else {
            get_diagonal(matrix, row_index)
        };
        diag_inv[row_index] = if NumericElement::abs(scale) > diagonal_epsilon() {
            <T as NumericElement>::ONE / scale
        } else {
            <T as NumericElement>::ONE
        };
    }
    DiagonalPreconditioner { diag_inv }
}

/// Block diagonal preconditioner for saddle-point systems
///
/// ```text
/// P = [ A_inv    0   ]
///     [ 0      S_inv ]
/// ```
///
/// where `S ≈ B A^{-1} B^T` (Schur complement approximation).
///
/// # Algorithm
///
/// For solving `P x = b`:
/// 1. Solve `A u_tilde = f` (momentum block)
/// 2. Solve `S p = g - B u_tilde` (pressure Schur complement)
/// 3. Solve `A u = f - B^T p` (pressure correction)
///
/// # Theorem (Block-Diagonal Preconditioning of Saddle-Point Systems)
///
/// For the saddle-point system with SPD momentum block `A` and full-rank `B`,
/// the exact block-diagonal preconditioner `P = diag(A, S)` with
/// `S = B A⁻¹ Bᵀ` yields a preconditioned system whose eigenvalues lie in
/// `{1} ∪ [1/φ, φ]` (with `φ` the golden ratio), so GMRES converges in at
/// most 3 iterations. When the Schur complement is approximated by
/// `S̃ ≈ B diag(A)⁻¹ Bᵀ`, the spectral bounds degrade gracefully with the
/// quality of the diagonal approximation.
///
/// **Proof sketch**: the preconditioned matrix satisfies
/// `(P⁻¹𝒜)³ = (P⁻¹𝒜)²` when `S` is exact, so the eigenvalues are the roots of
/// `λ² − λ − 1 = 0` plus `λ = 1`.
///
/// **Reference**: Murphy, Golub & Wathen (2000), Theorem 2.1; Elman,
/// Silvester & Wathen (2005), §6.2.
pub struct BlockDiagonalPreconditioner<T: RealField + FloatElement> {
    /// Momentum matrix approximation (diagonal of A)
    momentum_preconditioner: DiagonalPreconditioner<T>,
    /// Pressure Schur complement approximation (pressure mass matrix)
    pressure_preconditioner: DiagonalPreconditioner<T>,
    /// Velocity DOF count
    n_velocity: usize,
    /// Pressure DOF count
    n_pressure: usize,
}

impl<T: RealField + FloatElement + Copy + RealScalar> BlockDiagonalPreconditioner<T> {
    /// Create a block diagonal preconditioner.
    ///
    /// # Algorithm
    ///
    /// 1. Extract the momentum block diagonal (A block)
    /// 2. Approximate the Schur complement `S ≈ ν M_p` by signed row-sum
    ///    lumping of the pressure-pressure block; when a pressure row lumps
    ///    to zero, fall back to the average momentum diagonal (viscosity
    ///    scaling)
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::LengthMismatch`] when the matrix order is
    /// not `n_velocity + n_pressure`.
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

        // Momentum block diagonal (A block)
        let mut momentum_diag = Array1::zeros([n_velocity]);
        for index in 0..n_velocity {
            let d = get_diagonal(matrix, index);
            momentum_diag[index] = if NumericElement::abs(d) > diagonal_epsilon() {
                d
            } else {
                <T as NumericElement>::ONE
            };
        }

        // Approximate the Schur complement with the viscosity-scaled pressure
        // mass matrix: M_p_ii ≈ Σ_j |M_p_ij| (row-sum lumping).
        let mut pressure_diag = Array1::zeros([n_pressure]);
        for index in 0..n_pressure {
            let row = matrix.row(n_velocity + index);
            let mut row_sum = <T as NumericElement>::ZERO;
            for (&column, &value) in row.col_indices().iter().zip(row.values()) {
                if column >= n_velocity {
                    row_sum += NumericElement::abs(value);
                }
            }
            if row_sum > diagonal_epsilon() {
                pressure_diag[index] = row_sum;
            } else {
                // Fallback: viscosity-based scaling from the momentum block.
                let mut sum = <T as NumericElement>::ZERO;
                let mut count = 0usize;
                for idx in 0..n_velocity {
                    let value = NumericElement::abs(momentum_diag[idx]);
                    if value > diagonal_epsilon() {
                        sum += value;
                        count += 1;
                    }
                }
                let avg_momentum_diag = if count == 0 {
                    <T as NumericElement>::ONE
                } else {
                    sum / from_usize(count)
                };
                pressure_diag[index] = avg_momentum_diag;
            }
        }

        let mut momentum_diag_inv = Array1::zeros([n_velocity]);
        for idx in 0..n_velocity {
            let d = momentum_diag[idx];
            momentum_diag_inv[idx] = if NumericElement::abs(d) > diagonal_epsilon() {
                <T as NumericElement>::ONE / d
            } else {
                <T as NumericElement>::ONE
            };
        }

        let mut pressure_diag_inv = Array1::zeros([n_pressure]);
        for idx in 0..n_pressure {
            let d = pressure_diag[idx];
            pressure_diag_inv[idx] = if NumericElement::abs(d) > diagonal_epsilon() {
                <T as NumericElement>::ONE / d
            } else {
                <T as NumericElement>::ONE
            };
        }

        Ok(Self {
            momentum_preconditioner: DiagonalPreconditioner {
                diag_inv: momentum_diag_inv,
            },
            pressure_preconditioner: DiagonalPreconditioner {
                diag_inv: pressure_diag_inv,
            },
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
}

impl<T> Preconditioner<LetoBackend<T>> for BlockDiagonalPreconditioner<T>
where
    T: RealField + FloatElement + Copy + RealScalar,
{
    /// Block-diagonal apply straight over the borrowed views.
    ///
    /// The velocity and pressure blocks are contiguous index ranges, so this
    /// needs no scratch and no copy.
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
            output[index] = residual[index] * self.momentum_preconditioner.diag_inv()[index];
        }
        for index in 0..self.n_pressure {
            let offset = self.n_velocity + index;
            output[offset] = residual[offset] * self.pressure_preconditioner.diag_inv()[index];
        }
        Ok(())
    }
}
