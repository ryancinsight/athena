//! The chain's exact direct tier: sparse LU with a small-system dense retry.

use crate::LetoBackendError;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar, SparseLuSolver, solve_csr_via_dense_lu};
use std::fmt::Debug;

/// Dense-LU retry ceiling for the direct tier.
///
/// The dense path costs `O(n³)` work; at 1024 unknowns that is ~10⁹ fused
/// multiply-adds at `f64` — the order of a few failed sparse attempts — so
/// a small system whose sparse factorisation failed still gets its preferred
/// exact solve instead of falling to the iterative tiers.
const DENSE_RETRY_CEILING: usize = 1024;

/// Solve exactly with the atlas-native sparse LU, retrying small systems
/// through the dense bridge.
///
/// The retry is the chain's user-intent safety net, not a duplicate of the
/// sparse solver's internal dense dispatch (ADR 0031): it catches the
/// orthogonal case where the sparse path refuses or fails and the system is
/// small enough that the dense bridge's exact solve is still the cheapest
/// correct answer. Both paths verify the returned solution is finite.
///
/// # Errors
///
/// Propagates the sparse or dense failure when both paths fail, and
/// [`LetoBackendError::Leto`] naming the non-finite entry when a returned
/// solution carries one.
pub(super) fn solve_direct_with_dense_retry<T>(
    matrix: &CsrMatrix<T>,
    rhs: &Array1<T>,
    max_size: usize,
) -> Result<Array1<T>, LetoBackendError>
where
    T: RealField + Copy + FloatElement + RealScalar + Debug,
{
    let direct = SparseLuSolver {
        max_size,
        ..Default::default()
    };
    match direct.solve_view(matrix, &rhs.view()) {
        Ok(solution) => {
            ensure_finite_solution(&solution)?;
            Ok(solution)
        }
        Err(sparse_error) => {
            if matrix.nrows() <= DENSE_RETRY_CEILING {
                tracing::warn!(
                    size = matrix.nrows(),
                    "LinearSolverChain: sparse LU failed ({sparse_error}); retrying dense"
                );
                let dense = solve_csr_via_dense_lu(matrix, rhs).map_err(|dense_error| {
                    LetoBackendError::Leto(leto::LetoError::InvalidInput(format!(
                        "direct solve failed: sparse: {sparse_error}; dense: {dense_error}"
                    )))
                })?;
                ensure_finite_solution(&dense)?;
                return Ok(dense);
            }
            Err(LetoBackendError::Leto(leto::LetoError::InvalidInput(
                format!("direct solve failed: {sparse_error}"),
            )))
        }
    }
}

/// Reject a direct-solve result carrying a non-finite entry.
///
/// # Errors
///
/// Returns [`LetoBackendError::Leto`] naming the offending index.
pub(super) fn ensure_finite_solution<T: RealField + Copy + FloatElement>(
    solution: &Array1<T>,
) -> Result<(), LetoBackendError> {
    for (index, value) in solution.iter().enumerate() {
        if !NumericElement::to_f64(*value).is_finite() {
            return Err(LetoBackendError::Leto(leto::LetoError::InvalidInput(
                format!("direct solve entry {index} is not finite"),
            )));
        }
    }
    Ok(())
}
