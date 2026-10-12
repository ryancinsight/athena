//! Dense LU with partial pivoting for the small coarse-space system
//! `Uᵀ A U` of a deflation basis.
//!
//! The coarse matrix is `k × k` with `k` the deflation-subspace dimension —
//! a handful in practice — so a textbook factorisation is the right tool: the
//! allocation is one `k²` buffer at construction and each solve is
//! `O(k³)`/`O(k²)` scalar work on the host.

use eunomia::RealField;

/// Why a coarse-space factorisation was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CoarseFailure<T> {
    /// A pivot column vanished: the coarse matrix is singular.
    ZeroPivot {
        /// The step at which the elimination lost rank.
        step: usize,
        /// The largest remaining candidate magnitude.
        best: T,
    },
}

/// Factorise `matrix` in place with partial pivoting.
///
/// Row interchanges follow the textbook Doolittle scheme with a pivot search
/// (Golub & Van Loan, *Matrix Computations*, §3.2.1, Algorithm 3.2.1): at
/// each step the row holding the largest-magnitude candidate is moved to the
/// pivot position, so the factorisation is backward stable for the small,
/// well-conditioned coarse matrices deflation produces.
///
/// # Errors
///
/// Returns [`CoarseFailure::ZeroPivot`] when every remaining candidate at a
/// step is zero: the coarse matrix is singular, and no factorisation exists.
pub(super) fn factorise<T: RealField>(
    matrix: &mut [T],
    order: usize,
    pivots: &mut [usize],
) -> Result<(), CoarseFailure<T>> {
    debug_assert_eq!(matrix.len(), order * order);
    debug_assert_eq!(pivots.len(), order);
    let zero = T::ZERO;

    for (step, pivot) in pivots.iter_mut().enumerate() {
        // Pivot search: the largest-magnitude candidate below and including
        // the diagonal.
        let mut best_row = step;
        let mut best = matrix[step * order + step].abs();
        for row in (step + 1)..order {
            let magnitude = matrix[row * order + step].abs();
            if magnitude > best {
                best = magnitude;
                best_row = row;
            }
        }
        if best == zero {
            return Err(CoarseFailure::ZeroPivot { step, best });
        }
        *pivot = best_row;
        if best_row != step {
            for column in 0..order {
                let upper = step * order + column;
                let lower = best_row * order + column;
                matrix.swap(upper, lower);
            }
        }
        let pivot_value = matrix[step * order + step];
        for row in (step + 1)..order {
            let factor = matrix[row * order + step] / pivot_value;
            matrix[row * order + step] = factor;
            for column in (step + 1)..order {
                matrix[row * order + column] -= factor * matrix[step * order + column];
            }
        }
    }
    Ok(())
}

/// Solve `M·y = rhs` in place against a factorised coarse matrix.
///
/// Applies the stored row interchanges, then the triangular substitutions.
/// The factorisation is the one produced by [`factorise`]; `rhs` is `y` on
/// return.
pub(super) fn solve_in_place<T: RealField>(
    matrix: &[T],
    order: usize,
    pivots: &[usize],
    rhs: &mut [T],
) {
    debug_assert_eq!(matrix.len(), order * order);
    debug_assert_eq!(rhs.len(), order);
    // Interchange pass.
    for (step, pivot) in pivots.iter().enumerate() {
        rhs.swap(step, *pivot);
    }
    // Forward substitution (unit lower triangular).
    for row in 0..order {
        for column in 0..row {
            rhs[row] -= matrix[row * order + column] * rhs[column];
        }
    }
    // Back substitution (upper triangular).
    for row in (0..order).rev() {
        for column in (row + 1)..order {
            rhs[row] -= matrix[row * order + column] * rhs[column];
        }
        rhs[row] = rhs[row] / matrix[row * order + row];
    }
}
