use eunomia::RealField;

use super::super::rotation::{ScalarFailure, givens};

mod rank_deficient;
#[cfg(test)]
mod tests;
use super::breakdown::singular_pivot_floor;

pub(super) fn back_substitute<T: RealField, const RESTART: usize>(
    hessenberg: &mut [T],
    transformed_residual: &mut [T],
    coefficients: &mut [T],
    permutation: &mut [usize],
    count: usize,
) -> Result<(), ScalarFailure> {
    // The factor's own pivot scale, against which singularity is judged. A
    // pivot at the rounding level of the largest pivot carries no invertible
    // information; exact-zero comparison would hold for one rounding order
    // and fail for another equally valid one.
    let mut pivot_scale = T::ZERO;
    for row in 0..count {
        let magnitude = hessenberg[row * (RESTART + 1) + row].abs();
        if magnitude > pivot_scale {
            pivot_scale = magnitude;
        }
    }
    if !pivot_scale.is_finite() {
        return Err(ScalarFailure::NonFinite);
    }
    let pivot_floor = singular_pivot_floor(count, pivot_scale);
    for row in (0..count).rev() {
        let mut value = transformed_residual[row];
        for column in (row + 1)..count {
            value -= hessenberg[column * (RESTART + 1) + row] * coefficients[column];
        }
        let diagonal = hessenberg[row * (RESTART + 1) + row];
        if !value.is_finite() || !diagonal.is_finite() {
            return Err(ScalarFailure::NonFinite);
        }
        // A small unpivoted diagonal is not a rank certificate: an
        // off-diagonal column may still span a well-conditioned direction.
        // Preserve the ordinary arithmetic when every pivot is usable;
        // recover a basic reduced LS correction through column pivoting only
        // when this fast path encounters numerical rank loss.
        if diagonal.abs() <= pivot_floor {
            return rank_deficient::solve::<T, RESTART>(
                hessenberg,
                transformed_residual,
                coefficients,
                permutation,
                count,
            );
        }
        coefficients[row] = value / diagonal;
        if !coefficients[row].is_finite() {
            return Err(ScalarFailure::NonFinite);
        }
    }
    Ok(())
}
