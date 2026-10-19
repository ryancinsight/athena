use eunomia::RealField;

use super::super::rotation::ScalarFailure;
use super::breakdown::singular_pivot_floor;

pub(super) fn back_substitute<T: RealField, const RESTART: usize>(
    hessenberg: &[T],
    transformed_residual: &[T],
    coefficients: &mut [T],
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
        // Truncated triangular solve: a pivot at the rounding level of the
        // factor carries no invertible information, so its direction is
        // dropped — the minimum-norm least-squares solution of the
        // rank-deficient factor — instead of dividing by rounding noise.
        coefficients[row] = if diagonal.abs() <= pivot_floor {
            T::ZERO
        } else {
            value / diagonal
        };
        if !coefficients[row].is_finite() {
            return Err(ScalarFailure::NonFinite);
        }
    }
    Ok(())
}
