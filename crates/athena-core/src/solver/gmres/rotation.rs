use eunomia::RealField;

use super::super::rotation::ScalarFailure;

pub(super) fn back_substitute<T: RealField, const RESTART: usize>(
    hessenberg: &[T],
    transformed_residual: &[T],
    coefficients: &mut [T],
    count: usize,
) -> Result<(), ScalarFailure> {
    for row in (0..count).rev() {
        let mut value = transformed_residual[row];
        for column in (row + 1)..count {
            value -= hessenberg[column * (RESTART + 1) + row] * coefficients[column];
        }
        let diagonal = hessenberg[row * (RESTART + 1) + row];
        if !value.is_finite() || !diagonal.is_finite() {
            return Err(ScalarFailure::NonFinite);
        }
        if diagonal == T::ZERO {
            return Err(ScalarFailure::Breakdown);
        }
        coefficients[row] = value / diagonal;
        if !coefficients[row].is_finite() {
            return Err(ScalarFailure::NonFinite);
        }
    }
    Ok(())
}
