use eunomia::RealField;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScalarFailure {
    Breakdown,
    NonFinite,
}

pub(super) fn givens<T: RealField>(upper: T, lower: T) -> Result<(T, T), ScalarFailure> {
    if !upper.is_finite() || !lower.is_finite() {
        return Err(ScalarFailure::NonFinite);
    }

    let upper_magnitude = upper.abs();
    let lower_magnitude = lower.abs();
    let scale = if upper_magnitude > lower_magnitude {
        upper_magnitude
    } else {
        lower_magnitude
    };
    if scale == T::ZERO {
        return Ok((T::ONE, T::ZERO));
    }

    let scaled_upper = upper / scale;
    let scaled_lower = lower / scale;
    let radius = scale * (scaled_upper * scaled_upper + scaled_lower * scaled_lower).sqrt();
    if !radius.is_finite() || radius == T::ZERO {
        return Err(ScalarFailure::NonFinite);
    }
    Ok((upper / radius, lower / radius))
}

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
