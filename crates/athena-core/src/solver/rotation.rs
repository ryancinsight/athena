//! Plane-rotation primitives shared by the Krylov solvers that maintain an
//! incremental QR of their recurrence matrix: GMRES over the Hessenberg,
//! MINRES over the Lanczos tridiagonal, and LSQR over the bidiagonal.

use eunomia::RealField;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ScalarFailure {
    Breakdown,
    NonFinite,
}

/// The Givens rotation eliminating `lower` against `upper`.
///
/// Returns `(cosine, sine)` with `cosine = upper / radius` and
/// `sine = lower / radius` for `radius = √(upper² + lower²)`, scaled by the
/// larger magnitude so no squared magnitude overflows: a vector
/// `(upper, lower)` becomes `(radius, 0)` under the rotation
/// `[[cosine, sine], [−sine, cosine]]`.
///
/// A zero pair needs no rotation and gets the identity; the caller treats a
/// zero radius as breakdown where division by it would occur.
pub(super) fn givens<T: RealField>(upper: T, lower: T) -> Result<(T, T), ScalarFailure> {
    givens_with_radius(upper, lower).map(|(cosine, sine, _)| (cosine, sine))
}

/// Return the rotation and its nonnegative pivot. Compute the coefficients
/// in scaled coordinates so even a subnormal rounded pivot retains its angle.
pub(super) fn givens_with_radius<T: RealField>(
    upper: T,
    lower: T,
) -> Result<(T, T, T), ScalarFailure> {
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
        return Ok((T::ONE, T::ZERO, T::ZERO));
    }

    let scaled_upper = upper / scale;
    let scaled_lower = lower / scale;
    let scaled_radius = (scaled_upper * scaled_upper + scaled_lower * scaled_lower).sqrt();
    let radius = scale * scaled_radius;
    if !radius.is_finite() {
        return Err(ScalarFailure::NonFinite);
    }
    Ok((
        scaled_upper / scaled_radius,
        scaled_lower / scaled_radius,
        radius,
    ))
}

#[cfg(test)]
mod tests {
    use super::givens_with_radius;

    #[test]
    fn subnormal_pivot_retains_the_rotation_angle() {
        let smallest = f64::from_bits(1);
        let (cosine, sine, radius) =
            givens_with_radius(-smallest, smallest).expect("finite rotation");
        assert_eq!(radius.to_bits(), smallest.to_bits());
        assert!((cosine + core::f64::consts::FRAC_1_SQRT_2).abs() <= f64::EPSILON);
        assert!((sine - core::f64::consts::FRAC_1_SQRT_2).abs() <= f64::EPSILON);
    }
}
