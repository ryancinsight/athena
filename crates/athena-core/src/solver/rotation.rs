//! Plane-rotation primitives shared by the Krylov solvers that maintain an
//! incremental QR of their recurrence matrix: GMRES over the Hessenberg,
//! MINRES over the Lanczos tridiagonal.

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
