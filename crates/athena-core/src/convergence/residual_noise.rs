use eunomia::RealField;

/// Absolute accuracy floor of an explicitly recomputed residual norm.
///
/// Returns the magnitude below which a change in `‖b − Ax‖₂` carries no
/// information about the iteration, because it is smaller than the rounding
/// of the residual's own evaluation. Termination criteria that compare two
/// residuals — stagnation and divergence — are only meaningful against this
/// quantity, so it is computed once per solve and reused rather than
/// rediscovered as a literal at each comparison.
///
/// # Derivation
///
/// The floor is **absolute** and scales with `‖b‖₂`, not relative to the
/// current residual. Athena recomputes the residual explicitly as `b − Ax`
/// rather than carrying a recurrence value, and the error of that evaluation
/// is bounded by `O(u(‖b‖ + ‖A‖‖x‖))`; for an iterate approaching the
/// solution `‖A‖‖x‖ ≈ ‖b‖`, leaving `‖b‖` as the scale (Higham, *Accuracy and
/// Stability of Numerical Algorithms*, 2nd ed., §7.1). Near convergence
/// `‖r‖ ≪ ‖b‖`, so a floor expressed relative to `‖r‖` would understate the
/// true uncertainty by exactly the factor `‖b‖ / ‖r‖`.
///
/// The `len` dependence is the accumulation error of the length-`len`
/// summation inside the Euclidean norm and of the sparse product forming
/// `Ax`. Worst-case sequential accumulation is bounded by `len · u` (ibid.,
/// Lemma 3.1), but that bound is attained only when every rounding error
/// carries the same sign. Treating them as independent gives the `√len · u`
/// rule of thumb of ibid. §3.1, and the norm's summands are squares of one
/// sign, which removes the cancellation the worst case relies on. The
/// statistical form is used because `len · u` exceeds 1 in `f32` at roughly
/// 1.7e7 unknowns, which would declare every large single-precision system
/// stagnant while its residual still carried several correct digits.
///
/// With `u = EPSILON / 2` the returned `√len · EPSILON · ‖b‖` keeps a factor
/// of two of margin over `√len · u · ‖b‖`.
///
/// # Limits of this estimate
///
/// This is a statistical estimate of the evaluation error, not a worst-case
/// bound: an adversarial rounding pattern can exceed it by up to `√len`. It
/// also assumes the backend's norm accumulates in the scalar's own precision
/// and in an order no worse than sequential. A backend that accumulates
/// pairwise is bounded more tightly than this, so the floor stays
/// conservative for it.
///
/// # Examples
///
/// ```
/// use athena_core::residual_noise_floor;
///
/// // The floor grows with the right-hand-side scale, not with the residual.
/// let small = residual_noise_floor::<f64>(1024, 1.0);
/// let large = residual_noise_floor::<f64>(1024, 1.0e6);
///
/// assert!(small > 0.0);
/// assert!((large / small - 1.0e6).abs() <= 1.0e-6);
/// ```
#[must_use]
pub fn residual_noise_floor<T: RealField>(len: usize, right_hand_side_norm: T) -> T {
    #[expect(
        clippy::cast_precision_loss,
        reason = "len enters as the sample count of an error estimate; rounding a count above 2^53 to the nearest f64 perturbs the estimate by a relative 2^-53, far below the factor-of-two margin the estimate already carries"
    )]
    let samples = T::from_f64(len as f64);
    samples.sqrt() * T::EPSILON * right_hand_side_norm
}

#[cfg(test)]
mod tests {
    use super::residual_noise_floor;

    #[test]
    fn scales_as_the_square_root_of_the_unknown_count() {
        let single = residual_noise_floor::<f64>(1, 1.0);
        let hundred = residual_noise_floor::<f64>(100, 1.0);

        assert!((single - f64::EPSILON).abs() <= f64::EPSILON * f64::EPSILON);
        assert!((hundred / single - 10.0).abs() <= 1.0e-12);
    }

    #[test]
    fn stays_resolvable_for_large_single_precision_systems() {
        // At 1e8 unknowns the worst-case form is `1e8 * EPSILON ~ 11.9`, so a
        // floor derived from it would exceed every residual and declare any
        // such system stagnant. The statistical form is
        // `sqrt(1e8) * EPSILON ~ 1.2e-3`, which still resolves a residual
        // carrying three correct digits.
        let floor = residual_noise_floor::<f32>(100_000_000, 1.0);

        assert!(floor > 1.0e-3);
        assert!(floor < 2.0e-3);
    }

    #[test]
    fn a_zero_length_or_zero_scale_system_admits_no_noise() {
        assert!(residual_noise_floor::<f64>(0, 1.0) <= 0.0);
        assert!(residual_noise_floor::<f64>(1024, 0.0) <= 0.0);
    }
}
