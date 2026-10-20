use eunomia::RealField;

/// Model-based floor for modified Gram–Schmidt breakdown detection.
///
/// `sqrt(steps * len) * EPSILON * initial_norm` uses independent-rounding
/// accumulation as a statistical scale (Higham, *Accuracy and Stability of
/// Numerical Algorithms*, 2nd ed., §3.1). With unit roundoff `u = EPSILON/2`,
/// the factor two is a heuristic margin, not a probability guarantee.
///
/// This is **not** a deterministic worst-case bound or a certificate that a
/// nonzero direction contains no information. Aligned/correlated rounding
/// errors can exceed square-root accumulation; pairwise/FMA evaluation does
/// not by itself prove this particular floor conservative for all data.
/// There is no universal positive lower bound on a healthy post/pre norm
/// ratio. A small ratio is therefore a numerical-breakdown policy decision.
///
/// Callers reject non-finite input norms before classification and still
/// recompute the original residual after the candidate correction.
#[must_use]
pub(super) fn arnoldi_breakdown_floor<T: RealField>(
    steps: usize,
    len: usize,
    initial_norm: T,
) -> T {
    let samples = T::from_count(steps) * T::from_count(len);
    samples.sqrt() * T::EPSILON * initial_norm
}

/// Statistical numerical-rank cutoff relative to a factor scale.
///
/// `sqrt(count) * EPSILON * pivot_scale` models independent Givens/solve
/// rounding with a factor-two margin over `sqrt(count) * u`; it is not a
/// deterministic worst-case bound or an exact rank certificate.
///
/// The ordinary triangular path uses its largest diagonal as the scale.
/// Encountering a small diagonal triggers column-pivoted QR, which uses the
/// largest complete column norm instead. Small **unpivoted** diagonals can
/// coexist with well-separated nonzero singular values, so independently
/// zeroing their coefficients is not a least-squares algorithm. The recovery
/// returns a basic solution of the numerically rank-truncated factor, not a
/// minimum-coefficient-norm solution. Genuine ill-conditioning still needs
/// the caller's explicit original-residual check.
#[must_use]
pub(super) fn singular_pivot_floor<T: RealField>(count: usize, pivot_scale: T) -> T {
    let samples = T::from_count(count);
    samples.sqrt() * T::EPSILON * pivot_scale
}

#[cfg(test)]
mod tests {
    use super::{arnoldi_breakdown_floor, singular_pivot_floor};

    #[test]
    fn scales_as_the_square_root_of_steps_and_length() {
        let unit = arnoldi_breakdown_floor::<f64>(1, 1, 1.0);
        let long = arnoldi_breakdown_floor::<f64>(1, 100, 1.0);
        let many_steps = arnoldi_breakdown_floor::<f64>(4, 1, 1.0);

        assert!((unit - f64::EPSILON).abs() <= f64::EPSILON * f64::EPSILON);
        assert!((long / unit - 10.0).abs() <= 1.0e-12);
        assert!((many_steps / unit - 2.0).abs() <= 1.0e-12);
    }

    #[test]
    fn oracle_scaleadmits_single_ulp_residue_in_both_precisions() {
        // The FGMRES stall oracle orthogonalizes two length-two vectors; the
        // degenerate direction cancels exactly under scalar evaluation and
        // leaves ~1ulp under FMA-based SIMD. Both must read as breakdown.
        let f64_floor = arnoldi_breakdown_floor::<f64>(2, 2, 1.0);
        let f32_floor = arnoldi_breakdown_floor::<f32>(2, 2, 1.0);

        assert!((f64_floor - 2.0 * f64::EPSILON).abs() <= f64::EPSILON * f64::EPSILON);
        assert!(f64_floor > f64::EPSILON);
        assert!((f32_floor - 2.0 * f32::EPSILON).abs() <= f32::EPSILON * f32::EPSILON);
        assert!(f32_floor > f32::EPSILON);
    }

    #[test]
    fn zero_steps_length_or_scale_admits_no_noise() {
        assert!(arnoldi_breakdown_floor::<f64>(0, 1024, 1.0) <= 0.0);
        assert!(arnoldi_breakdown_floor::<f64>(4, 0, 1.0) <= 0.0);
        assert!(arnoldi_breakdown_floor::<f64>(4, 1024, 0.0) <= 0.0);
    }

    #[test]
    fn pivot_floor_scales_as_the_square_root_of_the_count() {
        let single = singular_pivot_floor::<f64>(1, 1.0);
        let hundred = singular_pivot_floor::<f64>(100, 1.0);

        assert!((single - f64::EPSILON).abs() <= f64::EPSILON * f64::EPSILON);
        assert!((hundred / single - 10.0).abs() <= 1.0e-12);
    }

    #[test]
    fn pivot_floor_admits_rounding_level_degeneracy() {
        // The stalled oracle factors a 2x2 Hessenberg whose degenerate pivot
        // is exactly zero under scalar evaluation and ~1ulp under SIMD. Both
        // must read as singular against a unit-scale factor.
        let floor = singular_pivot_floor::<f64>(2, 1.0);

        assert!(floor > f64::EPSILON);
        assert!(floor < 2.0 * f64::EPSILON);
    }

    #[test]
    fn zero_count_or_scale_admits_no_pivot_noise() {
        assert!(singular_pivot_floor::<f64>(0, 1.0) <= 0.0);
        assert!(singular_pivot_floor::<f64>(4, 0.0) <= 0.0);
    }
}
