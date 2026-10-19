use eunomia::RealField;

/// Rounding-level floor for modified Gram–Schmidt breakdown detection.
///
/// Returns the magnitude below which an orthogonalized Arnoldi vector carries
/// no direction, because it is smaller than the rounding its own
/// orthogonalization commits. The happy-breakdown test compares the
/// post-orthogonalization norm against this quantity instead of exact zero:
/// exact equality holds for one rounding order and fails for another equally
/// valid one (scalar mul-then-add cancels where FMA-based SIMD leaves a
/// ~1ulp residue), so it reports `Breakdown` on one backend and
/// `MaxIterations` on another for the same mathematics.
///
/// # Derivation
///
/// One Arnoldi step orthogonalizes `work` against `steps` orthonormal basis
/// vectors. Each projection forms a length-`len` dot product and one axpy
/// update. The dot error is bounded by `len · u` worst case (Higham,
/// *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1),
/// attained only when every rounding error carries the same sign; treating
/// them as independent gives the `√len · u` rule of thumb of ibid. §3.1, and
/// the axpy contributes a further `u` per element. Over `steps` sequential
/// projections the statistical accumulation is `√(steps · len) · u` relative
/// to the pre-orthogonalization norm `initial_norm`.
///
/// With `u = EPSILON / 2` the returned
/// `√(steps · len) · EPSILON · initial_norm` keeps a factor of two of margin
/// over that estimate. A healthy Arnoldi step keeps a post/pre ratio of order
/// `0.1`–`1`, far above the floor at any realistic restart width and vector
/// length, so the margin separates invariance from slow convergence rather
/// than conflating them.
///
/// # Limits of this estimate
///
/// This is a statistical estimate of the orthogonalization error, not a
/// worst-case bound: an adversarial rounding pattern can exceed it by up to
/// `√(steps · len)`. It also assumes the backend's dot and norm accumulate in
/// the scalar's own precision and in an order no worse than sequential. A
/// backend that accumulates pairwise, or an axpy that fuses multiply-add into
/// one rounding, is bounded more tightly than this, so the floor stays
/// conservative for it. Callers must still reject a non-finite
/// `initial_norm` before comparing: the floor of a non-finite scale is
/// meaningless and must report `NonFinite`, never breakdown.
#[must_use]
pub(super) fn arnoldi_breakdown_floor<T: RealField>(
    steps: usize,
    len: usize,
    initial_norm: T,
) -> T {
    let samples = T::from_count(steps) * T::from_count(len);
    samples.sqrt() * T::EPSILON * initial_norm
}

/// Rounding-level floor for triangular-pivot singularity detection.
///
/// Returns the magnitude below which a back-substitution pivot carries no
/// invertible information, because it is smaller than the rounding its own
/// Givens reduction commits. The truncated triangular solve drops pivots at
/// or below this quantity scaled to the factor — never against exact zero,
/// which holds for one rounding order and fails for another equally valid
/// one — yielding the minimum-norm least-squares solution of a rank-deficient
/// factor instead of dividing by rounding noise.
///
/// # Derivation
///
/// Each factor entry passes through roughly `count` Givens rotations, and
/// back-substitution accumulates over `count` rows. The per-entry error is
/// bounded by `count · u` worst case (Higham, *Accuracy and Stability of
/// Numerical Algorithms*, 2nd ed., §19.7 for the rotations, §8.1 for the
/// substitution); treating the roundings as independent gives the `√count · u`
/// rule of thumb of ibid. §3.1, relative to the factor's own pivot scale.
/// With `u = EPSILON / 2` the returned `√count · EPSILON · pivot_scale` keeps
/// a factor of two of margin over that estimate. A healthy factor keeps all
/// pivots within a modest factor of the largest, far above the floor, so the
/// margin separates singularity from ill-conditioning rather than conflating
/// them.
///
/// # Limits of this estimate
///
/// Like [`arnoldi_breakdown_floor`], this is a statistical estimate, not a
/// worst-case bound, and it assumes evaluation in the scalar's own precision
/// in an order no worse than sequential. A genuinely ill-conditioned factor
/// whose smallest pivot sits a few orders above the floor still divides by
/// it; the floor detects singularity, not ill-conditioning.
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
