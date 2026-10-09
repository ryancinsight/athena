//! Scalar conversions shared by the multigrid family.
//!
//! Counts and indices become scalars through the exact `f64` round trip the
//! stack's conversion discipline owns: `usize` fits `u64` on every supported
//! target, and `u64` values are exact in `f64` below 2⁵³ — grid and level
//! counts sit far below that.

use eunomia::{FloatElement, NumericElement};

/// Convert a count to the scalar type exactly.
#[inline]
pub(super) fn count_to_scalar<T: FloatElement>(value: usize) -> T {
    let value_u64 = u64::try_from(value).expect("invariant: usize count fits into u64");
    <T as FloatElement>::from_f64(<u64 as NumericElement>::to_f64(value_u64))
}

/// Ratio of two counts, `0.0` for a zero denominator (a rate over no points).
#[inline]
pub(super) fn ratio_of_counts(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        count_to_scalar::<f64>(numerator) / count_to_scalar::<f64>(denominator)
    }
}

/// The scale below which a diagonal counts as absent in this family.
///
/// The multigrid hierarchy assembles entries whose smallest physical scale
/// is O(1); a diagonal below `1e-15` is structural absence or cancellation,
/// not data - the recorded smoother and complexity fixtures pin this bound.
/// The saddle family keeps its own `1e-14` tolerance in `saddle/matrix`.
pub(super) fn diagonal_epsilon<T: FloatElement>() -> T {
    <T as FloatElement>::from_f64(1e-15)
}
