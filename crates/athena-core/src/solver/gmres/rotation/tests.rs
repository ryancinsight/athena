use super::{ScalarFailure, back_substitute};
use core::fmt::Debug;
use eunomia::{FloatElement, RealField};
use tracing_subscriber::fmt::format::FmtSpan;

fn with_trace(run: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        .finish();
    tracing::subscriber::with_default(subscriber, run);
}
fn full_rank<T: RealField + FloatElement + Debug>() {
    let arrange = tracing::info_span!("arrange", scalar = core::any::type_name::<T>()).entered();
    let zero = T::from_f64(0.0);
    let mut factor = [
        T::from_f64(2.0),
        zero,
        zero,
        T::from_f64(1.0),
        T::from_f64(4.0),
        zero,
    ];
    let mut rhs = [T::from_f64(5.0), T::from_f64(8.0), zero];
    let original_h = factor;
    let original_g = rhs;
    let mut coefficients = [zero; 2];
    let mut permutation = [usize::MAX; 2];
    drop(arrange);
    {
        let _act = tracing::info_span!("act").entered();
        back_substitute::<T, 2>(
            &mut factor,
            &mut rhs,
            &mut coefficients,
            &mut permutation,
            2,
        )
        .expect("finite full-rank solve");
    }
    let _assert = tracing::info_span!("assert", ?coefficients).entered();
    assert_eq!(coefficients, [T::from_f64(1.5), T::from_f64(2.0)]);
    assert_eq!(factor, original_h, "full-rank factor is not refactored");
    assert_eq!(rhs, original_g, "full-rank RHS is not transformed again");
    assert_eq!(
        permutation,
        [usize::MAX; 2],
        "full-rank path never uses recovery scratch"
    );
}
fn basic<T: RealField + FloatElement + Debug>() {
    let arrange = tracing::info_span!("arrange", scalar = core::any::type_name::<T>()).entered();
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    let half = T::from_f64(0.5);
    let mut factor = [one, zero, zero, one, zero, zero];
    let mut rhs = [one, zero, zero];
    let mut coefficients = [T::from_f64(7.0); 2];
    let mut permutation = [0; 2];
    drop(arrange);
    {
        let _act = tracing::info_span!("act").entered();
        back_substitute::<T, 2>(
            &mut factor,
            &mut rhs,
            &mut coefficients,
            &mut permutation,
            2,
        )
        .expect("rank-one solve");
    }
    let _assert = tracing::info_span!("assert", ?coefficients).entered();
    assert_eq!(
        coefficients,
        [one, zero],
        "stable first-column tie gives a basic solution"
    );
    assert_eq!(
        coefficients[0] + coefficients[1],
        one,
        "zero reduced residual"
    );
    assert_eq!(
        coefficients[0] * coefficients[0] + coefficients[1] * coefficients[1],
        one
    );
    assert_eq!(
        half * half + half * half,
        half,
        "another LS solution has smaller coefficient norm; minimum norm is not promised"
    );
}
fn off_diagonal<T: RealField + FloatElement + Debug>() {
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    let mut factor = [zero, zero, zero, one, one, zero];
    let mut rhs = [one, zero, zero];
    let mut coefficients = [T::from_f64(7.0); 2];
    let mut permutation = [0; 2];
    {
        let _act = tracing::info_span!("act", scalar = core::any::type_name::<T>()).entered();
        back_substitute::<T, 2>(
            &mut factor,
            &mut rhs,
            &mut coefficients,
            &mut permutation,
            2,
        )
        .expect("pivoted rank-one solve");
    }
    let _assert = tracing::info_span!("assert", ?coefficients).entered();
    let half = T::from_f64(0.5);
    let bound = T::from_f64(32.0) * <T as RealField>::EPSILON;
    // One two-row rotation and substitution: fewer than16 native rounded
    // operations on O(1) data;32 eps bounds the resulting component error.
    assert_eq!(coefficients[0], zero);
    assert!((coefficients[1] - half).abs() <= bound);
    let upper = one - coefficients[1];
    let lower = -coefficients[1];
    assert!((upper * upper + lower * lower - half).abs() <= bound);
}
fn invalid<T: RealField + FloatElement + Debug>() {
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    for (bad_entry, bad_rhs) in [
        (T::from_f64(f64::NAN), one),
        (one, T::from_f64(f64::INFINITY)),
    ] {
        let mut factor = [one, zero, zero, bad_entry, zero, zero];
        let mut rhs = [bad_rhs, zero, zero];
        let mut coefficients = [zero; 2];
        let mut permutation = [0; 2];
        assert_eq!(
            back_substitute::<T, 2>(
                &mut factor,
                &mut rhs,
                &mut coefficients,
                &mut permutation,
                2
            ),
            Err(ScalarFailure::NonFinite)
        );
    }
    let mut factor = [zero; 6];
    let mut rhs = [one, one, zero];
    let mut coefficients = [one; 2];
    let mut permutation = [0; 2];
    back_substitute::<T, 2>(
        &mut factor,
        &mut rhs,
        &mut coefficients,
        &mut permutation,
        2,
    )
    .expect("finite rank-zero factor");
    assert_eq!(coefficients, [zero, zero]);
}
#[test]
fn full_rank_preserves_ordinary_arithmetic_and_inputs() {
    with_trace(|| {
        full_rank::<f32>();
        full_rank::<f64>();
    });
}
#[test]
fn deficient_factor_returns_basic_not_minimum_norm_coefficients() {
    with_trace(|| {
        basic::<f32>();
        basic::<f64>();
    });
}
#[test]
fn column_pivoting_retains_off_diagonal_range() {
    with_trace(|| {
        off_diagonal::<f32>();
        off_diagonal::<f64>();
    });
}
#[test]
fn recovery_checks_nonfinite_data_and_handles_rank_zero() {
    with_trace(|| {
        invalid::<f32>();
        invalid::<f64>();
    });
}
