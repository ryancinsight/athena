//! Convergence policy construction and threshold computation.
//!
//! [`ConvergencePolicy`] encodes the stopping criterion for iterative solvers:
//! a residual `‖r‖₂ ≤ max(absolute_tol, relative_tol × ‖b‖₂)`.  Constructing
//! one is fallible — invalid tolerances (negative, NaN, infinity) or zero
//! iteration counts are rejected at the boundary.

use athena_core::{ConvergencePolicy, InvalidConvergencePolicy};

fn main() {
    // ── Valid policy ──
    let policy = ConvergencePolicy::<f64>::new(1e-10, 1e-8, 1000)
        .expect("valid tolerances and iteration count");
    println!("convergence policy: {policy:?}");

    // ── Threshold computation ──
    // For ‖b‖₂ = 1.0 the threshold is max(1e-10, 1e-8 × 1.0) = 1e-8.
    let rhs_norm = 1.0_f64;
    let threshold = policy.threshold(rhs_norm);
    println!("threshold at ‖b‖₂={rhs_norm}: {threshold:.3e}");
    assert!((threshold - 1e-8).abs() < 1e-15);

    // For ‖b‖₂ = 1e-5 the threshold is max(1e-10, 1e-8 × 1e-5) = 1e-10
    // (absolute tolerance dominates).
    let small_rhs = 1e-5_f64;
    let threshold_small = policy.threshold(small_rhs);
    println!("threshold at ‖b‖₂={small_rhs:.0e}: {threshold_small:.3e}");
    assert!((threshold_small - 1e-10).abs() < 1e-22);

    // ── Validation errors ──
    let result_neg = ConvergencePolicy::<f64>::new(-1.0, 1e-6, 100);
    assert!(
        matches!(result_neg, Err(InvalidConvergencePolicy::InvalidTolerance)),
        "negative tolerance must be rejected"
    );
    println!("negative tolerance rejected: {}", InvalidConvergencePolicy::InvalidTolerance);

    let result_zero = ConvergencePolicy::<f64>::new(1e-8, 1e-6, 0);
    assert!(
        matches!(result_zero, Err(InvalidConvergencePolicy::ZeroIterations)),
        "zero iterations must be rejected"
    );
    println!("zero iterations rejected: {}", InvalidConvergencePolicy::ZeroIterations);

    println!("all convergence-policy assertions passed");
}
