//! Solve an SPD system with `Chebyshev` semi-iteration over a known interval.
//!
//! The operator `diag(2, 3, 5)` has the exact spectral interval `[2, 5]`, so
//! `μ = 7/3` and each step damps the error by the Chebyshev factor. The
//! right-hand side `(2, 6, 15)` has the closed-form solution `(1, 2, 3)`.

// The book includes this file verbatim as a tested code sample. `mdbook test`
// spawns rustdoc against a staged library path rather than a cargo target, so
// every crate the sample names has to be declared; cargo itself resolves these
// from the extern prelude and ignores the declarations.
extern crate athena_core;
extern crate athena_leto;
extern crate leto;
extern crate leto_ops;

use athena_core::{Chebyshev, ChebyshevWorkspace, ConvergencePolicy, SpectralBounds, Termination};
use athena_leto::{CsrOperator, LetoBackend};
use leto::Array1;
use leto_ops::CsrMatrix;

/// Absolute error bound for each solution component, derived from the
/// stopping test, not from the recurrence. `solve_into` reports `Converged`
/// only after it recomputes `r̂ = fl(b − fl(A·x̂))` from the returned `x̂` and
/// finds `‖r̂‖ ≤ ε‖b‖`; the roundings of the 25 steps decide which `x̂` is
/// returned, but the test certifies `x̂` through that last residual alone.
/// Assume IEEE binary64, round to nearest, no overflow or underflow, a
/// native-precision dot product, and `u = ε/2 = 2^-53`.
///
/// - Each row holds one nonzero `aᵢ`; its sums with zero are exact, so
///   `fl(A·x̂)ᵢ = aᵢx̂ᵢ(1 + δ₁)` whether the product is fused or not, and
///   `r̂ᵢ = (bᵢ − aᵢx̂ᵢ(1 + δ₁))(1 + δ₂)` with `|δ| ≤ u`. With `A·x = b` this
///   gives `|x̂ᵢ − xᵢ| ≤ |r̂ᵢ| / (aᵢ(1 − u)) + u|x̂ᵢ|`.
/// - The squares of `b = (2, 6, 15)` sum to 265 exactly, so the computed
///   `‖b‖` is `√265·(1 + δ)` with `√265 ≈ 16.2788`; the threshold product
///   by `ε = 2^-52` is exact. A computed three-term norm has relative error
///   at most `γ₃/2 + u ≤ 3u`, `γ₃ = 3u / (1 − 3u)`, for any summation order,
///   with a fused or unfused accumulation counted as one rounding per
///   product and per sum. Hence `|r̂ᵢ| ≤ ‖r̂‖ ≤ ε√265·(1 + 5u)`.
/// - Per component, `|x̂ᵢ − xᵢ| / ε ≤ (1 + 7u)·(16.2788 / aᵢ + xᵢ/2)`, which
///   is 8.640, 6.427 and 4.756 for `a = (2, 3, 5)`, `x = (1, 2, 3)`. The
///   `1 + 7u` adds under `10^-14` to the largest, so 9 machine epsilons
///   covers it.
const BOUND: f64 = 9.0 * f64::EPSILON;

fn main() {
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(
        vec![2.0_f64, 3.0, 5.0],
        vec![0, 1, 2],
        vec![0, 1, 2, 3],
        3,
        3,
    )
    .expect("diagonal CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([3], vec![2.0_f64, 6.0, 15.0]).expect("vector shape is exact");
    let mut solution = Array1::zeros([3]);
    let mut workspace = ChebyshevWorkspace::new(&backend, 3).expect("workspace allocation");
    let bounds = SpectralBounds::new(2.0_f64, 5.0).expect("the interval is finite and ordered");
    // The solution starts at zero, so the residual starts at `b` and the
    // threshold is `ε‖b‖`. The bound `‖r_k‖ ≤ ‖b‖ / T_k(μ)` reaches it once
    // `T_k(μ) ≥ 1/ε`. With `T_k(μ) = cosh(k·acosh μ) ≈ e^(k·acosh μ) / 2`, that
    // is `k·acosh(μ) ≥ ln(2/ε)`. For `ε = 2^-52` the right side is about 36.74
    // and `acosh(7/3)` is about 1.491, so `k ≥ 24.64`, that is `k = 25`; the
    // budget of 28 covers it.
    let policy = ConvergencePolicy::new(f64::EPSILON, f64::EPSILON, 28)
        .expect("tolerances are finite and positive");

    let report = Chebyshev::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
        bounds,
    )
    .expect("dimensions match");

    assert_eq!(report.termination, Termination::Converged);
    let values = solution.as_slice().expect("solution is contiguous");
    for (got, want) in values.iter().zip([1.0_f64, 2.0, 3.0]) {
        assert!((got - want).abs() <= BOUND, "component {got} vs {want}");
    }
}
