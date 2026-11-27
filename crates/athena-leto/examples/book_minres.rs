//! Solve a symmetric indefinite system with `Minres`, where `Cg` fails.
//!
//! The operator `diag(1, -1)` has eigenvalues of opposite sign, so the first
//! curvature `dᵀ·A·d` of CG on the right-hand side `(1, 2)` is `1 - 4 < 0`.
//! MINRES needs only `A = Aᵀ`, and recovers the closed-form solution `(1, -2)`.

// The book includes this file verbatim as a tested code sample. `mdbook test`
// spawns rustdoc against a staged library path rather than a cargo target, so
// every crate the sample names has to be declared; cargo itself resolves these
// from the extern prelude and ignores the declarations.
extern crate athena_core;
extern crate athena_leto;
extern crate leto;
extern crate leto_ops;

use athena_core::{ConvergencePolicy, Minres, MinresWorkspace, Termination};
use athena_leto::{CsrOperator, LetoBackend};
use leto::Array1;
use leto_ops::CsrMatrix;

/// Absolute error bound for each solution component. Two Lanczos steps over
/// length-two vectors perform fewer than 48 rounded scalar operations with unit
/// amplification (the condition number is one); 256 machine epsilons is a
/// factor-of-five margin over that first-order count.
const BOUND: f64 = 256.0 * f64::EPSILON;

fn main() {
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(vec![1.0_f64, -1.0], vec![0, 1], vec![0, 1, 2], 2, 2)
        .expect("diagonal CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([2], vec![1.0_f64, 2.0]).expect("vector shape is exact");
    let mut solution = Array1::zeros([2]);
    let mut workspace = MinresWorkspace::new(&backend, 2).expect("workspace allocation");
    // Two distinct eigenvalues complete the Krylov space after two Lanczos
    // steps; a budget of three leaves the finish to the recurrence.
    let policy = ConvergencePolicy::new(f64::EPSILON, f64::EPSILON, 3)
        .expect("tolerances are finite and positive");

    let report = Minres::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("dimensions match");

    assert_eq!(report.termination, Termination::Converged);
    let values = solution.as_slice().expect("solution is contiguous");
    for (got, want) in values.iter().zip([1.0_f64, -2.0]) {
        assert!((got - want).abs() <= BOUND, "component {got} vs {want}");
    }
}
