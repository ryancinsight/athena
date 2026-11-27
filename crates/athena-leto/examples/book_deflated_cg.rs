//! Solve an ill-conditioned SPD system with `DeflatedCg` over an exact basis.
//!
//! The operator is the diagonal matrix with spectrum `{0.01, 0.02, 0.5, 1, 2}`.
//! The standard basis vectors `e₁`, `e₂` are the eigenvectors of its two small
//! eigenvalues, so the deflated CG phase sees only three eigenvalues and
//! terminates in at most `n - k = 3` iterations. The right-hand side
//! `(0.01, 0.04, 1.5, 4, 10)` has the closed-form solution `(1, 2, 3, 4, 5)`.

// The book includes this file verbatim as a tested code sample. `mdbook test`
// spawns rustdoc against a staged library path rather than a cargo target, so
// every crate the sample names has to be declared; cargo itself resolves these
// from the extern prelude and ignores the declarations.
extern crate athena_core;
extern crate athena_leto;
extern crate leto;
extern crate leto_ops;

use athena_core::{CgWorkspace, ConvergencePolicy, DeflatedCg, DeflationSubspace, Termination};
use athena_leto::{CsrOperator, LetoBackend};
use leto::Array1;
use leto_ops::CsrMatrix;

/// Absolute error bound for each solution component. The complement phase
/// runs three steps over length-five vectors (under 40 rounded operations per
/// entry, complement eigenvalues of unit amplification), and the coarse
/// correction adds one 2-by-2 solve whose entries have magnitude at most
/// `1/0.01`: 512 machine epsilons covers that first-order count with margin.
const BOUND: f64 = 512.0 * f64::EPSILON;

fn main() {
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(
        vec![0.01_f64, 0.02, 0.5, 1.0, 2.0],
        vec![0, 1, 2, 3, 4],
        vec![0, 1, 2, 3, 4, 5],
        5,
        5,
    )
    .expect("diagonal CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("matrix is square");
    let right_hand_side = Array1::from_shape_vec([5], vec![0.01_f64, 0.04, 1.5, 4.0, 10.0])
        .expect("vector shape is exact");

    // Scaled copies of `e₁` and `e₂`: construction orthonormalises its input.
    let first = Array1::from_shape_vec([5], vec![5.0_f64, 0.0, 0.0, 0.0, 0.0])
        .expect("vector shape is exact");
    let second = Array1::from_shape_vec([5], vec![0.0_f64, 3.0, 0.0, 0.0, 0.0])
        .expect("vector shape is exact");
    let subspace = DeflationSubspace::from_vectors(&backend, &operator, &[first, second])
        .expect("the eigenbasis is independent and its coarse matrix is nonsingular");

    let mut solution = Array1::zeros([5]);
    let mut workspace = CgWorkspace::new(&backend, 5).expect("workspace allocation");
    // The complement spectrum has three values, so the CG phase needs at most
    // three iterations; one spare lets finite termination, not the cap, end it.
    let policy = ConvergencePolicy::new(f64::EPSILON, f64::EPSILON, 4)
        .expect("tolerances are finite and positive");

    let report = DeflatedCg::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &subspace,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("dimensions match");

    assert_eq!(report.termination, Termination::Converged);
    assert!(
        report.iterations <= 3,
        "finite termination in n - k steps, ran {}",
        report.iterations
    );
    let values = solution.as_slice().expect("solution is contiguous");
    for (got, want) in values.iter().zip([1.0_f64, 2.0, 3.0, 4.0, 5.0]) {
        assert!((got - want).abs() <= BOUND, "component {got} vs {want}");
    }
}
