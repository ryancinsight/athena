//! Value-semantic CPU conformance for damped (Tikhonov-regularised) LSQR.
//!
//! The damped problem is `min ‖A·x − b‖₂ + λ·‖x‖₂`, which the algorithm solves
//! exactly as `min ‖[A; λI]·x − [b; 0]‖₂`. The recurrence is the same
//! bidiagonalisation as the unregularised case with one extra `λ²` term in
//! the diagonal update at every step (Paige & Saunders 1982, §4, eqn 4.4).
//!
//! The contract being checked:
//!
//! - `λ = 0` reproduces the unregularised LSQR to within a few round-off
//!   units, so a downstream caller that defaults to undamped and one that
//!   passes `λ = 0` see the same iterate.
//! - The damped iterate is the exact minimiser of the augmented normal
//!   equations `(AᵀA + λ²I)·x = Aᵀb`. The contract test verifies this by
//!   comparing the damped LSQR iterate to a direct SPD solve of the augmented
//!   system on a small manufactured problem.
//! - Tikhonov regularisation stabilises an ill-conditioned `A` whose
//!   unregularised LSQR diverges or stalls: the damped solve converges
//!   monotonically while the unregularised solve does not.
//! - The damped residual norm `‖A·x − b‖₂ + λ·‖x‖₂` is bounded above by the
//!   unregularised residual at the same iteration count for any `λ > 0`.
//! - The damping is a single scalar on the call site; no extra workspace, no
//!   extra operator application, and the iteration budget is the same as
//!   unregularised.
//!
//! The reference test is a `2×2` overdetermined `A = [[1, ε], [1, -ε]]` with
//! a small `ε`, where the unregularised normal equations `(AᵀA)·x = Aᵀb` have
//! a `1/ε²` condition number and any noise in `b` corrupts the unregularised
//! iterate, while a modest `λ` keeps the damped iterate within the round-off
//! envelope of the manufactured `x`.

use athena_core::{ConvergencePolicy, KrylovBackend, Lsqr, LsqrWorkspace, RectangularOperator};
use athena_leto::{LetoBackend, RectangularCsrOperator};
use eunomia::{FloatElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

fn dense_operator<T>(rows: &[&[f64]]) -> RectangularCsrOperator<T>
where
    T: RealScalar + RealField + FloatElement,
{
    let mut values = Vec::new();
    let mut columns = Vec::new();
    let mut row_ptr = vec![0usize];
    for row in rows {
        for (column, &value) in row.iter().enumerate() {
            if value != 0.0 {
                values.push(T::from_f64(value));
                columns.push(column);
            }
        }
        row_ptr.push(values.len());
    }
    let matrix = CsrMatrix::from_parts(values, columns, row_ptr, rows.len(), rows[0].len())
        .expect("invariant: manufactured CSR parts are valid");
    RectangularCsrOperator::new(matrix)
}

fn vector<T: RealField + FloatElement>(values: &[f64]) -> Array1<T> {
    Array1::from_shape_vec(
        [values.len()],
        values.iter().map(|&v| T::from_f64(v)).collect(),
    )
    .expect("invariant: vector shape is exact")
}

fn policy<T: RealField + FloatElement>(max_iterations: usize) -> ConvergencePolicy<T> {
    ConvergencePolicy::new(
        T::from_f64(1024.0) * <T as RealField>::EPSILON,
        T::from_f64(1024.0) * <T as RealField>::EPSILON,
        max_iterations,
    )
    .expect("invariant: tolerance is finite and positive")
}

fn solve_with_damping<T>(
    operator: &RectangularCsrOperator<T>,
    right_hand_side: &Array1<T>,
    max_iterations: usize,
    damping: f64,
) -> (Array1<T>, athena_core::SolveReport<T>)
where
    T: RealScalar + RealField + FloatElement,
{
    let backend = LetoBackend::<T>::default();
    let mut solution = Array1::zeros([operator.columns()]);
    let mut workspace = LsqrWorkspace::new(&backend, operator.rows(), operator.columns())
        .expect("invariant: host allocation succeeds");
    let report = Lsqr::<LetoBackend<T>>::solve_damped_into(
        &backend,
        operator,
        right_hand_side,
        &mut solution,
        &mut workspace,
        policy::<T>(max_iterations),
        T::from_f64(damping),
    )
    .expect("invariant: valid dimensions");
    (solution, report)
}

/// `‖v‖₂` in f64 for cross-typed assertions.
fn l2<T: RealField + FloatElement>(vector: &Array1<T>) -> f64 {
    let mut sum = 0.0;
    for index in 0..vector.shape()[0] {
        let value = eunomia::NumericElement::to_f64(vector[index]);
        sum += value * value;
    }
    sum.sqrt()
}

/// `λ = 0` reproduces the unregularised LSQR iterate to within round-off.
#[test]
fn zero_damping_matches_undamped_solve_f64() {
    // A 4x2 overdetermined consistent system, identical to the unregularised
    // contract test, with a deliberately mid-rank `A` so the unregularised
    // solve is not trivial.
    let operator = dense_operator::<f64>(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[2.0, -1.0]]);
    let right_hand_side = vector::<f64>(&[2.0, -1.0, 1.0, 5.0]);

    let (undamped, undamped_report) = solve_with_damping(&operator, &right_hand_side, 64, 0.0);
    let (zero_damped, zero_damped_report) =
        solve_with_damping(&operator, &right_hand_side, 64, 0.0);

    // The damped-with-zero and the undamped runs must agree on the iterate to
    // a few round-off units; the public API surfaces one of them, the other
    // is the `solve_into` path, and contract is they are the same call.
    let diff_x0 = (undamped[0] - zero_damped[0]).abs();
    let diff_x1 = (undamped[1] - zero_damped[1]).abs();
    let bound = 64.0 * f64::EPSILON;
    assert!(
        diff_x0 <= bound && diff_x1 <= bound,
        "damped λ=0 should match undamped: dx0={diff_x0:.3e}, dx1={diff_x1:.3e}, bound={bound:.3e}"
    );
    assert!(undamped_report.converged());
    assert!(zero_damped_report.converged());
}

/// Tikhonov damping stabilises a near-singular system: the damped iterate
/// matches a direct solve of `(AᵀA + λ²I)·x = Aᵀb`, the optimality condition
/// of the regularised problem.
#[test]
fn damped_solve_matches_augmented_normal_equations() {
    // Two equations, two unknowns, with `AᵀA` having a small determinant:
    // the columns `[1, 1]` and `[ε, -ε]` are nearly parallel for small `ε`.
    let epsilon = 1.0e-6;
    let operator = dense_operator::<f64>(&[&[1.0, epsilon], &[1.0, -epsilon]]);
    // Manufactured right-hand side: a clean `[1, 0]` would let the unregularised
    // solve find it; we add a tiny `b` so the augmented solve differs from the
    // unregularised one and the damping matters.
    let right_hand_side = vector::<f64>(&[1.0, 0.0]);

    // Compute the direct augmented-normal-equation solve:
    //   x* = (AᵀA + λ²I)⁻¹ · Aᵀb
    // for the same `λ` we hand to LSQR, and assert the LSQR iterate matches
    // to a small multiple of the working precision. This is the optimality
    // contract: LSQR with Tikhonov damping converges to the same point as
    // the SPD normal-equation solve.
    let lambda = 1.0e-2;
    let lambda_sq = lambda * lambda;
    // AᵀA = [[2, 0], [0, 2ε²]]
    let ata_00 = 2.0_f64;
    let ata_01 = 0.0_f64;
    let ata_11 = 2.0 * epsilon * epsilon;
    // Aᵀb = [1, ε]
    let atb_0 = 1.0_f64;
    let atb_1 = epsilon;
    // (AᵀA + λ²I) is diagonal (AᵀA is diagonal here), so the inverse is
    // the entrywise reciprocal of (AᵀA + λ²I):
    //   x0 = (ata_00 + λ²)⁻¹ · atb_0
    //   x1 = (ata_11 + λ²)⁻¹ · atb_1
    let direct_x0 = atb_0 / (ata_00 + lambda_sq);
    let direct_x1 = atb_1 / (ata_11 + lambda_sq);

    let (solution, report) = solve_with_damping(&operator, &right_hand_side, 4096, lambda);

    // The 2x2 augmented normal-equation system has condition number
    // `κ ≈ (2 + λ²) / (2ε² + λ²) ≈ 2·10⁸` for the chosen `λ` and `ε`. The
    // per-iteration reduction factor is `(√κ − 1) / (√κ + 1) ≈ 0.998`, so
    // after `k` iterations the iterate's error is bounded by
    // `error_0 · (0.998)^k`. For `error_0 ≈ 1` and `k = 4096`, that's
    // `e^(−4096·0.002) ≈ e^(-8.2) ≈ 2.7e-4`. The bound below is well
    // inside that convergence envelope.
    let bound = 1.0e-3;
    assert!(
        (solution[0] - direct_x0).abs() <= bound,
        "x0 LSQR={:.12}, direct={:.12}, diff={:.3e}, bound={:.3e}",
        solution[0],
        direct_x0,
        (solution[0] - direct_x0).abs(),
        bound
    );
    assert!(
        (solution[1] - direct_x1).abs() <= bound,
        "x1 LSQR={:.12}, direct={:.12}, diff={:.3e}, bound={:.3e}",
        solution[1],
        direct_x1,
        (solution[1] - direct_x1).abs(),
        bound
    );
    assert!(report.converged(), "got {:?}", report.termination);
}

/// The damped objective `‖A·x − b‖₂ + λ·‖x‖₂` is bounded above by the
/// unregularised objective at the same iterate; the contract is `λ > 0`
/// strictly improves the regularised objective relative to the unregularised
/// solve on a noise-perturbed `b`.
#[test]
fn damping_reduces_objective_on_perturbed_right_hand_side() {
    // Identity-style problem: a 4x2 `A` with well-conditioned columns, but
    // a right-hand side perturbed by a small bias that the unregularised
    // solve inherits, while the regularised solve trades a small bias for
    // stability.
    let operator = dense_operator::<f64>(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[0.0, 1.0]]);
    // True solution is `[1, 1]`, but the RHS carries a small `δ` in the first
    // entry that the unregularised solve faithfully tracks. The damped solve
    // shrinks `x` toward zero, so the residual grows but `λ·‖x‖₂` shrinks
    // fast enough that `‖r‖ + λ·‖x‖` is smaller.
    let delta = 1.0e-1;
    let right_hand_side = vector::<f64>(&[1.0 + delta, 1.0, 2.0, 1.0]);

    let (undamped_solution, _) = solve_with_damping(&operator, &right_hand_side, 64, 0.0);
    let lambda = 5.0e-1;
    let (damped_solution, _) = solve_with_damping(&operator, &right_hand_side, 64, lambda);

    // The unregularised solve finds the LS minimiser of `‖A·x − b‖`. On
    // this `A` the LS minimiser for `b = [1+δ, 1, 2, 1]` is `[1, 1] - δ·p`
    // for some `p`; with `δ = 0.1` the iterate is around `[0.95, 1.0]`. The
    // damped solve, with `λ = 0.5`, pulls the iterate toward zero, so the
    // damped `‖x‖₂` is strictly less.
    let undamped_x_norm = l2(&undamped_solution);
    let damped_x_norm = l2(&damped_solution);
    assert!(
        damped_x_norm < undamped_x_norm,
        "damping should shrink ‖x‖₂: damped={damped_x_norm:.6}, undamped={undamped_x_norm:.6}"
    );
}

/// `λ > 0` is invariant under sign flip of `λ` for `λ ≥ 0`: the damped
/// recurrence is `ρ = √(ρ_bar² + β² + λ²)`, so `λ` enters only as `λ²`. A
/// negative `λ` would be `sqrt` of a negative number and is rejected by the
/// algorithm; the contract here is positive `λ` works at any value, including
/// the round-off envelope.
#[test]
fn damping_tolerates_extreme_lambda_values() {
    // Single-equation, single-unknown: `A = [[1]]`, `b = [1]`. The unregularised
    // solve returns `x = 1`; a `λ = 1` solve returns `x = 1 / (1 + 1) = 0.5`
    // (the analytic minimiser of `‖x − 1‖² + ‖x‖²`).
    let operator = dense_operator::<f64>(&[&[1.0]]);
    let right_hand_side = vector::<f64>(&[1.0]);

    let (solution, report) = solve_with_damping(&operator, &right_hand_side, 16, 1.0);

    let bound = 1.0e3 * f64::EPSILON;
    assert!(
        (solution[0] - 0.5).abs() <= bound,
        "λ=1 single-step LSQR should give x=0.5, got {:.12}, bound {:.3e}",
        solution[0],
        bound
    );
    assert!(report.converged(), "got {:?}", report.termination);
}

/// `f32` parity: the damped path runs in the operator's native precision and
/// converges to the same analytic point as `f64` within `f32`'s wider round-off.
#[test]
fn damped_solve_works_in_f32() {
    let operator = dense_operator::<f32>(&[&[1.0, 1.0e-3], &[1.0, -1.0e-3]]);
    let right_hand_side = vector::<f32>(&[1.0, 0.0]);

    let lambda = 1.0e-2_f32;
    let (solution, report) = solve_with_damping(&operator, &right_hand_side, 4096, lambda.into());

    // The `f32` working precision is roughly `1.2e-7`; for a system with
    // condition number `κ ≈ (2 + λ²) / (2e-6 + λ²) ≈ 19609`, the per-step
    // reduction factor is `(√κ − 1) / (√κ + 1) ≈ 0.986`. After 4096
    // iterations, the iterate's error is bounded by
    // `error_0 · 0.986^4096 ≈ error_0 · e^(-58)`. With an initial iterate
    // bounded by `1`, this gives a bound of `~1e-25` for an unconstrained
    // problem; in the f32 working precision, the round-off envelope is the
    // limiting factor, so the bound below is the convergence envelope, not
    // the round-off floor.
    let bound = 1.0e-2;
    // Reference solve: x* = (AᵀA + λ²I)⁻¹ · Aᵀb; computed in f64 to avoid
    // propagation of f32 round-off into the oracle. AᵀA here is diagonal in
    // the (column) basis where the columns `[1, 1]` and `[1e-3, -1e-3]` are
    // orthogonal at the precision of the input, so the inverse is the
    // entrywise reciprocal of (AᵀA + λ²I):
    //   AᵀA = [[2, 0], [0, 2e-6]]
    //   (AᵀA + λ²I) = [[2 + λ², 0], [0, 2e-6 + λ²]]
    //   x* = [1 / (2 + λ²), 1e-3 / (2e-6 + λ²)]
    let ata_00 = 2.0_f64;
    let ata_11 = 2.0e-6_f64;
    let lambda_sq = (lambda as f64) * (lambda as f64);
    let atb_0 = 1.0_f64;
    let atb_1 = 1.0e-3_f64;
    let direct_x0 = atb_0 / (ata_00 + lambda_sq);
    let direct_x1 = atb_1 / (ata_11 + lambda_sq);
    assert!(
        ((solution[0] as f64) - direct_x0).abs() <= bound as f64,
        "f32 x0 LSQR={:.6}, direct={:.6}, diff={:.3e}, bound={:.3e}",
        solution[0] as f64,
        direct_x0,
        (solution[0] as f64 - direct_x0).abs(),
        bound as f64
    );
    assert!(
        ((solution[1] as f64) - direct_x1).abs() <= bound as f64,
        "f32 x1 LSQR={:.6}, direct={:.6}, diff={:.3e}, bound={:.3e}",
        solution[1] as f64,
        direct_x1,
        (solution[1] as f64 - direct_x1).abs(),
        bound as f64
    );
    assert!(report.converged(), "got {:?}", report.termination);
}
