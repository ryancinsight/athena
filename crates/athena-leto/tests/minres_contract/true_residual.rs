//! MINRES `Converged` certifies the recomputed residual of the returned
//! iterate, not the Lanczos recurrence estimate.
//!
//! The system is the ADR 0062 oracle `diag(1, -1) x = (1, 2)` from `x0 = 0`
//! with budget three and absolute tolerance zero, so the threshold is
//! `relative * ||b||` with `||b|| = sqrt(5)`. Each row sets `relative` to a
//! multiple of the machine epsilon, bracketing the recomputed residual of the
//! iterate after the second Lanczos step: on the exceeding rows the threshold
//! lies below it while the recurrence estimate `|phi_bar|` still meets the
//! threshold, so a solver that trusts the estimate reports `Converged` there.

use athena_core::{
    ConvergencePolicy, KrylovBackend, LinearOperator, Minres, MinresWorkspace, Termination,
};
use athena_leto::{CsrOperator, LetoBackend};
use eunomia::RealField;
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

use super::indefinite_matrix;

/// Where a row's threshold lies relative to the recomputed residual.
#[derive(Clone, Copy, Debug)]
enum Threshold {
    /// The recomputed residual meets the threshold: `Converged` at step two.
    Met,
    /// The recomputed residual exceeds the threshold although the recurrence
    /// estimate meets it.
    Exceeded,
}

/// Recompute `||b - A x||` through the backend, independent of the solver.
fn recomputed_residual_norm<T>(
    backend: LetoBackend<T>,
    operator: &CsrOperator<T>,
    right_hand_side: &Array1<T>,
    solution: &Array1<T>,
) -> T
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let mut image = backend.allocate(2).expect("invariant: host allocation");
    operator
        .apply(
            &backend,
            backend.view(solution),
            backend.view_mut(&mut image),
        )
        .expect("invariant: dimensions match");
    let mut residual = backend.allocate(2).expect("invariant: host allocation");
    backend
        .residual(
            backend.view(right_hand_side),
            backend.view(&image),
            backend.view_mut(&mut residual),
        )
        .expect("invariant: dimensions match");
    backend
        .norm_l2(backend.view(&residual))
        .expect("invariant: host norm")
}

/// Solve one row and return each contract violation, labelled by row.
fn row_violations<T>(relative_in_epsilons: f64, expected: Threshold) -> Vec<String>
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let operator = CsrOperator::new(indefinite_matrix()).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([2], vec![T::from_f64(1.0), T::from_f64(2.0)])
        .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        MinresWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let relative = T::from_f64(relative_in_epsilons) * <T as RealField>::EPSILON;
    let policy = ConvergencePolicy::new(T::from_f64(0.0), relative, 3)
        .expect("invariant: tolerances are finite and non-negative");

    let report = Minres::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    let recomputed = recomputed_residual_norm(backend, &operator, &right_hand_side, &solution);
    let row = format!(
        "{} with {relative_in_epsilons} epsilon: {report:?}, recomputed {recomputed:?}",
        core::any::type_name::<T>()
    );
    let mut violations = Vec::new();
    match expected {
        Threshold::Met => {
            if report.termination != Termination::Converged || report.iterations != 2 {
                violations.push(format!("{row}: expected Converged at iteration 2"));
            }
            if recomputed > report.threshold {
                violations.push(format!("{row}: Converged above the threshold"));
            }
        }
        Threshold::Exceeded => {
            // No stall test exists for MINRES, so a failed recompute leaves
            // the solve running until the budget ends at step three.
            if report.termination != Termination::MaxIterations || report.iterations != 3 {
                violations.push(format!("{row}: expected MaxIterations at iteration 3"));
            }
            if recomputed <= report.threshold {
                violations.push(format!("{row}: the row does not exceed its threshold"));
            }
        }
    }
    if report.final_residual_norm != recomputed {
        violations.push(format!(
            "{row}: reported residual is not the recomputed one"
        ));
    }
    violations
}

#[test]
fn minres_converged_certifies_true_residual() {
    let violations: Vec<String> = [
        row_violations::<f64>(0.25, Threshold::Exceeded),
        row_violations::<f64>(0.5, Threshold::Exceeded),
        row_violations::<f64>(4.0, Threshold::Met),
        row_violations::<f32>(1.0 / 3.0, Threshold::Exceeded),
        row_violations::<f32>(0.5, Threshold::Met),
    ]
    .into_iter()
    .flatten()
    .collect();
    assert_eq!(violations, Vec::<String>::new());
}

/// Pivot of the closed-space witness: `fl(107 * fl(1 / 107))` is `1 - EPSILON / 2`
/// in `f32` and in `f64`, the first pivot above 100 for which both round down.
const WITNESS_PIVOT: f64 = 107.0;

/// A closed Krylov space with a consistent, nonsingular system ends
/// `Breakdown` when the threshold lies below the attainable accuracy.
///
/// `diag(p, 2) x = (1, 0)` with `p = WITNESS_PIVOT`: `v1 = (1, 0)` is an
/// eigenvector, so `alpha = p`, the Lanczos remainder is exactly zero and the
/// recurrence estimate is exactly zero after one step. The iterate is
/// `x0 = fl(1 / p)`, and the recomputed residual `1 - fl(p * fl(1 / p))` is
/// `EPSILON / 2`, one unit in the last place below one. A relative tolerance
/// of `EPSILON / 4` lies below it, and the recurrence cannot continue because
/// the space is closed, so the run reports that the recurrence ended
/// (`Breakdown`) with the recomputed norm, not that the budget did. The
/// pre-fix solver reported `Converged` on the zero estimate.
fn closed_space_witness_violations<T>() -> Vec<String>
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let matrix = CsrMatrix::from_parts(
        vec![T::from_f64(WITNESS_PIVOT), T::from_f64(2.0)],
        vec![0, 1],
        vec![0, 1, 2],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([2], vec![T::from_f64(1.0), T::from_f64(0.0)])
        .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        MinresWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let relative = T::from_f64(0.25) * <T as RealField>::EPSILON;
    let policy = ConvergencePolicy::new(T::from_f64(0.0), relative, 3)
        .expect("invariant: tolerances are finite and non-negative");

    let report = Minres::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    let recomputed = recomputed_residual_norm(backend, &operator, &right_hand_side, &solution);
    let half_epsilon = T::from_f64(0.5) * <T as RealField>::EPSILON;
    let row = format!(
        "{}: {report:?}, recomputed {recomputed:?}",
        core::any::type_name::<T>()
    );
    let mut violations = Vec::new();
    if report.termination != Termination::Breakdown || report.iterations != 1 {
        violations.push(format!("{row}: expected Breakdown at iteration 1"));
    }
    // Seed, one step and the recompute.
    if report.operator_applications != 3 {
        violations.push(format!("{row}: expected 3 operator applications"));
    }
    if report.final_residual_norm != recomputed || recomputed != half_epsilon {
        violations.push(format!(
            "{row}: expected the recomputed residual EPSILON / 2"
        ));
    }
    if report.final_residual_norm <= report.threshold {
        violations.push(format!("{row}: the residual does not exceed the threshold"));
    }
    violations
}

#[test]
fn minres_reports_a_closed_space_above_the_threshold_as_breakdown() {
    let violations: Vec<String> = [
        closed_space_witness_violations::<f32>(),
        closed_space_witness_violations::<f64>(),
    ]
    .into_iter()
    .flatten()
    .collect();
    assert_eq!(violations, Vec::<String>::new());
}
