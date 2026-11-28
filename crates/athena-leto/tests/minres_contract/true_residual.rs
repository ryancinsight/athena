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
use leto_ops::RealScalar;

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
            if !matches!(
                report.termination,
                Termination::MaxIterations | Termination::Stagnated
            ) {
                violations.push(format!("{row}: expected MaxIterations or Stagnated"));
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
