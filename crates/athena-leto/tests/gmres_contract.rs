//! Value-semantic restarted-GMRES CPU conformance.

use athena_core::{ConvergencePolicy, Gmres, GmresWorkspace, Identity, SolveError, Termination};
use athena_leto::{CsrOperator, Jacobi, LetoBackend};
use eunomia::{FloatElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

fn nonsymmetric_matrix<T>() -> CsrMatrix<T>
where
    T: RealScalar + FloatElement,
{
    CsrMatrix::from_parts(
        vec![
            T::from_f64(4.0),
            T::from_f64(1.0),
            T::from_f64(2.0),
            T::from_f64(3.0),
            T::from_f64(1.0),
            T::from_f64(1.0),
            T::from_f64(2.0),
        ],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![0, 2, 5, 7],
        3,
        3,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

fn assert_manufactured_solution<T>(solution: &Array1<T>, bound: T)
where
    T: RealField + FloatElement + core::fmt::Debug,
{
    let values = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    assert!((values[0] - T::from_f64(1.0)).abs() <= bound);
    assert!((values[1] - T::from_f64(-2.0)).abs() <= bound);
    assert!((values[2] - T::from_f64(3.0)).abs() <= bound);

    // Independent direct substitution verifies A*x=b in addition to comparing
    // the solution vector. The matrix infinity norm is six, so this residual
    // bound is six times the forward-error bound plus one rounding per row.
    let residual_bound = T::from_f64(8.0) * bound;
    let first = T::from_f64(4.0) * values[0] + values[1] - T::from_f64(2.0);
    let second = T::from_f64(2.0) * values[0] + T::from_f64(3.0) * values[1] + values[2] + T::ONE;
    let third = values[1] + T::from_f64(2.0) * values[2] - T::from_f64(4.0);
    assert!(first.abs() <= residual_bound);
    assert!(second.abs() <= residual_bound);
    assert!(third.abs() <= residual_bound);
}

fn verifies_nonsymmetric_solution<T>()
where
    T: RealScalar + RealField + FloatElement + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let matrix = nonsymmetric_matrix();
    let operator = CsrOperator::new(matrix.clone()).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec(
        [3],
        vec![T::from_f64(2.0), T::from_f64(-1.0), T::from_f64(4.0)],
    )
    .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        GmresWorkspace::<_, 3>::new(&backend, 3).expect("invariant: host allocation succeeds");
    // One MGS pass, three Arnoldi columns, and one triangular backsolve give
    // O(n² ε) forward error for this well-conditioned manufactured system.
    let bound = T::from_f64(4096.0) * T::EPSILON;
    let policy = ConvergencePolicy::new(bound, bound, 6)
        .expect("invariant: tolerance is finite and non-negative");

    let report = Gmres::<LetoBackend<T>, 3>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert_eq!(report.termination, Termination::Converged);
    assert_manufactured_solution(&solution, bound);

    let jacobi = Jacobi::from_csr(&matrix).expect("invariant: diagonal is nonzero");
    solution.fill(T::ZERO);
    let report = Gmres::<LetoBackend<T>, 3>::solve_into(
        &backend,
        &operator,
        &jacobi,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: right-preconditioned solve has valid dimensions");
    assert_eq!(report.termination, Termination::Converged);
    assert_manufactured_solution(&solution, bound);
}

#[test]
fn gmres_solves_nonsymmetric_system_for_every_cpu_scalar() {
    verifies_nonsymmetric_solution::<f32>();
    verifies_nonsymmetric_solution::<f64>();
}

#[test]
fn restart_width_smaller_than_dimension_executes_multiple_cycles() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(nonsymmetric_matrix()).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([3], vec![2.0, -1.0, 4.0])
        .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        GmresWorkspace::<_, 2>::new(&backend, 3).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(4096.0 * f64::EPSILON, 4096.0 * f64::EPSILON, 30)
        .expect("invariant: valid policy");

    let report = Gmres::<LetoBackend<f64>, 2>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: restarted solve succeeds");

    assert_eq!(report.termination, Termination::Converged);
    assert!(report.iterations > workspace.restart_width());
    let values = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    let bound = 8192.0 * f64::EPSILON;
    assert!((values[0] - 1.0).abs() <= bound);
    assert!((values[1] + 2.0).abs() <= bound);
    assert!((values[2] - 3.0).abs() <= bound);
}

#[test]
fn gmres_algorithm_marker_is_zero_sized() {
    assert_eq!(core::mem::size_of::<Gmres<LetoBackend<f64>, 3>>(), 0);
}

#[test]
fn gmres_rejects_dimension_mismatch_before_dispatch() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(nonsymmetric_matrix()).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([2], vec![1.0, 2.0]).expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        GmresWorkspace::<_, 3>::new(&backend, 3).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(0.0, 0.0, 3).expect("invariant: zero tolerance is valid");

    let error = Gmres::<LetoBackend<f64>, 3>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect_err("right-hand-side length must be validated");

    assert!(matches!(
        error,
        SolveError::DimensionMismatch {
            context: "right-hand side",
            expected: 3,
            actual: 2,
        }
    ));
}

#[test]
fn gmres_reports_budget_and_breakdown_value_semantically() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(nonsymmetric_matrix()).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([3], vec![2.0, -1.0, 4.0])
        .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        GmresWorkspace::<_, 1>::new(&backend, 3).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(0.0, 0.0, 1).expect("invariant: exact policy is valid");
    let report = Gmres::<LetoBackend<f64>, 1>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: one Arnoldi step is valid");
    assert_eq!(report.termination, Termination::MaxIterations);
    assert_eq!(report.iterations, 1);

    let zero = CsrMatrix::from_parts(Vec::new(), Vec::new(), vec![0, 0, 0], 2, 2)
        .expect("invariant: empty square CSR is valid");
    let zero_operator = CsrOperator::new(zero).expect("invariant: matrix is square");
    let zero_rhs =
        Array1::from_shape_vec([2], vec![1.0, 0.0]).expect("invariant: vector shape is exact");
    let mut zero_solution = Array1::zeros([2]);
    let mut zero_workspace =
        GmresWorkspace::<_, 2>::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(0.0, 0.0, 2).expect("invariant: exact policy is valid");
    let report = Gmres::<LetoBackend<f64>, 2>::solve_into(
        &backend,
        &zero_operator,
        &Identity,
        &zero_rhs,
        &mut zero_solution,
        &mut zero_workspace,
        policy,
    )
    .expect("invariant: singular recurrence terminates value-semantically");
    assert_eq!(report.termination, Termination::Breakdown);
    assert_eq!(report.iterations, 1);
}

#[test]
fn gmres_zero_right_hand_side_converges_without_iteration() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(nonsymmetric_matrix()).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([3], vec![0.0; 3]).expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        GmresWorkspace::<_, 3>::new(&backend, 3).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(0.0, 0.0, 3).expect("invariant: exact policy is valid");

    let report = Gmres::<LetoBackend<f64>, 3>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: zero system is valid");

    assert_eq!(report.termination, Termination::InitialResidual);
    assert_eq!(report.iterations, 0);
}
