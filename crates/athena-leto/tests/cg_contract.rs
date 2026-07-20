//! Value-semantic CPU solver conformance.

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, IterationObserver, IterationState, SolveError,
    Termination,
};
use athena_leto::{BorrowedDenseOperator, CsrOperator, Jacobi, LetoBackend};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

fn spd_matrix<T>() -> CsrMatrix<T>
where
    T: RealScalar + FloatElement,
{
    CsrMatrix::from_parts(
        vec![
            T::from_f64(4.0),
            T::from_f64(1.0),
            T::from_f64(1.0),
            T::from_f64(3.0),
        ],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

fn verifies_spd_solution<T>()
where
    T: RealScalar + RealField + FloatElement + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let matrix = spd_matrix();
    let operator = CsrOperator::new(matrix.clone()).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([2], vec![T::from_f64(6.0), T::from_f64(7.0)])
        .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(
        T::from_f64(64.0) * <T as RealField>::EPSILON,
        T::from_f64(64.0) * <T as RealField>::EPSILON,
        4,
    )
    .expect("invariant: tolerance is finite and positive");

    let report = Cg::<LetoBackend<T>>::solve_into(
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
    let values = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    let bound = T::from_f64(256.0) * <T as RealField>::EPSILON;
    assert!((values[0] - T::from_f64(1.0)).abs() <= bound);
    assert!((values[1] - T::from_f64(2.0)).abs() <= bound);

    let jacobi = Jacobi::from_csr(&matrix).expect("invariant: diagonal is nonzero");
    solution.fill(<T as NumericElement>::ZERO);
    let report = Cg::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &jacobi,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: Jacobi solve has valid dimensions");
    assert_eq!(report.termination, Termination::Converged);
}

#[test]
fn cg_solves_spd_system_for_every_cpu_scalar() {
    verifies_spd_solution::<f32>();
    verifies_spd_solution::<f64>();
}

#[test]
fn borrowed_dense_operator_remains_zero_copy() {
    let backend = LetoBackend::<f64>::default();
    let coefficients = [4.0, 1.0, 1.0, 3.0];
    let operator =
        BorrowedDenseOperator::new(2, &coefficients).expect("invariant: dense shape is exact");
    assert!(operator.is_borrowed());

    let right_hand_side =
        Array1::from_shape_vec([2], vec![6.0, 7.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(64.0 * f64::EPSILON, 64.0 * f64::EPSILON, 4)
        .expect("invariant: valid policy");

    let report = Cg::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve succeeds");

    assert_eq!(report.termination, Termination::Converged);
    assert!(operator.is_borrowed());
}

#[test]
fn cg_reports_non_positive_curvature() {
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(vec![-1.0], vec![0], vec![0, 1], 1, 1)
        .expect("invariant: one-entry CSR is valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([1], vec![1.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([1]);
    let mut workspace = CgWorkspace::new(&backend, 1).expect("invariant: host allocation succeeds");
    let policy =
        ConvergencePolicy::new(f64::EPSILON, f64::EPSILON, 2).expect("invariant: valid policy");

    let report = Cg::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: numerical termination is value-semantic");

    assert_eq!(report.termination, Termination::NonPositiveCurvature);
    assert_eq!(report.iterations, 0);
}

#[test]
fn zero_right_hand_side_converges_without_iteration() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(spd_matrix()).expect("invariant: matrix is square");
    let right_hand_side = Array1::zeros([2]);
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(0.0, 0.0, 2).expect("invariant: valid policy");

    let report = Cg::<LetoBackend<f64>>::solve_into(
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
    assert!(
        solution
            .as_slice()
            .expect("invariant: Array1 is contiguous")
            .iter()
            .all(|value| value.to_bits() == 0.0_f64.to_bits())
    );
}

#[test]
fn iteration_budget_is_a_value_semantic_termination() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(spd_matrix()).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([2], vec![6.0, 7.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(0.0, 0.0, 1).expect("invariant: valid policy");

    let report = Cg::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("iteration exhaustion is not a backend failure");

    assert_eq!(report.termination, Termination::MaxIterations);
    assert_eq!(report.iterations, 1);
}

#[test]
fn cg_rejects_vector_dimension_mismatch_before_dispatch() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(spd_matrix()).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([1], vec![1.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy =
        ConvergencePolicy::new(f64::EPSILON, f64::EPSILON, 2).expect("invariant: valid policy");

    let error = Cg::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect_err("a length-one right-hand side must be rejected");

    assert!(matches!(
        error,
        SolveError::DimensionMismatch {
            context: "right-hand side",
            expected: 2,
            actual: 1,
        }
    ));
}

#[derive(Default)]
struct ResidualTrace {
    values: Vec<f64>,
}

impl IterationObserver<f64> for ResidualTrace {
    fn observe(&mut self, state: IterationState<f64>) {
        self.values.push(state.residual_norm);
    }
}

#[test]
fn observer_receives_decreasing_checked_residuals() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(spd_matrix()).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([2], vec![6.0, 7.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(64.0 * f64::EPSILON, 64.0 * f64::EPSILON, 4)
        .expect("invariant: valid policy");
    let mut trace = ResidualTrace::default();

    let report = Cg::<LetoBackend<f64>>::solve_with_observer(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
        &mut trace,
    )
    .expect("invariant: manufactured solve succeeds");

    assert_eq!(report.termination, Termination::Converged);
    assert!(!trace.values.is_empty());
    assert!(trace.values.windows(2).all(|pair| pair[1] <= pair[0]));
}

#[test]
fn backend_and_algorithm_markers_are_zero_sized() {
    assert_eq!(core::mem::size_of::<LetoBackend<f64>>(), 0);
    assert_eq!(core::mem::size_of::<Cg<LetoBackend<f64>>>(), 0);
    assert_eq!(core::mem::size_of::<Identity>(), 0);
}
