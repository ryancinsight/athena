//! Value-semantic CPU solver conformance.

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, IterationObserver, IterationState, KrylovBackend,
    SolveError, SolveReport, Termination,
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

fn assert_manufactured_solution<T>(solution: &Array1<T>, bound: T)
where
    T: RealField + core::fmt::Debug,
{
    let values = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    assert_eq!(values.len(), 2, "manufactured system dimension");
    assert!((values[0] - T::from_f64(1.0)).abs() <= bound);
    assert!((values[1] - T::from_f64(2.0)).abs() <= bound);

    // Direct substitution verifies A*x=b independently of PCG. The matrix
    // infinity norm is five, so six times the forward bound covers propagated
    // solution error plus the final row-operation rounding.
    let residual_bound = T::from_f64(6.0) * bound;
    let first = T::from_f64(4.0) * values[0] + values[1] - T::from_f64(6.0);
    let second = values[0] + T::from_f64(3.0) * values[1] - T::from_f64(7.0);
    assert!(first.abs() <= residual_bound);
    assert!(second.abs() <= residual_bound);
}

fn verifies_spd_solution<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
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
    // Two length-two recurrence steps perform fewer than 32 rounded scalar
    // operations. Multiplying by κ∞(A)=25/11 and a factor-four safety margin
    // remains below 256 first-order rounding units.
    let bound = T::from_f64(256.0) * <T as RealField>::EPSILON;
    assert_manufactured_solution(&solution, bound);

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
    assert_manufactured_solution(&solution, bound);
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

/// `||b - A x||` for `A = [[4, 1], [1, 3]]`, by direct substitution.
///
/// For two entries the backend norm is one rounding of `fl(a0*a0) + fl(a1*a1)`
/// in every dot dispatch arm, so it equals this value bitwise.
fn original_residual_norm<T: RealField>(x: &[T], rhs: [T; 2]) -> T {
    let first = rhs[0] - (T::from_f64(4.0) * x[0] + x[1]);
    let second = rhs[1] - (x[0] + T::from_f64(3.0) * x[1]);
    (first * first + second * second).sqrt()
}

fn budget_exit_reports_the_original_residual<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let operator = CsrOperator::new(spd_matrix()).expect("invariant: matrix is square");
    let rhs = [T::from_f64(6.0), T::from_f64(7.0)];
    let right_hand_side =
        Array1::from_shape_vec([2], rhs.to_vec()).expect("invariant: exact shape");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let zero = <T as NumericElement>::ZERO;
    let policy = ConvergencePolicy::new(zero, zero, 1).expect("invariant: valid policy");

    let report = Cg::<LetoBackend<T>>::solve_into(
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
    // The initial residual, the CG step and the budget-exit verification
    // each apply A once; the recurrence norm alone would give two.
    assert_eq!(report.operator_applications, 3);
    let x = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    assert_eq!(report.final_residual_norm, original_residual_norm(x, rhs));
}

#[test]
fn iteration_budget_is_a_value_semantic_termination() {
    budget_exit_reports_the_original_residual::<f32>();
    budget_exit_reports_the_original_residual::<f64>();
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

    let backend = LetoBackend::<f64>::default();
    let left = Array1::zeros([2]);
    let right = Array1::zeros([2]);
    let dot = backend
        .prepare_dot(left.view(), right.view())
        .expect("equal-length host views must prepare");
    let norm = backend
        .prepare_norm_l2(left.view())
        .expect("host norm preparation cannot fail");
    assert_eq!(core::mem::size_of_val(&dot), 0);
    assert_eq!(core::mem::size_of_val(&norm), 0);
}

/// Thread-scoped subscriber: installing it cannot fail or leak across tests.
fn test_subscriber() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(tracing_subscriber::fmt().with_test_writer().finish())
}

#[derive(Default)]
struct Trace<T>(Vec<IterationState<T>>);

impl<T> IterationObserver<T> for Trace<T> {
    fn observe(&mut self, state: IterationState<T>) {
        self.0.push(state);
    }
}

fn exact_identity_solution<T: RealScalar + RealField + core::fmt::Debug>() {
    let _span = tracing::info_span!("exact_cg_terminal_residual").entered();
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let two = one + one;
    let backend = LetoBackend::<T>::default();
    let coefficients = [one, zero, zero, zero, one, zero, zero, zero, one];
    let operator = BorrowedDenseOperator::new(3, &coefficients)
        .expect("the identity is a square dense operator");
    let rhs = Array1::from_shape_vec([3], vec![one, two, -two])
        .expect("three entries match the vector shape");

    for interval in [1, 2, 4] {
        let mut solution = Array1::zeros([3]);
        let mut workspace = CgWorkspace::new(&backend, 3).expect("valid host workspace");
        let policy = ConvergencePolicy::with_check_interval(zero, zero, 4, interval)
            .expect("zero tolerances and positive counts are valid");
        let mut trace = Trace(Vec::new());
        let report = Cg::solve_with_observer(
            &backend,
            &operator,
            &Identity,
            &rhs,
            &mut solution,
            &mut workspace,
            policy,
            &mut trace,
        )
        .expect("matching dimensions and real CPU operations");

        // A=I gives alpha=1 and x=b after one step. All products/sums
        // are small integers, hence exact in both native scalar formats.
        assert_eq!(
            solution.as_slice().expect("contiguous solution"),
            [one, two, -two]
        );
        assert_eq!(
            report.termination,
            Termination::Converged,
            "interval={interval}, report={report:?}"
        );
        assert_eq!(report.iterations, 1);
        assert_eq!(
            report.operator_applications, 3,
            "initial, CG step and original residual"
        );
        assert_eq!(report.initial_residual_norm, one + two);
        assert_eq!(report.final_residual_norm, zero);
        assert_eq!(report.threshold, zero);
        assert_eq!(trace.0.len(), 1, "one terminal check, no duplicate sample");
        assert_eq!(trace.0[0].iteration, 1);
        assert_eq!(trace.0[0].residual_norm, zero);
    }
}

#[test]
fn cg_reports_exact_solution_between_residual_checks() {
    let _guard = test_subscriber();
    exact_identity_solution::<f32>();
    exact_identity_solution::<f64>();
}

fn underflow_is_not_convergence<T: RealScalar + RealField + core::fmt::Debug>(tiny: T) {
    let _span = tracing::info_span!("nonzero_cg_terminal_residual").entered();
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let backend = LetoBackend::<T>::default();
    let coefficients = [one, tiny, tiny, one];
    let operator = BorrowedDenseOperator::new(2, &coefficients).expect("SPD operator");
    let diagonal = CsrMatrix::from_parts(vec![one, one / tiny], vec![0, 1], vec![0, 1, 2], 2, 2)
        .expect("positive diagonal matrix");
    let preconditioner = Jacobi::from_csr(&diagonal).expect("finite nonzero diagonal");
    let rhs = Array1::from_shape_vec([2], vec![one, zero]).expect("exact vector shape");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("valid workspace");
    let policy = ConvergencePolicy::with_check_interval(zero, zero, 4, 4).expect("valid policy");
    let mut trace = Trace(Vec::new());
    let report = Cg::solve_with_observer(
        &backend,
        &operator,
        &preconditioner,
        &rhs,
        &mut solution,
        &mut workspace,
        policy,
        &mut trace,
    )
    .expect("matching dimensions");

    // First step gives x=(1,0), r=(0,-tiny), z=(0,-tiny²).
    // tiny² is representable, tiny³ underflows: rho=0 is not a zero residual.
    assert_eq!(solution.as_slice().expect("contiguous"), [one, zero]);
    assert_eq!(report.termination, Termination::Breakdown);
    assert_eq!(report.iterations, 1);
    assert_eq!(report.final_residual_norm, tiny);
    assert_eq!(trace.0.len(), 1);
    assert_eq!(trace.0[0].iteration, 1);
    assert_eq!(trace.0[0].residual_norm, tiny);
}

#[test]
fn cg_checks_the_residual_before_classifying_a_zero_product() {
    let _guard = test_subscriber();
    underflow_is_not_convergence(2.0_f32.powi(-60));
    underflow_is_not_convergence(2.0_f64.powi(-400));
}

fn checked_original_system<T: RealScalar + RealField + core::fmt::Debug>() -> SolveReport<T> {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let three = one + one + one;
    let four = three + one;
    let backend = LetoBackend::<T>::default();
    let coefficients = [four, one, one, three];
    let operator = BorrowedDenseOperator::new(2, &coefficients).expect("SPD operator");
    let rhs = Array1::from_shape_vec([2], vec![one, one]).expect("exact vector shape");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("valid workspace");
    let policy = ConvergencePolicy::new(zero, zero, 32).expect("valid policy");
    let report = Cg::solve_into(
        &backend,
        &operator,
        &Identity,
        &rhs,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("matching dimensions");
    let x = solution.as_slice().expect("contiguous solution");
    let true_norm = original_residual_norm(x, [one, one]);
    // Zero tolerance: Converged exactly when the original residual is zero.
    assert_eq!(
        report.termination == Termination::Converged,
        true_norm == zero,
        "x={x:?}, true residual norm={true_norm:?}, report={report:?}",
    );
    assert_eq!(report.final_residual_norm, true_norm);
    report
}

#[test]
fn cg_checks_original_system_before_declaring_convergence() {
    let _guard = test_subscriber();
    // f32 reaches the exact solution; f64 recurrence drift reaches a zero
    // product with a true residual of 2^-52 (observed at the locked
    // backend revision), which the unverified solver reported as Converged.
    let report = checked_original_system::<f32>();
    assert_eq!(report.termination, Termination::Converged);
    assert_eq!(report.final_residual_norm.to_bits(), 0.0_f32.to_bits());
    let report = checked_original_system::<f64>();
    assert_eq!(report.termination, Termination::Breakdown);
    assert_eq!(report.iterations, 21);
    assert_eq!(report.final_residual_norm.to_bits(), f64::EPSILON.to_bits());
}

fn curvature_exit_reports_current_residual<T: RealScalar + RealField + core::fmt::Debug>() {
    let zero = <T as NumericElement>::ZERO;
    let one = <T as NumericElement>::ONE;
    let two = one + one;
    let backend = LetoBackend::<T>::default();
    // After x=(1,0), r=(0,-2), p=(4,-2). The next curvature
    // is negative for A22=1 and zero for A22=4.
    for (diagonal, termination) in [
        (one, Termination::NonPositiveCurvature),
        (two + two, Termination::Breakdown),
    ] {
        let coefficients = [one, two, two, diagonal];
        let operator = BorrowedDenseOperator::new(2, &coefficients).expect("square operator");
        let rhs = Array1::from_shape_vec([2], vec![one, zero]).expect("exact vector shape");
        let mut solution = Array1::zeros([2]);
        let mut workspace = CgWorkspace::new(&backend, 2).expect("valid workspace");
        let policy =
            ConvergencePolicy::with_check_interval(zero, zero, 4, 4).expect("valid policy");
        let mut trace = Trace(Vec::new());
        let report = Cg::solve_with_observer(
            &backend,
            &operator,
            &Identity,
            &rhs,
            &mut solution,
            &mut workspace,
            policy,
            &mut trace,
        )
        .expect("matching dimensions");
        assert_eq!(solution.as_slice().expect("contiguous"), [one, zero]);
        assert_eq!(report.termination, termination);
        assert_eq!(report.iterations, 1);
        assert_eq!(report.operator_applications, 4);
        assert_eq!(report.final_residual_norm, two);
        assert_eq!(trace.0.len(), 1);
        assert_eq!(trace.0[0].iteration, 1);
        assert_eq!(trace.0[0].residual_norm, two);
    }
}

#[test]
fn cg_reports_current_residual_when_the_next_direction_breaks_down() {
    let _guard = test_subscriber();
    curvature_exit_reports_current_residual::<f32>();
    curvature_exit_reports_current_residual::<f64>();
}
