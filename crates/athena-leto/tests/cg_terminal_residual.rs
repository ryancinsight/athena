//! Exact CG termination must not depend on the scheduled check interval.

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, IterationObserver, IterationState, Termination,
};
use athena_leto::{BorrowedDenseOperator, Jacobi, LetoBackend};
use eunomia::{NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

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
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
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
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    underflow_is_not_convergence(2.0_f32.powi(-60));
    underflow_is_not_convergence(2.0_f64.powi(-400));
}

fn checked_original_system<T: RealScalar + RealField + core::fmt::Debug>() {
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
    let residual = [one - (four * x[0] + x[1]), one - (x[0] + three * x[1])];
    assert!(
        report.termination != Termination::Converged || residual == [zero, zero],
        "zero tolerance cannot accept x={x:?}, true residual={residual:?}, report={report:?}",
    );
    let true_norm = (residual[0] * residual[0] + residual[1] * residual[1]).sqrt();
    assert_eq!(report.final_residual_norm, true_norm);
}

#[test]
fn cg_checks_original_system_before_declaring_convergence() {
    checked_original_system::<f32>();
    checked_original_system::<f64>();
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
    curvature_exit_reports_current_residual::<f32>();
    curvature_exit_reports_current_residual::<f64>();
}
