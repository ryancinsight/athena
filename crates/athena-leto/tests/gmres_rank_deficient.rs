//! Attainable rank-deficient Arnoldi factors and checked GMRES telemetry.
use athena_core::{
    ConvergencePolicy, Gmres, GmresWorkspace, Identity, IterationObserver, IterationState,
    SolveReport, Termination,
};
use athena_leto::{CsrOperator, LetoBackend};
use eunomia::RealField;
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

#[derive(Debug)]
struct History<T>(Vec<IterationState<T>>);
impl<T> IterationObserver<T> for History<T> {
    fn observe(&mut self, state: IterationState<T>) {
        self.0.push(state);
    }
}
#[derive(Debug)]
struct Outcome<T> {
    solution: Vec<T>,
    report: SolveReport<T>,
    history: Vec<IterationState<T>>,
    residual_squared: T,
}
fn with_trace(run: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(FmtSpan::ENTER | FmtSpan::CLOSE)
        .finish();
    tracing::subscriber::with_default(subscriber, run);
}
/// Aδ has rank two, Arnoldi basis e1,e2,e3 and exact minimum residual²1/2:
/// ‖e1−Aδx‖² = 1/2 + 2(δx0+x2−1/2)² + x1². Both nonzero singular
/// values stay O(1), even when the unpivoted leading diagonal is tiny.
fn solve<T>(delta: T, interval: usize) -> Outcome<T>
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let (backend, operator, rhs, mut solution, mut workspace, policy) = {
        let _arrange = info_span!(
            "arrange",
            scalar = core::any::type_name::<T>(),
            ?delta,
            interval
        )
        .entered();
        let zero = T::from_f64(0.0);
        let one = T::from_f64(1.0);
        let matrix = CsrMatrix::from_parts(
            vec![delta, one, delta, one, one],
            vec![0, 2, 0, 2, 1],
            vec![0, 2, 4, 5],
            3,
            3,
        )
        .expect("analytic CSR");
        let backend = LetoBackend::<T>::default();
        let operator = CsrOperator::new(matrix).expect("square analytic operator");
        let rhs = Array1::from_shape_vec([3], vec![one, zero, zero]).expect("exact shape");
        let solution = Array1::zeros([3]);
        let workspace = GmresWorkspace::<_, 3>::new(&backend, 3).expect("host workspace");
        let tolerance = T::from_f64(32.0) * <T as RealField>::EPSILON;
        let policy = ConvergencePolicy::with_check_interval(tolerance, tolerance, 6, interval)
            .expect("valid policy");
        (backend, operator, rhs, solution, workspace, policy)
    };
    let mut history = History(Vec::new());
    let report = {
        let _act = info_span!("act").entered();
        Gmres::<LetoBackend<T>, 3>::solve_with_observer(
            &backend,
            &operator,
            &Identity,
            &rhs,
            &mut solution,
            &mut workspace,
            policy,
            &mut history,
        )
        .expect("valid solve")
    };
    let values = solution.as_slice().expect("contiguous vector").to_vec();
    let range = delta * values[0] + values[2];
    let offset = range - T::from_f64(0.5);
    let residual_squared =
        T::from_f64(0.5) + T::from_f64(2.0) * offset * offset + values[1] * values[1];
    Outcome {
        solution: values,
        report,
        history: history.0,
        residual_squared,
    }
}
fn assert_minimum<T>(outcome: &Outcome<T>)
where
    T: RealField + core::fmt::Debug,
{
    let _assert = info_span!("assert", ?outcome).entered();
    assert_eq!(
        outcome.report.termination,
        Termination::Breakdown,
        "inconsistent system"
    );
    assert_eq!(
        outcome.report.iterations, 3,
        "full attained basis before exact breakdown"
    );
    let bound = T::from_f64(32.0) * <T as RealField>::EPSILON;
    assert!(
        (outcome.residual_squared - T::from_f64(0.5)).abs() <= bound,
        "reachable minimum residual²1/2: {outcome:?}"
    );
    assert!(
        (outcome.report.final_residual_norm * outcome.report.final_residual_norm
            - outcome.residual_squared)
            .abs()
            <= bound,
        "checked final residual: {outcome:?}"
    );
    assert!(
        outcome.solution.iter().all(|x| x.is_finite()),
        "finite basic correction"
    );
}
fn rank_recovery<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let eps = <T as RealField>::EPSILON;
    for delta in [T::from_f64(1.0), eps * eps, -eps * eps] {
        assert_minimum(&solve(delta, 1));
    }
}
fn telemetry<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let delta = <T as RealField>::EPSILON * <T as RealField>::EPSILON;
    for interval in [1, 2, 4] {
        let outcome = solve(delta, interval);
        let _assert = info_span!("assert", ?outcome, interval).entered();
        let bound = T::from_f64(32.0) * <T as RealField>::EPSILON;
        let last = outcome.history.last().expect("terminal checked sample");
        assert_eq!(last.iteration, outcome.report.iterations);
        assert!(
            (last.residual_norm - outcome.report.final_residual_norm).abs() <= bound,
            "terminal observer must use checked residual: {outcome:?}"
        );
        for sample in &outcome.history {
            assert!(
                sample.residual_norm + bound >= T::from_f64(0.5).sqrt(),
                "no iterate can beat this system's exact lower bound: {outcome:?}"
            );
        }
        let expected_samples = outcome.report.iterations / interval
            + usize::from(!outcome.report.iterations.is_multiple_of(interval));
        assert_eq!(
            outcome.history.len(),
            expected_samples,
            "one sample per check, no earlier false zero"
        );
    }
}
#[test]
fn attained_rank_deficiency_retains_the_least_squares_minimum() {
    with_trace(|| {
        rank_recovery::<f32>();
        rank_recovery::<f64>();
    });
}
#[test]
fn breakdown_observer_uses_the_checked_cycle_residual() {
    with_trace(|| {
        telemetry::<f32>();
        telemetry::<f64>();
    });
}
