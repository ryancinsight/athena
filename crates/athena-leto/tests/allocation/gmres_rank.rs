//! The existing allocation-instrument lane explicitly runs this ignored
//! strict contract alongside its full-rank GMRES contract.
use super::{assert_steady_state, warm};
use athena_core::{ConvergencePolicy, Gmres, GmresWorkspace, Identity, Termination};
use athena_leto::{CsrOperator, LetoBackend};
use leto::Array1;
use leto_ops::CsrMatrix;
use mnemosyne::counting::measure;

#[test]
#[ignore = "strict zero-traffic contract; explicitly run by the existing allocation-instrument job with ignored-only"]
fn rank_recovery_reuses_workspace_without_heap_traffic() {
    warm();
    let subscriber = tracing_subscriber::fmt().with_test_writer().finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let arrange = tracing::info_span!("arrange").entered();
    let delta = f64::EPSILON * f64::EPSILON;
    let matrix = CsrMatrix::from_parts(
        vec![delta, 1.0, delta, 1.0, 1.0],
        vec![0, 2, 0, 2, 1],
        vec![0, 2, 4, 5],
        3,
        3,
    )
    .expect("analytic rank-two CSR");
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(matrix).expect("square");
    let rhs = Array1::from_shape_vec([3], vec![1.0, 0.0, 0.0]).expect("exact shape");
    let mut solution = Array1::zeros([3]);
    let mut workspace = GmresWorkspace::<_, 3>::new(&backend, 3).expect("host workspace");
    let policy =
        ConvergencePolicy::new(32.0 * f64::EPSILON, 32.0 * f64::EPSILON, 6).expect("valid policy");
    let report = Gmres::<LetoBackend<f64>, 3>::solve_into(
        &backend,
        &operator,
        &Identity,
        &rhs,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("warm solve");
    assert_eq!(report.termination, Termination::Breakdown);
    assert!(
        (report.final_residual_norm * report.final_residual_norm - 0.5).abs()
            <= 32.0 * f64::EPSILON
    );
    solution.fill(0.0);
    drop(arrange);
    let act = tracing::info_span!("act").entered();
    let ((), change) = measure(|| {
        for _ in 0..32 {
            let report = Gmres::<LetoBackend<f64>, 3>::solve_into(
                &backend,
                &operator,
                &Identity,
                &rhs,
                &mut solution,
                &mut workspace,
                policy,
            )
            .expect("measured solve");
            assert_eq!(report.termination, Termination::Breakdown);
            assert!(
                (report.final_residual_norm * report.final_residual_norm - 0.5).abs()
                    <= 32.0 * f64::EPSILON
            );
            solution.fill(0.0);
        }
    });
    drop(act);
    let _assert = tracing::info_span!("assert", ?change).entered();
    assert_steady_state(change);
}
