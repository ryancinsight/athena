//! Value-semantic CPU solver conformance for MINRES on symmetric indefinite
//! systems.
//!
//! The oracle is ADR 0062's Phase-2 table under ADR 0056's new-construction
//! rule: MINRES recovers the closed-form solution of a constructed
//! symmetric-indefinite system on which CG must fail (its first curvature
//! test is negative), at `f32` and `f64`.

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, Minres, MinresWorkspace, SolveError, SolveReport,
    Termination,
};
use athena_leto::{CsrOperator, LetoBackend};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

/// The ADR 0062 oracle operator: `diag(1, −1)`, symmetric indefinite with
/// eigenvalues of mixed sign and condition number one.
fn indefinite_matrix<T>() -> CsrMatrix<T>
where
    T: RealScalar + FloatElement,
{
    CsrMatrix::from_parts(
        vec![T::from_f64(1.0), T::from_f64(-1.0)],
        vec![0, 1],
        vec![0, 1, 2],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

/// Assert both components of the closed-form solution `(1, −2)`, and verify
/// them by direct substitution so the oracle does not trust the solver's own
/// residual path.
fn assert_closed_form<T>(solution: &Array1<T>, bound: T)
where
    T: RealField + core::fmt::Debug,
{
    let values = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    assert_eq!(values.len(), 2, "manufactured system dimension");
    assert!(
        (values[0] - T::from_f64(1.0)).abs() <= bound,
        "first component of the closed form: {:?}",
        values[0]
    );
    assert!(
        (values[1] - T::from_f64(-2.0)).abs() <= bound,
        "second component of the closed form: {:?}",
        values[1]
    );
    // Direct substitution of A·x against b: the diagonal rows give
    // x₀ − 1 = 0 and −x₁ − 2 = 0. The row entries have magnitude one, so the
    // same bound covers the propagated solution error plus one final
    // row-operation rounding.
    let first = values[0] - T::from_f64(1.0);
    let second = -values[1] - T::from_f64(2.0);
    assert!(
        first.abs() <= T::from_f64(2.0) * bound,
        "substitution x0-1 = {first:?}"
    );
    assert!(
        second.abs() <= T::from_f64(2.0) * bound,
        "substitution -x1-2 = {second:?}"
    );
}

/// Error bound for one MINRES solve of the `diag(1, −1)` oracle.
///
/// Two Lanczos steps over length-two vectors perform fewer than 48 rounded
/// scalar operations with unit amplification (κ(A) = 1); 256 machine
/// epsilons is a factor-of-five margin over that first-order count.
fn indefinite_oracle_bound<T>() -> T
where
    T: RealField,
{
    T::from_f64(256.0) * <T as RealField>::EPSILON
}

fn solves_the_indefinite_oracle<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let matrix = indefinite_matrix();
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([2], vec![T::from_f64(1.0), T::from_f64(2.0)])
        .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        MinresWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    // The operator has two distinct eigenvalues, so the Krylov space is
    // complete after two Lanczos steps; the policy budget is set to three to
    // leave the completion of the space to the recurrence rather than the cap.
    let policy = ConvergencePolicy::new(<T as RealField>::EPSILON, <T as RealField>::EPSILON, 3)
        .expect("invariant: tolerance is finite and positive");

    let report = Minres::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert_eq!(report.termination, Termination::Converged);
    assert_closed_form(&solution, indefinite_oracle_bound());

    // CG must fail on this operator: its first denominator is the Rayleigh
    // quotient of the residual direction, dᵀ·A·d = 1 − 4 = −3 < 0, reported
    // value-semantically as NonPositiveCurvature.
    let mut cg_solution = Array1::zeros([2]);
    let mut cg_workspace =
        CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let cg_report = Cg::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut cg_solution,
        &mut cg_workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");
    assert_eq!(
        cg_report.termination,
        Termination::NonPositiveCurvature,
        "CG must fail on the indefinite oracle: the ADR 0062 oracle statement"
    );
}

#[test]
fn minres_solves_the_indefinite_oracle_for_every_cpu_scalar() {
    solves_the_indefinite_oracle::<f32>();
    solves_the_indefinite_oracle::<f64>();
}

/// One solve of the `diag(1, −1)` oracle with `b = (1, 2)` from `x₀ = 0` on
/// the given workspace, returning the solution and its report.
fn solve_indefinite_oracle<T>(
    backend: LetoBackend<T>,
    workspace: &mut MinresWorkspace<LetoBackend<T>>,
) -> (Array1<T>, SolveReport<T>)
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let operator = CsrOperator::new(indefinite_matrix()).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([2], vec![T::from_f64(1.0), T::from_f64(2.0)])
        .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([2]);
    let policy = ConvergencePolicy::new(<T as RealField>::EPSILON, <T as RealField>::EPSILON, 3)
        .expect("invariant: tolerance is finite and positive");
    let report = Minres::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");
    (solution, report)
}

/// Solve the oracle on `workspace`, then on a fresh workspace, and require
/// the two to agree.
///
/// A fresh workspace executes the identical operation sequence on identical
/// inputs, so the reused solve must match it bit for bit (evaluation order
/// is the same, so no reduction-order tolerance applies) and end after the
/// same number of steps, on the closed form within the oracle bound.
fn assert_oracle_matches_fresh_workspace<T>(
    backend: LetoBackend<T>,
    workspace: &mut MinresWorkspace<LetoBackend<T>>,
) where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let (reused, reused_report) = solve_indefinite_oracle(backend, workspace);
    assert_eq!(reused_report.termination, Termination::Converged);
    assert_closed_form(&reused, indefinite_oracle_bound());

    let mut fresh = MinresWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let (reference, reference_report) = solve_indefinite_oracle(backend, &mut fresh);
    assert_eq!(reused_report.termination, reference_report.termination);
    assert_eq!(reused_report.iterations, reference_report.iterations);
    assert_eq!(reused.as_slice(), reference.as_slice());
}

/// A workspace that has finished one solve must start the next solve from
/// the same recurrence state as a fresh workspace: the update vectors enter
/// the first step scaled by `λ₁ = β ≠ 0`, so a stale `wⱼ₋₁` corrupts `x`
/// while the residual estimate still reports convergence.
fn reused_workspace_matches_fresh_workspace<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let mut workspace =
        MinresWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let (first, first_report) = solve_indefinite_oracle(backend, &mut workspace);
    assert_eq!(first_report.termination, Termination::Converged);
    assert_closed_form(&first, indefinite_oracle_bound());

    assert_oracle_matches_fresh_workspace(backend, &mut workspace);
}

#[test]
fn minres_reused_workspace_matches_fresh_workspace() {
    reused_workspace_matches_fresh_workspace::<f32>();
    reused_workspace_matches_fresh_workspace::<f64>();
}

/// Solve `diag(s, −s)·x = (1, 2)` with `s` the first power of the machine
/// epsilon whose reciprocal overflows.
///
/// The rotation radius is `|s|`, so `1/ρ₁` is infinite and the first update
/// vector `w₁` leaves the solve non-finite; the returned solution is
/// infinite and the solve ends after that one step.
fn solve_with_overflowing_update_vector<T>(
    backend: LetoBackend<T>,
    workspace: &mut MinresWorkspace<LetoBackend<T>>,
) -> (Array1<T>, SolveReport<T>)
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let mut scale = <T as NumericElement>::ONE;
    while (<T as NumericElement>::ONE / scale).is_finite() {
        scale *= <T as RealField>::EPSILON;
    }
    let matrix = CsrMatrix::from_parts(vec![scale, -scale], vec![0, 1], vec![0, 1, 2], 2, 2)
        .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([2], vec![T::from_f64(1.0), T::from_f64(2.0)])
        .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([2]);
    let policy = ConvergencePolicy::new(<T as RealField>::EPSILON, <T as RealField>::EPSILON, 3)
        .expect("invariant: tolerance is finite and positive");
    let report = Minres::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");
    (solution, report)
}

/// Zeroing by scaling leaves a NaN entry NaN and turns an infinite one into
/// NaN, so a workspace left non-finite by an earlier solve must be
/// overwritten, not scaled, to start the next solve from the fresh state.
fn workspace_reused_after_non_finite_solve_matches_fresh_workspace<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let mut workspace =
        MinresWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let (poisoned, report) = solve_with_overflowing_update_vector(backend, &mut workspace);
    assert_eq!(
        report.iterations, 1,
        "the step that writes the first update vector"
    );
    assert!(
        poisoned
            .as_slice()
            .expect("invariant: Array1 is contiguous")
            .iter()
            .all(|component| !component.is_finite()),
        "precondition: the overflowing solve leaves non-finite state: {poisoned:?}"
    );

    assert_oracle_matches_fresh_workspace(backend, &mut workspace);
}

#[test]
fn minres_workspace_reused_after_non_finite_solve_matches_fresh_workspace() {
    workspace_reused_after_non_finite_solve_matches_fresh_workspace::<f32>();
    workspace_reused_after_non_finite_solve_matches_fresh_workspace::<f64>();
}

/// A three-unknown symmetric-indefinite operator with three distinct
/// eigenvalues of mixed sign, exercising more than two Lanczos steps.
fn indefinite_tridiagonal<T>() -> CsrMatrix<T>
where
    T: RealScalar + FloatElement,
{
    CsrMatrix::from_parts(
        vec![
            T::from_f64(2.0),
            T::from_f64(1.0),
            T::from_f64(1.0),
            T::from_f64(-2.0),
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

fn solves_the_tridiagonal_system<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let matrix = indefinite_tridiagonal();
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    // Manufactured solution (1, 2, 3): b = A·x* = (4, 0, 8).
    let right_hand_side = Array1::from_shape_vec(
        [3],
        vec![T::from_f64(4.0), T::from_f64(0.0), T::from_f64(8.0)],
    )
    .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        MinresWorkspace::new(&backend, 3).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(<T as RealField>::EPSILON, <T as RealField>::EPSILON, 4)
        .expect("invariant: tolerance is finite and positive");

    let report = Minres::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert_eq!(report.termination, Termination::Converged);
    // Three distinct eigenvalues complete the Krylov space at step three.
    assert!(report.iterations <= 3, "Lanczos completion bound");
    // Three steps over length-three vectors perform fewer than 72 rounded
    // scalar operations; the spectral spread is [−2.7, 2.7] with the smallest
    // magnitude ≈ 1.4, so the amplification stays below two. 512 machine
    // epsilons covers the first-order count with a factor-of-three margin.
    let bound = T::from_f64(512.0) * <T as RealField>::EPSILON;
    let values = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    for (component, expected) in values.iter().zip([1.0, 2.0, 3.0]) {
        let delta = *component - T::from_f64(expected);
        assert!(
            delta.abs() <= bound,
            "manufactured component {component:?} vs {expected}"
        );
    }
}

#[test]
fn minres_solves_a_three_step_indefinite_system_for_every_cpu_scalar() {
    solves_the_tridiagonal_system::<f32>();
    solves_the_tridiagonal_system::<f64>();
}

#[test]
fn minres_reports_inconsistency_as_breakdown() {
    let backend = LetoBackend::<f64>::default();
    // Singular operator with b outside its range: the Krylov space closes on
    // the first step (A·v = 0), the residual cannot decrease, and the run
    // must report the rank loss value-semantically rather than looping.
    let matrix = CsrMatrix::from_parts(vec![0.0, 1.0], vec![0, 1], vec![0, 1, 2], 2, 2)
        .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([2], vec![1.0, 0.0]).expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        MinresWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy =
        ConvergencePolicy::new(f64::EPSILON, f64::EPSILON, 10).expect("invariant: valid policy");

    let report = Minres::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert_eq!(report.termination, Termination::Breakdown);
    assert_eq!(report.iterations, 1, "the Krylov space closes at step one");
}

#[test]
fn minres_reports_dimension_mismatch() {
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(vec![1.0], vec![0], vec![0, 1], 1, 1)
        .expect("invariant: one-entry CSR is valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([2], vec![1.0, 1.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([1]);
    let mut workspace =
        MinresWorkspace::new(&backend, 1).expect("invariant: host allocation succeeds");
    let policy =
        ConvergencePolicy::new(f64::EPSILON, f64::EPSILON, 2).expect("invariant: valid policy");

    let failure = Minres::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    );

    assert!(matches!(failure, Err(SolveError::DimensionMismatch { .. })));
}
