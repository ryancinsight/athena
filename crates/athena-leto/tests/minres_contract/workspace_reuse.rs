//! Workspace reuse for MINRES: a solve on a workspace that an earlier solve
//! left behind must equal a solve on a fresh workspace.

use athena_core::{ConvergencePolicy, Minres, MinresWorkspace, SolveReport, Termination};
use athena_leto::{CsrOperator, LetoBackend};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

use super::{
    assert_closed_form, assert_tridiagonal_closed_form, indefinite_matrix, indefinite_oracle_bound,
    solve_tridiagonal_oracle,
};

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
/// inputs, so the reused solution must equal it under IEEE value equality
/// (`-0 == +0`, NaN equals nothing; evaluation order is the same, so no
/// reduction-order tolerance applies), with the same termination and step
/// count, and must lie on the closed form within the oracle bound.
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
pub(super) fn reused_workspace_matches_fresh_workspace<T>()
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
pub(super) fn workspace_reused_after_non_finite_solve_matches_fresh_workspace<T>()
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

/// Poisoning operator `[[a, b, 0], [b, d, f], [0, f, 1]]` with `a`, `b`, `d`,
/// `f` powers of two.
///
/// Solved for `b = e₁` it rotates with `ρ₁ ≈ b` and `s₁ ≈ 1`, so `w₁ ≈ 1/b`
/// and `λ₂ = c₁c₂β₂ + s₁α₂ ≈ d`. The step-two numerator `λ₂·w₁ ≈ d/b`
/// overflows when its exponent exceeds the scalar's: `2⁶⁰⁰·2⁵⁰⁰ = 2¹¹⁰⁰ >
/// 2¹⁰²⁴` at `f64`, `2⁸⁰·2⁶⁰ = 2¹⁴⁰ > 2¹²⁸` at `f32`. `w₂` is then infinite,
/// step three (`β₃ = f`, `β₄ = 0`) copies it into `wⱼ₋₂`, and the solve ends
/// with a non-finite `x[0]` and a non-finite `update_previous_2`.
fn overflowing_second_update_matrix<T>(exponents: [i32; 4]) -> CsrMatrix<T>
where
    T: RealScalar + FloatElement,
{
    let [a, b, d, f] = exponents.map(|exponent| T::from_f64(2.0_f64.powi(exponent)));
    CsrMatrix::from_parts(
        vec![a, b, b, d, f, f, T::from_f64(1.0)],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![0, 2, 5, 7],
        3,
        3,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

/// A first solve whose step-two update overflows and which then runs a third
/// step leaves `wⱼ₋₂` non-finite. A finite stale `wⱼ₋₂` is annihilated by
/// `ε₁ = 0`, but `−ε₁·NaN` is NaN, so the next solve must not inherit it.
pub(super) fn workspace_reused_after_overflowing_second_update_matches_fresh_workspace<T>(
    exponents: [i32; 4],
) where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let mut workspace =
        MinresWorkspace::new(&backend, 3).expect("invariant: host allocation succeeds");
    let operator = CsrOperator::new(overflowing_second_update_matrix::<T>(exponents))
        .expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec(
        [3],
        vec![T::from_f64(1.0), T::from_f64(0.0), T::from_f64(0.0)],
    )
    .expect("invariant: vector shape is exact");
    let mut poisoned = Array1::zeros([3]);
    let policy = ConvergencePolicy::new(<T as RealField>::EPSILON, <T as RealField>::EPSILON, 3)
        .expect("invariant: tolerance is finite and positive");
    let report = Minres::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut poisoned,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");
    assert_eq!(report.iterations, 3, "the step that moves w₂ into wⱼ₋₂");
    assert!(
        !poisoned
            .as_slice()
            .expect("invariant: Array1 is contiguous")[0]
            .is_finite(),
        "precondition: the overflowing update reaches x: {poisoned:?}"
    );

    let (reused, reused_report) = solve_tridiagonal_oracle(backend, &mut workspace);
    assert_tridiagonal_closed_form(&reused, &reused_report);
    let mut fresh = MinresWorkspace::new(&backend, 3).expect("invariant: host allocation succeeds");
    let (reference, reference_report) = solve_tridiagonal_oracle(backend, &mut fresh);
    assert_eq!(reused_report.iterations, reference_report.iterations);
    assert_eq!(reused.as_slice(), reference.as_slice());
}
