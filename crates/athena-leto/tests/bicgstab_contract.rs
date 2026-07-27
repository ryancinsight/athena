//! Value-semantic CPU conformance for right-preconditioned `BiCGSTAB`.

use athena_core::{
    BiCgStab, BiCgStabWorkspace, ConvergencePolicy, Identity, KrylovBackend, LinearOperator,
    SolveError, Termination,
};
use athena_leto::{CsrOperator, Jacobi, LetoBackend};
use eunomia::{FloatElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

/// Nonsymmetric, well-conditioned 2x2 system. CG cannot solve it; the
/// asymmetry is what distinguishes this contract from `cg_contract`.
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
        ],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

/// Nonsymmetric tridiagonal of dimension `n`, diagonally dominant so `BiCGSTAB`
/// is not asked to survive a case where it has no convergence guarantee.
fn tridiagonal_matrix<T>(n: usize) -> CsrMatrix<T>
where
    T: RealScalar + FloatElement,
{
    let mut values = Vec::new();
    let mut columns = Vec::new();
    let mut row_offsets = vec![0usize];
    for row in 0..n {
        if row > 0 {
            values.push(T::from_f64(-1.0));
            columns.push(row - 1);
        }
        values.push(T::from_f64(4.0));
        columns.push(row);
        if row + 1 < n {
            // Asymmetric: the superdiagonal differs from the subdiagonal.
            values.push(T::from_f64(2.0));
            columns.push(row + 1);
        }
        row_offsets.push(values.len());
    }
    CsrMatrix::from_parts(values, columns, row_offsets, n, n)
        .expect("invariant: manufactured CSR parts are valid")
}

/// Ascending test data `1..=n`. The dimensions used here are single digits,
/// so every index converts to `f64` exactly.
fn ascending(n: usize) -> Vec<f64> {
    (0..n)
        .map(|index| 1.0 + f64::from(u16::try_from(index).expect("invariant: small dimension")))
        .collect()
}

fn vector<T: RealField + FloatElement>(values: &[f64]) -> Array1<T> {
    Array1::from_shape_vec(
        [values.len()],
        values.iter().map(|&v| T::from_f64(v)).collect(),
    )
    .expect("invariant: vector shape is exact")
}

fn policy<T: RealField + FloatElement>(max_iterations: usize) -> ConvergencePolicy<T> {
    ConvergencePolicy::new(
        T::from_f64(256.0) * <T as RealField>::EPSILON,
        T::from_f64(256.0) * <T as RealField>::EPSILON,
        max_iterations,
    )
    .expect("invariant: tolerance is finite and positive")
}

/// Manufacture `b = A·x*` through the operator, solve, and require `x*` back.
fn recovers_manufactured_solution<T>(n: usize, max_iterations: usize)
where
    T: RealScalar + RealField + FloatElement + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let operator = CsrOperator::new(tridiagonal_matrix::<T>(n)).expect("invariant: square matrix");

    let exact: Array1<T> = vector(&ascending(n));
    let mut right_hand_side = Array1::zeros([n]);
    operator
        .apply(
            &backend,
            backend.view(&exact),
            backend.view_mut(&mut right_hand_side),
        )
        .expect("invariant: manufactured application has valid dimensions");

    let mut solution = Array1::zeros([n]);
    let mut workspace =
        BiCgStabWorkspace::new(&backend, n).expect("invariant: host allocation succeeds");

    let report = BiCgStab::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy::<T>(max_iterations),
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert!(
        report.converged(),
        "expected convergence, got {:?} at residual {:?}",
        report.termination,
        report.final_residual_norm
    );
    assert!(report.final_residual_norm <= report.threshold);

    // The reported residual is the recomputed one, so the solution is checked
    // independently against the manufactured vector.
    let bound = T::from_f64(1024.0)
        * <T as RealField>::EPSILON
        * T::from_f64(f64::from(
            u16::try_from(n).expect("invariant: small dimension"),
        ));
    let got = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    let want = exact.as_slice().expect("invariant: Array1 is contiguous");
    for (index, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= bound,
            "component {index}: got {g:?}, want {w:?}"
        );
    }
}

#[test]
fn recovers_manufactured_solution_f64() {
    recovers_manufactured_solution::<f64>(8, 64);
}

#[test]
fn recovers_manufactured_solution_f32() {
    recovers_manufactured_solution::<f32>(8, 64);
}

#[test]
fn solves_a_nonsymmetric_system() {
    // A = [[4, 1], [2, 3]], x = [1, 2] gives b = [6, 8].
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(nonsymmetric_matrix::<f64>()).expect("invariant: square");
    let right_hand_side = vector::<f64>(&[6.0, 8.0]);
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        BiCgStabWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");

    let report = BiCgStab::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy::<f64>(16),
    )
    .expect("invariant: valid dimensions");

    assert!(report.converged());
    let values = solution.as_slice().expect("invariant: contiguous");
    assert!((values[0] - 1.0).abs() <= 1e-12);
    assert!((values[1] - 2.0).abs() <= 1e-12);
}

#[test]
fn right_preconditioning_preserves_the_solution() {
    // Jacobi enters on the right, so it must change the iteration path without
    // changing the fixed point.
    let backend = LetoBackend::<f64>::default();
    let matrix = tridiagonal_matrix::<f64>(12);
    let operator = CsrOperator::new(matrix.clone()).expect("invariant: square");
    let preconditioner = Jacobi::<f64>::from_csr(&matrix).expect("invariant: nonzero diagonal");

    let exact = vector::<f64>(&ascending(12));
    let mut right_hand_side = Array1::zeros([12]);
    operator
        .apply(
            &backend,
            backend.view(&exact),
            backend.view_mut(&mut right_hand_side),
        )
        .expect("invariant: valid dimensions");

    let mut plain = Array1::zeros([12]);
    let mut plain_workspace =
        BiCgStabWorkspace::new(&backend, 12).expect("invariant: host allocation succeeds");
    let plain_report = BiCgStab::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut plain,
        &mut plain_workspace,
        policy::<f64>(128),
    )
    .expect("invariant: valid dimensions");

    let mut scaled = Array1::zeros([12]);
    let mut scaled_workspace =
        BiCgStabWorkspace::new(&backend, 12).expect("invariant: host allocation succeeds");
    let scaled_report = BiCgStab::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &preconditioner,
        &right_hand_side,
        &mut scaled,
        &mut scaled_workspace,
        policy::<f64>(128),
    )
    .expect("invariant: valid dimensions");

    assert!(plain_report.converged() && scaled_report.converged());
    assert!(scaled_report.preconditioner_applications > 0);
    let plain_values = plain.as_slice().expect("invariant: contiguous");
    let scaled_values = scaled.as_slice().expect("invariant: contiguous");
    for (&p, &q) in plain_values.iter().zip(scaled_values.iter()) {
        assert!((p - q).abs() <= 1e-8, "{p} vs {q}");
    }
}

#[test]
fn a_converged_initial_guess_terminates_without_iterating() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(nonsymmetric_matrix::<f64>()).expect("invariant: square");
    let right_hand_side = vector::<f64>(&[6.0, 8.0]);
    let mut solution = vector::<f64>(&[1.0, 2.0]);
    let mut workspace =
        BiCgStabWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");

    let report = BiCgStab::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy::<f64>(16),
    )
    .expect("invariant: valid dimensions");

    assert_eq!(report.termination, Termination::InitialResidual);
    assert_eq!(report.iterations, 0);
}

#[test]
fn an_exhausted_budget_reports_max_iterations() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(tridiagonal_matrix::<f64>(32)).expect("invariant: square");
    let right_hand_side = vector::<f64>(&vec![1.0; 32]);
    let mut solution = Array1::zeros([32]);
    let mut workspace =
        BiCgStabWorkspace::new(&backend, 32).expect("invariant: host allocation succeeds");
    let strict = ConvergencePolicy::new(1e-300_f64, 0.0, 1).expect("invariant: valid policy");

    let report = BiCgStab::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        strict,
    )
    .expect("invariant: valid dimensions");

    assert_eq!(report.termination, Termination::MaxIterations);
    assert!(!report.converged());
    assert_eq!(report.iterations, 1);
}

#[test]
fn a_singular_operator_never_reports_convergence() {
    // A ≡ 0 makes ⟨r̂₀, v⟩ vanish on the first step. The recurrence must
    // surface that rather than report a solve it did not perform.
    let matrix = CsrMatrix::<f64>::from_parts(vec![0.0], vec![0], vec![0, 1, 1], 2, 2)
        .expect("invariant: manufactured CSR parts are valid");
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(matrix).expect("invariant: square");
    let right_hand_side = vector::<f64>(&[1.0, 1.0]);
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        BiCgStabWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");

    let report = BiCgStab::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy::<f64>(8),
    )
    .expect("invariant: valid dimensions");

    assert!(!report.converged(), "got {:?}", report.termination);
}

#[test]
fn dimension_mismatch_is_rejected() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(nonsymmetric_matrix::<f64>()).expect("invariant: square");
    let right_hand_side = vector::<f64>(&[6.0, 8.0, 1.0]);
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        BiCgStabWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");

    let outcome = BiCgStab::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy::<f64>(8),
    );

    assert!(matches!(outcome, Err(SolveError::DimensionMismatch { .. })));
}
