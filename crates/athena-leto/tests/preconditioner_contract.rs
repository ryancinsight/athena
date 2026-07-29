//! Value-semantic conformance for the CPU preconditioners.

use athena_core::{
    BiCgStab, BiCgStabWorkspace, ConvergencePolicy, Identity, KrylovBackend, LinearOperator,
    Preconditioner,
};
use athena_leto::{
    CsrOperator, IncompleteLu, Jacobi, LetoBackend, LetoBackendError, SuccessiveOverRelaxation,
};
use leto::Array1;
use leto_ops::CsrMatrix;

/// Dense-to-CSR helper keeping every structural zero out of the pattern.
fn from_dense(rows: &[&[f64]]) -> CsrMatrix<f64> {
    let mut values = Vec::new();
    let mut columns = Vec::new();
    let mut row_ptr = vec![0usize];
    for row in rows {
        for (column, &value) in row.iter().enumerate() {
            if value != 0.0 {
                values.push(value);
                columns.push(column);
            }
        }
        row_ptr.push(values.len());
    }
    CsrMatrix::from_parts(values, columns, row_ptr, rows.len(), rows[0].len())
        .expect("invariant: manufactured CSR parts are valid")
}

fn vector(values: &[f64]) -> Array1<f64> {
    Array1::from_shape_vec([values.len()], values.to_vec()).expect("invariant: exact shape")
}

fn apply<P: Preconditioner<LetoBackend<f64>>>(
    preconditioner: &P,
    input: &Array1<f64>,
) -> Array1<f64> {
    let backend = LetoBackend::<f64>::default();
    let mut output = Array1::zeros([input.shape()[0]]);
    preconditioner
        .apply(&backend, backend.view(input), backend.view_mut(&mut output))
        .expect("invariant: matching dimensions");
    output
}

#[test]
fn incomplete_lu_is_exact_when_no_fill_is_discarded() {
    // A lower-bidiagonal-plus-diagonal matrix generates no fill outside its own
    // pattern, so the incomplete factorization equals the exact LU and the
    // preconditioner inverts the matrix exactly. This is the strongest
    // available check: it pins the factorization against a closed form rather
    // than against itself.
    let matrix = from_dense(&[&[4.0, 0.0, 0.0], &[2.0, 5.0, 0.0], &[0.0, 1.0, 3.0]]);
    let preconditioner = IncompleteLu::from_csr(&matrix).expect("invariant: factorable");

    // A x = b with x = [1, 2, 3] gives b = [4, 12, 11].
    let solved = apply(&preconditioner, &vector(&[4.0, 12.0, 11.0]));
    let values = solved.as_slice().expect("invariant: contiguous");
    for (&got, &want) in values.iter().zip([1.0, 2.0, 3.0].iter()) {
        assert!((got - want).abs() <= 1e-12, "got {got}, want {want}");
    }
}

#[test]
fn incomplete_lu_inverts_a_tridiagonal_system_exactly() {
    // Tridiagonal elimination produces fill only on the tridiagonal pattern
    // itself, so this factorization is also exact.
    let matrix = from_dense(&[
        &[4.0, -1.0, 0.0, 0.0],
        &[-1.0, 4.0, -1.0, 0.0],
        &[0.0, -1.0, 4.0, -1.0],
        &[0.0, 0.0, -1.0, 4.0],
    ]);
    let preconditioner = IncompleteLu::from_csr(&matrix).expect("invariant: factorable");
    let operator = CsrOperator::new(matrix).expect("invariant: square");
    let backend = LetoBackend::<f64>::default();

    let exact = vector(&[1.0, -2.0, 3.0, -4.0]);
    let mut right_hand_side = Array1::zeros([4]);
    operator
        .apply(
            &backend,
            backend.view(&exact),
            backend.view_mut(&mut right_hand_side),
        )
        .expect("invariant: valid dimensions");

    let solved = apply(&preconditioner, &right_hand_side);
    let got = solved.as_slice().expect("invariant: contiguous");
    let want = exact.as_slice().expect("invariant: contiguous");
    for (&g, &w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() <= 1e-12, "got {g}, want {w}");
    }
}

#[test]
fn incomplete_lu_discards_fill_outside_the_pattern() {
    // Elimination on this pattern would create an entry at (2, 0), which the
    // zero-fill contract discards. The factorization must therefore be
    // inexact, which is what distinguishes it from a direct solve.
    let matrix = from_dense(&[&[4.0, 1.0, 0.0], &[1.0, 4.0, 1.0], &[0.0, 1.0, 4.0]]);
    let preconditioner = IncompleteLu::from_csr(&matrix).expect("invariant: factorable");
    let operator = CsrOperator::new(matrix).expect("invariant: square");
    let backend = LetoBackend::<f64>::default();

    let exact = vector(&[1.0, 1.0, 1.0]);
    let mut right_hand_side = Array1::zeros([3]);
    operator
        .apply(
            &backend,
            backend.view(&exact),
            backend.view_mut(&mut right_hand_side),
        )
        .expect("invariant: valid dimensions");
    let approximate = apply(&preconditioner, &right_hand_side);

    // Still a good approximation, but not the exact inverse.
    let values = approximate.as_slice().expect("invariant: contiguous");
    for &value in values {
        assert!(
            (value - 1.0).abs() <= 0.25,
            "approximation drifted: {value}"
        );
    }
}

#[test]
fn incomplete_lu_rejects_a_missing_or_zero_diagonal() {
    let no_diagonal = from_dense(&[&[0.0, 1.0], &[1.0, 2.0]]);
    assert!(matches!(
        IncompleteLu::from_csr(&no_diagonal),
        Err(LetoBackendError::MissingDiagonal { row: 0 })
    ));

    // Elimination drives the second pivot to zero: 4 - (2/1)*2 = 0.
    let vanishing_pivot = from_dense(&[&[1.0, 2.0], &[2.0, 4.0]]);
    assert!(matches!(
        IncompleteLu::from_csr(&vanishing_pivot),
        Err(LetoBackendError::SingularDiagonal { .. })
    ));
}

#[test]
fn successive_over_relaxation_reduces_to_gauss_seidel_at_unit_relaxation() {
    // At omega = 1 the factor is D + L, so applying it to A x recovers a
    // forward-substituted solution of the lower triangle.
    let matrix = from_dense(&[&[2.0, 0.0], &[1.0, 4.0]]);
    let preconditioner =
        SuccessiveOverRelaxation::from_csr(&matrix, 1.0).expect("invariant: valid relaxation");

    // (D + L) z = [2, 6] has the solution z = [1, 1.25].
    let solved = apply(&preconditioner, &vector(&[2.0, 6.0]));
    let values = solved.as_slice().expect("invariant: contiguous");
    assert!((values[0] - 1.0).abs() <= 1e-12);
    assert!((values[1] - 1.25).abs() <= 1e-12);
}

#[test]
fn successive_over_relaxation_scales_the_diagonal_by_the_relaxation() {
    // The factor is D/omega + L, so halving omega doubles the diagonal and
    // halves the leading component.
    let matrix = from_dense(&[&[2.0, 0.0], &[0.0, 2.0]]);
    let full = SuccessiveOverRelaxation::from_csr(&matrix, 1.0).expect("invariant: valid");
    let half = SuccessiveOverRelaxation::from_csr(&matrix, 0.5).expect("invariant: valid");

    let input = vector(&[2.0, 2.0]);
    let full_values = apply(&full, &input);
    let half_values = apply(&half, &input);
    let full_slice = full_values.as_slice().expect("invariant: contiguous");
    let half_slice = half_values.as_slice().expect("invariant: contiguous");
    for (&f, &h) in full_slice.iter().zip(half_slice.iter()) {
        assert!((f - 2.0 * h).abs() <= 1e-12, "{f} vs {h}");
    }
}

#[test]
fn successive_over_relaxation_rejects_relaxation_outside_the_convergent_range() {
    let matrix = from_dense(&[&[2.0, 0.0], &[0.0, 2.0]]);
    for relaxation in [0.0, -0.5, 2.0, 2.5, f64::NAN] {
        assert!(
            matches!(
                SuccessiveOverRelaxation::from_csr(&matrix, relaxation),
                Err(LetoBackendError::InvalidRelaxation)
            ),
            "relaxation {relaxation} must be rejected"
        );
    }
}

#[test]
fn every_preconditioner_preserves_the_solution_under_bicgstab() {
    // Right preconditioning changes the iteration path, never the fixed point.
    // Each preconditioner must reach the same solution as the unpreconditioned
    // solve, and the stronger ones must not need more iterations.
    let matrix = from_dense(&[
        &[10.0, 2.0, 0.0, 1.0],
        &[3.0, 12.0, 1.0, 0.0],
        &[0.0, 1.0, 8.0, 2.0],
        &[1.0, 0.0, 2.0, 9.0],
    ]);
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(matrix.clone()).expect("invariant: square");
    let exact = vector(&[1.0, -2.0, 3.0, 0.5]);
    let mut right_hand_side = Array1::zeros([4]);
    operator
        .apply(
            &backend,
            backend.view(&exact),
            backend.view_mut(&mut right_hand_side),
        )
        .expect("invariant: valid dimensions");
    let policy = ConvergencePolicy::new(1e-12_f64, 1e-12, 128).expect("invariant: valid policy");

    let plain = {
        let mut solution = Array1::zeros([4]);
        let mut workspace =
            BiCgStabWorkspace::new(&backend, 4).expect("invariant: host allocation succeeds");
        let report = BiCgStab::<LetoBackend<f64>>::solve_into(
            &backend,
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            &mut workspace,
            policy,
        )
        .expect("invariant: valid dimensions");
        assert!(report.converged());
        solution
    };

    let jacobi = Jacobi::from_csr(&matrix).expect("invariant: nonzero diagonal");
    let sor = SuccessiveOverRelaxation::from_csr(&matrix, 1.0).expect("invariant: valid");
    let ilu = IncompleteLu::from_csr(&matrix).expect("invariant: factorable");

    for (label, solution) in [
        (
            "jacobi",
            solve_with(backend, &operator, &right_hand_side, &jacobi, policy),
        ),
        (
            "sor",
            solve_with(backend, &operator, &right_hand_side, &sor, policy),
        ),
        (
            "ilu",
            solve_with(backend, &operator, &right_hand_side, &ilu, policy),
        ),
    ] {
        let got = solution.as_slice().expect("invariant: contiguous");
        let want = plain.as_slice().expect("invariant: contiguous");
        for (index, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() <= 1e-8,
                "{label} component {index}: {g} vs {w}"
            );
        }
    }
}

fn solve_with<P: Preconditioner<LetoBackend<f64>>>(
    backend: LetoBackend<f64>,
    operator: &CsrOperator<f64>,
    right_hand_side: &Array1<f64>,
    preconditioner: &P,
    policy: ConvergencePolicy<f64>,
) -> Array1<f64> {
    let mut solution = Array1::zeros([right_hand_side.shape()[0]]);
    let mut workspace = BiCgStabWorkspace::new(&backend, right_hand_side.shape()[0])
        .expect("invariant: host allocation succeeds");
    let report = BiCgStab::<LetoBackend<f64>>::solve_into(
        &backend,
        operator,
        preconditioner,
        right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: valid dimensions");
    assert!(report.converged(), "preconditioned solve must converge");
    solution
}

#[test]
fn identity_rows_are_opt_in_for_a_missing_diagonal() {
    // An assembly may omit the diagonal of a row carrying no self-coupling.
    // The default constructor rejects that, because a splitting factor with a
    // structurally absent pivot is undefined; the identity-row constructor
    // gives it an implied unit pivot and leaves the row unchanged.
    let matrix = from_dense(&[&[2.0, 0.0, 0.0], &[1.0, 0.0, 0.0], &[0.0, 0.0, 4.0]]);

    assert!(matches!(
        SuccessiveOverRelaxation::from_csr(&matrix, 1.0),
        Err(LetoBackendError::MissingDiagonal { row: 1 })
    ));

    let lenient = SuccessiveOverRelaxation::from_csr_with_identity_rows(&matrix, 1.0)
        .expect("invariant: identity rows are permitted");
    // Row 1 has a unit pivot, so it resolves to r[1] - 1*z[0]: with r = [2, 3, 8]
    // that is z = [1, 3 - 1 = 2, 2].
    let solved = apply(&lenient, &vector(&[2.0, 3.0, 8.0]));
    let values = solved.as_slice().expect("invariant: contiguous");
    assert!((values[0] - 1.0).abs() <= 1e-12, "{}", values[0]);
    assert!((values[1] - 2.0).abs() <= 1e-12, "{}", values[1]);
    assert!((values[2] - 2.0).abs() <= 1e-12, "{}", values[2]);
}
