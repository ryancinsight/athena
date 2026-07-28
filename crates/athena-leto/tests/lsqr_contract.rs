//! Value-semantic CPU conformance for LSQR.

use athena_core::{
    ConvergencePolicy, KrylovBackend, Lsqr, LsqrWorkspace, RectangularOperator, SolveError,
    Termination,
};
use athena_leto::{LetoBackend, RectangularCsrOperator};
use eunomia::{FloatElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

fn dense_operator<T>(rows: &[&[f64]]) -> RectangularCsrOperator<T>
where
    T: RealScalar + RealField + FloatElement,
{
    let mut values = Vec::new();
    let mut columns = Vec::new();
    let mut row_ptr = vec![0usize];
    for row in rows {
        for (column, &value) in row.iter().enumerate() {
            if value != 0.0 {
                values.push(T::from_f64(value));
                columns.push(column);
            }
        }
        row_ptr.push(values.len());
    }
    let matrix = CsrMatrix::from_parts(values, columns, row_ptr, rows.len(), rows[0].len())
        .expect("invariant: manufactured CSR parts are valid");
    RectangularCsrOperator::new(matrix)
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
        T::from_f64(1024.0) * <T as RealField>::EPSILON,
        T::from_f64(1024.0) * <T as RealField>::EPSILON,
        max_iterations,
    )
    .expect("invariant: tolerance is finite and positive")
}

/// `‖Aᵀ(b − A·x)‖₂`, the quantity a least-squares solution drives to zero.
///
/// This is the independent optimality oracle: for an inconsistent system the
/// residual itself stays bounded away from zero, so only the normal-equation
/// residual can confirm the minimiser.
fn normal_equation_residual<T>(
    backend: LetoBackend<T>,
    operator: &RectangularCsrOperator<T>,
    right_hand_side: &Array1<T>,
    solution: &Array1<T>,
) -> f64
where
    T: RealScalar + RealField + FloatElement,
{
    let mut image = Array1::zeros([operator.rows()]);
    operator
        .apply(
            &backend,
            backend.view(solution),
            backend.view_mut(&mut image),
        )
        .expect("invariant: valid dimensions");
    let mut residual = Array1::zeros([operator.rows()]);
    for index in 0..operator.rows() {
        residual[index] = right_hand_side[index] - image[index];
    }
    let mut gradient = Array1::zeros([operator.columns()]);
    operator
        .apply_transpose(
            &backend,
            backend.view(&residual),
            backend.view_mut(&mut gradient),
        )
        .expect("invariant: valid dimensions");
    let mut sum = 0.0;
    for index in 0..operator.columns() {
        let value = eunomia::NumericElement::to_f64(gradient[index]);
        sum += value * value;
    }
    sum.sqrt()
}

fn solve<T>(
    operator: &RectangularCsrOperator<T>,
    right_hand_side: &Array1<T>,
    max_iterations: usize,
) -> (Array1<T>, athena_core::SolveReport<T>)
where
    T: RealScalar + RealField + FloatElement,
{
    let backend = LetoBackend::<T>::default();
    let mut solution = Array1::zeros([operator.columns()]);
    let mut workspace = LsqrWorkspace::new(&backend, operator.rows(), operator.columns())
        .expect("invariant: host allocation succeeds");
    let report = Lsqr::<LetoBackend<T>>::solve_into(
        &backend,
        operator,
        right_hand_side,
        &mut solution,
        &mut workspace,
        policy::<T>(max_iterations),
    )
    .expect("invariant: valid dimensions");
    (solution, report)
}

fn recovers_a_consistent_overdetermined_system<T>()
where
    T: RealScalar + RealField + FloatElement + core::fmt::Debug,
{
    // Four equations, two unknowns, consistent by construction: b is exactly
    // A·[2, -1], so the least-squares solution is the exact one and the
    // residual reaches zero.
    let operator = dense_operator::<T>(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[2.0, -1.0]]);
    let right_hand_side = vector::<T>(&[2.0, -1.0, 1.0, 5.0]);

    let (solution, report) = solve(&operator, &right_hand_side, 64);

    assert!(report.converged(), "got {:?}", report.termination);
    let bound = T::from_f64(4096.0) * <T as RealField>::EPSILON;
    assert!(
        (solution[0] - T::from_f64(2.0)).abs() <= bound,
        "{:?}",
        solution[0]
    );
    assert!(
        (solution[1] + T::from_f64(1.0)).abs() <= bound,
        "{:?}",
        solution[1]
    );
}

#[test]
fn recovers_a_consistent_overdetermined_system_f64() {
    recovers_a_consistent_overdetermined_system::<f64>();
}

#[test]
fn recovers_a_consistent_overdetermined_system_f32() {
    recovers_a_consistent_overdetermined_system::<f32>();
}

#[test]
fn minimises_an_inconsistent_system() {
    // Three equations in one unknown with no exact solution: A = [1, 1, 1]ᵀ,
    // b = [1, 2, 6]. The least-squares solution is the mean, x = 3, with a
    // residual norm of sqrt(14) that never reaches zero. Only the
    // normal-equation criterion can terminate this.
    let operator = dense_operator::<f64>(&[&[1.0], &[1.0], &[1.0]]);
    let right_hand_side = vector::<f64>(&[1.0, 2.0, 6.0]);

    let (solution, report) = solve(&operator, &right_hand_side, 32);

    assert!(report.converged(), "got {:?}", report.termination);
    assert_eq!(report.termination, Termination::NormalEquations);
    assert!((solution[0] - 3.0).abs() <= 1e-10, "{}", solution[0]);
    // The residual is bounded away from zero, which is exactly why the
    // residual test alone could not have terminated here.
    assert!(report.final_residual_norm > 1.0);
    let backend = LetoBackend::<f64>::default();
    assert!(normal_equation_residual(backend, &operator, &right_hand_side, &solution) <= 1e-10);
}

#[test]
fn solves_an_underdetermined_system_to_normal_equation_optimality() {
    // One equation, three unknowns: x + y + z = 3. LSQR started from zero
    // converges to the minimum-norm solution [1, 1, 1].
    let operator = dense_operator::<f64>(&[&[1.0, 1.0, 1.0]]);
    let right_hand_side = vector::<f64>(&[3.0]);

    let (solution, report) = solve(&operator, &right_hand_side, 32);

    assert!(report.converged(), "got {:?}", report.termination);
    for index in 0..3 {
        assert!(
            (solution[index] - 1.0).abs() <= 1e-10,
            "{}",
            solution[index]
        );
    }
}

#[test]
fn a_least_squares_solution_beats_its_neighbours() {
    // Optimality checked directly rather than against a quoted answer: any
    // perturbation of the returned solution must increase the residual norm.
    let operator = dense_operator::<f64>(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0], &[1.0, 4.0]]);
    let right_hand_side = vector::<f64>(&[6.0, 5.0, 7.0, 10.0]);
    let (solution, report) = solve(&operator, &right_hand_side, 64);
    assert!(report.converged());

    let backend = LetoBackend::<f64>::default();
    let residual_norm = |candidate: &Array1<f64>| {
        let mut image = Array1::zeros([4]);
        operator
            .apply(
                &backend,
                backend.view(candidate),
                backend.view_mut(&mut image),
            )
            .expect("invariant: valid dimensions");
        let mut sum = 0.0;
        for index in 0..4 {
            let value = right_hand_side[index] - image[index];
            sum += value * value;
        }
        sum.sqrt()
    };

    let optimum = residual_norm(&solution);
    for component in 0..2 {
        for step in [-1e-3, 1e-3] {
            let mut perturbed = solution.clone();
            perturbed[component] += step;
            assert!(
                residual_norm(&perturbed) >= optimum,
                "perturbing component {component} by {step} reduced the residual"
            );
        }
    }
}

#[test]
fn a_warm_start_is_honoured() {
    // The recurrence solves for a correction to the supplied solution, so
    // starting at the answer must terminate immediately without iterating.
    let operator = dense_operator::<f64>(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
    let right_hand_side = vector::<f64>(&[2.0, 3.0, 5.0]);
    let backend = LetoBackend::<f64>::default();
    let mut solution = vector::<f64>(&[2.0, 3.0]);
    let mut workspace =
        LsqrWorkspace::new(&backend, 3, 2).expect("invariant: host allocation succeeds");

    let report = Lsqr::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy::<f64>(32),
    )
    .expect("invariant: valid dimensions");

    assert_eq!(report.termination, Termination::InitialResidual);
    assert_eq!(report.iterations, 0);
}

#[test]
fn an_exhausted_budget_reports_max_iterations() {
    let operator = dense_operator::<f64>(&[&[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0], &[1.0, 4.0]]);
    let right_hand_side = vector::<f64>(&[6.0, 5.0, 7.0, 10.0]);
    let backend = LetoBackend::<f64>::default();
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        LsqrWorkspace::new(&backend, 4, 2).expect("invariant: host allocation succeeds");
    let strict = ConvergencePolicy::new(1e-300_f64, 0.0, 1).expect("invariant: valid policy");

    let report = Lsqr::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        strict,
    )
    .expect("invariant: valid dimensions");

    assert_eq!(report.termination, Termination::MaxIterations);
    assert!(!report.converged());
}

#[test]
fn dimension_mismatch_is_rejected() {
    let operator = dense_operator::<f64>(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
    let backend = LetoBackend::<f64>::default();
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        LsqrWorkspace::new(&backend, 3, 2).expect("invariant: host allocation succeeds");

    // Right-hand side must have the operator row count.
    let outcome = Lsqr::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &vector::<f64>(&[1.0, 2.0]),
        &mut solution,
        &mut workspace,
        policy::<f64>(8),
    );
    assert!(matches!(outcome, Err(SolveError::DimensionMismatch { .. })));

    // Workspace must match the operator shape.
    let mut wrong = LsqrWorkspace::new(&backend, 4, 2).expect("invariant: host allocation");
    let outcome = Lsqr::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &vector::<f64>(&[1.0, 2.0, 3.0]),
        &mut solution,
        &mut wrong,
        policy::<f64>(8),
    );
    assert!(matches!(outcome, Err(SolveError::DimensionMismatch { .. })));
}

#[test]
fn transpose_application_matches_the_dense_product() {
    // The adjoint is scattered rather than materialised, so it is checked
    // against the explicit definition.
    let rows: &[&[f64]] = &[&[1.0, 2.0, 0.0], &[0.0, 3.0, 4.0]];
    let operator = dense_operator::<f64>(rows);
    let backend = LetoBackend::<f64>::default();
    let input = vector::<f64>(&[2.0, -1.0]);
    let mut output = Array1::zeros([3]);

    operator
        .apply_transpose(
            &backend,
            backend.view(&input),
            backend.view_mut(&mut output),
        )
        .expect("invariant: valid dimensions");

    for column in 0..3 {
        let expected: f64 = (0..2).map(|row| rows[row][column] * input[row]).sum();
        assert!(
            (output[column] - expected).abs() <= 1e-12,
            "column {column}"
        );
    }
}
