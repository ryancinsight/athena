//! Value-semantic CPU solver conformance for two-level deflated CG.
//!
//! The oracle is ADR 0062's Phase-2 table under ADR 0056's new-construction
//! rule: with an exact invariant deflation subspace, the deflated solve
//! terminates in at most `n − k` iterations where plain CG stalls on the
//! small eigenvalues — finite termination, at `f32` and `f64`.

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, DeflatedCg, DeflatedOperator, DeflationSubspace,
    DeflationSubspaceError, Identity, KrylovBackend, LinearOperator, Termination,
};
use athena_leto::{CsrOperator, LetoBackend};
use eunomia::{FloatElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

/// SPD diagonal operator with the spectrum `{0.01, 0.02, 0.5, 1.0, 2.0}`:
/// two near-null directions dominate the plain-CG condition number (200),
/// and the standard basis vectors are the exact eigenvectors — the deflation
/// subspace `{e₁, e₂}` is exact.
fn ill_conditioned_matrix<T>() -> CsrMatrix<T>
where
    T: RealScalar + FloatElement,
{
    CsrMatrix::from_parts(
        vec![
            T::from_f64(0.01),
            T::from_f64(0.02),
            T::from_f64(0.5),
            T::from_f64(1.0),
            T::from_f64(2.0),
        ],
        vec![0, 1, 2, 3, 4],
        vec![0, 1, 2, 3, 4, 5],
        5,
        5,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

fn solves_by_finite_termination<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let matrix = ill_conditioned_matrix();
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec(
        [5],
        vec![
            T::from_f64(0.01),
            T::from_f64(0.04),
            T::from_f64(1.5),
            T::from_f64(4.0),
            T::from_f64(10.0),
        ],
    )
    .expect("invariant: vector shape is exact");

    // The deflation basis: the exact eigenvectors of the two smallest
    // eigenvalues. Scaled copies prove construction orthonormalises rather
    // than assuming orthonormal input.
    let first = Array1::from_shape_vec(
        [5],
        vec![T::from_f64(5.0), T::ZERO, T::ZERO, T::ZERO, T::ZERO],
    )
    .expect("invariant: exact shape");
    let second = Array1::from_shape_vec(
        [5],
        vec![T::ZERO, T::from_f64(3.0), T::ZERO, T::ZERO, T::ZERO],
    )
    .expect("invariant: exact shape");
    let subspace = DeflationSubspace::from_vectors(&backend, &operator, &[first, second])
        .expect("invariant: exact eigenbasis assembles");

    let mut solution = Array1::zeros([5]);
    let mut workspace = CgWorkspace::new(&backend, 5).expect("invariant: host allocation succeeds");
    // n - k = 3: the complement spectrum {0.5, 1.0, 2.0} has three values,
    // so the CG phase terminates in at most three iterations; the budget
    // carries one spare so the finite termination, not the cap, ends the
    // solve.
    let policy = ConvergencePolicy::new(<T as RealField>::EPSILON, <T as RealField>::EPSILON, 4)
        .expect("invariant: tolerance is finite and positive");

    let report = DeflatedCg::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &subspace,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert_eq!(report.termination, Termination::Converged);
    assert!(
        report.iterations <= 3,
        "finite termination: the complement has three eigenvalues, ran {}",
        report.iterations
    );
    // The complement phase runs three steps over length-five vectors (under
    // 40 rounded operations per entry with unit-amplification complement
    // eigenvalues), and the coarse correction adds one k-by-k solve whose
    // entries have magnitude at most 1/0.01: 512 machine epsilons covers
    // the first-order count with margin.
    let bound = T::from_f64(512.0) * <T as RealField>::EPSILON;
    let values = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    for (component, expected) in values.iter().zip([1.0, 2.0, 3.0, 4.0, 5.0]) {
        let delta = *component - T::from_f64(expected);
        assert!(
            delta.abs() <= bound,
            "manufactured component {component:?} vs {expected}"
        );
    }

    // The stall half: plain CG on the same system with the same budget
    // cannot converge — the 0.01 and 0.02 directions need hundreds of
    // iterations at this condition number, and three iterations leave the
    // residual far above an epsilon-scaled threshold.
    let mut plain = Array1::zeros([5]);
    let mut plain_workspace =
        CgWorkspace::new(&backend, 5).expect("invariant: host allocation succeeds");
    let report = Cg::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut plain,
        &mut plain_workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");
    assert_eq!(
        report.termination,
        Termination::MaxIterations,
        "plain CG stalls on the small eigenvalues at the same budget"
    );
}

#[test]
fn deflated_cg_terminates_finitely_for_every_cpu_scalar() {
    solves_by_finite_termination::<f32>();
    solves_by_finite_termination::<f64>();
}

#[test]
fn deflation_rejects_degenerate_and_invalid_inputs() {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(ill_conditioned_matrix()).expect("invariant: matrix is square");
    let vector =
        Array1::from_shape_vec([5], vec![1.0, 0.0, 0.0, 0.0, 0.0]).expect("invariant: exact shape");

    // A repeated candidate collapses to zero norm under Gram–Schmidt.
    let repeated =
        Array1::from_shape_vec([5], vec![2.0, 0.0, 0.0, 0.0, 0.0]).expect("invariant: exact shape");
    let failure = DeflationSubspace::from_vectors(&backend, &operator, &[vector, repeated])
        .expect_err("invariant: a repeated candidate is degenerate");
    assert!(
        matches!(failure, DeflationSubspaceError::Degenerate { index: 1 }),
        "the second candidate cannot be orthonormalised: {failure:?}"
    );

    // An empty or oversized candidate list is a typed dimension failure.
    let failure = DeflationSubspace::<LetoBackend<f64>>::from_vectors(&backend, &operator, &[])
        .expect_err("invariant: an empty deflation is invalid");
    assert!(matches!(
        failure,
        DeflationSubspaceError::Invalid {
            width: 0,
            dimension: 5
        }
    ));
}

/// A noninvariant basis must retain the coupling A e1 = (2, 1).
/// The inverse is (1/3) [[2, -1], [-1, 2]], so b=e1 has the exact
/// solution (2/3, -1/3), whereas the old coarse-only result (1/2, 0)
/// has true residual norm 1/2.
fn solves_coupled_subspace<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let matrix = CsrMatrix::from_parts(
        [2.0, 1.0, 1.0, 2.0].map(T::from_f64).to_vec(),
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )
    .expect("invariant: SPD coupled matrix");
    let operator = CsrOperator::new(matrix).expect("invariant: square");
    let first = Array1::from_shape_vec([2], vec![T::ONE, T::ZERO]).expect("invariant: exact shape");
    let rhs = first.clone();
    let subspace = DeflationSubspace::from_vectors(&backend, &operator, &[first])
        .expect("invariant: independent basis and invertible coarse matrix");
    let policy = ConvergencePolicy::new(T::from_f64(8.0) * T::EPSILON, T::ZERO, 2)
        .expect("invariant: rounding allowance for the two-by-two solve");
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: allocation");
    for start in [[0.0, 0.0], [1.0, -1.0]] {
        let mut solution = Array1::from_shape_vec([2], start.map(T::from_f64).to_vec())
            .expect("invariant: exact shape");
        let report = DeflatedCg::<LetoBackend<T>>::solve_into(
            &backend,
            &operator,
            &subspace,
            &rhs,
            &mut solution,
            &mut workspace,
            policy,
        )
        .expect("invariant: valid dimensions");
        let values = solution.as_slice().expect("invariant: contiguous");
        let tolerance = T::from_f64(8.0) * T::EPSILON;
        for (value, exact) in values.iter().zip([2.0 / 3.0, -1.0 / 3.0]) {
            assert!(
                (*value - T::from_f64(exact)).abs() <= tolerance,
                "coupled solution {values:?}, report {report:?}"
            );
        }
        let r0 = T::ONE - (T::from_f64(2.0) * values[0] + values[1]);
        let r1 = -(values[0] + T::from_f64(2.0) * values[1]);
        let residual = (r0 * r0 + r1 * r1).sqrt();
        assert_eq!(report.termination, Termination::Converged);
        assert_eq!(report.iterations, 1);
        assert!(residual <= report.threshold, "true residual {residual:?}");
        assert!((report.final_residual_norm - residual).abs() <= tolerance);
        assert!(report.initial_residual_norm > report.threshold);
    }
}

#[test]
fn deflated_cg_solves_noninvariant_subspace_for_every_cpu_scalar() {
    solves_coupled_subspace::<f32>();
    solves_coupled_subspace::<f64>();
}

fn checks_coupled_levels<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    // SPD leading minors 4, 11, 17. With x=(1,2,3), b=(9,10,9).
    let matrix = CsrMatrix::from_parts(
        [4.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0]
            .map(T::from_f64)
            .to_vec(),
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![0, 3, 6, 9],
        3,
        3,
    )
    .expect("invariant: symmetric positive definite");
    let operator = CsrOperator::new(matrix).expect("invariant: square");
    let rhs = Array1::from_shape_vec([3], [9.0, 10.0, 9.0].map(T::from_f64).to_vec())
        .expect("invariant: shape");
    let candidates: Vec<_> = (0..3)
        .map(|column| {
            let mut vector = vec![T::ZERO; 3];
            vector[column] = T::ONE;
            Array1::from_shape_vec([3], vector).expect("invariant: shape")
        })
        .collect();
    let tolerance = T::from_f64(32.0) * T::EPSILON;
    let policy = ConvergencePolicy::new(tolerance, T::ZERO, 1).expect("invariant: valid");
    let mut workspace = CgWorkspace::new(&backend, 3).expect("invariant: allocation");
    for width in 1..=3 {
        let subspace = DeflationSubspace::from_vectors(&backend, &operator, &candidates[..width])
            .expect("invariant: full-rank coarse system");
        let mut solution = Array1::zeros([3]);
        let report = DeflatedCg::<LetoBackend<T>>::solve_into(
            &backend,
            &operator,
            &subspace,
            &rhs,
            &mut solution,
            &mut workspace,
            policy,
        )
        .expect("invariant: dimensions");
        let mut image = Array1::zeros([3]);
        operator
            .apply(
                &backend,
                backend.view(&solution),
                backend.view_mut(&mut image),
            )
            .expect("invariant: apply");
        let mut residual = Array1::zeros([3]);
        backend
            .residual(
                backend.view(&rhs),
                backend.view(&image),
                backend.view_mut(&mut residual),
            )
            .expect("invariant: residual");
        let norm = backend
            .norm_l2(backend.view(&residual))
            .expect("invariant: norm");
        assert!((report.final_residual_norm - norm).abs() <= tolerance);
        assert_eq!(report.converged(), norm <= report.threshold);
        assert!((report.initial_residual_norm - T::from_f64(262.0).sqrt()).abs() <= tolerance);
        if width == 1 {
            assert_eq!(report.termination, Termination::MaxIterations);
            assert_eq!(report.iterations, 1);
        } else {
            assert_eq!(report.termination, Termination::Converged);
            assert_eq!(report.iterations, usize::from(width == 2));
            for (value, exact) in solution
                .as_slice()
                .expect("invariant: contiguous")
                .iter()
                .zip([1.0, 2.0, 3.0])
            {
                assert!((*value - T::from_f64(exact)).abs() <= tolerance);
            }
            // An already-converged initial guess must survive unchanged.
            let initial = solution.clone();
            let report = DeflatedCg::<LetoBackend<T>>::solve_into(
                &backend,
                &operator,
                &subspace,
                &rhs,
                &mut solution,
                &mut workspace,
                policy,
            )
            .expect("invariant: dimensions");
            assert_eq!(report.termination, Termination::InitialResidual);
            assert_eq!(report.iterations, 0);
            assert_eq!(report.operator_applications, 1);
            assert_eq!(initial.as_slice(), solution.as_slice());
        }
    }
}

#[test]
fn deflated_cg_checks_recovered_residual_and_budget_for_every_cpu_scalar() {
    checks_coupled_levels::<f32>();
    checks_coupled_levels::<f64>();
}

#[test]
fn deflated_operator_is_the_coupled_schur_complement() {
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(
        vec![2.0, 1.0, 1.0, 2.0],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )
    .expect("invariant: SPD");
    let operator = CsrOperator::new(matrix).expect("invariant: square");
    let basis = Array1::from_shape_vec([2], vec![1.0, 0.0]).expect("invariant: shape");
    let subspace =
        DeflationSubspace::from_vectors(&backend, &operator, &[basis]).expect("invariant: basis");
    let deflated =
        DeflatedOperator::new(&backend, &operator, &subspace).expect("invariant: allocation");
    let input = Array1::from_shape_vec([2], vec![1.0, 1.0]).expect("invariant: shape");
    let mut output = Array1::zeros([2]);
    deflated
        .apply(
            &backend,
            backend.view(&input),
            backend.view_mut(&mut output),
        )
        .expect("invariant: apply");
    // Schur complement is diag(0, 2-1/2). All quantities are dyadic.
    assert_eq!(
        output.as_slice().expect("invariant: contiguous"),
        &[0.0, 1.5]
    );
}
