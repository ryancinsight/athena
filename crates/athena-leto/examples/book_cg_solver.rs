//! Solve a 3×3 symmetric positive-definite system with CG.
//!
//! Conjugate gradient is the canonical Krylov solver for SPD linear systems.
//! `Cg::solve_into` takes a [`KrylovBackend`], a [`LinearOperator`], a
//! [`Preconditioner`], the right-hand side, a mutable solution buffer, a
//! pre-allocated workspace, and a [`ConvergencePolicy`].
//!
//! This example manufactures a simple tridiagonal SPD system, applies the
//! operator to a known solution vector to get `b`, and then solves to recover
//! the original solution.

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, KrylovBackend, LinearOperator, Termination,
};
use athena_leto::{CsrOperator, LetoBackend};
use leto::Array1;
use leto_ops::CsrMatrix;

fn tridiagonal_spd(n: usize) -> CsrMatrix<f64> {
    // Tridiagonal: diagonal = 4, sub-/super-diagonal = -1. SPD for any n.
    let mut values = Vec::new();
    let mut columns = Vec::new();
    let mut row_offsets = vec![0usize];
    for row in 0..n {
        if row > 0 {
            values.push(-1.0_f64);
            columns.push(row - 1);
        }
        values.push(4.0_f64);
        columns.push(row);
        if row + 1 < n {
            values.push(-1.0_f64);
            columns.push(row + 1);
        }
        row_offsets.push(values.len());
    }
    CsrMatrix::from_parts(values, columns, row_offsets, n, n).expect("valid tridiagonal CSR")
}

fn main() {
    let n = 5_usize;
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(tridiagonal_spd(n)).expect("square SPD operator");

    // Manufacture b = A × x* where x* = [1, 2, 3, 4, 5].
    let x_exact: Array1<f64> =
        Array1::from_shape_vec([n], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).expect("valid vector");
    let mut b = Array1::zeros([n]);
    operator
        .apply(&backend, backend.view(&x_exact), backend.view_mut(&mut b))
        .expect("apply valid dimensions");

    println!("b = {:?}", b.as_slice().expect("contiguous"));

    // Solve A × x = b with CG.
    let policy = ConvergencePolicy::<f64>::new(256.0 * f64::EPSILON, 256.0 * f64::EPSILON, 200)
        .expect("valid policy");
    let mut solution = Array1::zeros([n]);
    let mut workspace = CgWorkspace::new(&backend, n).expect("workspace allocation");

    let report = Cg::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &b,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("valid dimensions");

    let iter = report.iterations;
    let resid = report.final_residual_norm;
    println!("converged: {}", report.converged());
    println!("iterations: {iter}");
    println!("residual: {resid:.3e}");
    assert!(report.converged(), "CG must converge on SPD system");
    assert!(matches!(report.termination, Termination::Converged));

    // Verify solution ≈ x*.
    let sol = solution.as_slice().expect("contiguous");
    for (i, (&got, &want)) in sol
        .iter()
        .zip(x_exact.as_slice().expect("contiguous"))
        .enumerate()
    {
        assert!(
            (got - want).abs() < 1e-9,
            "solution[{i}]: got {got}, want {want}"
        );
    }
    println!("solution = {sol:?}");
    println!("CG solve assertions passed");
}
