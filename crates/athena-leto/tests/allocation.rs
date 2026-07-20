//! Allocation contract for initialized CPU solves.

use std::alloc::System;

use athena_core::{Cg, CgWorkspace, ConvergencePolicy, Gmres, GmresWorkspace, Identity};
use athena_leto::{CsrOperator, LetoBackend};
use leto::Array1;
use leto_ops::CsrMatrix;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

#[test]
fn repeated_cpu_solves_allocate_nothing_after_initialization() {
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(
        vec![4.0_f64, 1.0, 1.0, 3.0],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([2], vec![6.0, 7.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(64.0 * f64::EPSILON, 64.0 * f64::EPSILON, 4)
        .expect("invariant: valid policy");

    Cg::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("warm-up solve must succeed");
    solution.fill(0.0);

    let region = Region::new(GLOBAL);
    for _ in 0..16 {
        let report = Cg::<LetoBackend<f64>>::solve_into(
            &backend,
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            &mut workspace,
            policy,
        )
        .expect("measured solve must succeed");
        assert!(report.converged());
        solution.fill(0.0);
    }
    let change = region.change();

    assert_eq!(change.allocations, 0);
    assert_eq!(change.reallocations, 0);
    assert_eq!(change.deallocations, 0);
}

#[test]
fn repeated_gmres_solves_allocate_nothing_after_initialization() {
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(
        vec![4.0_f64, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![0, 2, 5, 7],
        3,
        3,
    )
    .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([3], vec![2.0, -1.0, 4.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        GmresWorkspace::<_, 3>::new(&backend, 3).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(4096.0 * f64::EPSILON, 4096.0 * f64::EPSILON, 6)
        .expect("invariant: valid policy");

    Gmres::<LetoBackend<f64>, 3>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("warm-up solve must succeed");
    solution.fill(0.0);

    let region = Region::new(GLOBAL);
    for _ in 0..16 {
        let report = Gmres::<LetoBackend<f64>, 3>::solve_into(
            &backend,
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            &mut workspace,
            policy,
        )
        .expect("measured solve must succeed");
        assert!(report.converged());
        solution.fill(0.0);
    }
    let change = region.change();

    assert_eq!(change.allocations, 0);
    assert_eq!(change.reallocations, 0);
    assert_eq!(change.deallocations, 0);
}
