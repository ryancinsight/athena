//! Real-device WGPU solver conformance.

mod support;

use athena_core::{Cg, CgWorkspace, ConvergencePolicy, Identity, Termination};
use athena_wgpu::{WgpuBackend, WgpuCsrOperator};
use hephaestus_core::ComputeDevice;
use leto_ops::CsrMatrix;

use support::device;

#[test]
fn wgpu_cg_matches_manufactured_spd_solution() {
    let Some(device) = device("athena-cg-contract") else {
        return;
    };
    let backend = WgpuBackend::new(device).expect("WGPU kernels must prepare");
    let matrix = CsrMatrix::from_parts(
        vec![4.0_f32, 1.0, 1.0, 3.0],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid");
    let operator = WgpuCsrOperator::from_cpu(&backend, &matrix).expect("CSR upload must succeed");
    let right_hand_side = backend
        .device()
        .upload(&[6.0_f32, 7.0])
        .expect("right-hand side upload must succeed");
    let mut solution = backend
        .device()
        .alloc_zeroed(2)
        .expect("solution allocation must succeed");
    let mut workspace = CgWorkspace::new(&backend, 2).expect("workspace allocation must succeed");
    let policy = ConvergencePolicy::new(64.0 * f32::EPSILON, 64.0 * f32::EPSILON, 4)
        .expect("invariant: valid policy");

    let report = Cg::<WgpuBackend>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("manufactured GPU solve must execute");

    assert_eq!(report.termination, Termination::Converged);
    let mut host_solution = [0.0_f32; 2];
    backend
        .device()
        .download(&solution, &mut host_solution)
        .expect("solution download must succeed");
    // Two length-two PCG iterations plus GPU reduction reordering stay within
    // 512 first-order f32 rounding units for this well-conditioned matrix.
    let bound = 512.0 * f32::EPSILON;
    assert!((host_solution[0] - 1.0).abs() <= bound);
    assert!((host_solution[1] - 2.0).abs() <= bound);

    // Direct substitution verifies A*x=b independently of the GPU recurrence.
    // The matrix infinity norm is five; six also covers row-operation rounding.
    let residual_bound = 6.0 * bound;
    let first = 4.0 * host_solution[0] + host_solution[1] - 6.0;
    let second = host_solution[0] + 3.0 * host_solution[1] - 7.0;
    assert!(first.abs() <= residual_bound);
    assert!(second.abs() <= residual_bound);
}
