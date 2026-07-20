//! Real-device WGPU restarted-GMRES conformance.

mod support;

use athena_core::{ConvergencePolicy, Gmres, GmresWorkspace, Identity, Termination};
use athena_wgpu::{WgpuBackend, WgpuCsrOperator};
use hephaestus_core::ComputeDevice;
use leto_ops::CsrMatrix;

use support::device;

#[test]
fn wgpu_gmres_matches_manufactured_nonsymmetric_solution() {
    let Some(device) = device("athena-gmres-contract") else {
        return;
    };
    let backend = WgpuBackend::new(device).expect("WGPU kernels must prepare");
    let matrix = CsrMatrix::from_parts(
        vec![4.0_f32, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![0, 2, 5, 7],
        3,
        3,
    )
    .expect("invariant: manufactured CSR parts are valid");
    let operator = WgpuCsrOperator::from_cpu(&backend, &matrix).expect("CSR upload must succeed");
    let right_hand_side = backend
        .device()
        .upload(&[2.0_f32, -1.0, 4.0])
        .expect("right-hand side upload must succeed");
    let mut solution = backend
        .device()
        .alloc_zeroed(3)
        .expect("solution allocation must succeed");
    let mut workspace =
        GmresWorkspace::<_, 3>::new(&backend, 3).expect("workspace allocation must succeed");
    // Three MGS columns, WGPU reductions, and the triangular update contribute
    // O(n² ε) error for this well-conditioned manufactured system.
    let bound = 8192.0 * f32::EPSILON;
    let policy =
        ConvergencePolicy::new(bound, bound, 6).expect("invariant: valid convergence policy");

    let report = Gmres::<WgpuBackend, 3>::solve_into(
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
    let mut host_solution = [0.0_f32; 3];
    backend
        .device()
        .download(&solution, &mut host_solution)
        .expect("solution download must succeed");
    assert!((host_solution[0] - 1.0).abs() <= bound);
    assert!((host_solution[1] + 2.0).abs() <= bound);
    assert!((host_solution[2] - 3.0).abs() <= bound);

    // Direct substitution is independent of the solver recurrence. The matrix
    // infinity norm is six; eight accounts for propagation plus row rounding.
    let residual_bound = 8.0 * bound;
    let first = 4.0 * host_solution[0] + host_solution[1] - 2.0;
    let second = 2.0 * host_solution[0] + 3.0 * host_solution[1] + host_solution[2] + 1.0;
    let third = host_solution[1] + 2.0 * host_solution[2] - 4.0;
    assert!(first.abs() <= residual_bound);
    assert!(second.abs() <= residual_bound);
    assert!(third.abs() <= residual_bound);
}
