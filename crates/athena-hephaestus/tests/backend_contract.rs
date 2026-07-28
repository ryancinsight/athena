//! End-to-end conformance for the device-neutral Hephaestus backend.
//!
//! The backend is generic over the device; these cases instantiate it at WGPU
//! because that is the backend testable on this host. The operator is built
//! here from WGPU sparse storage rather than in the library, because
//! `GpuCsrMatrix` and `spmv_into` are still per-backend and no device-neutral
//! sparse seam exists yet.

use athena_core::{
    BiCgStab, BiCgStabWorkspace, Cg, CgWorkspace, ConvergencePolicy, Gmres, GmresWorkspace,
    Identity, KrylovBackend, LinearOperator, SolveError,
};
use athena_hephaestus::HephaestusBackend;
use hephaestus_core::{ComputeDevice, Result};
use hephaestus_wgpu::{GpuCsrMatrix, WgpuDevice, WgpuVectorOps, spmv_into};
use leto_ops::CsrMatrix;

type Backend = HephaestusBackend<WgpuDevice, WgpuVectorOps, f32>;

fn backend_or_skip() -> Option<Backend> {
    let device = match WgpuDevice::try_default("athena-hephaestus-contract") {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping athena-hephaestus contract: {error}");
            return None;
        }
    };
    let operations = WgpuVectorOps::new(&device).expect("invariant: kernels compile");
    Some(HephaestusBackend::new(device, operations))
}

/// Square CSR operator resident in WGPU buffers.
struct DeviceCsrOperator {
    matrix: GpuCsrMatrix<f32>,
    dimension: usize,
}

impl DeviceCsrOperator {
    fn new(backend: &Backend, matrix: &CsrMatrix<f32>) -> Result<Self> {
        let (rows, _) = matrix.shape();
        Ok(Self {
            matrix: GpuCsrMatrix::from_cpu(backend.device(), matrix)?,
            dimension: rows,
        })
    }
}

impl LinearOperator<Backend> for DeviceCsrOperator {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(
        &self,
        backend: &Backend,
        input: <Backend as KrylovBackend>::View<'_>,
        output: <Backend as KrylovBackend>::ViewMut<'_>,
    ) -> Result<()> {
        spmv_into(backend.device(), &self.matrix, input, output)
    }
}

/// Symmetric positive definite: `[[4, 1], [1, 3]]`, with `x = [1, 2]` giving
/// `b = [6, 7]`.
fn spd_matrix() -> CsrMatrix<f32> {
    CsrMatrix::from_parts(
        vec![4.0, 1.0, 1.0, 3.0],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

/// Nonsymmetric: `[[4, 1], [2, 3]]`, with `x = [1, 2]` giving `b = [6, 8]`.
fn nonsymmetric_matrix() -> CsrMatrix<f32> {
    CsrMatrix::from_parts(
        vec![4.0, 1.0, 2.0, 3.0],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

fn policy() -> ConvergencePolicy<f32> {
    ConvergencePolicy::new(1e-5, 1e-5, 32).expect("invariant: tolerance is finite and positive")
}

fn assert_solution(backend: &Backend, solution: &<Backend as KrylovBackend>::Vector) {
    let mut host = [0.0_f32; 2];
    backend
        .device()
        .download(solution, &mut host)
        .expect("invariant: readback succeeds");
    assert!((host[0] - 1.0).abs() <= 1e-3, "first component {}", host[0]);
    assert!(
        (host[1] - 2.0).abs() <= 1e-3,
        "second component {}",
        host[1]
    );
}

fn upload(backend: &Backend, values: &[f32]) -> <Backend as KrylovBackend>::Vector {
    backend
        .device()
        .upload(values)
        .expect("invariant: upload succeeds")
}

#[test]
fn cg_solves_an_spd_system_on_device() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let operator = DeviceCsrOperator::new(&backend, &spd_matrix()).expect("invariant: upload");
    let right_hand_side = upload(&backend, &[6.0, 7.0]);
    let mut solution = backend.allocate(2).expect("invariant: allocation");
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: workspace allocation");

    let report = Cg::<Backend>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy(),
    )
    .expect("invariant: valid dimensions");

    assert!(report.converged(), "got {:?}", report.termination);
    assert_solution(&backend, &solution);
}

#[test]
fn gmres_solves_a_nonsymmetric_system_on_device() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let operator =
        DeviceCsrOperator::new(&backend, &nonsymmetric_matrix()).expect("invariant: upload");
    let right_hand_side = upload(&backend, &[6.0, 8.0]);
    let mut solution = backend.allocate(2).expect("invariant: allocation");
    let mut workspace =
        GmresWorkspace::<Backend, 2>::new(&backend, 2).expect("invariant: workspace allocation");

    let report = Gmres::<Backend, 2>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy(),
    )
    .expect("invariant: valid dimensions");

    assert!(report.converged(), "got {:?}", report.termination);
    assert_solution(&backend, &solution);
}

#[test]
fn bicgstab_solves_a_nonsymmetric_system_on_device() {
    // The recurrence added under ADR 0033 stage A, exercised on the
    // device-neutral backend without a line of device-specific solver code.
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let operator =
        DeviceCsrOperator::new(&backend, &nonsymmetric_matrix()).expect("invariant: upload");
    let right_hand_side = upload(&backend, &[6.0, 8.0]);
    let mut solution = backend.allocate(2).expect("invariant: allocation");
    let mut workspace =
        BiCgStabWorkspace::new(&backend, 2).expect("invariant: workspace allocation");

    let report = BiCgStab::<Backend>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy(),
    )
    .expect("invariant: valid dimensions");

    assert!(report.converged(), "got {:?}", report.termination);
    assert_solution(&backend, &solution);
}

#[test]
fn repeated_solves_reuse_the_retained_reductions() {
    // The retained handles live in the workspace across solves. A second solve
    // must reach the same answer through the same handles, which is the
    // property the borrowing form could not provide.
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let operator = DeviceCsrOperator::new(&backend, &spd_matrix()).expect("invariant: upload");
    let right_hand_side = upload(&backend, &[6.0, 7.0]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: workspace allocation");

    for _ in 0..3 {
        let mut solution = backend.allocate(2).expect("invariant: allocation");
        let report = Cg::<Backend>::solve_into(
            &backend,
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            &mut workspace,
            policy(),
        )
        .expect("invariant: valid dimensions");
        assert!(report.converged());
        assert_solution(&backend, &solution);
    }
}

#[test]
fn dimension_mismatch_is_rejected() {
    let Some(backend) = backend_or_skip() else {
        return;
    };
    let operator = DeviceCsrOperator::new(&backend, &spd_matrix()).expect("invariant: upload");
    let right_hand_side = upload(&backend, &[6.0, 7.0, 1.0]);
    let mut solution = backend.allocate(2).expect("invariant: allocation");
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: workspace allocation");

    let outcome = Cg::<Backend>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy(),
    );

    assert!(matches!(outcome, Err(SolveError::DimensionMismatch { .. })));
}
