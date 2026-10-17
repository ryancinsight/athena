//! Mandatory device regressions: CI provisions a headless Vulkan adapter.
//! An unavailable adapter fails these checks instead of silently skipping them.

use athena_core::{
    CgWorkspace, Chebyshev, ChebyshevWorkspace, ConvergencePolicy, DeflatedCg, DeflationSubspace,
    KrylovBackend, SpectralBounds, Termination,
};
use athena_hephaestus::{CsrOperator, HephaestusBackend};
use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{GpuCsrMatrix, WgpuDevice, WgpuSparseOps, WgpuVectorOps};

type Backend = HephaestusBackend<WgpuDevice, WgpuVectorOps<f32>, f32>;
type Operator = CsrOperator<WgpuSparseOps, GpuCsrMatrix<f32>>;

fn backend() -> Backend {
    let device = WgpuDevice::try_default("athena-solver-correctness")
        .expect("a real adapter is required for solver correctness coverage");
    let operations = WgpuVectorOps::new(&device).expect("invariant: kernels compile");
    HephaestusBackend::new(device, operations)
}

#[test]
fn deflated_cg_recovers_noninvariant_coarse_coupling_on_device() {
    let backend = backend();
    let operator = Operator::from_parts(
        WgpuSparseOps,
        backend.device(),
        &[2.0, 1.0, 1.0, 2.0],
        &[0, 1, 0, 1],
        &[0, 2, 4],
        2,
        2,
    )
    .expect("invariant: SPD matrix");
    let rhs = backend
        .device()
        .upload(&[1.0, 0.0])
        .expect("invariant: upload");
    let basis = backend
        .device()
        .upload(&[1.0, 0.0])
        .expect("invariant: upload");
    let subspace = DeflationSubspace::from_vectors(&backend, &operator, &[basis])
        .expect("invariant: independent basis");
    let mut solution = backend.allocate(2).expect("invariant: allocation");
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: workspace");
    // Two-by-two Schur elimination has O(1) operations per component;
    // eight machine epsilons covers the rounded multiply/add/solve path.
    let tolerance = 8.0 * f32::EPSILON;
    let policy = ConvergencePolicy::new(tolerance, 0.0, 2).expect("invariant: valid");
    let report = DeflatedCg::<Backend>::solve_into(
        &backend,
        &operator,
        &subspace,
        &rhs,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: valid dimensions");
    let mut host = [0.0; 2];
    backend
        .device()
        .download(&solution, &mut host)
        .expect("invariant: readback");
    assert!((host[0] - 2.0 / 3.0).abs() <= tolerance);
    assert!((host[1] + 1.0 / 3.0).abs() <= tolerance);
    let r0 = 1.0 - (2.0 * host[0] + host[1]);
    let r1 = -(host[0] + 2.0 * host[1]);
    let norm = (r0 * r0 + r1 * r1).sqrt();
    assert_eq!(report.termination, Termination::Converged);
    assert_eq!(report.iterations, 1);
    assert!(norm <= report.threshold);
    assert!((report.final_residual_norm - norm).abs() <= tolerance);
}

#[test]
fn chebyshev_reports_the_exact_singleton_spectrum_solution_on_device() {
    let backend = backend();
    let operator =
        Operator::from_parts(WgpuSparseOps, backend.device(), &[1.0], &[0], &[0, 1], 1, 1)
            .expect("invariant: identity");
    let rhs = backend.device().upload(&[1.0]).expect("invariant: upload");
    let mut solution = backend.allocate(1).expect("invariant: allocation");
    let mut workspace = ChebyshevWorkspace::new(&backend, 1).expect("invariant: workspace");
    let policy = ConvergencePolicy::new(f32::EPSILON, 0.0, 1).expect("invariant: valid");
    let bounds = SpectralBounds::new(1.0, 1.0).expect("invariant: singleton spectrum");
    let report = Chebyshev::<Backend>::solve_into(
        &backend,
        &operator,
        &rhs,
        &mut solution,
        &mut workspace,
        policy,
        bounds,
    )
    .expect("invariant: dimensions");
    let mut host = [0.0];
    backend
        .device()
        .download(&solution, &mut host)
        .expect("invariant: readback");
    assert_eq!(host[0].to_bits(), 1.0_f32.to_bits());
    assert_eq!(report.termination, Termination::Converged);
    assert_eq!(report.iterations, 1);
    assert_eq!(report.operator_applications, 2);
    assert_eq!(report.final_residual_norm.to_bits(), 0.0_f32.to_bits());
}
