//! Mandatory device regressions: CI provisions a headless Vulkan adapter.
//! An unavailable adapter fails these checks instead of silently skipping them.

use athena_core::{
    Cg, CgWorkspace, Chebyshev, ChebyshevWorkspace, ConvergencePolicy, DeflatedCg,
    DeflationSubspace, Identity, KrylovBackend, Minres, MinresWorkspace, SpectralBounds,
    Termination,
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
fn cg_checks_exact_convergence_between_device_residual_checks() {
    let backend = backend();
    let operator =
        Operator::from_parts(WgpuSparseOps, backend.device(), &[1.0], &[0], &[0, 1], 1, 1)
            .expect("identity operator");
    let rhs = backend.device().upload(&[2.0]).expect("upload");
    let mut solution = backend.allocate(1).expect("allocation");
    let mut workspace = CgWorkspace::new(&backend, 1).expect("workspace");
    let policy = ConvergencePolicy::with_check_interval(0.0, 0.0, 4, 4).expect("valid policy");
    let report = Cg::solve_into(
        &backend,
        &operator,
        &Identity,
        &rhs,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("valid dimensions and real device operations");
    let mut host = [0.0];
    backend
        .device()
        .download(&solution, &mut host)
        .expect("readback");
    assert_eq!(host[0].to_bits(), 2.0_f32.to_bits());
    assert_eq!(report.termination, Termination::Converged);
    assert_eq!(report.iterations, 1);
    assert_eq!(report.operator_applications, 3);
    assert_eq!(report.final_residual_norm.to_bits(), 0.0_f32.to_bits());
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

/// Absolute error bound, in machine epsilons of `f32`, for the MINRES solution
/// components and its final residual estimate.
///
/// Two Lanczos steps over length-two vectors perform fewer than 48 rounded
/// scalar operations (`crates/athena-leto/tests/minres_contract.rs`), each
/// amplified by at most `κ(diag(1, −1)) = 1`, so the first-order error is
/// below `48 ε`. The device recurrence performs no more operations: the
/// two-element dot and squared-norm reductions have one addition, so the
/// sequential host order and any device workgroup tree evaluate the same two
/// products and one sum, and the axpy kernels fuse `factor * source +
/// destination` into one rounding in place of two. The bound `256 ε` is the
/// host contract's factor-of-five margin over that count (`240 ε`, rounded
/// up), which also covers a per-operation device error of up to five
/// epsilons. No source read for this derivation states the device square-root
/// and division accuracy, so that per-operation allowance is an assumption: a
/// result beyond the bound reports a device accuracy outside it, and the
/// bound is never widened to absorb it.
const ORACLE_BOUND_EPSILONS: u16 = 256;

fn oracle_bound() -> f32 {
    f32::from(ORACLE_BOUND_EPSILONS) * f32::EPSILON
}

#[test]
fn minres_solves_the_indefinite_oracle_on_device() {
    let backend = backend();
    // `diag(1, -1)`: symmetric indefinite with condition number one, on which
    // `Cg` fails its first curvature test (`dᵀAd = 1 - 4 < 0` for `d = b`).
    let operator = Operator::from_parts(
        WgpuSparseOps,
        backend.device(),
        &[1.0, -1.0],
        &[0, 1],
        &[0, 1, 2],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid");
    // `x = (1, -2)` gives `b = (1, 2)`; the Krylov space of `b` is complete
    // after two Lanczos steps because `A` has two distinct eigenvalues.
    let rhs = backend.device().upload(&[1.0, 2.0]).expect("upload");
    let mut solution = backend.allocate(2).expect("allocation");
    let mut workspace = MinresWorkspace::new(&backend, 2).expect("workspace");
    let bound = oracle_bound();
    let policy = ConvergencePolicy::new(bound, bound, 3).expect("invariant: valid policy");
    let report = Minres::<Backend>::solve_into(
        &backend,
        &operator,
        &rhs,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: valid dimensions and real device operations");
    let mut host = [0.0_f32; 2];
    backend
        .device()
        .download(&solution, &mut host)
        .expect("readback");

    assert!((host[0] - 1.0).abs() <= bound, "x0 = {}", host[0]);
    assert!((host[1] + 2.0).abs() <= bound, "x1 = {}", host[1]);
    // Direct substitution in `f64`, exact for `f32` operands: the rows give
    // `x0 - 1` and `-x1 - 2`, so the solver's own residual path is not trusted.
    let r0 = f64::from(host[0]) - 1.0;
    let r1 = -f64::from(host[1]) - 2.0;
    assert!(r0.abs() <= f64::from(bound), "row 0 residual {r0}");
    assert!(r1.abs() <= f64::from(bound), "row 1 residual {r1}");
    assert_eq!(report.termination, Termination::Converged);
    // One step leaves residual `‖b - c·A·b‖ = √3.2` (`c = -3/5`), far above
    // the threshold, so convergence is reported by the second step.
    assert_eq!(report.iterations, 2);
    assert_eq!(report.operator_applications, 3);
    assert!(report.final_residual_norm <= report.threshold);
}
