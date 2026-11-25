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

/// Bound, in machine epsilons of `f32`, on the error of each MINRES solution
/// component for `diag(1, -1)`, `b = (1, 2)`, `x₀ = 0`.
///
/// First-order propagation over the rounded operations `k` that form `x`:
/// `|Δxᵢ| ≤ Σₖ |∂xᵢ/∂δₖ| τₖ`, with `δₖ` the relative rounding error of
/// operation `k` and `τₖ` its tolerance (`ε` is `f32::EPSILON`):
///
/// - host `+ - * / √`: `ε/2` (Rust `f32` docs: `roundTiesToEven`; `sqrt` is
///   the rounded infinite-precision result; `eunomia` 0.12.1 enables `std` in
///   this dependency graph, so its `sqrt` is `f32::sqrt`);
/// - device `+ - *` and `fma`: `ε`. WGSL §15.7.4.1 calls these "correctly
///   rounded", which admits either neighbour of the exact result;
///   §15.7.5 and the note in §17.5.32 admit an unfused multiply then add,
///   so each `fma` counts as two;
/// - device `sqrt`: `4.5 ε`. WGSL §15.7.4.1 lists `sqrt` as inherited from
///   `1.0 / inverseSqrt(x)`, `inverseSqrt` as 2 ULP and `x / y` as 2.5 ULP,
///   and `ULP(y) ≤ ε|y|`.
///
/// WGSL sections are those of <https://www.w3.org/TR/WGSL/>
/// (`#concrete-float-accuracy`, `#reassociation-and-fusion`).
///
/// Exact, so unlisted: the CSR rows `0 + (±1)·v` (hephaestus-wgpu 0.30.0
/// `application/sparse/spmv.rs:93-97`); `b - A·0`; `1² + 2² = 5`; `1·g` and
/// `2·g` in the seed scale; every product with a zero start vector; the zero
/// band entries of the first two rotated columns; `λ₁ = ‖r₀‖`; `β/β = 1` in
/// the first rotation. In the second rotation the remainder `β₃` is `O(ε)`,
/// so `1 + (β₃/ρ)²` rounds to 1 (`β₃/ρ < 2⁻¹²`, see the residual bound),
/// giving `cos = -1`, `ρ₂ = |ᾱ₂|`, `φ₂ = -φ̄₁` exactly, and `β₃` reaches `x`
/// only at second order. The device dot of two elements is two products and
/// one sum (the lanes past `len` hold the identity,
/// `application/linalg/map_reduction/shader.rs:32-43`), the host order.
///
/// 55 operations form `x`. Three have zero first-order sensitivity because
/// `λ₂ = 0` (the sum forming it and the two products `λ₂·w₁`); 52 remain:
/// seed 2, first Lanczos step 11, first rotation 11, first update 4, shift 3,
/// second Lanczos step 7, second rotation 6, second update 8. With
/// `α₁ = -3/5`, `β₂ = 4/5`, `ρ₂ = 1`, `φ₂ = 4√5/5`, their sensitivities sum to
/// `38.01 ε` for `x₀` (`15.1` device `sqrt`, `14.8` device, `8.1` host) and
/// `72.35 ε` for `x₁` (`40.3 + 20.4 + 11.6`).
/// Second-order terms are below `10⁻³ ε` (`(72 ε)²`), so `⌈72.35⌉ = 73`.
const ORACLE_BOUND_EPSILONS: u16 = 73;

/// Bound, in machine epsilons, on the true residual `‖b - A x‖₂` and on the
/// reported residual estimate `|φ̄₃|`.
///
/// The true residual is at most `hypot(38.01, 72.35) ε = 81.73 ε`. The
/// estimate is `|s₂ φ̄₂|` with `φ̄₂ = 4√5/5` and `s₂ = β₃/ρ₂`, where `β₃` is
/// the norm of the second Lanczos remainder; the same propagation bounds its
/// components by `15.09 ε` and `34.10 ε`, so `β₃ ≤ 37.29 ε` and the estimate
/// is at most `1.789 · 37.29 ε = 66.7 ε`. Both are nonnegative and below
/// `82 ε`, so they differ by at most `82 ε`. The estimate also lies below the
/// convergence threshold `73 ε · ‖b‖ = 163 ε`, so the second step converges,
/// while the first step's estimate `4√5/5` exceeds it.
const ORACLE_RESIDUAL_BOUND_EPSILONS: u16 = 82;

fn oracle_bound() -> f32 {
    f32::from(ORACLE_BOUND_EPSILONS) * f32::EPSILON
}

fn oracle_residual_bound() -> f64 {
    f64::from(ORACLE_RESIDUAL_BOUND_EPSILONS) * f64::from(f32::EPSILON)
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
    let rhs = backend
        .device()
        .upload(&[1.0, 2.0])
        .expect("invariant: upload");
    let mut solution = backend.allocate(2).expect("invariant: allocation");
    let mut workspace = MinresWorkspace::new(&backend, 2).expect("invariant: workspace");
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
        .expect("invariant: readback");

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
    // `b - A x = (-r0, -r1)`, so the true residual norm of the read-back
    // iterate is `hypot(r0, r1)`; the estimate must agree within the bound.
    let true_residual = r0.hypot(r1);
    let residual_bound = oracle_residual_bound();
    assert!(
        true_residual <= residual_bound,
        "true residual {true_residual}"
    );
    let estimate = f64::from(report.final_residual_norm);
    assert!(
        (estimate - true_residual).abs() <= residual_bound,
        "estimate {estimate} against true residual {true_residual}"
    );

    // The oracle's discriminating half: `Cg` meets `dᵀAd = 1 - 4 = -3` on its
    // first direction `d = b` and stops.
    let mut cg_solution = backend.allocate(2).expect("invariant: allocation");
    let mut cg_workspace = CgWorkspace::new(&backend, 2).expect("invariant: workspace");
    let cg_report = Cg::<Backend>::solve_into(
        &backend,
        &operator,
        &Identity,
        &rhs,
        &mut cg_solution,
        &mut cg_workspace,
        policy,
    )
    .expect("invariant: valid dimensions and real device operations");
    assert_eq!(cg_report.termination, Termination::NonPositiveCurvature);
}

/// The empty second CSR column keeps x[1] out of the residual. Exercise
/// exceptional values directly on the device, independently of how its
/// arithmetic treats the subnormals in the host overflow fixtures.
#[test]
fn minres_checks_nullspace_components_on_device() {
    let backend = backend();
    let operator = Operator::from_parts(
        WgpuSparseOps,
        backend.device(),
        &[1.0],
        &[0],
        &[0, 1, 1],
        2,
        2,
    )
    .expect("square sparse diagonal");
    let mut workspace = MinresWorkspace::new(&backend, 2).expect("workspace");
    let policy = ConvergencePolicy::new(0.0, 0.0, 1).expect("exact convergence");

    for (rhs0, finite_exit, steps) in [
        (0.0, Termination::InitialResidual, 0),
        (1.0, Termination::Converged, 1),
    ] {
        let rhs = backend.device().upload(&[rhs0, 0.0]).expect("upload");
        for (value, expected) in [
            (f32::MAX, finite_exit),
            (f32::NAN, Termination::NonFinite),
            (f32::INFINITY, Termination::NonFinite),
            (f32::NEG_INFINITY, Termination::NonFinite),
        ] {
            let mut x = backend.device().upload(&[0.0, value]).expect("upload");
            let report =
                Minres::solve_into(&backend, &operator, &rhs, &mut x, &mut workspace, policy)
                    .expect("valid solve");
            assert_eq!(report.termination, expected, "x[1]={value}, rhs[0]={rhs0}");
            assert_eq!(report.iterations, steps);
            assert_eq!(report.operator_applications, steps + 1);
        }
    }
}
