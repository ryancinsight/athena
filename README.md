# Athena

[![CI](https://github.com/ryancinsight/athena/actions/workflows/ci.yml/badge.svg)](https://github.com/ryancinsight/athena/actions/workflows/ci.yml)

Athena is Atlas's iterative-solver provider. Its complete vertical contracts
are preconditioned conjugate gradient (PCG) for symmetric positive-definite
linear systems and restarted right-preconditioned GMRES for general
nonsymmetric systems. Leto CPU and Hephaestus WGPU execution share one
backend-neutral recurrence per solver family.

## Ownership boundary

Athena owns:

- matrix-free square-operator and preconditioner contracts;
- validated absolute-plus-relative convergence policy;
- PCG and restarted GMRES recurrences, numerical termination, and scalar
  telemetry;
- reusable, allocation-stable solver workspaces;
- Leto and Hephaestus implementations of the solver-specific vector
  recurrences.

Athena does not own:

- scalar law, which remains in Eunomia;
- host arrays, layouts, CSR storage, SpMV, or reductions, which remain in
  Leto;
- accelerator devices, buffers, transfers, sparse kernels, reductions, or
  dispatch, which remain in Hephaestus;
- domain equations, discretization, nonlinear residual construction, or time
  integration.

This boundary removes CG and GMRES orchestration from Leto without wrapping or
mirroring Leto's APIs. `leto_ops::CsrMatrix` remains the single CSR
representation used by both CPU and WGPU operators.

## Architecture

```text
crates/
├── athena/                    # curated facade and runnable examples
├── athena-core/
│   └── src/
│       ├── backend/           # GAT-based borrowed vector family
│       ├── convergence/       # validated residual policy
│       ├── operator/          # matrix-free operator seam
│       ├── preconditioner/    # preconditioner seam and Identity ZST
│       ├── report/            # allocation-free scalar telemetry
│       └── solver/
│           ├── cg/            # generic PCG recurrence and workspace
│           └── gmres/         # generic restarted GMRES recurrence/workspace
├── athena-leto/
│   └── src/
│       ├── backend/           # Array1 and zero-copy array views
│       ├── operator/          # CSR and borrowed CowStorage dense operators
│       └── preconditioner/    # Jacobi inverse diagonal
└── athena-wgpu/
    └── src/
        ├── backend/kernels/   # prepared fused/vector WGSL kernels
        └── operator/          # Hephaestus GpuCsrMatrix adapter
```

Every `lib.rs` and `mod.rs` is a manifest. Operation families live in leaf
modules and no source file exceeds the repository's 500-line target.

`KrylovBackend` uses generic associated view types so the core recurrence
borrows Leto `ArrayView1`/`ArrayViewMut1` on CPU and typed
`WgpuBuffer<f32>` references on WGPU. `Cg<B>`,
`Gmres<B, const RESTART: usize>`, `Identity`, and `LetoBackend<T>` are
zero-sized policy markers. Static dispatch monomorphizes each recurrence at
the backend and restart-width boundary; no trait object or per-element backend
branch exists.

The CPU implementation performs no allocation after `CgWorkspace` or
`GmresWorkspace` construction. GMRES stores exactly `RESTART + 1` Arnoldi
vectors, `RESTART` preconditioned vectors, and const-bounded host scalar
storage. `BorrowedDenseOperator` holds Leto `CowStorage` and does not detach
because operator application is read-only. On WGPU, full vectors remain
device-resident and Athena uses prepared fused PCG kernels plus prepared scale
and AXPY kernels for GMRES. Current Hephaestus `dot` and `norm_l2` allocate
provider-local reduction buffers and transfer one scalar for each requested
convergence value, so this release makes no zero-allocation or
zero-submission-overhead GPU claim.

## Example

```rust
use athena::{
    Cg, CgWorkspace, ConvergencePolicy, Identity,
    cpu::{CsrOperator, LetoBackend},
};
use leto::Array1;
use leto_ops::CsrMatrix;

# // dyn exception: top-level example error aggregation is outside solver paths.
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let matrix = CsrMatrix::from_parts(
    vec![2.0_f64, -1.0, -1.0, 2.0],
    vec![0, 1, 0, 1],
    vec![0, 2, 4],
    2,
    2,
)?;
let operator = CsrOperator::new(matrix)?;
let backend = LetoBackend::<f64>::default();
let right_hand_side = Array1::from_shape_vec([2], vec![0.0, 3.0])?;
let mut solution = Array1::zeros([2]);
let mut workspace = CgWorkspace::new(&backend, 2)?;
let policy =
    ConvergencePolicy::new(64.0 * f64::EPSILON, 64.0 * f64::EPSILON, 4)?;

let report = Cg::<LetoBackend<f64>>::solve_into(
    &backend,
    &operator,
    &Identity,
    &right_hand_side,
    &mut solution,
    &mut workspace,
    policy,
)?;
assert!(report.converged());
# Ok(())
# }
```

The runnable `poisson_cpu` and `poisson_wgpu` examples solve the same
manufactured two-cell Poisson system with PCG. `nonsymmetric_cpu` and
`nonsymmetric_wgpu` solve the same three-variable nonsymmetric system with
restarted GMRES. The WGPU examples acquire a real Hephaestus device and have
no CPU fallback.

## Numerical contract

PCG requires a symmetric positive-definite operator and preconditioner.
Athena reports non-positive curvature, exact recurrence breakdown,
non-finite values, or iteration-budget exhaustion as explicit terminal values.
GMRES uses right preconditioning, modified Gram--Schmidt Arnoldi
orthogonalization, scaled Givens rotations, and true-residual recomputation
before convergence. Its restart width is a structural const generic.
Convergence uses

```text
||r||₂ <= max(absolute_tolerance, relative_tolerance * ||b||₂).
```

The recurrence follows Hestenes and Stiefel,
[*Methods of Conjugate Gradients for Solving Linear Systems*,
Journal of Research of the National Bureau of Standards 49(6), 1952,
pp. 409–436](https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_a1b.pdf),
and the preconditioned algorithm in Netlib's
[*Templates for the Solution of Linear Systems*, §2.3.1](https://www.netlib.org/templates/templates.html).
Restarted GMRES and right preconditioning follow §§2.3.4 and 3.1.2 of the same
reference.

## Verification

```sh
cargo fmt --check
cargo check --workspace --all-targets --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run --workspace --all-features
cargo test --doc --workspace --all-features
cargo doc --workspace --no-deps --all-features
cargo build --examples --all-features
cargo deny check
```

The CPU contract suite is generic over every shipped CPU scalar (`f32` and
`f64`) and checks manufactured SPD and nonsymmetric solutions, identity and
Jacobi preconditioning, forced multi-cycle restart, borrowed dense storage,
observer semantics, termination values, dimension errors, allocation
stability, and zero-sized markers. The WGPU suite runs both manufactured
systems through real Hephaestus allocation, CSR upload, SpMV, reductions, and
Athena kernels, then downloads each final solution once. A local machine
without an adapter records the unavailable lane; CI treats adapter acquisition
failure as an infrastructure failure.

## Roadmap

The next dependency-ordered increments are:

1. Add Hephaestus `dot_into`/prepared reductions and prepared fused solver
   dispatch so WGPU iterations reuse scalar and bind-group resources.
2. Extract consumer-owned CG/GMRES recurrences from CFDrs and Kwavers and
   migrate their operator/preconditioner implementations to Athena.
3. Add nonlinear solver policy only when a second concrete residual/Jacobian
   consumer establishes the shared contract.
