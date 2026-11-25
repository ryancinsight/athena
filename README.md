# Athena

[![CI](https://github.com/ryancinsight/athena/actions/workflows/ci.yml/badge.svg)](https://github.com/ryancinsight/athena/actions/workflows/ci.yml)

Athena is Atlas's iterative-solver provider. `athena-core` exports seven
solver recurrences:

| Solver | System | Operator contract | Storage per solve |
| --- | --- | --- | --- |
| `Cg` | symmetric positive-definite | `LinearOperator` | 4 vectors |
| `Gmres<_, RESTART>` | general nonsymmetric | `LinearOperator` | `2·RESTART + 3` vectors |
| `BiCgStab` | general nonsymmetric | `LinearOperator` | 7 vectors |
| `Lsqr` | rectangular least squares | `RectangularOperator` | 5 vectors (2 × `rows`, 3 × `columns`) |
| `Minres` | symmetric, possibly indefinite | `LinearOperator` | 7 vectors |
| `Chebyshev` | symmetric positive-definite, with spectral bounds | `LinearOperator` | 3 vectors |
| `DeflatedCg` | symmetric positive-definite, with a deflation basis | `LinearOperator` | 4 workspace vectors; per solve, 6 vectors and 3 basis-width scalar arrays (2 vectors and no arrays when the initial residual already meets the policy); and the basis and its images |

`Cg`, `Gmres` and `BiCgStab` accept a `Preconditioner`; `Lsqr`, `Minres`,
`Chebyshev` and `DeflatedCg` take none. `Minres` is deliberately unpreconditioned
until a consumer needs the symmetric-positive-definite factor seam it requires.

Preconditioned conjugate gradient (PCG) and restarted GMRES are the
symmetric and general square recurrences. `BiCGSTAB` solves the same general
systems as GMRES in constant storage — no restart parameter and no `O(n·m)`
Arnoldi basis — at the cost of a non-monotone residual and two operator
applications per iteration; the choice between them trades storage against
residual smoothness, and neither is a fallback for the other. LSQR minimises
`‖A·x − b‖₂` for rectangular
operators by Golub–Kahan bidiagonalisation, never forming `AᵀA` and so never
squaring the condition number.

Leto CPU and Hephaestus accelerator execution share one backend-neutral
recurrence per solver family. The accelerator suites run `Cg`, `Gmres`,
`BiCgStab`, `DeflatedCg`, `Chebyshev` and `Minres`.
`Lsqr` runs on any backend, but `RectangularOperator` is implemented for Leto
only, so its shipped execution is CPU.

The CPU backend ships these preconditioners: `Jacobi`, `IncompleteLu`,
`SuccessiveOverRelaxation`, `AlgebraicMultigrid` with its smoother, coarsening,
interpolation and restriction components, and the saddle-point
`BlockDiagonalPreconditioner`, `ComponentBlockPreconditioner` and
`SimplePreconditioner`. `LinearSolverChain::solve` tries a direct sparse LU for
systems of fewer than `direct_threshold` unknowns (default 2,048); GMRES under
the component-block, SIMPLE, multigrid and block-diagonal preconditioners;
unpreconditioned GMRES, only when the block-diagonal preconditioner could not
be built; GMRES under incomplete LU; and unpreconditioned `BiCgStab`.
`solve_with_guess_state` starts from the caller's optional guess (from zeros
when it is `None`), tries GMRES under SOR before a cached multigrid tier, and
orders the remaining tiers differently.
Both return the first tier that succeeds and a `LetoBackendError` when every
tier fails. `LinearSolverState` caches the multigrid transfer operators and
the component-block symbolic factors for warm solves. The accelerator backend
ships `Jacobi` only.

Athena is independently versioned and consumed by the
[Atlas multiphysics stack](https://github.com/ryancinsight/atlas). The
cross-repository ownership and migration boundary is recorded in
[Atlas ADR 0022](https://github.com/ryancinsight/atlas/blob/main/docs/adr/0022-horae-athena-provider-extraction.md).

## Deflation and spectral iteration

`DeflatedCg` solves SPD systems over any supported deflation basis, including
noninvariant bases. With `E = Uᵀ A U` and `C = U E⁻¹ Uᵀ`, it solves the Schur
system `(A - A C A)y = (I - A C)(b - A x₀)` on the complement and recovers
`x = x₀ + y + C(b - A x₀ - A y)`. The final report checks the true residual
`b - A x` against the original RHS policy. The subspace must be built from
the same operator used for the solve. Cached `A U` images require one extra
basis-sized block and avoid a second base operator application per iteration.
Coarse coordinate scratch is retained in `DeflatedOperator`, so repeated
applications allocate no memory after construction.

`Chebyshev` accepts positive ordered spectral bounds, including equal
endpoints. A singleton spectrum implies `A = λ I`, so the normalised
recurrence becomes a Richardson update with scale `1/λ`, exact in one step
in exact arithmetic. The general coefficients use the bounded inverse centre
`η = (1 - λ_min/λ_max)/(1 + λ_min/λ_max)` instead of forming an infinite
centre for zero-width or narrow intervals. Reports always check the updated
iterate before preparing another step.

## Rank-deficient GMRES

The full-rank reduced factor uses ordinary triangular back-substitution.
Numerical rank loss switches to in-place column-pivoted QR with permutation
scratch prepared in the workspace, preserving warm-solve allocation freedom.
It returns a basic least-squares correction for the rank-truncated factor,
without promising minimum coefficient or physical solution norm. The rank
cutoff is a statistical numerical policy, not a worst-case rank certificate.
Terminal-cycle telemetry and termination use the recomputed original residual;
inner-cycle telemetry can use the recurrence estimate. See the [GMRES
chapter](docs/book/gmres.md) for the attainable rank-deficient example.

## Ownership boundary

Athena owns:

- matrix-free square-operator, rectangular-operator, and preconditioner
  contracts;
- validated absolute-plus-relative convergence policy;
- PCG, restarted GMRES, `BiCGSTAB`, LSQR, MINRES, Chebyshev, and deflated-CG
  recurrences, numerical termination, and scalar telemetry;
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

This boundary removes Krylov orchestration from Leto without wrapping or
mirroring Leto's APIs. `leto_ops::CsrMatrix` remains the single CSR
representation used by both CPU and accelerator operators.

## Architecture

```text
crates/
├── athena/                    # curated facade and runnable examples
├── athena-core/
│   └── src/
│       ├── backend/           # GAT-based borrowed vector and block family
│       ├── convergence/       # validated residual policy
│       ├── operator/          # square and rectangular operator seams
│       ├── preconditioner/    # preconditioner seam and Identity ZST
│       ├── report/            # allocation-free scalar telemetry
│       └── solver/
│           ├── bicgstab/      # generic BiCGSTAB recurrence and workspace
│           ├── cg/            # generic PCG recurrence and workspace
│           ├── chebyshev/     # Chebyshev semi-iteration and spectral bounds
│           ├── deflation/     # two-level deflated CG and its basis
│           ├── gmres/         # generic restarted GMRES recurrence/workspace
│           ├── lsqr/          # generic LSQR recurrence and workspace
│           └── minres/        # generic MINRES recurrence and workspace
├── athena-leto/
│   └── src/
│       ├── backend/           # Array1, zero-copy views, contiguous blocks
│       ├── krylov.rs          # runtime-restart GMRES workspace
│       ├── operator/          # CSR, rectangular CSR, dense, Poisson stencil
│       ├── preconditioner/    # Jacobi, incomplete LU, SOR, multigrid, saddle-point
│       └── solver_chain/      # tiered direct-then-GMRES fallback chain
└── athena-hephaestus/
    └── src/
        ├── backend.rs         # device-neutral DenseVectorOps binding
        ├── operator.rs        # CsrOperator<S, M> over SparseOperatorOps
        └── preconditioner/    # device Jacobi
```

Operation families live in leaf modules. No source file under `crates/` exceeds
the repository's 500-line target; the largest is
`athena-core/src/solver/bicgstab/execution.rs`, at 487 lines.

The `athena` facade enables the Leto CPU backend by default through the `cpu`
feature; the `accelerator` feature adds Hephaestus execution. `athena-core` is
the `no_std + alloc`, infrastructure-independent contract crate. Consumers
that only need convergence policy, iteration observation, or backend-neutral
solver traits do not acquire Leto or Hephaestus dependencies.

`KrylovBackend` uses generic associated view types so the core recurrence
borrows Leto `ArrayView1`/`ArrayViewMut1` on CPU and typed device-buffer
references on an accelerator. `Cg<B>`, `Gmres<B, const RESTART: usize>`,
`BiCgStab<B>`, `Lsqr<B>`, `Minres<B>`, `Chebyshev<B>`, `DeflatedCg<B>`,
`Identity`, and `LetoBackend<T>` are zero-sized policy markers. Static dispatch
monomorphizes each recurrence at the backend
and restart-width boundary; no trait object or per-element backend branch
exists.

The `Cg`, `Gmres`, `BiCgStab`, `Lsqr`, `Minres` and `Chebyshev` recurrences
allocate nothing after workspace construction: `crates/athena-core/src/solver`
calls `allocate`, `allocate_block`, `vec!`, `Vec::new`, `Vec::with_capacity`
and `to_vec` only in the workspace constructors and the deflation module, and
the allocation tests cover `Cg`, `Gmres` and `BiCgStab`. `DeflatedCg::solve_into`
allocates on every solve (see the table), and so does `LinearSolverChain`: each
solve builds the solution vector, and `solve_with_guess_state` clones a given
guess into it. A solve the direct tier ends runs one sparse LU solve, with a
dense retry for small systems, and builds no iterative-tier preconditioner or
workspace. Otherwise each iterative tier a solve attempts constructs its
preconditioner (the unpreconditioned and `BiCgStab` tiers use `Identity` and
build none) and, once that succeeds, a workspace.
GMRES stores exactly `RESTART + 1` Arnoldi vectors, `RESTART` preconditioned
vectors, and const-bounded host scalar storage. Those two vector sets are
`KrylovBackend::VectorBlock` values rather than `Vec<Vector>`, so each backend
chooses its own residency: the Leto block is one contiguous allocation lent
out as offset subviews, while an accelerator block stays a set of independent
device buffers, which is what its allocator and bind-group model require.
`BorrowedDenseOperator` holds Leto `CowStorage` and does not detach because
operator application is read-only. On an accelerator, full vectors remain
device-resident and Athena uses prepared scale and AXPY kernels plus
Hephaestus prepared dot/norm reductions. Workspace
construction fixes each reduction's input allocations and creates its output,
scratch, pipeline, and bind-group resources once. Iterations dispatch those
plans without rebuilding provider resources; only the convergence scalar
crosses to the host. Solver initialization copies the right-hand side into the
residual workspace before measuring its norm, so repeated solves reuse the
same prepared norm instead of allocating a one-shot reduction.

## Prepared-reduction performance

The provider-owned Criterion instrument compares identical 65,536-element
operations before and after preparation with 100 samples. On an Intel Core
Ultra 9 285K host and NVIDIA RTX 5080 (driver 610.47), prepared dot measured
107.65 us versus 144.79 us one-shot (25.7% lower point estimate), and prepared
L2 norm measured 122.28 us versus 158.89 us one-shot (23.0% lower). The 95%
confidence intervals were 105.19--110.40 us and 141.65--148.34 us for dot,
and 119.79--124.79 us and 150.88--169.24 us for L2 norm. This evidence covers
uncontended provider dispatch latency on that machine; it does not establish
whole-solver speedup or portability to other adapters.

Reproduce the instrument in Hephaestus with:

```sh
cargo bench -p hephaestus-wgpu --bench prepared_map_reduction
```

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

The runnable `poisson_cpu` and `poisson_accelerator` examples solve the same
manufactured two-cell Poisson system with PCG. `nonsymmetric_cpu` and
`nonsymmetric_accelerator` solve the same three-variable nonsymmetric system
with restarted GMRES. The accelerator examples acquire a real Hephaestus
device and have no CPU fallback.

## Numerical contract

Every solver except `Minres` shares one convergence test:

```text
||r||₂ <= max(absolute_tolerance, relative_tolerance * ||b||₂).
```

`Minres` scales its relative tolerance by the initial residual norm
`||b - A x₀||₂` instead of `||b||₂`; the two agree when `x₀ = 0`.

PCG requires a symmetric positive-definite operator and preconditioner.
Athena reports non-positive curvature, exact recurrence breakdown,
non-finite values, or iteration-budget exhaustion as explicit terminal values.

PCG recomputes the original-system residual `b - A x` before accepting
convergence and at curvature and budget terminations, so recurrence drift
cannot produce a false success and exact convergence between scheduled checks
succeeds. A zero preconditioned product with a nonzero residual remains a
breakdown. Each verification counts in `operator_applications`, reuses
workspace scratch, and allocates nothing in a warm solve.

GMRES uses right preconditioning, modified Gram--Schmidt Arnoldi
orthogonalization, scaled Givens rotations, and true-residual recomputation
before convergence. Its restart width is a structural const generic.

`BiCGSTAB` also preconditions on the right, so the recurrence residual is the
residual of the original system and the convergence policy applies to it
directly; left preconditioning would instead measure `‖M⁻¹(b − A·x)‖`, which
differs from the true residual by up to `κ(M)`. Its residual is non-monotone
by construction, so it too recomputes the true residual before declaring
convergence, and it reports the `ρ` and `ω` breakdowns of the two-term
recurrence as explicit terminal values.

LSQR terminates on two distinct criteria because a least-squares system has
two. A consistent system is detected on the residual itself. An inconsistent
one — the genuine least-squares case — has a residual bounded away from zero,
so its criterion is the normal-equation residual `‖Aᵀr‖` relative to `‖r‖`,
reported as `Termination::NormalEquations`. Testing only the residual would
run such a solve to the iteration cap despite it having found the exact
minimiser.

The CG recurrence follows Hestenes and Stiefel,
[*Methods of Conjugate Gradients for Solving Linear Systems*,
Journal of Research of the National Bureau of Standards 49(6), 1952,
pp. 409–436](https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_a1b.pdf),
and the preconditioned algorithm in Netlib's
[*Templates for the Solution of Linear Systems*, §2.3.1](https://www.netlib.org/templates/templates.html).
Restarted GMRES and right preconditioning follow §§2.3.4 and 3.1.2 of the same
reference. `BiCGSTAB` follows van der Vorst (1992), *Bi-CGSTAB: a fast and
smoothly converging variant of Bi-CG for the solution of nonsymmetric linear
systems*, SIAM J. Sci. Stat. Comput. 13(2), 631–644. LSQR follows Paige and
Saunders (1982), *LSQR: An algorithm for sparse linear equations and sparse
least squares*, ACM TOMS 8(1), 43–71.

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
`f64`) and checks manufactured SPD, nonsymmetric, consistent overdetermined,
inconsistent, and underdetermined solutions; identity, Jacobi, incomplete-LU,
and SOR preconditioning; forced multi-cycle restart; borrowed
dense storage; adjoint application against the dense product; observer
semantics; termination values; dimension errors; allocation stability for CG,
GMRES, and `BiCGSTAB`; contiguity, zero-initialization, and non-aliasing of
the vector blocks; and zero-sized markers. The accelerator suite runs the
manufactured SPD and nonsymmetric systems through real Hephaestus allocation,
CSR upload, SpMV, reductions, and Athena kernels with CG, GMRES, and
`BiCGSTAB`, then downloads each final solution once; separate device
regressions run `DeflatedCg`, `Chebyshev` and `Minres`. A local machine without
an adapter records the unavailable lane; CI treats adapter acquisition failure
as an infrastructure failure.

The allocation cases wrap Mnemosyne in `mnemosyne::counting::CountingAllocator`,
which counts per thread, so each case reads only the allocations of its own
thread and is independent under both `cargo nextest run` and the threaded
`cargo test` harness.

## Roadmap

The 1.0.0 milestone, solving overdetermined least-squares systems with `Lsqr`
on a Hephaestus device, is scoped in
[ADR 0005](docs/adr/0005-release-scope.md). The next dependency-ordered
increments beyond it are:

1. Extract consumer-owned CG/GMRES recurrences from CFDrs and Kwavers and
   migrate their operator/preconditioner implementations to Athena.
2. Add nonlinear solver policy only when a second concrete residual/Jacobian
   consumer establishes the shared contract.

### Pre-push hook

`.githooks/` holds the Atlas member gates (`pre-commit`, `pre-push`,
`rescue-push`). They are copies of one stack-owned source that
`scripts/atlas-lock-form.py sync-hooks` deploys, so change them in the Atlas
repository, never here. Git applies no tracked hook on its own; from the Atlas
stack root, run once per clone:

```sh
python scripts/atlas-lock-form.py install-hooks
```

It points each member's `core.hooksPath` at shared source-bound hook shims. The
`pre-push` hook refuses a `Cargo.lock` that does not resolve under `--locked`,
using the stack's own `lockfile.py --check` and never this repository's copy.
CI's lockfile job calls the Atlas `lockfile-guard.yml` workflow at the commit
`.github/workflows/ci.yml` pins, which runs this repository's
`scripts/lockfile.py --check`. The check matters most inside the Atlas stack: the
stack's `[patch]` overlay makes cargo resolve first-party dependencies to
local paths and write a `Cargo.lock` with every `source = "git+..."` line
stripped. That lock resolves fine under the overlay and fails every
`--locked` job in CI, so without the hook the corruption is invisible until a
runner reports it. Repair with `python3 scripts/lockfile.py --regenerate`.

## License

Licensed under either the MIT License or Apache License 2.0.
