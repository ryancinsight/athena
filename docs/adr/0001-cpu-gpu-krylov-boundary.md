# ADR 0001: Own Krylov orchestration above Leto and Hephaestus

- Status: Accepted
- Date: 2026-07-19
- Class: [arch] [minor]

## Context

Leto publicly owned CG and GMRES beside arrays and sparse kernels, while CFDrs
and Kwavers own additional solver recurrences. That topology mixes algorithm
policy with storage and prevents one recurrence from substituting host and
accelerator execution.

The first extraction must prove a real vertical contract without creating a
second CSR type, a consumer-local GPU substrate, a dynamic-dispatch hot path,
or a CPU fallback labeled as GPU execution.

The CG recurrence and its symmetric-positive-definite domain follow Hestenes
and Stiefel, *Methods of Conjugate Gradients for Solving Linear Systems*,
Journal of Research of the National Bureau of Standards 49(6), 1952,
pp. 409–436:
<https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_a1b.pdf>.
The preconditioned form follows *Templates for the Solution of Linear
Systems*, §2.3.1: <https://www.netlib.org/templates/templates.html>.

## Decision

Create a four-crate workspace with inward dependency direction:

```text
athena facade
├── athena-leto ──> athena-core <── athena-wgpu
│       │                              │
│       └── Leto                       └── Hephaestus + Leto CSR host form
└──────────────────────────────────────────────────────────────────────
```

- `athena-core` owns `KrylovBackend`, operator and preconditioner contracts,
  convergence policy, reports, PCG and restarted-GMRES workspaces, and one
  recurrence per solver family.
- `KrylovBackend` uses GAT view families. Solver code borrows provider-native
  storage and monomorphizes per backend.
- `athena-leto` maps vectors to `Array1<T>`, delegates CSR SpMV and dot to
  Leto, supplies a Jacobi preconditioner, and exposes a read-only
  `CowStorage` dense operator.
- `athena-wgpu` maps vectors to typed Hephaestus WGPU buffers, delegates CSR
  storage, SpMV, reductions, transfer, and dispatch to Hephaestus, and owns
  only solver-specific prepared WGSL kernels.
- GPU PCG initially supports `f32`, the scalar for which Hephaestus currently
  implements the required WGPU norm contract.
- Numerical termination is a value, not a backend error. Contract or dispatch
  failures remain typed errors.
- No solver allocates implicit residual history. Callers receive scalar samples
  through an `IterationObserver`.
- Leto's CG and restarted-GMRES implementations and exports are deleted after
  Athena's CPU and WGPU conformance suites pass. No compatibility wrapper is
  introduced.

## Rejected alternatives

### Keep solver algorithms in Leto and add WGPU methods

Rejected because it makes host storage own cross-backend algorithm policy and
pulls accelerator concerns toward the CPU array provider.

### Duplicate CG in CPU and WGPU crates

Rejected because recurrence and termination semantics would drift. A GAT
backend contract preserves provider-native borrowing under one static
implementation.

### Use `dyn` operator, preconditioner, or backend traits

Rejected because the hot path has a finite compile-time strategy set. Generic
dispatch is monomorphic and keeps invalid backend/view combinations out of the
runtime.

### Download vectors for WGPU updates

Rejected as a fake GPU path. Full vectors stay resident; only scalar reduction
results cross the host boundary for control flow.

### Implement all existing solver families in the bootstrap change

Rejected because CG is the first complete CPU/GPU conformance slice. Restarted
GMRES requires a different basis/workspace contract, and nonlinear solvers
require residual/Jacobian consumers. Those contracts cannot be inferred from
CG without speculative generality.

## Consequences

- CPU solve iterations allocate nothing after workspace construction.
- WGPU vectors remain resident, but current provider reductions allocate
  scalar/scratch buffers and synchronize a scalar readback. A later
  Hephaestus increment must add prepared/reusable reductions before Athena can
  claim zero-allocation GPU iteration.
- The WGPU backend is a real Hephaestus consumer-authored-kernel path and does
  not depend directly on raw `wgpu`.
- `leto_ops::CsrMatrix` stays the CSR SSOT used for host execution and
  Hephaestus upload.
- Consumer migrations remain as dependency-ordered follow-up work after Atlas
  promotes Athena to a current package.

## Verification

- one generic CPU contract suite instantiates PCG for `f32` and `f64`;
- one generic CPU contract suite instantiates GMRES for `f32` and `f64`;
- a manufactured SPD system has exact solution `[1, 2]`;
- a manufactured nonsymmetric system has exact solution `[1, -2, 3]`;
- CPU Jacobi and identity-preconditioned paths meet epsilon-derived bounds;
- negative tests check shape and non-positive-curvature semantics;
- layout tests pin every policy marker claimed to be zero-sized;
- a real WGPU contract executes the same system through Hephaestus and checks
  the downloaded solution against an `f32` reduction-order bound;
- a real WGPU GMRES contract executes the nonsymmetric system through
  Hephaestus and checks the downloaded solution against its derived bound;
- local adapter absence is reported as unavailable coverage, while CI adapter
  absence fails the GPU lane.
