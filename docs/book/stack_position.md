# 7. Position in the Stack

Athena is one repository in the [Atlas](https://github.com/ryancinsight/atlas)
multiphysics stack. This chapter says what it owns, what it deliberately does
not, and why the boundaries fall where they do — which is the information you
need to know whether a change belongs here or somewhere else.

## The layering

Atlas layers one way, foundation to integrator, and a dependency pointing the
other way is an architectural defect rather than a preference:

```text
eunomia        scalar law: RealField, FloatElement, precision contracts
   ↓
leto           host arrays, layouts, views, CSR storage, SpMV, reductions
hephaestus     accelerator devices, buffers, transfers, sparse kernels
   ↓
athena         iterative solver law  ←  you are here
   ↓
domain crates  discretisation, equations, time integration
   ↓
integrators    end-user applications
```

Athena sits above the array and device providers and below anything that knows
what the equations mean. The layer boundary is exactly the statement made in
the introduction: a Krylov method needs \\(A v\\) and vector arithmetic, and
nothing else. Everything below supplies those. Everything above decides what
\\(A\\) is.

## What Athena owns

- the matrix-free `LinearOperator`, `RectangularOperator`, and `Preconditioner`
  contracts;
- the validated absolute-plus-relative `ConvergencePolicy`, the derived
  `residual_noise_floor`, and the `Termination` vocabulary;
- the CG, restarted GMRES, BiCGSTAB, and LSQR recurrences, their numerical
  termination conditions, and their scalar telemetry;
- reusable, allocation-stable workspaces;
- the Leto and Hephaestus implementations of the solver-specific vector
  recurrences.

## What Athena does not own

- **Scalar law** — `RealField`, precision contracts, and accumulator policy are
  Eunomia's. Athena's `Scalar` is `eunomia::RealField`.
- **Host arrays** — layouts, views, CSR storage, SpMV, and reductions are
  Leto's. Athena's CSR operator wraps `leto_ops::CsrMatrix`; it does not define
  a second CSR representation.
- **Devices** — buffers, transfers, sparse kernels, reductions, and dispatch
  are Hephaestus's.
- **Equations** — discretisation, nonlinear residual construction, and time
  integration belong to the domain crates above.

The last exclusion is the one that shapes the API. A solver that knew it was
solving a pressure Poisson equation could exploit that; one that does not can
be reused by every consumer. The price is that the operator is a trait the
caller implements, and the benefit is that Athena has no domain dependencies at
all.

## Crate layout

| Crate | Role | Dependencies |
| --- | --- | --- |
| `athena-core` | contracts and recurrences | `no_std + alloc`, Eunomia only |
| `athena-leto` | CPU backend, operators, preconditioners | Leto |
| `athena-hephaestus` | accelerator backend and operator adapter | Hephaestus |
| `athena` | curated facade and runnable examples | the above, by feature |

`athena-core` forbids `unsafe` and denies missing documentation. It is
infrastructure-independent by construction: a consumer needing only the
convergence policy, the iteration observer, or the backend-neutral solver traits
acquires neither Leto nor Hephaestus. The `athena` facade enables the Leto CPU
backend through its default `cpu` feature; the `accelerator` feature adds
Hephaestus execution.

Backend selection is a *type* decision, not a runtime one. There is no
`Backend::detect()`; a program instantiates `Cg<LetoBackend<f64>>` or the
Hephaestus equivalent and the recurrence monomorphises to it. A program needing
to choose at runtime makes that choice at its own boundary.

## The extraction boundary

Athena's contents were previously split between Leto and a consumer. Leto
exported a GMRES recurrence, which made the array provider a second owner of
solver policy and meant the same recurrence could not run over Hephaestus
buffers. The extraction removed `gmres`, `GmresResult`, and the solver module
from Leto entirely — no re-export, no forwarding helper, no compatibility
module — and Leto kept CSR, SpMV, dot, arrays, and views.

The cross-repository ownership and migration boundary is recorded in
[Atlas ADR 0022](https://github.com/ryancinsight/atlas/blob/main/docs/adr/0022-horae-athena-provider-extraction.md).
Athena's own decisions live in `docs/adr/`:

| ADR | Decision |
| --- | --- |
| 0001 | Own Krylov orchestration above Leto and Hephaestus |
| 0002 | Add restarted right-preconditioned GMRES |
| 0003 | Move Krylov vector-set residency onto the backend seam |
| 0004 | Terminate on derived stagnation and divergence criteria |

## Adding a backend

Implement `KrylovBackend` for it. The associated types name the vector, the
block, the borrowed views, and the prepared reductions; the methods are the
operation set from the [Krylov Backend](krylov_backend.md) chapter. Every
solver then instantiates over it with no change to any recurrence, and the
generic conformance suites in `crates/athena-leto/tests/` are the contract the
new backend must satisfy.
