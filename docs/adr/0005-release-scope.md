# ADR 0005: Athena release scope

- Status: Proposed
- Date: 2026-11-13
- Class: [arch] [patch]

## Context

Athena is the Atlas stack's iterative-solver provider over Eunomia, Leto and
Hephaestus ([Atlas ADR 0022](https://github.com/ryancinsight/atlas/blob/main/docs/adr/0022-horae-athena-provider-extraction.md),
[Atlas ADR 0062](https://github.com/ryancinsight/atlas/blob/main/docs/adr/0062-iterative-solver-and-preconditioner-ownership.md)).
The workspace ships four crates: `athena-core`, `athena-leto`,
`athena-hephaestus` and the `athena-krylov` facade (library name `athena`).
Manifest and tags agree at 0.3.3: `crate-athena-core-v0.3.3`,
`crate-athena-leto-v0.3.3`, `crate-athena-hephaestus-v0.3.3` and
`crate-athena-krylov-v0.3.3`, all dated 2026-10-22.

The repository has never had a milestone ladder. Its README names four solver
contracts, while `athena-core` exports seven recurrences and the
`athena-leto` crate ships multigrid, saddle-point preconditioners and a tiered
solver chain that no README line or book chapter describes. This ADR fixes
what 1.0 contains from the crate export lists, the tests and the ADRs; the
ladder at the head of `backlog.md` orders the work.

## Decision

Each capability area is Delivered (a named test or example exercises it at the
basis), In scope (a ladder exit or a current-milestone item serves it) or
Non-goal for 1.0. A shipped area with no solve-level test is In scope, because
a public API without an oracle is not a 1.0 contract.

| Crate | Delivered | In scope for 1.0 | Non-goal for 1.0 |
|---|---|---|---|
| athena-core | `KrylovBackend` seam with GAT views and `VectorBlock`; `LinearOperator`, `RectangularOperator`, `Preconditioner`, `Identity`; `ConvergencePolicy`, `residual_noise_floor`, `Termination` with `Stagnated` and `Diverged` (ADR 0004); allocation-free reports and observers; `Cg`, `Gmres` (restarted, right-preconditioned, flexible-preconditioner property, rank-deficient recovery), `BiCgStab`, `Lsqr` (plain and damped), `Minres`, `Chebyshev`, `DeflatedCg` | Original-residual verification of CG termination (item `ATHENA-CG-TERMINAL-RESIDUAL-20261103`); a tour of every solver | Nonlinear solver policy; QMR, TFQMR, IDR, LSMR and Krylov recycling; Arnoldi and Lanczos eigenvalue machinery; complex scalars |
| athena-leto | `LetoBackend` and `LetoVectorBlock`; runtime-restart `KrylovWorkspace`; `CsrOperator`, `BorrowedCsrOperator`, `BorrowedDenseOperator`, `PoissonStencilOperator`, `RectangularCsrOperator`; `Jacobi`, `IncompleteLu`, triangular and `SuccessiveOverRelaxation` preconditioners | Solve-level contract for `AlgebraicMultigrid` (two unit tests today); solve-level contract for `LinearSolverChain` (no test today); a book chapter for the preconditioners, multigrid and chain | Coarsening over unstructured meshes beyond the shipped strategies |
| athena-hephaestus | `HephaestusBackend` over `DenseVectorOps`; prepared dot and norm; `CsrOperator` over `SparseOperatorOps`; device `Jacobi`; `Cg`, `Gmres`, `BiCgStab`, `DeflatedCg` and `Chebyshev` on device | `Minres` on device; `Lsqr` on device through a device `RectangularOperator` | Device incomplete-LU, multigrid and saddle-point preconditioners; a device stencil operator; device scalars other than `f32` |
| athena-krylov | Re-export of `athena-core`; features `cpu` (default) and `accelerator`; examples `poisson_cpu`, `nonsymmetric_cpu`, `poisson_accelerator`, `nonsymmetric_accelerator`; benches `cg_poisson`, `gmres_poisson`, `kernels`, `stencil_vs_csr`; a four-solver book | One self-checking example per milestone; README and book that describe every shipped solver | A crate per device (ADR 0034); runtime backend detection; a CPU fallback behind an accelerator example |

Sources for the non-goals: README Roadmap 2 (nonlinear policy waits for a
second residual and Jacobian consumer); Atlas ADR 0062 Non-goals (eigenvalue
machinery) and its list of absent methods (QMR, TFQMR, IDR, LSMR, recycling),
none of which any consumer or exit requires; the book's stack chapter (the
scalar is `eunomia::RealField`; no `Backend::detect()`); Atlas ADR 0034 (one
accelerator backend over Hephaestus); Atlas ADR 0062 Phase 3 (unstructured
coarsening waits for a mesh-hierarchy owner or a second consumer).

## Ladder

The milestones, in dependency order, are:

- `M-preconditioned-solves` (0.4.0): solve Poisson and saddle-point systems
  with multigrid and block preconditioning through the facade and the chain.
- `M-accelerator-solver-parity` (0.5.0): solve indefinite and least-squares
  systems on a Hephaestus device.
- `M-stable-solver-suite` (1.0.0): use every shipped solver from a tagged
  release on a stable API.

Each exit names a test or example that does not exist at the basis commit
(`git grep` on the named identifier finds nothing). Tightening, verification
and correctness items join the current milestone and gate no exit.

## Version basis

Planned versions are successive minors above 0.3.3, the manifest version and
the version of every crate tag, then 1.0.0. Crates version together
(`[workspace.package]`) and release per crate tag
(`crate-<package>-v<version>`). Release and publication need explicit
authority; the 1.0.0 block closes only its exit and standing conditions.

## Veto points

Choices no source determines, taken as the recommended option:

1. Order: preconditioned solves before accelerator parity. The meta item
   `ATLAS-CFDRS-ATHENA-MIGRATION-001` names `chain.rs` and multigrid as its
   next conversion step, so the consumer path needs those contracts first.
2. Consumer migration is not a milestone gate. CFDrs and Kwavers conversions
   (README Roadmap 1, `ATLAS-CFDRS-ATHENA-MIGRATION-001`,
   `ATLAS-JFNK-MATRIX-FREE-GMRES-001`) are meta-owned and land when each
   consumer is ready; Athena serves them with contract items, not exits.
3. `Lsqr` on device is in scope. The design is open: Hephaestus
   `SparseOperatorOps` exposes `upload_csr` and `apply` but no transposed
   apply, so `Aᵀ` is either a second uploaded CSR (twice the device storage)
   or an upstream kernel. The item records the choice in its own ADR.
4. `LinearOperator::apply` and `Preconditioner` stay `&self` through 1.0. The
   flexible-GMRES contract already drives a varying preconditioner through
   interior mutability, and a `&mut self` change would break every
   implementor. A consumer whose Jacobian-vector product needs mutation uses
   the same pattern; `ATLAS-JFNK-MATRIX-FREE-GMRES-001` asks the open question.
5. QMR, TFQMR, IDR, LSMR, recycling and eigenvalue machinery are non-goals;
   a consumer requirement reopens each.
6. Complex scalars are a non-goal; the scalar law is `RealField`.
7. Device scalar coverage stays at `f32`, the only scalar the device suites
   exercise; Hephaestus owns the scalar set.
8. Device preconditioners other than `Jacobi` and the device stencil operator
   are non-goals.
9. Planned versions 0.4.0, 0.5.0 and 1.0.0.

## Rejected alternatives

- One release without a ladder: rejected. The preconditioner and device
  slices touch different crates and each needs its own runnable check.
- A milestone for the book and README: rejected as a quality state. Each
  milestone's example and its book chapter ship with the capability, and the
  final block's standing conditions require the book and README builds.
- A milestone that gates 1.0 on consumer migration: rejected. It makes
  Athena's release wait on two repositories it does not own.
- Milestones for delivered solvers (`Minres`, `Chebyshev`, `DeflatedCg`
  on CPU): rejected. Each has a contract suite in `crates/athena-leto/tests/`;
  recording them as exits would count delivered work as progress.

## Consequences

- Every open item carries one `milestone:`; a feature item names its exit.
- `ATHENA-CG-TERMINAL-RESIDUAL-20261103` is a `correctness` item, so the
  standing condition "no open correctness item" is false until it lands.
- Overturning evidence: a consumer requirement for a non-goal; a Hephaestus
  transposed sparse kernel (changes the `Lsqr` device design); a consumer
  whose operator cannot be expressed as `&self` (reopens veto point 4).

## Verification

Basis: `main` at `0e250fdfa9a8ecff85bc3810993952e9d1f78876`. Read: README,
ADRs 0001-0004, the book, the four `lib.rs` export lists and manifests, the
test and example lists, the release tags, and Atlas ADRs 0022, 0034 and 0062.
The `solver_chain` module has zero `#[test]` items; the facade examples and
the `athena-leto` tests do not name `LinearSolverChain`.
