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
Non-goal for 1.0. A shipped area with no solve-level test is In scope as a
verification item, because a public API without an oracle is not a 1.0
contract; such an item gates no exit.

| Crate | Delivered | In scope for 1.0 | Non-goal for 1.0 |
|---|---|---|---|
| athena-core | `KrylovBackend` seam with GAT views and `VectorBlock`; `LinearOperator`, `RectangularOperator`, `Preconditioner`, `Identity`; `ConvergencePolicy`, `residual_noise_floor`, `Termination` with `Stagnated` and `Diverged` (ADR 0004); allocation-free reports and observers; `Cg`, `Gmres` (restarted, right-preconditioned, flexible-preconditioner property, rank-deficient recovery), `BiCgStab`, `Lsqr` (plain and damped), `Minres`, `Chebyshev`, `DeflatedCg` | Original-residual verification of CG termination (`ATHENA-CG-TERMINAL-RESIDUAL-20261103`); README and book that describe every shipped solver (`ATHENA-DOC-DRIFT-20261113`) | Nonlinear solver policy; QMR, TFQMR, IDR, LSMR and Krylov recycling; Arnoldi and Lanczos eigenvalue machinery; complex scalars |
| athena-leto | `LetoBackend` and `LetoVectorBlock`; runtime-restart `KrylovWorkspace`; `CsrOperator`, `BorrowedCsrOperator`, `BorrowedDenseOperator`, `PoissonStencilOperator`, `RectangularCsrOperator`; `Jacobi`, `IncompleteLu` and `SuccessiveOverRelaxation` preconditioners; `BlockDiagonalPreconditioner`, `ComponentBlockPreconditioner` and `SimplePreconditioner` (7 unit tests); `AlgebraicMultigrid` (2 unit tests) and its component API: five smoothers, five coarsening functions with `analyze_coarsening_quality`, the classical and direct interpolation builders, the restriction builders and both validators, `restrict_matrix` and `restrict_vector` (28 unit tests) | Solve-level contracts for `AlgebraicMultigrid`, the three saddle-point preconditioners and the two functions no test names, `hmis_coarsening` and `create_standard_interpolation` (`ATHENA-PRECONDITIONER-CONTRACT-20261113`) and for `LinearSolverChain`, which has no test (`ATHENA-SOLVER-CHAIN-CONTRACT-20261113`); a book chapter for the preconditioners, multigrid and chain | Coarsening over unstructured meshes beyond the shipped `*_coarsening` strategies |
| athena-hephaestus | `HephaestusBackend` over `DenseVectorOps`; prepared dot and norm; `CsrOperator` over `SparseOperatorOps`; device `Jacobi`; `Cg`, `Gmres`, `BiCgStab`, `DeflatedCg` and `Chebyshev` on device | `Lsqr` on device through a device `RectangularOperator` (the milestone exit); a device oracle for `Minres`, whose recurrence is backend-generic (`ATHENA-DEVICE-MINRES-20261113`) | Device incomplete-LU, SOR, multigrid and saddle-point preconditioners; a device stencil operator; device scalars other than `f32` |
| athena-krylov | Re-export of `athena-core`; features `cpu` (default) and `accelerator`; examples `poisson_cpu`, `nonsymmetric_cpu`, `poisson_accelerator`, `nonsymmetric_accelerator`; benches `cg_poisson`, `gmres_poisson`, `kernels`, `stencil_vs_csr`; a four-solver book | One self-checking example for the milestone (`least_squares_accelerator`), which solves with `Lsqr` through the device `RectangularOperator` and asserts the closed-form least-squares solution; README and book that describe every shipped solver | A crate per device (ADR 0034); runtime backend detection; a CPU fallback behind an accelerator example |

`triangular` is a private kernel of `IncompleteLu` and `SuccessiveOverRelaxation`,
not a preconditioner, though README, the book and module docs list it as one.

Sources for the non-goals: README Roadmap 2 (nonlinear policy waits for a
second residual and Jacobian consumer); Atlas ADR 0062 Non-goals (eigenvalue
machinery) and its 2026-10-12 revision, which closes Phase 2 with MINRES,
FGMRES, Chebyshev and two-level deflated CG, so recycling (named in the
Phase 2 decision text) and QMR, TFQMR, IDR and LSMR are not delivered and no
consumer or exit requires them; the book's stack chapter (the scalar is
`eunomia::RealField`; no `Backend::detect()`); Atlas ADR 0034 (one accelerator
backend over Hephaestus); Atlas ADR 0062 Phase 3 as narrowed by its 2026-10-11
revision (unstructured coarsening waits for a mesh-hierarchy owner or a second
consumer).

## Ladder

One milestone, the final block: `M-device-least-squares` (1.0.0), solve
overdetermined least-squares systems on a Hephaestus device.

Device `Lsqr` is the one capability this scope names that is absent at the
basis: README:26-28 records `RectangularOperator` as Leto-only,
`git grep RectangularOperator` finds no impl under `crates/athena-hephaestus/src`
(control: `crates/athena-leto/src/operator/rectangular_csr.rs`), and
`CsrOperator::from_parts` rejects `rows != columns` (`operator.rs:53`). Each
exit fails at the basis for lack of production code, not of a test file: the
example must solve with `Lsqr` through that operator, since host-formed normal
equations under device `Cg` would pass at base. Correctness, verification and
tightening items join the milestone and gate no exit; its standing conditions
require no open `correctness` or `verification` item, so the solve-level
contracts gate the release without being exits.

## Version basis

The milestone is the final block, so it plans 1.0.0, a major above the 0.3.3
manifest version and every crate tag. Crates version together and release per
crate tag (`crate-<package>-v<version>`); release needs explicit authority.

## Veto points

Choices no source determines, taken as the recommended option. Each states the
rejected alternative and the evidence that overturns the choice.

1. One milestone, released as 1.0.0. Rejected: a leading preconditioned-solves
   milestone (its exits test AMG and the chain, which ship at base, ADR 0062
   revision 2026-10-11) and a closing stable-suite milestone (a quality
   state); a 0.4.0 device block before a 1.0.0 block that would hold no
   feature absent at the basis. Overturned by: a second absent capability that
   a consumer requires adds a block before 1.0.0.
2. Consumer migration is not a milestone gate. Rejected: gating 1.0 on CFDrs
   and Kwavers conversions, which makes Athena wait on two repositories it
   does not own. Overturned by: a conversion that names a missing Athena
   capability (`ATLAS-CFDRS-ATHENA-MIGRATION-001` names `chain.rs` and
   multigrid as its next step), which then becomes an item or a block.
3. `Lsqr` on device is in scope. Grounds: README:25-28 states the backend-neutral
   contract and names this pair as the one that does not ship; no consumer
   requirement was found in the README roadmap or the Atlas items read.
   Rejected: a non-goal with CPU-only `Lsqr`. Design open:
   Hephaestus `SparseOperatorOps` has `upload_csr` and `apply` but no
   transposed apply, so `Aᵀ` is a second uploaded CSR (twice the device
   storage) or an upstream kernel; the item records the choice in its own ADR.
   Overturned by (scope): Hephaestus ruling device rectangular operators out
   of its contract, or no consumer needing device least squares, after which
   the ladder is re-derived; (design): an upstream transposed sparse kernel.
4. `LinearOperator::apply` and `Preconditioner` stay `&self` through 1.0.
   Rejected: `&mut self`, which breaks every implementor while flexible GMRES
   already drives a varying preconditioner through interior mutability.
   Overturned by: a consumer whose Jacobian-vector product cannot be expressed
   so (`ATLAS-JFNK-MATRIX-FREE-GMRES-001` asks the question).
5. QMR, TFQMR, IDR, LSMR, recycling and eigenvalue machinery are non-goals.
   Rejected: adding methods for coverage. Overturned by: a consumer
   requirement for the method.
6. Complex scalars are a non-goal; the scalar law is `RealField`. Rejected:
   generalising the scalar. Overturned by: a consumer needing complex systems
   and an Eunomia complex field.
7. Device scalar coverage stays at `f32`, the only scalar the device suites
   exercise. Rejected: `f64` device claims without a suite. Overturned by:
   Hephaestus owning an `f64` device path and a consumer using it.
8. Device preconditioners beyond `Jacobi` and a device stencil operator are
   non-goals. Rejected: incomplete-LU and SOR (sequential substitution,
   `athena-hephaestus` module docs); multigrid and saddle-point (the pinned
   Hephaestus `SparseOperatorOps` has no sparse-sparse product or block
   extraction); `StencilOps` under `PoissonStencilOperator` (its Dirichlet wall
   rows drop the inward neighbour the operator subtracts). Overturned by: a
   parallel triangular formulation, those sparse kernels or a matching stencil.

## Rejected alternatives

- A milestone for the book and README: rejected as a quality state; the
  example and book chapter ship with the capability, and the final block's
  standing conditions require the book and README builds.
- Milestones whose exits are new tests or examples of shipped capability (AMG,
  the chain, device `Minres`, CPU `Minres`, `Chebyshev`, `DeflatedCg`):
  rejected. That is verification work, ranked above features by the closed
  priority order; an exit counts a missing feature, and recording delivered
  work as exits would count it as progress.

## Consequences

- Every open item carries `milestone: M-device-least-squares`; a feature item
  names its exit.
- `ATHENA-CG-TERMINAL-RESIDUAL-20261103` is a `correctness` item, so the
  standing condition "no open correctness item" is false until it lands.

## Verification

Basis: `main` at `0e250fdfa9a8ecff85bc3810993952e9d1f78876`. Read: README,
ADRs 0001-0004, the book, the four `lib.rs` export lists and manifests, the
test and example lists, the release tags, and Atlas ADRs 0022, 0034 and 0062.
The `solver_chain` module has zero `#[test]` items. No Atlas conformance
detector reads `include.path` or `core.hooksPath` and athena tracks no
`.gitconfig` (`ATHENA-GITCONFIG-20261113`).
