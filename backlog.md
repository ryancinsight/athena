# Backlog — athena

Board for the Athena iterative-solver provider. Items are DoR-shaped: outcome,
scope/non-goals, acceptance oracle, dependencies, risk/change class, status.
Priority order is the order below: correctness and verification first, then
architecture required for correctness, then CI conformance, then documentation
and PM hygiene.

Created 2026-08-20 by the Atlas gap audit against `f6608be` (detached, four
peer-modified files preserved). Every item cites the evidence that opened it.

---

## AT-001 — Fail the accelerator lane when no adapter is acquired [patch] — todo

- Outcome: an adapter-less run reports the accelerator lane as not executed
  instead of green, and a CI run without an adapter fails.
- Evidence: `crates/athena-hephaestus/tests/backend_contract.rs:19` returns
  `None` on `WgpuDevice::try_default` failure and all seven cases early-return
  (`:87`, `:112`, `:140`, `:169`, `:195`, `:218`, `:241`). No source in
  `crates/` reads the `CI` environment variable that `.github/workflows/ci.yml`
  sets, so the CI and local paths are identical.
- Scope: `crates/athena-hephaestus/tests/backend_contract.rs` and, if the
  signal must be explicit, one CI step. Non-goals: changing solver or backend
  code; adding a CPU fallback to the accelerator examples.
- Acceptance oracle: with an adapter present the seven cases run and pass; with
  the adapter removed, a run under `CI=true` fails with a message naming the
  acquisition error, and a local run without `CI` reports the lane as skipped
  in a way `cargo nextest run` surfaces (skip, not silent pass).
- Dependencies: none.
- Risk/change class: [verification] [patch]. Effort M.
- Re-open trigger: a second accelerator device family is added and needs its
  own lane signal.

## AT-002 — Differential-test the accelerator path against the CPU path [patch] — todo

- Outcome: every operation the Hephaestus backend implements is compared to the
  Leto implementation of the same operation on the same input, within a
  tolerance derived from `f32` epsilon, vector length, and reduction depth.
- Evidence: the accelerator suite asserts against hand-known solutions with a
  hardcoded absolute epsilon —
  `crates/athena-hephaestus/tests/backend_contract.rs:70` and `:72` use
  `1e-3` on a two-component `f32` solution. There is no CPU reference in that
  file and no derivation for `1e-3`; `athena_core::residual_noise_floor`
  (`crates/athena-core/src/convergence/residual_noise.rs:59`) already publishes
  the derived scale the comparison should use.
- Scope: a new differential test target covering `copy`, `scale`, `axpy`,
  `dot`, `norm_l2`, `residual`, `fused_cg_update`, `combine_direction`,
  `allocate_block`/`block_view`, and CSR SpMV, plus end-to-end CG/GMRES/
  BiCGSTAB solutions. Non-goals: changing kernels; adding a new backend.
- Acceptance oracle: for each operation, `|gpu - cpu| <= derived_bound` with the
  bound's derivation cited at the assertion site; the test fails when the bound
  is halved on a genuinely divergent kernel and passes on the shipped one.
  No literal epsilon without a derivation remains in the accelerator suite.
- Dependencies: AT-001 (a skipped lane cannot be differential evidence).
- Risk/change class: [verification] [patch]. Effort M.
- Re-open trigger: a new `KrylovBackend` method lands without a differential
  case.

## AT-003 — Extend derived stagnation and divergence to CG, BiCGSTAB, and LSQR [minor] — todo

- Outcome: a stalled or diverging solve in each family reports the condition
  instead of consuming its whole budget and reporting `MaxIterations`.
- Evidence: `Termination::Stagnated`/`Diverged` are declared on the shared enum
  (`crates/athena-core/src/report/solve.rs`) but detected only by GMRES
  (`crates/athena-core/src/solver/gmres/algorithm.rs:248`, `:251`). CG
  (`crates/athena-core/src/solver/cg/algorithm.rs:256`), BiCGSTAB
  (`crates/athena-core/src/solver/bicgstab/algorithm.rs:206`), and LSQR
  (`crates/athena-core/src/solver/lsqr/algorithm.rs:264`) terminate at the bare
  cap. ADR 0004 "Consequences" records the deferral and states each family
  needs its own derivation because BiCGSTAB's residual is not monotone.
- Scope: per-family criteria and their derivations, recorded as a revision of
  ADR 0004 or a new ADR; the three algorithm modules and their contract tests.
  Non-goals: a tuned window length or threshold; changing `SolveError`.
- Acceptance oracle: each family has (a) a written derivation of its progress
  criterion measured against `residual_noise_floor`, (b) a manufactured stalling
  system that terminates early with the new variant, and (c) a companion
  system that does make progress and still reports `Converged`, so the detector
  is not a false positive. CG's A-norm monotonicity and BiCGSTAB's
  non-monotonicity are each addressed explicitly.
- Dependencies: none.
- Risk/change class: [correctness] [minor] (behavioural change to a published
  terminal value). Effort M.
- Re-open trigger: a consumer reports a budget-exhausted solve that had stopped
  progressing.

## AT-004 — Provide a preconditioner for the Hephaestus backend [minor] — todo

- Outcome: an accelerator PCG solve can use a real preconditioner rather than
  `Identity` only.
- Evidence: the only `Preconditioner<B>` implementations are
  `crates/athena-core/src/preconditioner/identity.rs:7` (blanket over any
  backend) and three bound to `LetoBackend<T>` —
  `crates/athena-leto/src/preconditioner/jacobi.rs:45`,
  `incomplete_lu.rs:102`, `successive_over_relaxation.rs:105`. The
  `athena-hephaestus` crate exports only `HephaestusBackend` and `CsrOperator`
  (`crates/athena-hephaestus/src/lib.rs:14-15`). README describes the shipped
  solvers as "preconditioned conjugate gradient"; on the accelerator that
  reduces to unpreconditioned CG.
- Scope: at minimum a device Jacobi preconditioner (diagonal extraction plus an
  element-wise divide kernel binding) in `athena-hephaestus`, its conformance
  case, and a README statement of which preconditioners each backend has.
  Non-goals: device ILU or SOR, whose triangular sweeps are sequential and need
  their own design; AMG.
- Acceptance oracle: an accelerator CG solve of the manufactured SPD system
  with the device Jacobi preconditioner converges, its iteration count is not
  greater than the `Identity` run's on an ill-conditioned manufactured system,
  and the preconditioned residual matches the CPU Jacobi result within the
  AT-002 derived bound.
- Dependencies: AT-002 (the parity oracle), and a Hephaestus element-wise
  divide or reciprocal-scale kernel — confirm before claiming, and if absent it
  is upstream work in Hephaestus, not a downstream approximation.
- Risk/change class: [arch] [minor]. Effort L.
- Re-open trigger: a consumer runs a stiff system on the accelerator lane.

## AT-005 — Verify convergence behaviour, not only solution recovery [patch] — todo

- Outcome: the suite pins the methods' convergence properties, not only that
  each solver reproduces a hand-picked small solution.
- Evidence: every CPU system in `crates/athena-leto/tests/` is two or three
  variables (for example `cg_contract.rs` `recovers_manufactured_solution_f32`
  / `_f64`); no test asserts an iteration-count bound, a residual reduction
  rate, or exactness at dimension `n`. `grep -rn 'proptest\|quickcheck' crates`
  returns 0 hits, so no algebraic law is exercised as a property.
- Scope: convergence-property tests over a parameterized SPD family (for
  example a 1-D Poisson stencil at several sizes) plus property tests of the
  laws the solvers claim. Non-goals: benchmarks (AT-011); a PDE discretization
  layer, which is outside Athena's ownership boundary.
- Acceptance oracle: (a) CG on an `n x n` SPD system terminates within `n`
  iterations in exact-arithmetic terms and its observed residual reduction is
  within the `(sqrt(k)-1)/(sqrt(k)+1)` bound for the system's known condition
  number; (b) GMRES(n) on an `n`-dimensional system converges in at most `n`
  iterations; (c) LSQR's normal-equation residual on an inconsistent system
  matches the dense normal-equation solution within a derived bound; (d)
  properties: right preconditioning preserves the solution, a warm start from
  the exact solution terminates at `InitialResidual`, and the observed residual
  sequence is non-increasing for CG and GMRES. Every tolerance carries its
  derivation at the assertion site.
- Dependencies: none.
- Risk/change class: [verification] [patch]. Effort M.
- Re-open trigger: a consumer reports a convergence-rate regression the suite
  did not catch.

## AT-006 — Reconcile the CI toolchain with the committed pin [patch] — todo

- Outcome: the verification gate runs on the compiler the repository pins.
- Evidence: `rust-toolchain.toml:3` pins `channel = "1.97.0"`, while
  `.github/workflows/ci.yml` installs `dtolnay/rust-toolchain@1.95.0` for both
  the `verify` and `supply-chain` jobs. `.github/workflows/rust-release.yml`
  and `.github/workflows/book-pages.yml` both pass `rust-toolchain: "1.97.0"`,
  so `ci.yml` is the outlier. `Cargo.toml` separately declares
  `rust-version = "1.95"` as the MSRV; there is no job that builds at that
  floor, so the MSRV claim is unverified either way.
- Scope: `.github/workflows/ci.yml`; optionally an explicit MSRV job.
  Non-goals: changing the pin or the MSRV value.
- Acceptance oracle: the CI toolchain matches `rust-toolchain.toml`, and either
  a job builds the workspace at the declared `rust-version` floor or the floor
  is corrected to what is actually verified.
- Dependencies: none.
- Risk/change class: [verification] [patch]. Effort S.
- Re-open trigger: any future pin advance that does not move every workflow.

## AT-007 — Bring `ci.yml` to the workflow-hygiene floor [patch] — todo

- Outcome: the verification workflow matches the standard the release and book
  workflows already meet.
- Evidence: `.github/workflows/ci.yml` pins `actions/checkout@v4`,
  `dtolnay/rust-toolchain@1.95.0`, `taiki-e/install-action@nextest`, and
  `EmbarkStudios/cargo-deny-action@v2` by mutable tag rather than commit SHA;
  neither job declares `timeout-minutes`; the file declares no `concurrency`
  group; no cargo invocation passes `--locked` despite a committed
  `Cargo.lock`; and no dependency or toolchain cache is restored, so every run
  cold-builds the full Atlas provider graph.
  `.github/workflows/rust-release.yml:36` and
  `.github/workflows/book-pages.yml:16` both pin the reusable workflow by SHA
  and both declare `concurrency`, which is the target shape.
- Scope: `.github/workflows/ci.yml`. Non-goals: changing which gates run.
- Acceptance oracle: every third-party action is SHA-pinned; both jobs carry
  `timeout-minutes` derived from measured runtime; a `concurrency` group with
  `cancel-in-progress: true` covers verification; every cargo invocation uses
  `--locked`; a lockfile-keyed cache is restored. `actionlint` passes.
- Dependencies: AT-006 (the same file changes).
- Risk/change class: [security] [patch]. Effort S.
- Re-open trigger: a new workflow lands without these properties.

## AT-008 — Ship per-crate READMEs and complete registry metadata [patch] — todo

- Outcome: each published crate has the landing page and metadata a registry
  listing needs.
- Evidence: `ls crates/*/README.md` finds nothing; all four members inherit
  `publish = true` from `Cargo.toml`'s `[workspace.package]`. Each member
  declares only `description`; none declares `readme`, `keywords`,
  `categories`, or `documentation`
  (`crates/athena/Cargo.toml:1-18`, `crates/athena-core/Cargo.toml:1-10`).
- Scope: four `README.md` files and four `[package]` metadata blocks, with
  shared fields hoisted into `[workspace.package]` where they are identical.
  Non-goals: publishing.
- Acceptance oracle: `cargo package` for each member includes a README;
  the crate-level `//!` docs and the README are single-sourced (generated or
  `#![doc = include_str!]`) so they cannot drift; `athena-krylov`'s README
  states the `package = "athena-krylov"` / `use athena::` name split that
  `crates/athena/Cargo.toml:2-8` documents only in a manifest comment.
- Dependencies: none.
- Risk/change class: [docs] [patch]. Effort S.
- Re-open trigger: a fifth crate is added.

## AT-009 — Gate the public surface with `cargo-semver-checks` [patch] — todo

- Outcome: a breaking change to a published surface fails the gate instead of
  being classified by hand.
- Evidence: `.github/workflows/ci.yml` runs fmt, check, clippy, nextest,
  doctests, doc, examples, and `cargo-deny`, with no semver gate, while four
  crates are `publish = true` and the current `CHANGELOG.md` "Unreleased"
  section already records one **Breaking** `KrylovBackend` change (ADR 0003).
- Scope: one CI step plus, if needed, a baseline configuration. Non-goals:
  cutting a release.
- Acceptance oracle: the gate runs on any change touching a `pub` item and
  fails on a manufactured breaking change; its classification overrides the
  hand-assigned change class in the commit subject when they disagree.
- Dependencies: the crates must be resolvable as a baseline; with unpublished
  0.1.1 members the baseline may need to be the previous tag rather than the
  registry — establish which before claiming.
- Risk/change class: [verification] [patch]. Effort S.
- Re-open trigger: any hand-classified `[major]` item.

## AT-010 — Restore or retire the ADR index generator [patch] — todo

- Outcome: the ADR index's stated provenance is true.
- Evidence: `docs/adr/README.md:3-5` carries the banner "Generated by
  `scripts/adr-index.py` — do not hand-edit", with regenerate and check
  commands; the repository has no `scripts/` directory, and no workflow runs
  the check. The index is therefore hand-maintained under a do-not-hand-edit
  banner, and a new ADR can land unindexed with nothing failing.
- Scope: either commit the generator and a CI `check` step, or replace the
  banner with an accurate statement and add the index freshness check by other
  means. Non-goals: changing ADR content.
- Acceptance oracle: adding an ADR file without updating the index fails a
  gate; the banner names a mechanism that exists in this repository.
- Dependencies: none.
- Risk/change class: [pm-hygiene] [patch]. Effort S.
- Re-open trigger: a fifth ADR.

## AT-011 — Establish a solver-level performance baseline [patch] — todo

- Outcome: a Krylov performance regression has an instrument in this
  repository.
- Evidence: no `benches/` directory exists in any member. The only performance
  evidence in `README.md` ("Prepared-reduction performance") is produced by
  `cargo bench -p hephaestus-wgpu --bench prepared_map_reduction`, a
  Hephaestus-owned instrument measuring provider dispatch latency; README
  states it "does not establish whole-solver speedup". The allocation-stability
  tests pin allocation counts but not time.
- Scope: a Criterion instrument over CG and GMRES on a parameterized SPD family
  with a committed baseline and a per-binary wall-clock budget; a smoke run in
  CI at single-iteration mode. Non-goals: accelerator benchmarking, whose
  quiet-host requirements make it a separate item.
- Acceptance oracle: a stored baseline exists; the smoke run completes inside
  the `.config/nextest.toml` 30-second slow bound; a manufactured algorithmic
  regression shows as a statistically significant change.
- Dependencies: AT-005 (the parameterized system family is shared).
- Risk/change class: [perf] [patch]. Effort M.
- Re-open trigger: any change claiming a solver speedup.

## AT-012 — Complete the book's example coverage and preconditioner wording [patch] — todo

- Outcome: the book's example chapters cover the solvers it teaches, and the
  README's preconditioner list distinguishes public preconditioners from
  internal sweeps.
- Evidence: `docs/book/SUMMARY.md` maps example pages only under Convergence
  Policy and CG; the BiCGSTAB, GMRES, LSQR, and Krylov Backend chapters have
  none, and `crates/athena-leto/examples/` contains exactly the two included
  sources. Separately, `README.md` lists "identity, Jacobi, incomplete-LU,
  triangular, and SOR preconditioning" among the tested preconditioners, but
  `crates/athena-leto/src/preconditioner/mod.rs` exports only `IncompleteLu`,
  `Jacobi`, and `SuccessiveOverRelaxation`; `triangular` is a `pub(super)`
  sweep helper (`triangular.rs:13`) exercised through ILU and SOR, not a
  `Preconditioner` implementation.
- Scope: example sources plus SUMMARY entries for the uncovered chapters, and a
  one-line README correction. Non-goals: rewriting chapter theory.
- Acceptance oracle: each solver chapter includes a runnable example compiled
  by `mdbook test` through the book-pages workflow; the README sentence names
  only the preconditioners a consumer can construct.
- Dependencies: the peer-owned `mdbook test` wiring currently uncommitted in
  `.github/workflows/book-pages.yml` and `docs/book/book.toml` must land first.
- Risk/change class: [docs] [patch]. Effort S.
- Re-open trigger: a new solver family.

## AT-013 — Bring `bicgstab/algorithm.rs` under the 500-line target [patch] — todo

- Outcome: no source file exceeds the repository's 500-line target.
- Evidence: `crates/athena-core/src/solver/bicgstab/algorithm.rs` is 575 lines;
  the next largest source is `lsqr/algorithm.rs` at 487. `README.md` records
  the file as debt and as the only exception, so the claim is honest but the
  debt is open.
- Scope: split the two-term recurrence's step families into leaf modules
  (the `rho`/`omega` breakdown tests and the half-step update are the natural
  seams). Non-goals: changing the recurrence or its terminal values.
- Acceptance oracle: every file is at or under 500 lines with no incoherent
  fragment; the BiCGSTAB contract suite is unchanged and still passes; the
  README exception sentence is deleted in the same change.
- Dependencies: none.
- Risk/change class: [arch] [patch]. Effort S.
- Re-open trigger: any file crossing the target.
