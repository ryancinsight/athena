# Athena Backlog

Milestone ladder; scope in [ADR 0005](docs/adr/0005-release-scope.md). The one milestone is the final block, so it plans 1.0.0, a major above the 0.3.3 manifest version and every crate tag. Each exit fails at the basis commit because a production symbol it needs is absent (`RectangularOperator` has no Hephaestus impl), not because its test file is missing. Release and publication need explicit authority.

<a id="M-device-least-squares"></a>
## M-device-least-squares — 1.0.0 — solve overdetermined least-squares systems on a Hephaestus device
- needs: none
- exit: run an example that solves an overdetermined system with `Lsqr` through the device `RectangularOperator`, asserts the closed-form least-squares solution and fails without an adapter: `cargo run -p athena-krylov --example least_squares_accelerator --features accelerator`
- exit: solve an overdetermined and an inconsistent system with LSQR on a device; the test lives in `solver_correctness.rs`, whose `backend()` panics without an adapter: `cargo nextest run -p athena-hephaestus --no-tests=fail -E 'package(=athena-hephaestus) & test(=lsqr_solves_an_overdetermined_system_on_device)'`
- standing: no open `correctness` or `verification` item; `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo nextest run --workspace --locked --profile ci --all-features`, `cargo test --workspace --locked --doc --all-features` and `cargo doc --workspace --locked --no-deps --all-features` pass; the SemVer gate and publish dry run of engineering_gates pass for each declared release; README, example (`cargo build --workspace --locked --examples --all-features`) and book (`mdbook test docs/book`) builds pass.
- standing: `member_lockfile_copies` in `python -I scripts/atlas-conformance.py report --repo athena --json` stays at or below 1 (run from the Atlas root with `--member-path` at the athena revision; read 1 at the basis export); installed state: `target_forks`, `excess_worktrees`, `worktrees_outside_lane_root`, `detached_lanes`, `root_sprawl_untracked` and `second_output_root` read 0 in the same report, and `git config --get core.hooksPath` names a directory holding an executable `pre-push` (the scan has no detector for hook or include paths and athena tracks no `.gitconfig`; `ATHENA-GITCONFIG-20261113` files it and adds the `git config --get include.path` assertion here).
- non-goals: device incomplete-LU, SOR, multigrid and saddle-point preconditioners; a device stencil operator; device scalars other than `f32`; nonlinear solver policy; QMR, TFQMR, IDR, LSMR and recycling; eigenvalue machinery; complex scalars; coarsening over unstructured meshes beyond the shipped strategies; publication without release authority.

<a id="athena-cg-terminal-residual-20261103"></a>
## ATHENA-CG-TERMINAL-RESIDUAL-20261103 — Verify CG convergence against the original residual [patch] — todo

- priority: correctness; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/cg/`, `crates/athena-leto/tests/`, `crates/athena-hephaestus/tests/solver_correctness.rs`
- outcome: CG reports `Termination::Converged` only when the recomputed `b - A x` meets the policy, and exact convergence between scheduled samples ends the solve; GMRES and BiCGSTAB already recompute (README numerical contract).
- acceptance: a CPU regression where the recurrence residual passes and the original residual does not reports a non-converged termination; an exactly solved identity system reports success, not breakdown; warm solves allocate nothing; the same case passes on a device.
- state: released claim. The work sits at `refs/pull/141/head` (PR 141, closed `released`). On this board's basis, `check_residual` in `solver/cg/algorithm.rs` accepts `Converged` from the recurrence residual: a hypothesis until the claim's regression reproduces on unpatched main.
- next step: `git fetch origin pull/141/head:fix/cg-terminal-residual`, rebase onto main, run its regression against unpatched main, then land.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-preconditioner-contract-20261113"></a>
## ATHENA-PRECONDITIONER-CONTRACT-20261113 — Pin multigrid and saddle-point preconditioners to solve-level oracles [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-leto/tests/preconditioner_contract.rs` (existing harness, no new test binary), `docs/book/`
- outcome: `AlgebraicMultigrid` has two unit tests (`zero_rhs_yields_zero`, `diagonal_system_solve_is_exact`), the three saddle-point preconditioners have seven, and none runs inside a Krylov solve; add the contracts and a book chapter for the preconditioners and multigrid. `hmis_coarsening` and `create_standard_interpolation` are named by no test and are reached only through `CoarseningStrategy::HMIS` and `InterpolationStrategy::Standard`; the default configuration selects RugeStueben and Classical. Production code changes only if a test exposes a defect.
- acceptance: generic over `f32` and `f64`, a manufactured 2-D Poisson system solves with Jacobi and with multigrid-preconditioned CG to the closed form within a derived bound, with strictly fewer multigrid iterations; `BlockDiagonalPreconditioner`, `ComponentBlockPreconditioner` and `SimplePreconditioner` each solve a manufactured saddle-point system under GMRES to its closed form; `AlgebraicMultigrid` configured with `CoarseningStrategy::HMIS` and with `InterpolationStrategy::Standard` solves the same Poisson system to the closed form, `hmis_coarsening` maps every listed coarse point to its own position in `coarse_points` and every `Some` in `fine_to_coarse_map` is below `coarse_points.len()` (the invariant `test_mapping_correctness` asserts for the Ruge-Stueben, aggregation, Falgout and PMIS strategies; F-points hold `Some` by design), and in the standard interpolation operator of that system each C-row holds exactly one entry, of value 1, at that point's position in `coarse_points`, read directly from its CSR; `mdbook test docs/book` passes.
- next step: derive the Poisson bound from the conditioning of the five-point Laplacian, then write the multigrid test first.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-solver-chain-contract-20261113"></a>
## ATHENA-SOLVER-CHAIN-CONTRACT-20261113 — Test the tiered solver chain [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-leto/src/solver_chain/` (a `#[cfg(test)]` module; no new test binary), `crates/athena-leto/tests/preconditioner_contract.rs`
- outcome: `LinearSolverChain` and `LinearSolverState` are public and carry no test (`solver_chain/` holds zero `#[test]` items); `cold.rs` tries the direct tier, then GMRES with `ComponentBlock`, SIMPLE, AMG and `BlockDiagonal`, then unpreconditioned GMRES and GMRES+ILU. The preconditioner item covers the block tiers' solves; `IncompleteLu` runs in a Krylov solve only under BiCGSTAB (`preconditioner_contract.rs:192`), so this item adds its GMRES solve and pins the routing, the warm path and the failure variant.
- acceptance: a manufactured mixed saddle-point system solves to its closed form at `f32` and `f64` with the default threshold and with `with_direct_threshold(0)`; `solve_with_guess_state` from the exact solution returns it and a changed sparsity pattern rebuilds the cached hierarchy; the chain's error variant is asserted when every tier fails; `IncompleteLu` under GMRES solves a manufactured system to its closed form.
- open question: the chain reports which tier solved only through the log; whether a typed result should carry it is a 1.0 API decision, recorded in this item's PR.
- next step: build the manufactured saddle-point matrix once as a shared fixture, assert the default-threshold solve against its closed form, then the `with_direct_threshold(0)` solve.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-doc-drift-20261113"></a>
## ATHENA-DOC-DRIFT-20261113 — Reconcile README, book, changelog and gap audit with the shipped crates [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `README.md`, `docs/book/`, `CHANGELOG.md`, `gap_audit.md`, `docs/verification/`, `crates/athena-hephaestus/src/preconditioner/mod.rs`
- outcome: README claims four contracts, a 575-line `bicgstab/algorithm.rs` (now 105 lines; `oversized_files` is 0) and omits `Minres`, multigrid, saddle-point preconditioners and the chain; README (lines 115, 285), `docs/book/krylov_backend.md:142` and the hephaestus module docs list `triangular` as a preconditioner though it is a private kernel; README:316 tells contributors `git config core.hooksPath .githooks` while the installed path is the stack hook directory; the book has four solver chapters; `CHANGELOG.md` heads `0.3.2 - Unreleased` although the 0.3.3 tags exist; `gap_audit.md` holds two fixed defects; `docs/verification/` holds two ledgers for landed work.
- acceptance: each claim matches `lib.rs` and the tests; fixed-defect sections leave `gap_audit.md` except the rank-recovery performance-hold lines ("does not clear Athena's affected graph" paragraph and the Harmonia clause), which `ATHENA-GMRES-PERFORMANCE-HOLD-20261113` removes in its own PR; the ledgers relocate to the commit history; `mdbook test docs/book` passes.
- next step: rewrite the README solver table from the `athena-core` export list, then the book chapters as each contract and the device least-squares capability land.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-device-minres-20261113"></a>
## ATHENA-DEVICE-MINRES-20261113 — Pin MINRES on a Hephaestus device to its oracle [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-hephaestus/tests/solver_correctness.rs`
- outcome: `Minres<B: KrylovBackend>` is backend-generic (`athena-core/src/solver/minres/algorithm.rs:46`); the device suites run `Cg`, `Gmres`, `BiCgStab`, `DeflatedCg` and `Chebyshev` and none runs `Minres`. The recurrence is expected to run unchanged; this item verifies that and builds nothing.
- acceptance: the `diag(1, -1)` oracle of `crates/athena-leto/tests/minres_contract.rs` solves on a device at `f32` within a bound derived from the reduction order; the test sits in `solver_correctness.rs`, whose `backend()` panics without an adapter, so an unavailable adapter fails it.
- next step: derive the `f32` bound from the dot-product reduction depth of a two-element system, then write the test first beside the Chebyshev device regression.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-gitconfig-20261113"></a>
## ATHENA-GITCONFIG-20261113 — Commit the conflict-hygiene repository config [chore] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `.gitconfig` (new), the commit template it names (new), `README.md` hook-install section, `backlog.md` standing line
- outcome: athena tracks no `.gitconfig`, so `git config --get include.path` exits 1 and the commit template, rerere, zdiff3, blame ignore-revs and push settings the stack names are per-clone. `core.hooksPath` points at the stack hook directory on the checked host. No Atlas detector reads `include.path` or `core.hooksPath` and no Atlas board item adds one.
- acceptance: a committed `.gitconfig` carries `commit.template` with an `Item:` skeleton, `core.hooksPath`, `rerere.enabled`, `rerere.autoupdate`, `merge.conflictStyle=zdiff3`, `rebase.autostash=false`, `push.useForceIfIncludes`, `fetch.writeCommitGraph` and `blame.ignoreRevsFile`; a fresh clone installs it with one documented command and `git config --get include.path` names it; the standing line of M-device-least-squares gains that assertion in the same PR.
- next step: copy the key list from the stack's conflict-hygiene rule into `.gitconfig`, then script the one-command install and document it in README beside the hook step.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-device-lsqr-20261113"></a>
## ATHENA-DEVICE-LSQR-20261113 — Run LSQR on a Hephaestus device [feature] [arch] — todo

- priority: feature; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-hephaestus/src/operator.rs`, `crates/athena-hephaestus/tests/solver_correctness.rs`, `crates/athena/examples/least_squares_accelerator.rs` (new), `crates/athena/Cargo.toml`, `docs/book/`
- outcome: `RectangularOperator` has no Hephaestus impl and `CsrOperator::from_parts` rejects `rows != columns` (`operator.rs:53`, pinned by `a_rectangular_matrix_is_rejected`), so `Lsqr` ships on CPU only (README:26-28).
- exit: `lsqr_solves_an_overdetermined_system_on_device` and the `least_squares_accelerator` example of M-device-least-squares.
- open question: Hephaestus `SparseOperatorOps` has `upload_csr` and `apply` but no transposed apply. `Aᵀ` is either a second uploaded CSR (twice the device storage, no upstream change) or an upstream Hephaestus kernel; the choice is recorded in its own ADR.
- acceptance: an overdetermined and an inconsistent system solve on a device with `Lsqr` to the closed-form least-squares solution within a derived `f32` bound, with `Termination::NormalEquations` on the inconsistent one; the example calls `Lsqr` through the device `RectangularOperator` (never normal equations formed on the host), asserts the closed-form solution, needs an adapter, has no CPU fallback and exits nonzero on a miss; the rectangular-rejection test follows the chosen design.
- next step: spike: read `SparseOperatorOps` at the locked Hephaestus revision, verify `upload_csr` accepts a non-square matrix, price the second copy of `Aᵀ`, then write the ADR and the test first.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-fused-update-20261020"></a>
## ATHENA-FUSED-UPDATE-20261020 — Next CG lever: fused vector update [perf] — todo

- priority: tightening; needs: none; milestone: M-device-least-squares
- basis: 80402fbd0b825bc4eed7b346c0b265ef6d477979
- scope: `crates/athena/benches/kernels.rs`, `crates/athena-leto/src/backend/krylov.rs`
- outcome: profile the fused update (memory traffic against the cache-level ceiling, SIMD coverage), then cut it
  back-to-back against the `kernels` bench; land once it no longer co-leads or a faster form is proven unnecessary.
- acceptance: recorded diagnosis plus a measured win with the same convergence and iterations; tests, strict gates
  and hosted CI green.
- evidence: the 2026-10-22 diagnosis and re-measure are in `git show 80402fb:backlog.md`; stencil cut 16f33eb (#130);
  shares bc16603.
- next step: run `kernels` on a quiescent host, on isolated cores or with instruction counters, and read
  `matvec_stencil`, `fused_cg_update` and `stream_ceiling` from one run. Fused at the ceiling: memory-bound, apply the
  stop clause; well above it: inspect codegen and cut. Then recompute the shares with the stencil matvec and file the next leader.

<a id="athena-gmres-performance-hold-20261113"></a>
## ATHENA-GMRES-PERFORMANCE-HOLD-20261113 — Assess GMRES performance after rank recovery [perf] — todo

- priority: tightening; needs: none; milestone: M-device-least-squares
- scope: `crates/athena/benches/gmres_poisson.rs`, `gap_audit.md`
- outcome: PR 126 (`a71cd7c`) landed the rank recovery (`c9ddce7`) and the provider renewal to Leto/LetoOps 0.46.3 and Hephaestus 0.25. `gap_audit.md` records that native provider acceptance "does not clear Athena's affected graph, whole-cohort or absolute application-performance review" and that "the separate Harmonia performance hold persists"; `ATHENA-DOC-DRIFT-20261113` leaves those lines in place, so this item owns and removes them. The fix leaves full-rank arithmetic unchanged, so a regression is not expected (hypothesis).
- acceptance: a stored `gmres_poisson` comparison of `606b4f9` (the first parent of `a71cd7c`) against `a71cd7c`, which spans the rank recovery and the provider renewal the hold names; main is 41 commits further on (median, confidence interval, machine class, load) on isolated cores or with instruction counters; the Harmonia hold's disposition is read from its owner and recorded in the PR; the hold text leaves `gap_audit.md` with that PR.
- next step: build `606b4f9` and `a71cd7c` into the shared target, run `gmres_poisson` on isolated cores, and find the Harmonia hold's owner on the Atlas board.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876
