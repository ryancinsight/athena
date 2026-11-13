# Athena Backlog

Milestone ladder; scope in [ADR 0005](docs/adr/0005-release-scope.md). Planned versions are successive minors above 0.3.3, the manifest version and the version of every crate tag, then 1.0.0. Each exit names the test or example that proves it, and each fails at the basis commit because its named identifier does not exist yet. Release and publication need explicit authority.

<a id="M-preconditioned-solves"></a>
## M-preconditioned-solves — 0.4.0 — solve Poisson and saddle-point systems with multigrid and block preconditioning
- needs: none
- exit: run a multigrid-preconditioned CG solve that checks its own error: `cargo run -p athena-krylov --example multigrid_poisson`
- exit: reach the manufactured solution with fewer iterations than Jacobi: `cargo nextest run -p athena-leto --no-tests=fail -E 'binary_id(=athena-leto::multigrid_contract) & test(=amg_preconditioned_cg_needs_fewer_iterations_than_jacobi)'`
- exit: solve a saddle-point system by the chain's direct tier: `cargo nextest run -p athena-leto --no-tests=fail -E 'binary_id(=athena-leto::solver_chain_contract) & test(=direct_tier_solves_a_saddle_point_system_to_the_manufactured_solution)'`
- exit: solve the same system by the chain's iterative tiers: `cargo nextest run -p athena-leto --no-tests=fail -E 'binary_id(=athena-leto::solver_chain_contract) & test(=iterative_tiers_solve_the_same_system_when_the_direct_threshold_is_zero)'`
- standing: no open `correctness` item; `member_lockfile_copies` in `python -I scripts/atlas-conformance.py report --repo athena --json` stays at or below 1 (run from the Atlas root with `--member-path` at the athena revision; read 1 at the basis export); the conformance scan's installed-state assertions pass.
- non-goals: multigrid on device; coarsening over unstructured meshes beyond the shipped strategies.

<a id="M-accelerator-solver-parity"></a>
## M-accelerator-solver-parity — 0.5.0 — solve indefinite and least-squares systems on a Hephaestus device
- needs: M-preconditioned-solves
- exit: run a symmetric indefinite solve on a device that checks its own result: `cargo run -p athena-krylov --example symmetric_indefinite_accelerator --features accelerator`
- exit: solve a symmetric indefinite system with MINRES on a device: `cargo nextest run -p athena-hephaestus --no-tests=fail -E 'binary_id(=athena-hephaestus::backend_contract) & test(=minres_solves_a_symmetric_indefinite_system_on_device)'`
- exit: solve an overdetermined system with LSQR on a device: `cargo nextest run -p athena-hephaestus --no-tests=fail -E 'binary_id(=athena-hephaestus::least_squares_contract) & test(=lsqr_solves_an_overdetermined_system_on_device)'`
- standing: no open `correctness` item; `member_lockfile_copies` stays at or below 1 (command above).
- non-goals: device incomplete-LU, multigrid and saddle-point preconditioners; a device stencil operator; device scalars other than `f32`.

<a id="M-stable-solver-suite"></a>
## M-stable-solver-suite — 1.0.0 — use every shipped solver from a tagged release on a stable API
- needs: M-accelerator-solver-parity
- exit: run every shipped solver on one manufactured system and check each result against the closed form: `cargo run -p athena-krylov --example solver_suite_tour`
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo nextest run --workspace --locked --profile ci --all-features`, `cargo test --workspace --locked --doc --all-features` and `cargo doc --workspace --locked --no-deps --all-features` pass; the SemVer gate and publish dry run of engineering_gates pass for each declared release; README, example (`cargo build --workspace --locked --examples --all-features`) and book (`mdbook test docs/book`) builds pass; no open `correctness` item; `member_lockfile_copies` stays at or below 1 (command above).
- non-goals: nonlinear solver policy; QMR, TFQMR, IDR, LSMR and recycling; eigenvalue machinery; complex scalars; publication without release authority.

<a id="athena-cg-terminal-residual-20261103"></a>
## ATHENA-CG-TERMINAL-RESIDUAL-20261103 — Verify CG convergence against the original residual [patch] — todo

- priority: correctness; needs: none; milestone: M-preconditioned-solves
- scope: `crates/athena-core/src/solver/cg/`, `crates/athena-leto/tests/`, `crates/athena-hephaestus/tests/solver_correctness.rs`
- outcome: CG reports `Termination::Converged` only when the recomputed `b - A x` meets the policy, and exact convergence between scheduled samples ends the solve; GMRES and BiCGSTAB already recompute (README numerical contract).
- acceptance: a CPU regression where the recurrence residual passes and the original residual does not reports a non-converged termination; an exactly solved identity system reports success, not breakdown; warm solves allocate nothing; the same case passes on a device.
- state: released claim. The work sits at `refs/pull/141/head` (PR 141, closed `released`). On this board's basis, `check_residual` in `solver/cg/algorithm.rs` accepts `Converged` from the recurrence residual: a hypothesis until the claim's regression reproduces on unpatched main.
- next step: `git fetch origin pull/141/head:fix/cg-terminal-residual`, rebase onto main, run its regression against unpatched main, then land.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-amg-contract-20261113"></a>
## ATHENA-AMG-CONTRACT-20261113 — Pin the multigrid preconditioner to a solve-level oracle [test] — todo

- priority: verification; needs: none; milestone: M-preconditioned-solves
- scope: `crates/athena-leto/tests/multigrid_contract.rs` (new), `crates/athena/examples/multigrid_poisson.rs` (new), `crates/athena/Cargo.toml`, `docs/book/`
- outcome: `AlgebraicMultigrid` has two unit tests (`zero_rhs_yields_zero`, `diagonal_system_solve_is_exact`) and no solve-level test; add the contract, the facade example and a book chapter for the preconditioners and multigrid.
- exit: `amg_preconditioned_cg_needs_fewer_iterations_than_jacobi` and the `multigrid_poisson` example of M-preconditioned-solves.
- acceptance: the test, generic over `f32` and `f64`, solves a manufactured 2-D Poisson system with Jacobi and with multigrid and asserts the solution against the closed form within a derived bound and strictly fewer multigrid iterations; the example asserts the same and exits nonzero on a miss; `mdbook test docs/book` passes.
- next step: derive the bound from the conditioning of the Poisson matrix, then write the test first.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-solver-chain-contract-20261113"></a>
## ATHENA-SOLVER-CHAIN-CONTRACT-20261113 — Test the tiered solver chain [test] — todo

- priority: verification; needs: none; milestone: M-preconditioned-solves
- scope: `crates/athena-leto/tests/solver_chain_contract.rs` (new), `crates/athena-leto/src/solver_chain/` only if a test exposes a defect
- outcome: `LinearSolverChain` and `LinearSolverState` are public and carry no test (`solver_chain/` holds zero `#[test]` items); pin the direct tier, the iterative tiers and the warm path.
- exit: `direct_tier_solves_a_saddle_point_system_to_the_manufactured_solution` and `iterative_tiers_solve_the_same_system_when_the_direct_threshold_is_zero` of M-preconditioned-solves.
- acceptance: a manufactured mixed saddle-point system solves to its closed form at `f32` and `f64` with the default threshold and with `with_direct_threshold(0)`; `solve_with_guess_state` from the exact solution returns it and a changed sparsity pattern rebuilds the cached hierarchy; the chain's error variant is asserted when every tier fails.
- open question: the chain reports which tier solved only through the log; whether a typed result should carry it is a 1.0 API decision, recorded in this item's PR.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-doc-drift-20261113"></a>
## ATHENA-DOC-DRIFT-20261113 — Reconcile README, book, changelog and gap audit with the shipped crates [docs] — todo

- priority: verification; needs: none; milestone: M-preconditioned-solves
- scope: `README.md`, `docs/book/`, `CHANGELOG.md`, `gap_audit.md`, `docs/verification/`
- outcome: README claims four contracts, a 575-line `bicgstab/algorithm.rs` (now 105 lines; `oversized_files` is 0) and omits `Minres`, multigrid, saddle-point preconditioners and the chain; the book has four solver chapters; `CHANGELOG.md` heads `0.3.2 - Unreleased` although the 0.3.3 tags exist; `gap_audit.md` holds two fixed defects; `docs/verification/` holds two ledgers for landed work.
- acceptance: each claim matches `lib.rs` and the tests; fixed-defect sections leave `gap_audit.md`; the ledgers relocate to the commit history; `mdbook test docs/book` passes.
- next step: rewrite the README solver table from the `athena-core` export list, then the book chapters per M-preconditioned-solves and M-accelerator-solver-parity as each capability lands.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-device-minres-20261113"></a>
## ATHENA-DEVICE-MINRES-20261113 — Run MINRES on a Hephaestus device [feature] — todo

- priority: feature; needs: none; milestone: M-accelerator-solver-parity
- scope: `crates/athena-hephaestus/tests/backend_contract.rs`, `crates/athena/examples/symmetric_indefinite_accelerator.rs` (new), `crates/athena/Cargo.toml`, `docs/book/`
- outcome: `Minres` has CPU contracts only; the device suites run `Cg`, `Gmres`, `BiCgStab`, `DeflatedCg` and `Chebyshev`.
- exit: `minres_solves_a_symmetric_indefinite_system_on_device` and the `symmetric_indefinite_accelerator` example of M-accelerator-solver-parity.
- acceptance: the `diag(1, -1)` oracle of the CPU suite solves on a device at `f32` within a bound derived from the reduction order; the example needs an adapter and has no CPU fallback.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-device-lsqr-20261113"></a>
## ATHENA-DEVICE-LSQR-20261113 — Run LSQR on a Hephaestus device [feature] [arch] — todo

- priority: feature; needs: none; milestone: M-accelerator-solver-parity
- scope: `crates/athena-hephaestus/src/operator.rs`, `crates/athena-hephaestus/tests/least_squares_contract.rs` (new)
- outcome: `RectangularOperator` is implemented for Leto only, so `Lsqr` ships on CPU only (README).
- exit: `lsqr_solves_an_overdetermined_system_on_device` of M-accelerator-solver-parity.
- open question: Hephaestus `SparseOperatorOps` has `upload_csr` and `apply` but no transposed apply. `Aᵀ` is either a second uploaded CSR (twice the device storage, no upstream change) or an upstream Hephaestus kernel. Spike first: verify `upload_csr` accepts a non-square matrix, price the second copy, record the choice in its own ADR.
- acceptance: an overdetermined and an inconsistent system solve on a device to the closed-form least-squares solution within a derived `f32` bound, with `Termination::NormalEquations` on the inconsistent one.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-fused-update-20261020"></a>
## ATHENA-FUSED-UPDATE-20261020 — Next CG lever: fused vector update [perf] — todo

- priority: tightening; needs: none; milestone: M-preconditioned-solves
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
