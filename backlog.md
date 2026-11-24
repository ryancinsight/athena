# Athena Backlog

<a id="athena-stable-qualification"></a>
## ATHENA-STABLE-QUALIFICATION — athena stable-release qualification [patch] — todo
- priority: verification; milestone: M-device-least-squares; basis: `0ad161c90b07`.
- Outcome: Close the rectangular device-operator gap for LSQR. Follow the [release scope](docs/adr/0005-release-scope.md) and `M-device-least-squares`. Planning targets for `athena-krylov` are 0.5.1, then 0.6.0 proposed for additive device support, then 1.0.0. Verification alone can advance the current version.
- Representative manifest `athena-krylov 0.5.0`. Target 1.0.0.
- [ ] [patch] Pin the solver chain and preconditioner to solution and residual values. Include a system with an inconsistent right-hand side and compare its normal residual with the analytical optimum.
- [ ] [minor] Implement the signed-off rectangular operator route and run the least_squares_accelerator example and device LSQR tests on an actual adapter. Retain evidence of the selected device.
- [ ] [major] Run the M-device-least-squares standing gates and a consumer of the independently versioned Core and adapter packages against the released array and device providers.
- [ ] [patch] **0.5.1.** Use the host LSQR contract to measure both residual and normal-equation residual on a rectangular inconsistent system. Check rejection of mismatched dimensions before device submission. [Source and baseline](gap_audit.md#incremental-release-baseline).
- [ ] [minor] **0.6.0 proposed for additive device support.** Exercise the real rectangular operator and its adjoint on a Hephaestus device. Compare the minimizer with the host solution and record the adapter, transfers and solver termination.
- [ ] [major] **1.0.0-beta.1 → 1.0.0-rc.1 → 1.0.0.** Freeze the supported surface at beta. A packaged Krylov consumer repeats the rectangular solve and the final device least-squares milestone passes. Repeat the applicable exits on the exact RC before stable promotion. Attach exact-revision evidence for the local exits and this member’s [Atlas qualification criteria](https://github.com/ryancinsight/atlas/blob/17b0124a7548ad81e817284afc5cb115c1d1bc78/docs/parity/repositories.json).

Milestone ladder; scope in [ADR 0005](docs/adr/0005-release-scope.md). The one milestone is the final block, so it plans 1.0.0, a major above every manifest version and every crate tag. Each exit fails at the basis commit because a production symbol it needs is absent (`RectangularOperator` has no Hephaestus impl), not because its test file is missing. Release and publication need explicit authority.

<a id="M-device-least-squares"></a>
## M-device-least-squares — 1.0.0 — solve overdetermined least-squares systems on a Hephaestus device
- needs: none
- exit: run an example that solves an overdetermined system with `Lsqr` through the device `RectangularOperator`, asserts the closed-form least-squares solution and fails without an adapter: `cargo run -p athena-krylov --example least_squares_accelerator --features accelerator`
- exit: solve an overdetermined and an inconsistent system with LSQR on a device; the test lives in `solver_correctness.rs`, whose `backend()` panics without an adapter: `cargo nextest run -p athena-hephaestus --no-tests=fail -E 'package(=athena-hephaestus) & test(=lsqr_solves_an_overdetermined_system_on_device)'`
- standing: no open `correctness` or `verification` item; `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo nextest run --workspace --locked --profile ci --all-features`, `cargo test --workspace --locked --doc --all-features` and `cargo doc --workspace --locked --no-deps --all-features` pass; the SemVer gate and publish dry run of engineering_gates pass for each declared release; README, example (`cargo build --workspace --locked --examples --all-features`) and book (`mdbook test docs/book`) builds pass.
- standing: `member_lockfile_copies` in `python -I scripts/atlas-conformance.py report --repo athena --json` stays at or below 1 (run from the Atlas root with `--member-path` at the athena revision; read 1 at the basis export); installed state: `target_forks`, `excess_worktrees`, `worktrees_outside_lane_root`, `detached_lanes`, `root_sprawl_untracked` and `second_output_root` read 0 in the same report, and `git config --get core.hooksPath` names a directory holding an executable `pre-push` (the scan has no detector for hook or include paths and athena tracks no `.gitconfig`; `ATHENA-GITCONFIG-20261113` files it and adds the `git config --get include.path` assertion here).
- non-goals: device incomplete-LU, SOR, multigrid and saddle-point preconditioners; a device stencil operator; device scalars other than `f32`; nonlinear solver policy; QMR, TFQMR, IDR, LSMR and recycling; eigenvalue machinery; complex scalars; coarsening over unstructured meshes beyond the shipped strategies; publication without release authority.

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
- outcome: `LinearSolverChain` and `LinearSolverState` are public and carry no test (`solver_chain/` holds zero `#[test]` items); `cold.rs` tries the direct tier, then GMRES with `ComponentBlock`, SIMPLE, AMG and `BlockDiagonal`, then unpreconditioned GMRES, GMRES+ILU and a `BiCgStab` last resort (`cold.rs:143`). The preconditioner item covers the block tiers' solves; `IncompleteLu` runs in a Krylov solve only under BiCGSTAB (`preconditioner_contract.rs:192`), so this item adds its GMRES solve and pins the routing, the warm path and the failure variant.
- acceptance: a manufactured mixed saddle-point system solves to its closed form at `f32` and `f64` with the default threshold and with `with_direct_threshold(0)`; `solve_with_guess_state` from the exact solution returns it and a changed sparsity pattern rebuilds the cached hierarchy; the chain's error variant is asserted when every tier fails; `IncompleteLu` under GMRES solves a manufactured system to its closed form.
- open question: the chain reports which tier solved only through the log; whether a typed result should carry it is a 1.0 API decision, recorded in this item's PR.
- next step: build the manufactured saddle-point matrix once as a shared fixture, assert the default-threshold solve against its closed form, then the `with_direct_threshold(0)` solve.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-doc-drift-20261113"></a>
## ATHENA-DOC-DRIFT-20261113 — Finish the doc-drift remainders: book derivations, citations [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `docs/book/further_solvers.md`, doc comments in `crates/athena-core/src/solver/minres/algorithm.rs` and `chebyshev/algorithm.rs`
- outcome: two remainders after the README, CHANGELOG, gap-audit and module-doc reconciliation and the deletion of the GMRES rank-recovery ledger. (1) `further_solvers.md` names `Minres`, `Chebyshev` and `DeflatedCg` and states their contracts but derives none: it has no recurrences, no compiled code and no references section. (2) `minres/algorithm.rs:40-42` gives the Paige and Saunders (1975) title unverified, and `chebyshev/algorithm.rs:41` cites Golub and Van Loan section 11.3 with no edition.
- acceptance: (1) each of the three sections of `further_solvers.md` states its recurrence and carries a compiled `rust` block, the chapter ends in a References section, and `mdbook test docs/book` passes; (2) both comments cite a resolved title, journal and pages, or an edition and section, or drop the locator.
- next step: derive the MINRES recurrence in `further_solvers.md` with a compiled block, then the Chebyshev and deflated CG sections and the References section; the two source comments are `.rs` edits and take the cargo gates.
- basis: 9741050161a53f882f9ca8e4e8cfbb303579a9e2

<a id="athena-gitconfig-20261113"></a>
## ATHENA-GITCONFIG-20261113 — Commit the conflict-hygiene repository config [chore] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `.gitconfig` (new), the commit template it names (new), `README.md` hook-install section, `backlog.md` standing line
- outcome: athena tracks no `.gitconfig`, so `git config --get include.path` exits 1 and the commit template, rerere, zdiff3, blame ignore-revs and push settings the stack names are per-clone. `core.hooksPath` points at the stack hook directory on the checked host. No Atlas detector reads `include.path` or `core.hooksPath` and no Atlas board item adds one.
- acceptance: a committed `.gitconfig` carries `commit.template` with an `Item:` skeleton, `core.hooksPath`, `rerere.enabled`, `rerere.autoupdate`, `merge.conflictStyle=zdiff3`, `rebase.autostash=false`, `push.useForceIfIncludes`, `fetch.writeCommitGraph` and `blame.ignoreRevsFile`; a fresh clone installs it with one documented command and `git config --get include.path` names it; the standing line of M-device-least-squares gains that assertion in the same PR.
- next step: copy the key list from the stack's conflict-hygiene rule into `.gitconfig`, then script the one-command install and document it in README beside the hook step.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-noise-floor-scale-20261116"></a>
## ATHENA-NOISE-FLOOR-SCALE-20261116 — Test GMRES progress classification when `Ax` cancels [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/gmres/execution.rs` (`progress_failure`), `crates/athena-core/src/convergence/residual_noise.rs`, `crates/athena-leto/tests/gmres_contract.rs`
- outcome: `residual_noise_floor` scales with `‖b‖`, while the evaluation error of `b - Ax` scales with `‖|A||x|‖`, about 1.46e3 times larger for the order-62 stiffness matrix with a uniform load (its Rustdoc). GMRES `Diverged` (`‖r_k‖ > ‖r_0‖ + η`) and `Stagnated` (reduction `≤ η`) compare against that `η`; no test shows whether an understated `η` misclassifies a cycle.
- acceptance: a GMRES contract case on a system whose `‖|A||x|‖ / ‖b‖` exceeds `√n` either asserts the derived classification at a residual near its evaluation floor, or exposes a misclassification that this item then fixes with a computable scale (the public `residual_noise_floor(len, ‖b‖)` signature changes only under a [major] classification).
- next step: start from a warm start `x_0` near the solution of the stiffness system, where `‖r_0‖` is near the evaluation floor, and record `progress_failure`'s inputs per cycle through the observer.
- basis: b92a0fd339586e08fb1a94d4766dbdc15eba200a

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
- outcome: PR 126 (`a71cd7c`) landed the rank recovery (`c9ddce7`) and the provider renewal to Leto/LetoOps 0.46.3 and Hephaestus 0.25. `gap_audit.md` records that native provider acceptance "does not clear Athena's affected graph, whole-cohort or absolute application-performance review" and that "the separate Harmonia performance hold persists"; the doc-drift reconciliation (`Item: athena-doc-drift-20261113`) left those lines in place under `## GMRES performance hold`, so this item owns and removes them. The fix leaves full-rank arithmetic unchanged, so a regression is not expected (hypothesis).
- acceptance: a stored `gmres_poisson` comparison of `606b4f9` (the first parent of `a71cd7c`) against `a71cd7c`, which spans the rank recovery and the provider renewal the hold names; main is 41 commits further on (median, confidence interval, machine class, load) on isolated cores or with instruction counters; the Harmonia hold's disposition is read from its owner and recorded in the PR; the hold text leaves `gap_audit.md` with that PR.
- next step: build `606b4f9` and `a71cd7c` into the shared target, run `gmres_poisson` on isolated cores, and find the Harmonia hold's owner on the Atlas board.
- basis: 0e250fdfa9a8ecff85bc3810993952e9d1f78876

<a id="athena-adr-0004-budget-20261124"></a>
## ATHENA-ADR-0004-BUDGET-20261124 — Compact ADR 0004 to the ADR line budget [docs] — todo

- priority: tightening; needs: none; milestone: M-device-least-squares
- scope: `docs/adr/0004-derived-progress-termination.md`
- outcome: ADR 0004 is 168 lines (`docs/adr/0004-derived-progress-termination.md`, Context at :13, Verification at :149) against the budget of about 150 lines per ADR; the in-place revisions of 2026-11-16 (CG stall detection, noise-floor premise) and the Verification list grew it. Compact it without dropping the decision, the four rejected alternatives, the evidence or the revision notes; delivery narrative belongs to commits. ADR 0005 is within that budget and stays outside this item.
- acceptance: `test "$(wc -l < docs/adr/0004-derived-progress-termination.md)" -le 150` exits 0 (exits 1 at the basis, 168 lines); the Context, Decision, Rejected alternatives, Consequences and Verification headings and the four rejected-alternative headings remain; `python scripts/adr-index.py check --directory docs/adr` from an Atlas checkout passes.
- next step: trim statements Consequences repeats from Decision, then shorten the Verification list to each test's name and oracle; keep every cited source.
- basis: 68d618e87e42d1b4d9f77e56e2e8e05f7653b7f9

<a id="athena-lsqr-rotation-overflow-20261124"></a>
## ATHENA-LSQR-ROTATION-OVERFLOW-20261124 — Scale both LSQR plane rotations so a large damping converges [fix] — todo

- priority: correctness; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/lsqr/execution.rs`, `crates/athena-core/src/solver/rotation.rs` (only if the radius is exposed there), `crates/athena-leto/tests/lsqr_damped_contract.rs`
- outcome: `Lsqr::solve_damped_into` ends `Termination::Breakdown` at iteration 1 for a finite `λ` whose square overflows (`λ` above `√MAX`: about 1.34e154 at `f64`, 1.84e19 at `f32`; derived), although the solution exists. The first rotation forms `ρ̂ = (ρ̄² + λ²).sqrt()` unscaled and returns `Breakdown` when it is not finite (`execution.rs:225-227`); the second forms `ρ = (ρ̂² + β²).sqrt()` unscaled (`:238-240`), so scaling only the first still ends `Breakdown` at `ρ̂ ≈ λ` (derived from reading, not run). The exact radius is finite. `Breakdown` is documented as a zero denominator and a non-finite coefficient as `NonFinite` (`crates/athena-core/src/report/solve.rs:11-12`, `:15-16`), and the crate's `givens` scales by the larger magnitude so that no square overflows (`crates/athena-core/src/solver/rotation.rs:13-46`), the rule ADR 0002 sets for GMRES (`docs/adr/0002-restarted-gmres-contract.md:73-74`). Judge probe in the athena#195 review: `λ = 1e200` at `f64` ends `Breakdown` at iteration 1 (not rerun here). No committed test or text makes that outcome intended (`damping_tolerates_extreme_lambda_values`, `lsqr_damped_contract.rs:224`, solves at `λ = 1`). SciPy 1.18.0 `lsqr.py:446` is unscaled in the damping rotation too and scales the second (`_sym_ortho`, `:458`; read locally). Consumer: kwavers maps `converged()` into its result flag (`crates/kwavers-math/src/linear_algebra/sparse/matfree.rs:230,240` at kwavers `7e9ac5d97f`); the dampings 1e-4 and 100 recorded in the athena#195 review are far below the threshold.
- acceptance: a test `damped_solve_survives_a_damping_whose_square_overflows` in `lsqr_damped_contract.rs`, with a body generic over `f32` and `f64` beside `damped_solution_and_residual_match_closed_form_at` (`:386`), solves that test's line-fit system (rows `(1, 0)`, `(1, 1)`, `(1, 2)`, `b = (1, 0, 2)`, `Aᵀb = (3, 4)`) through `solve_with_damping` (`:72`, which applies `policy` with tolerances `1024·ε`; zero tolerances would end `MaxIterations` at `f32`, where the normal-residual estimate is about 5e-40 and not zero, derived by hand and not run) from `x₀ = 0` at `λ = 1e200` (`f64`) and `λ = 1e20` (`f32`, `λ² = 1e40` above `f32::MAX`), asserts `report.termination.converged()`, and compares `x` with the closed-form minimiser `(AᵀA + λ²I)⁻¹Aᵀb` evaluated without squaring `λ` (`damped_oracle` squares it in `f64`, which overflows at `λ = 1e200`) within a bound derived from an itemized rounding count that includes the `f32` subnormal spacing `2⁻¹⁴⁹`, about 1.4e-45, because the expected entries are subnormal. The minimiser is `(3, 4)/λ²` to leading order: below the `f64` subnormal range, so `x = 0` at `f64`; `(3, 4)·1e-40` at `f32`, subnormal and nonzero, so `x = x₀` is not the `f32` oracle. The test fails at the basis with `Breakdown`, and reverting each of the two rotations separately to its unscaled form also fails it, each run recorded in the PR: `cargo nextest run -p athena-leto --no-tests=fail -E 'test(=damped_solve_survives_a_damping_whose_square_overflows)'` (no test of that name exists at the basis). The same test pins the termination of an infinite `λ` on a nonzero residual to the variant that the delivering PR states (`Breakdown` as at the basis, or `NonFinite`, `crates/athena-core/src/report/solve.rs:15-16`); the choice is the owner's and is open.
- next step: scale both radii by the larger magnitude as `givens` does. `givens` returns the cosine and sine, not the radius, so `ρ̂` and `ρ` need the radius exposed, and it returns `ScalarFailure::NonFinite` for a non-finite input where the basis ends `Breakdown`; ATHENA-LSQR-DAMPED-CONTRACT-20261124 and ATHENA-LSQR-DAMPED-DOC-20261124 then pin and document it for every non-finite path. If the owner instead rejects such `λ`, the decision is a typed result recorded in an ADR and the test asserts it.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156

<a id="athena-minres-nonfinite-converged-20261124"></a>
## ATHENA-MINRES-NONFINITE-CONVERGED-20261124 — Report a non-finite MINRES solution as non-converged [fix] — todo

- priority: correctness; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/minres/execution.rs`, `crates/athena-leto/tests/minres_contract.rs`, `crates/athena-leto/tests/minres_contract/workspace_reuse.rs`
- outcome: `Minres::solve_into` reports `Termination::Converged` while `x` holds infinite entries, because the loop checks only the scalar `φ̄` (`execution.rs:286-299`) and never the update vector or `x` (judge-observed in the athena#194 review, not run here). Input 1, `f64` (and `f32` with exponents `(-120, -60, 80, 20)`, `minres_contract.rs:258-260`, where `2⁸⁰·2⁶⁰ = 2¹⁴⁰` exceeds `2¹²⁸` and `1/a = 2¹²⁰` is representable; by analogy with `f64`, not run): the 3×3 CSR `[[a, b, 0], [b, d, f], [0, f, 1]]` with `(a, b, d, f) = (2⁻¹⁰⁰⁰, 2⁻⁵⁰⁰, 2⁶⁰⁰, 2¹⁰⁰)`, `b = e₁`, `x₀ = 0`, policy `(ε, ε, 3)`: `λ₂·w₁ ≈ 2¹¹⁰⁰` overflows, the solve reports `Converged` after 3 iterations with `x[0] = inf` although the exact `x[0]` is about `2¹⁰⁰⁰ ≈ 1.07e301`, which is representable. Input 2, `f64` and `f32`: `diag(s, -s)` with `s` the first power of `EPSILON` whose reciprocal overflows, `b = (1, 2)`: `Converged` after 1 iteration with `x = [-inf, -inf]` (the test `minres_workspace_reused_after_non_finite_solve_matches_fresh_workspace` of athena#194 asserts one iteration and a non-finite `x`, `workspace_reuse.rs:133-144`, and no termination).
- acceptance: a test `minres_never_reports_a_non_finite_solution_as_converged` in `minres_contract.rs`, generic over `f32` and `f64`, solves both inputs and asserts `report.termination == Termination::NonFinite` (the existing variant for a non-finite recurrence value; its doc is widened to cover vectors) and that `x` is not reported converged; it fails at the basis with `Converged` and passes after the fix: `cargo nextest run -p athena-leto --no-tests=fail -E 'test(=minres_never_reports_a_non_finite_solution_as_converged)'` (no test of that name exists at the basis). The fix also re-derives `minres_workspace_reused_after_overflowing_second_update_matches_fresh_workspace` (`workspace_reuse.rs:176-218`): on input 1 `x[0]` is already infinite after step 2, so a finite-`x` check ends the solve at iteration 2 (judge-derived, not run), its `iterations == 3` precondition (`:203`) fails, and the poisoned `update_previous_2` that its mutant (drop that reset) needs must come from a witness that still reaches step 3.
- next step: after the update, test the solution vector (or the update vector) for finiteness where `φ̄` is tested (`execution.rs:286-289`), then add the test and rebuild the second-update reuse witness so it still poisons `update_previous_2`.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156

<a id="athena-device-minres-band-20261124"></a>
## ATHENA-DEVICE-MINRES-BAND-20261124 — Pin the MINRES band-correction updates on a device [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-hephaestus/tests/solver_correctness.rs`
- outcome: the device MINRES oracle (athena#186, `ATHENA-DEVICE-MINRES-20261113`) solves `diag(1, -1)`, where `λ₂ = c₁β₂ + s₁α₂ = 0`, `ε₁ = ε₂ = 0` and `λ₁` multiplies a zero vector, so deleting either band-correction axpy of `apply_update` (`execution.rs:376-389`) moves `x` by O(ε) and the test survives. athena#186 landed the device MINRES test, `minres_solves_the_indefinite_oracle_on_device`. The host tridiagonal (`minres_contract.rs:148-168`) does not close the gap alone: a Python transcription of the recurrence gives `ε₃ ≈ 0.249` there but `λ₂ = 0` and `λ₃ ≈ 1e-16`, so deleting the λ axpy changes `x` by under 1e-15 (derived; not run on the Rust code).
- acceptance: a 3×3 device instance with `λⱼ ≠ 0` against a nonzero update vector and `ε₃ ≠ 0`, with an integer manufactured solution, solves within a bound derived from its operation count (transcription candidate: `[[1, 1, 0], [1, -2, 1], [0, 1, -2]]`, `b = (3, 0, -4)`, `x* = (1, 2, 3)`; judge-derived from a Python copy of the recurrence, not run on the Rust code: `λ₂ = -1.198` multiplies `w₁` (norm 0.58), `ε₃ = 1.359`; deleting the λ axpy shifts `x` by 1.867 and the ε axpy by 1.478, against 8.9e-16 intact); deleting the λ axpy and, separately, the ε axpy makes the test fail, each deletion run and recorded in the PR.
- next step: add the instance beside `minres_solves_the_indefinite_oracle_on_device` and run the two deletions.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156

<a id="athena-minres-host-bounds-20261124"></a>
## ATHENA-MINRES-HOST-BOUNDS-20261124 — Derive the host MINRES tolerances from an operation count [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-leto/tests/minres_contract.rs`
- outcome: the bounds `256·EPSILON` (`:76-83`) and `512·EPSILON` (`:210-214`) rest on "fewer than 48" and "fewer than 72" rounded operations, a "factor-of-five" and "factor-of-three margin" and an amplification claim, with no tally. A tally from `execution.rs` shows the rotation scalars, `φ` and `φ̄`, and the update recurrence alone (`:231-233`, `:250-251`, `:370-402`) take 20 rounded operations per step on two-element vectors (1 + 4 + 4 + 2 + 9, each axpy element as one rounding, a lower bound), 40 over two steps, before the Lanczos step, the Givens scalars and the seed, so "48" does not count the operations that form `x`. The athena#186 review counted 98 for the 2×2 oracle (63 once operations exact on that instance are dropped); only the 20 per step above are confirmed here. Whether the constants still hold is open (hypothesis). The device bound was re-derived in athena#186 (`ORACLE_BOUND_EPSILONS = 73`, `solver_correctness.rs:183`).
- acceptance: each constant is derived from an itemized, source-cited count of the rounded operations forming `x`, each host operation at relative error at most `ε/2` (`RealField::EPSILON` is `f32::EPSILON` and `f64::EPSILON`, eunomia `crates/eunomia/src/impls/field.rs:19,46`), propagated through the recurrence by running-error analysis, with the derivation in the constant's doc; `cargo nextest run -p athena-leto -E 'test(~minres)'` passes at `f32` and `f64`.
- next step: list the rounded operations of the two-step and three-step runs from `execution.rs`, then bound the error of `x` through the rotation scalars.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156

<a id="athena-minres-reuse-doc-20261124"></a>
## ATHENA-MINRES-REUSE-DOC-20261124 — Correct the MINRES reuse comment and Rustdoc, compare termination in one reuse test [docs] — todo

- priority: verification; needs: ATHENA-MINRES-NONFINITE-CONVERGED-20261124; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/minres/execution.rs` (seed Rustdoc), `crates/athena-leto/tests/minres_contract/workspace_reuse.rs`
- outcome: three small defects from the athena#194 review, each read at the basis. (1) `workspace_reuse.rs:88` says the rotation radius is `abs(s)`; for `diag(s, -s)` and `b = (1, 2)` the exact `β₂` is `0.8·abs(s)` but the computed norm is zero because the squared image entries are below the smallest subnormal, so `givens(α₁, 0)` gives radius `abs(α₁) = 0.6·abs(s)` (derived by hand and by the judge, not run; the conclusion that `1/ρ₁` overflows holds). (2) The seed Rustdoc (`execution.rs:107-109`) says entries beneath a negative residual entry are `-0`; a `-0` residual entry also yields `-0`, because the sign of a product is the XOR of the operand signs even for a zero result (IEEE 754; derived). (3) `workspace_reused_after_overflowing_second_update_matches_fresh_workspace` (`workspace_reuse.rs:176-218`) compares iterations (`:216`) and solution (`:217`) with the fresh solve but not the termination, which the sibling helper compares at `:62`.
- acceptance: the comment states radius `0.6·abs(s)`, the Rustdoc names the `-0` residual entry, and the third test asserts equal terminations: `git grep -c -F 'reused_report.termination, reference_report.termination' -- crates/athena-leto/tests/minres_contract/workspace_reuse.rs` prints 2 (prints 1 at the basis); `cargo nextest run -p athena-leto -E 'test(~minres)'` passes at `f32` and `f64`, and the mutant that drops the `update_previous_2` reset still fails the third test.
- next step: reword the comment and the Rustdoc from the derivation above, then add the termination assertion beside `:216`.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156

<a id="athena-chebyshev-reuse-test-20261124"></a>
## ATHENA-CHEBYSHEV-REUSE-TEST-20261124 — Compare the Chebyshev reuse solution bitwise and name its tolerance [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-leto/tests/chebyshev_contract.rs`
- outcome: `reused_workspace_after_overflow_matches_a_fresh_one` compares the reused and fresh solutions with `assert_eq!` on `Array1` (`:353`), which equates `-0` and `+0`, so a signed-zero divergence between the two workspaces passes; bitwise equality holds at the basis (judge probe over four inputs in the athena#196 review, not rerun). It also repeats the literal `T::from_f64(512.0) * EPSILON` (`:354`) beside the same literal and its comment in `solves_the_manufactured_system` (`:96-100`), which the lint floor counts as a float-literal tolerance in an assertion.
- acceptance: `git grep -c -E '^ +[^/ ].*to_bits' -- crates/athena-leto/tests/chebyshev_contract.rs` exits 0 (exits 1 at the basis; a code line, not a comment, calls `to_bits`); `git grep -c -F '512.0' -- crates/athena-leto/tests/chebyshev_contract.rs` prints 1 (prints 2 at the basis), the one literal sitting in a named constant or helper that carries the derivation in its doc; `cargo nextest run -p athena-leto -E 'test(~chebyshev)'` passes at `f32` and `f64`.
- next step: compare the two solutions through `to_bits` and lift the shared bound into one named constant used by both tests.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156

<a id="athena-lsqr-damped-contract-20261124"></a>
## ATHENA-LSQR-DAMPED-CONTRACT-20261124 — Pin damped LSQR warm start and damping edge cases, derive its bound [test] — todo

- priority: verification; needs: ATHENA-LSQR-ROTATION-OVERFLOW-20261124; milestone: M-device-least-squares
- scope: `crates/athena-leto/tests/lsqr_damped_contract.rs` (production code changes only if a test exposes a defect)
- outcome: four gaps in the damped LSQR contract, read at the basis unless marked. (1) The Rustdoc objective `‖A·x − b‖² + λ²‖x − x₀‖²` (`algorithm.rs:104-111`) is pinned only undamped (`lsqr_contract.rs:233`, `a_warm_start_is_honoured`); a judge probe on a 3×2 system at `x₀ ≠ 0` matched `x₀ + (AᵀA + λ²I)⁻¹Aᵀ(b − A·x₀)` to 1.1e-16, not committed. (2) A non-finite and a negative `damping` have no test; judge probes: `A·x₀ = b` with NaN or infinite `λ` ends `InitialResidual` at iteration 0, and `-0.5` is bitwise equal to `+0.5`. A finite `λ` whose square overflows is a defect (`Breakdown` at iteration 1 for `λ = 1e200` at `f64`), filed and tested under ATHENA-LSQR-ROTATION-OVERFLOW-20261124, which this item needs because that fix settles the termination of a non-finite `λ` after the first step. (3) `STEP_ROUNDING_UNITS = 8.0` (`:302`) asserts a count that its own list exceeds (two `γ₃` matvecs are 6 units before the norms, scalings, rotations and updates); the bound holds by margin (judge measurement: error at most 0.054 of the bound) but scales with `‖x*‖`, and a warm-started 8×6 `f32` case at `λ = 0.3` transcribes to 50.2 against 8·k = 48, so (1) cannot reuse it unchanged. (4) The bounds `1.0e-3` (`:176`), `1.0e-2` (`:256`), `64.0 * f64::EPSILON` (`:126`) and `1.0e3 * f64::EPSILON` (`:230`) are float-literal tolerances; the closed-form test supersedes the first two for the defect class and leaves the ill-conditioned 2×2 regime loosely checked.
- acceptance: new tests, generic over `f32` and `f64`: a warm start at `λ` of 0.5 and 1 solves to the closed form above and reports `sqrt(‖b − A·x‖² + λ²‖x − x₀‖²)` within a bound derived from an itemized rounding count that replaces `STEP_ROUNDING_UNITS`; NaN and infinite `λ` end at iteration 1 on a nonzero residual with the termination that ATHENA-LSQR-ROTATION-OVERFLOW-20261124 settles (`Breakdown` at the basis, while `crates/athena-core/src/report/solve.rs:15-16` assigns a non-finite coefficient to `NonFinite`) and `InitialResidual` at iteration 0 when `A·x₀ = b`; `-λ` gives bitwise the iterate of `λ`; `git grep -n -E -e 'let bound = 1\.0e-[23]' -e 'let bound = 64\.0 \* f64::EPSILON' -e 'let bound = 1\.0e3 \* f64::EPSILON' -- crates/athena-leto/tests/lsqr_damped_contract.rs` prints nothing (prints four lines at the basis); `cargo nextest run -p athena-leto --no-tests=fail -E 'test(=damped_warm_start_matches_closed_form) or test(=non_finite_damping_ends_the_solve) or test(=negative_damping_acts_as_its_absolute_value)'` passes, and `git grep -c -F -e 'fn damped_warm_start_matches_closed_form()' -e 'fn non_finite_damping_ends_the_solve()' -e 'fn negative_damping_acts_as_its_absolute_value()' -- crates/athena-leto/tests/lsqr_damped_contract.rs` prints 3 (exits 1 at the basis), so one new test cannot satisfy the filter alone.
- next step: write the warm-start case from the 3×2 system of `damped_solution_and_residual_match_closed_form`, then the damping edge cases, then replace the constant and the two literals with counts.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156

<a id="athena-lsqr-damped-doc-20261124"></a>
## ATHENA-LSQR-DAMPED-DOC-20261124 — Correct the damped LSQR Rustdoc, comment, test text and changelog entry [docs] — todo

- priority: verification; needs: ATHENA-LSQR-ROTATION-OVERFLOW-20261124; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/lsqr/algorithm.rs`, `crates/athena-core/src/solver/lsqr/execution.rs`, `crates/athena-leto/tests/lsqr_damped_contract.rs`, `CHANGELOG.md`
- outcome: four statements that the sources contradict and one consumer record, read at the basis unless marked. (1) `algorithm.rs:127-129` says a non-finite `damping` ends the solve with `Breakdown`; a solve that leaves before its first step ends `InitialResidual` (`execution.rs:80-87`) or `NormalEquations` (`:106-113`) instead (judge probe). The termination after the first step follows ATHENA-LSQR-ROTATION-OVERFLOW-20261124, which this item needs; a `λ` whose square overflows is a defect filed there, never documented behaviour. (2) `execution.rs:272-275` writes the augmented residual as `[b − A·x; −λ·x]` and the normal residual as `Aᵀr − λ²x`; with a warm start they are `−λ(x − x₀)` and `Aᵀr − λ²(x − x₀)`. (3) `lsqr_damped_contract.rs:196` states the objective as `‖A·x − b‖₂ + λ·‖x‖₂`, unsquared, and the message at `:416` carries a run of 18 spaces. (4) `CHANGELOG.md:11-12` credits the two-rotation elimination to Paige and Saunders (1982) §2.1; §2.1 states the damped problem (eq. 2.4) and the projected problem (eq. 2.7) but no rotation (judge read of the paper, not re-read), so the SciPy recurrence is its source; the entry also names the `NormalEquations` criterion but not the `Converged` check and the observer, which see the augmented residual too. (5) Consumers: `kwavers-math` calls `Lsqr::solve_damped_into` (`crates/kwavers-math/src/linear_algebra/sparse/matfree.rs:215`) and stores `final_residual_norm`, and `crates/kwavers-diagnostics/src/reconstruction/sound_speed_shift/tests/lsqr.rs:118` documents the old rotation formula (both at kwavers `7e9ac5d97f`). The affected-consumer record belongs to the delivering PR body, and the kwavers-side check to the kwavers board, not to `CHANGELOG.md`.
- acceptance: `git grep -c -F 'λ·x]' -- crates/athena-core/src/solver/lsqr/execution.rs`, `git grep -c -F 'Aᵀr − λ²x' -- crates/athena-core/src/solver/lsqr/execution.rs`, `git grep -c -E 'non-finite .damping. ends the solve' -- crates/athena-core/src/solver/lsqr/algorithm.rs`, `git grep -c -F '‖A·x − b‖₂ + λ·‖x‖₂' -- crates/athena-leto/tests/lsqr_damped_contract.rs` and `git grep -c -E ' {10,}error' -- crates/athena-leto/tests/lsqr_damped_contract.rs` each exit 1 (each prints 1 at the basis); `git grep -c -F 'Termination::InitialResidual' -- crates/athena-core/src/solver/lsqr/algorithm.rs` prints 1 or more (exits 1 at the basis), so a reworded sentence that still names only `Breakdown` fails; the delivering PR body names the two kwavers sites of (5); `cargo doc -p athena-core --no-deps --locked` stays warning-clean.
- next step: correct the four code-side texts from the evidence above, then amend the changelog entry with the source split and the other two criteria; the consumer's own check runs on the kwavers board.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156

<a id="athena-krylov-zero-reset-20261124"></a>
## ATHENA-KRYLOV-ZERO-RESET-20261124 — Hold the copy-then-scale-by-zero reset in one place [refactor] — todo

- priority: tightening; needs: ATHENA-MINRES-REUSE-DOC-20261124; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/minres/execution.rs`, `crates/athena-core/src/solver/chebyshev/execution.rs`, `crates/athena-core/src/backend/krylov.rs` (only if a backend operation is chosen)
- outcome: the seed paths of MINRES (`execution.rs:178-195`, three vectors) and Chebyshev (`execution.rs:162-178`, the step vector) both copy the seed residual into a vector and scale it by `ZERO`: `KrylovBackend` declares no operation that writes zeros (`copy` at `krylov.rs:94`, `scale` at `:101`), and scaling alone keeps NaN and turns infinity into NaN. Each site pays two passes per vector and relies on the residual being finite because its norm passed (on a device whose norm does not propagate NaN that premise is a hypothesis, athena194-1 F7, unverified). At the Chebyshev site the scale is bitwise redundant: step one scales `step` by `state.tau - one` (`:224`) with `τ₁` the literal `ONE` (`:195`), exactly `+0`, so the copy alone gives the same iterate and report (a copy-only mutant survived the reuse test in the athena#196 review; judge-run, not rerun). MINRES needs true zeros, since step one multiplies its stale vectors by `λ₁ = β ≠ 0`.
- acceptance: the pair lives in one place or in none: `git grep -c -E '^ +<B::Scalar as NumericElement>::ZERO,$' -- crates/athena-core/src/solver/minres/execution.rs crates/athena-core/src/solver/chebyshev/execution.rs` prints at most one line (two lines, one per file, at the basis; the pattern matches the line that passes `ZERO` to `scale` and no field initializer); `cargo nextest run -p athena-leto -E 'test(=chebyshev_workspace_reuse_after_overflow_matches_a_fresh_workspace) or test(~minres_workspace_reused) or test(=minres_reused_workspace_matches_fresh_workspace)'` passes at `f32` and `f64`.
- next step: delete the Chebyshev scale and state the `τ₁ − 1 = +0` argument in its comment; then decide for MINRES between keeping the pair and a backend zero write, which adds a required method to the public `KrylovBackend` and so needs an ADR and the SemVer gate.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156

<a id="athena-bicgstab-book-claims-20261124"></a>
## ATHENA-BICGSTAB-BOOK-CLAIMS-20261124 — Correct the BiCGSTAB chapter's breakdown claim and the book's Templates note [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `docs/book/bicgstab.md`, `docs/book/README.md`
- outcome: three statements the sources contradict or leave stale, read at the basis. (1) `bicgstab.md:71` says "the report says which recurrence stopped"; `Termination::Breakdown` is a unit variant (`report/solve.rs:12`), `SolveReport` has no field naming the recurrence (`:81-96`), and `bicgstab/execution.rs` returns the same `Breakdown` at four sites (`:142` zero shadow product `ρ`, `:223` zero `ω`, `:308` zero shadow-image product, `:365` zero stabilizer norm square). (2) `:53` names two denominators; that matches Figure 2.10 of Templates but understates the four zero tests. (3) `README.md:171-172` lists the Templates sections cited by the CG, GMRES and convergence-policy chapters (2.3.1, 2.3.4, 4.2) and omits §2.3.8, which `bicgstab.md:118` cites.
- acceptance: `git grep -c -F 'which recurrence stopped' -- docs/book/bicgstab.md` exits 1 (prints 1 at the basis); `git grep -c -F '2.3.8' -- docs/book/README.md` prints 1 (exits 1 at the basis); `mdbook test docs/book` passes.
- next step: state that `Breakdown` does not name the recurrence and list the four zero tests in the chapter, then add §2.3.8 to the README sentence.
- basis: 7aadfda2acd842abdfe563bbec8d2177a874d156
