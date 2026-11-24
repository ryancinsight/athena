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
## ATHENA-DOC-DRIFT-20261113 — Finish the doc-drift remainders: ledger, book derivations, citations [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `docs/verification/`, `docs/book/further_solvers.md`, doc comments in `crates/athena-core/src/solver/minres/algorithm.rs` and `chebyshev/algorithm.rs`
- outcome: three remainders after the README, CHANGELOG, gap-audit and module-doc reconciliation. (1) `docs/verification/gmres-rank-recovery.md` (99 lines) records landed work pinned by `crates/athena-leto/tests/gmres_rank_deficient.rs`, whose counterexample `docs/book/gmres.md` states, so the ledger belongs to the commit history. (2) `further_solvers.md` names `Minres`, `Chebyshev` and `DeflatedCg` and states their contracts but derives none: it has no recurrences, no compiled code and no references section. (3) `minres/algorithm.rs:40-42` gives the Paige and Saunders (1975) title unverified, and `chebyshev/algorithm.rs:41` cites Golub and Van Loan section 11.3 with no edition.
- acceptance: (1) the file is deleted, no tracked file links it, and `git log --follow -- docs/verification/gmres-rank-recovery.md` still reaches it; (2) each of the three sections of `further_solvers.md` states its recurrence and carries a compiled `rust` block, the chapter ends in a References section, and `mdbook test docs/book` passes; (3) both comments cite a resolved title, journal and pages, or an edition and section, or drop the locator.
- next step: `git rm` the ledger, then `git grep -n "docs/verification"` to confirm no link remains.
- basis: a3b9b0472096796313c10512eb7dbc36cc3c3c79

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
- outcome: ADR 0004 is 168 lines (`docs/adr/0004-derived-progress-termination.md`, Context at :13, Verification at :149) against the budget of about 150 lines per ADR; the in-place revisions of 2026-11-16 (CG stall detection, noise-floor premise) and the Verification list grew it. Compact it without dropping the decision, the four rejected alternatives, the evidence or the revision notes; delivery narrative belongs to commits. ADR 0005 is 164 lines at `9741050` and stays outside this item (`ATHENA-ADR-0005-FORM-20261124`).
- acceptance: `test "$(wc -l < docs/adr/0004-derived-progress-termination.md)" -le 150` exits 0 (exits 1 at the basis, 168 lines); the Context, Decision, Rejected alternatives, Consequences and Verification headings and the four rejected-alternative headings remain; `python scripts/adr-index.py check --directory docs/adr` from an Atlas checkout passes.
- next step: trim statements Consequences repeats from Decision, then shorten the Verification list to each test's name and oracle; keep every cited source.
- basis: 68d618e87e42d1b4d9f77e56e2e8e05f7653b7f9

<a id="athena-minres-workspace-reset-20261124"></a>
## ATHENA-MINRES-WORKSPACE-RESET-20261124 — Reset every MINRES recurrence vector at the seed [fix] — todo

- priority: correctness; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/minres/execution.rs`, `crates/athena-leto/tests/minres_contract.rs`
- outcome: `MinresWorkspace` is documented as reusable (`workspace.rs:3-9`) and `solve_into` takes it by `&mut`, but `seed` zeroes only `previous_direction` (`execution.rs:162-168`); `update_previous` and `update_previous_2` keep the last solve's update vectors. At step one `c₁ = c₂ = 1` and `s₁ = 0`, so `λ₁ = β = ‖r₀‖ ≠ 0` (`execution.rs:174, 205`) and `apply_update` subtracts `λ₁·update_previous` (`execution.rs:349-355`). A second solve then returns a wrong `x` while the estimate `φ̄`, which never reads an update vector, still reports `Converged`. `update_previous_2` enters scaled by `ε₁ = 0` at step one (`execution.rs:204`), and at step two the stale `update_previous` enters again through it scaled by `ε₂ = 0`, so a stale `update_previous_2` corrupts only through non-finite entries: judge-derived from a Python copy of the recurrence, not run on the Rust code (hypothesis), resetting `update_previous` alone gives the right second solve with finite stale entries but `Converged` with `x = (NaN, NaN)` when the stale `update_previous_2` holds NaN.
- acceptance: `minres_reused_workspace_matches_fresh_workspace` in `minres_contract.rs` solves the `diag(1, -1)` oracle twice on one `MinresWorkspace` with the solution reset to zero, at `f32` and `f64` as the file's other tests do (`..._for_every_cpu_scalar`), asserts the second solution `(1, -2)` within the bound that file derives, and fails at the basis commit (judge-derived from a Python copy of the recurrence, not run on the Rust code: the second solve of `b = (1, 2)` returns `(3.68328157, -3.34164079)` and reports `Converged` at step two, while zeroing both update vectors gives `(1, -2)`); then `seed` resets every recurrence vector, which the test alone does not pin for `update_previous_2` (the fields are `pub(super)`, `workspace.rs:24-26`), so the PR also records a read of `seed` against those three fields: `cargo nextest run -p athena-leto -E 'test(=minres_reused_workspace_matches_fresh_workspace)'`.
- next step: write the test first and confirm it fails at the basis, then zero `update_previous` and `update_previous_2` beside `previous_direction` in `seed`.
- basis: 9741050161a53f882f9ca8e4e8cfbb303579a9e2

<a id="athena-device-minres-band-20261124"></a>
## ATHENA-DEVICE-MINRES-BAND-20261124 — Pin the MINRES band-correction updates on a device [test] — todo

- priority: verification; needs: ATHENA-DEVICE-MINRES-20261113; milestone: M-device-least-squares
- scope: `crates/athena-hephaestus/tests/solver_correctness.rs`
- outcome: the device MINRES oracle (athena#186, `ATHENA-DEVICE-MINRES-20261113`) solves `diag(1, -1)`, where `λ₂ = c₁β₂ + s₁α₂ = 0`, `ε₁ = ε₂ = 0` and `λ₁` multiplies a zero vector, so deleting either band-correction axpy of `apply_update` (`execution.rs:349-362`) moves `x` by O(ε) and the test survives. `main` has no device MINRES test until that item lands, hence the `needs:` edge. The host tridiagonal (`minres_contract.rs:136-156`) does not close the gap alone: a Python transcription of the recurrence gives `ε₃ ≈ 0.249` there but `λ₂ = 0` and `λ₃ ≈ 1e-16`, so deleting the λ axpy changes `x` by under 1e-15 (derived; not run on the Rust code).
- acceptance: a 3×3 device instance with `λⱼ ≠ 0` against a nonzero update vector and `ε₃ ≠ 0`, with an integer manufactured solution, solves within a bound derived from its operation count (transcription candidate: `[[1, 1, 0], [1, -2, 1], [0, 1, -2]]`, `b = (3, 0, -4)`, `x* = (1, 2, 3)`; judge-derived from a Python copy of the recurrence, not run on the Rust code: `λ₂ = -1.198` multiplies `w₁` (norm 0.58), `ε₃ = 1.359`; deleting the λ axpy shifts `x` by 1.867 and the ε axpy by 1.478, against 8.9e-16 intact); deleting the λ axpy and, separately, the ε axpy makes the test fail, each deletion run and recorded in the PR.
- next step: after athena#186 lands, add the instance beside its MINRES test and run the two deletions.
- basis: 9741050161a53f882f9ca8e4e8cfbb303579a9e2

<a id="athena-minres-host-bounds-20261124"></a>
## ATHENA-MINRES-HOST-BOUNDS-20261124 — Derive the host MINRES tolerances from an operation count [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-leto/tests/minres_contract.rs`
- outcome: the bounds `256·EPSILON` (`:99-102`) and `512·EPSILON` (`:190-194`) rest on "fewer than 48" and "fewer than 72" rounded operations, a "factor-of-five" and "factor-of-three margin" and an amplification claim, with no tally. A tally from `execution.rs` shows the rotation scalars, `φ` and `φ̄`, and the update recurrence alone (`:204-206`, `:223-224`, `:343-375`) take 20 rounded operations per step on two-element vectors (1 + 4 + 4 + 2 + 9, each axpy element as one rounding, a lower bound), 40 over two steps, before the Lanczos step, the Givens scalars and the seed, so "48" does not count the operations that form `x`. The athena#186 review counted 98 for the 2×2 oracle (63 once operations exact on that instance are dropped); only the 20 per step above are confirmed here. Whether the constants still hold is open (hypothesis). The device bound of athena#186 is re-derived in that PR.
- acceptance: each constant is derived from an itemized, source-cited count of the rounded operations forming `x`, each host operation at relative error at most `ε/2` (`RealField::EPSILON` is `f32::EPSILON` and `f64::EPSILON`, eunomia `crates/eunomia/src/impls/field.rs:19,46`), propagated through the recurrence by running-error analysis, with the derivation in the constant's doc; `cargo nextest run -p athena-leto -E 'test(~minres)'` passes at `f32` and `f64`.
- next step: list the rounded operations of the two-step and three-step runs from `execution.rs`, then bound the error of `x` through the rotation scalars.
- basis: 9741050161a53f882f9ca8e4e8cfbb303579a9e2

<a id="athena-book-krylov-claim-20261124"></a>
## ATHENA-BOOK-KRYLOV-CLAIM-20261124 — Correct the BiCGSTAB chapter's Templates section [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `docs/book/bicgstab.md`
- outcome: `docs/book/bicgstab.md:118` cites Templates §2.3.7 for the preconditioned algorithm. In the netlib text of <https://www.netlib.org/templates/templates.html> (read 2026-11-24), §2.3.7 is "Conjugate Gradient Squared Method (CGS)" and §2.3.8 is "BiConjugate Gradient Stabilized (BiCGSTAB)", whose Figure 2.10 is the preconditioned method.
- acceptance: the chapter cites the section that carries the preconditioned BiCGSTAB algorithm: `git grep -n "2\.3\.7" -- docs/book` is empty and `git grep -n "2\.3\.8" -- docs/book/bicgstab.md` prints the citation; `mdbook test docs/book` passes.
- next step: change §2.3.7 to §2.3.8 in `bicgstab.md:118` and re-read that section's figure.
- basis: 9741050161a53f882f9ca8e4e8cfbb303579a9e2

<a id="athena-readme-chain-wording-20261124"></a>
## ATHENA-README-CHAIN-WORDING-20261124 — State the chain's guess and allocation behaviour exactly [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `README.md`
- outcome: `README.md:47` says `solve_with_guess_state` "starts from a caller guess"; the guess is `Option<&Array1<T>>` and `None` starts from zeros (`solver_chain/warm.rs:26, 31-33`). `README.md:175-177` says `LinearSolverChain` "builds the solution vector and, for each iterative tier, a workspace", which reads as exhaustive: a solve that reaches an iterative tier also builds preconditioners (component-block, SIMPLE, multigrid and incomplete LU at `cold.rs:66, 78, 90, 131`; block-diagonal twice at `:103, :110`; SOR at `warm.rs:48`) and a solve the direct tier ends builds only direct-LU factors (`warm.rs:35-37`, `tiers.rs:183-190`); the warm path clones the guess (`warm.rs:31-33`).
- acceptance: both sentences match those sources: the guess is optional with a zero start, and the allocation sentence lists the preconditioner and factor construction or says only that the chain allocates on every solve.
- next step: edit the two sentences, then `git grep -n "caller guess" -- README.md docs` to confirm no other copy.
- basis: 9741050161a53f882f9ca8e4e8cfbb303579a9e2

<a id="athena-adr-0005-form-20261124"></a>
## ATHENA-ADR-0005-FORM-20261124 — Reconcile ADR 0005 with the ladder and bring it to the ADR form [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `docs/adr/0005-release-scope.md`, `docs/adr/README.md`, `backlog.md` (ladder preamble)
- outcome: (1) veto 1 (`:82-87`) rejects "a 0.4.0 device block before a 1.0.0 block", while the ladder preamble plans `athena-krylov` 0.5.1, then 0.6.0, before 1.0.0 (`backlog.md:6, 11-12`); the manifest already reads 0.5.0 (ADR 0005 `:14-16`), so "0.4.0" names a released version. I read the veto as rejecting a milestone block and the 0.5.1 and 0.6.0 lines as checkpoints inside the one milestone, so this is a judgment call the ADR must state. (2) The revision note (`:153-157`) cites neither the delivering PR nor evidence (ADR governance). (3) The ADR is 164 lines against about 150.
- acceptance: veto 1 states how the 0.5.1 and 0.6.0 checkpoints relate to the single milestone, naming the version it rejects correctly; the revision note cites PR 185 (merge `9741050`) and the verification read; `test "$(wc -l < docs/adr/0005-release-scope.md)" -le 150` exits 0 (exits 1 at the basis); the decision, alternatives and evidence remain; `python scripts/adr-index.py check --directory docs/adr` from an Atlas checkout passes.
- next step: move the 2026-11-24 delivery narrative and the Aequitas revision text to their owners, then rewrite veto 1.
- basis: 9741050161a53f882f9ca8e4e8cfbb303579a9e2

<a id="athena-changelog-gating-wording-20261124"></a>
## ATHENA-CHANGELOG-GATING-WORDING-20261124 — Remove "remain required" from released changelog entries [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `CHANGELOG.md`
- outcome: two bullets of released versions still read as pending gates: `[0.3.1]` (`CHANGELOG.md:120-123`, "final provider ordering, exact-head review and hosted/publication gates remain required") and `[0.3.2]` (`:93-96`, "Athena dependency-delta review and lifecycle gates remain required"); tags `crate-*-v0.3.1` and `crate-*-v0.3.2` exist. The file is hand-edited here: the tree holds no changelog generator configuration and recent commits edit it directly (`git log -- CHANGELOG.md`).
- acceptance: `git grep -n "remain required" -- CHANGELOG.md` is empty at the head and each bullet states only the shipped change.
- next step: delete the two gating clauses and keep the shipped-change text.
- basis: 9741050161a53f882f9ca8e4e8cfbb303579a9e2
