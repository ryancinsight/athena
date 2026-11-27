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

<a id="athena-device-minres-band-20261124"></a>
## ATHENA-DEVICE-MINRES-BAND-20261124 — Pin the MINRES band-correction updates on a device [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-hephaestus/tests/solver_correctness.rs`
- outcome: the device MINRES oracle (athena#186, `ATHENA-DEVICE-MINRES-20261113`) solves `diag(1, -1)`, where `λ₂ = c₁β₂ + s₁α₂ = 0`, `ε₁ = ε₂ = 0` and `λ₁` multiplies a zero vector, so deleting either band-correction axpy of `apply_update` (`execution.rs:398-411`) moves `x` by O(ε) and the test survives. athena#186 landed the device MINRES test, `minres_solves_the_indefinite_oracle_on_device`. The host tridiagonal (`minres_contract.rs:148-168`) does not close the gap alone: a Python transcription of the recurrence gives `ε₃ ≈ 0.249` there but `λ₂ = 0` and `λ₃ ≈ 1e-16`, so deleting the λ axpy changes `x` by under 1e-15 (derived; not run on the Rust code).
- acceptance: a 3×3 device instance with `λⱼ ≠ 0` against a nonzero update vector and `ε₃ ≠ 0`, with an integer manufactured solution, solves within a bound derived from its operation count (transcription candidate: `[[1, 1, 0], [1, -2, 1], [0, 1, -2]]`, `b = (3, 0, -4)`, `x* = (1, 2, 3)`; judge-derived from a Python copy of the recurrence, not run on the Rust code: `λ₂ = -1.198` multiplies `w₁` (norm 0.58), `ε₃ = 1.359`; deleting the λ axpy shifts `x` by 1.867 and the ε axpy by 1.478, against 8.9e-16 intact); deleting the λ axpy and, separately, the ε axpy makes the test fail, each deletion run and recorded in the PR.
- next step: add the instance beside `minres_solves_the_indefinite_oracle_on_device` and run the two deletions.
- basis: 170bf682f26f135ea15fd78e8d6ed9006b3443b2

<a id="athena-minres-host-bounds-20261124"></a>
## ATHENA-MINRES-HOST-BOUNDS-20261124 — Derive the host MINRES tolerances from an operation count [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-leto/tests/minres_contract.rs`
- outcome: the bounds `256·EPSILON` (`:76-83`) and `512·EPSILON` (`:210-214`) rest on "fewer than 48" and "fewer than 72" rounded operations, a "factor-of-five" and "factor-of-three margin" and an amplification claim, with no tally. A tally from `execution.rs` shows the rotation scalars, `φ` and `φ̄`, and the update recurrence alone (`:253-255`, `:272-273`, `:392-424`) take 20 rounded operations per step on two-element vectors (1 + 4 + 4 + 2 + 9, each axpy element as one rounding, a lower bound), 40 over two steps, before the Lanczos step, the Givens scalars and the seed, so "48" does not count the operations that form `x`. The athena#186 review counted 98 for the 2×2 oracle (63 once operations exact on that instance are dropped); only the 20 per step above are confirmed here. Whether the constants still hold is open (hypothesis). The device bound was re-derived in athena#186 (`ORACLE_BOUND_EPSILONS = 73`, `solver_correctness.rs:183`).
- acceptance: each constant is derived from an itemized, source-cited count of the rounded operations forming `x`, each host operation at relative error at most `ε/2` (`RealField::EPSILON` is `f32::EPSILON` and `f64::EPSILON`, eunomia `crates/eunomia/src/impls/field.rs:19,46`), propagated through the recurrence by running-error analysis, with the derivation in the constant's doc; `cargo nextest run -p athena-leto -E 'test(~minres)'` passes at `f32` and `f64`.
- next step: list the rounded operations of the two-step and three-step runs from `execution.rs`, then bound the error of `x` through the rotation scalars.
- basis: 170bf682f26f135ea15fd78e8d6ed9006b3443b2

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

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-leto/tests/lsqr_damped_contract.rs` (production code changes only if a test exposes a defect)
- outcome: four gaps in the damped LSQR contract, read at the basis unless marked. (1) The Rustdoc objective `‖A·x − b‖² + λ²‖x − x₀‖²` (`algorithm.rs:104-111`) is pinned only undamped (`lsqr_contract.rs:233`, `a_warm_start_is_honoured`); a judge probe on a 3×2 system at `x₀ ≠ 0` matched `x₀ + (AᵀA + λ²I)⁻¹Aᵀ(b − A·x₀)` to 1.1e-16, not committed. (2) A non-finite and a negative `damping` have no test; judge probes at 7aadfda2acd8, not rerun after athena#200: `A·x₀ = b` with NaN or infinite `λ` ends `InitialResidual` at iteration 0, and `-0.5` is bitwise equal to `+0.5`. A finite `λ` whose square overflows converges since athena#200 (`damped_solve_survives_a_damping_whose_square_overflows`, `:478`), and that test asserts `Breakdown` at iteration 1 for `λ = +∞` (`:473`). (3) `STEP_ROUNDING_UNITS = 8.0` (`:302`) asserts a count that its own list exceeds (two `γ₃` matvecs are 6 units before the norms, scalings, rotations and updates); the bound holds by margin (judge measurement: error at most 0.054 of the bound) but scales with `‖x*‖`, and a warm-started 8×6 `f32` case at `λ = 0.3` transcribes to 50.2 against 8·k = 48, so (1) cannot reuse it unchanged. (4) The bounds `1.0e-3` (`:176`), `1.0e-2` (`:256`), `64.0 * f64::EPSILON` (`:126`) and `1.0e3 * f64::EPSILON` (`:230`) are float-literal tolerances; the closed-form test supersedes the first two for the defect class and leaves the ill-conditioned 2×2 regime loosely checked.
- acceptance: new tests, generic over `f32` and `f64`: a warm start at `λ` of 0.5 and 1 solves to the closed form above and reports `sqrt(‖b − A·x‖² + λ²‖x − x₀‖²)` within a bound derived from an itemized rounding count that replaces `STEP_ROUNDING_UNITS`; NaN and infinite `λ` end at iteration 1 on a nonzero residual with `Breakdown`, which `lsqr_damped_contract.rs:473` asserts for `λ = +∞` and which NaN and `−∞` reach by the same early return (`rotation.rs:33-35`, `lsqr/execution.rs:226-230`; derived by reading, not run), while `crates/athena-core/src/report/solve.rs:15-16` assigns a non-finite coefficient to `NonFinite`, so the test pins the current behaviour and the PR records whether it is intended and `InitialResidual` at iteration 0 when `A·x₀ = b`; `-λ` gives bitwise the iterate of `λ`; `git grep -n -E -e 'let bound = 1\.0e-[23]' -e 'let bound = 64\.0 \* f64::EPSILON' -e 'let bound = 1\.0e3 \* f64::EPSILON' -- crates/athena-leto/tests/lsqr_damped_contract.rs` prints nothing (prints four lines at the basis); `cargo nextest run -p athena-leto --no-tests=fail -E 'test(=damped_warm_start_matches_closed_form) or test(=non_finite_damping_ends_the_solve) or test(=negative_damping_acts_as_its_absolute_value)'` passes, and `git grep -c -F -e 'fn damped_warm_start_matches_closed_form()' -e 'fn non_finite_damping_ends_the_solve()' -e 'fn negative_damping_acts_as_its_absolute_value()' -- crates/athena-leto/tests/lsqr_damped_contract.rs` prints 3 (exits 1 at the basis), so one new test cannot satisfy the filter alone.
- next step: write the warm-start case from the 3×2 system of `damped_solution_and_residual_match_closed_form`, then the damping edge cases, then replace the constant and the two literals with counts.
- basis: 170bf682f26f135ea15fd78e8d6ed9006b3443b2

<a id="athena-lsqr-damped-doc-20261124"></a>
## ATHENA-LSQR-DAMPED-DOC-20261124 — Correct the damped LSQR Rustdoc, comment, test text and changelog entry [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/lsqr/algorithm.rs`, `crates/athena-core/src/solver/lsqr/execution.rs`, `crates/athena-leto/tests/lsqr_damped_contract.rs`, `CHANGELOG.md`
- outcome: four statements that the sources contradict and one consumer record, read at the basis unless marked. (1) `algorithm.rs:127-129` says a non-finite `damping` ends the solve with `Breakdown`; a solve that leaves before its first step ends `InitialResidual` (`execution.rs:81-88`) or `NormalEquations` (`:107-114`) instead (judge probe). After the first step `λ = +∞` ends `Breakdown` at iteration 1 (`lsqr_damped_contract.rs:473`); athena#200 fixed the finite `λ` whose square overflows, so the sentence names no such case. (2) `execution.rs:272-275` writes the augmented residual as `[b − A·x; −λ·x]` and the normal residual as `Aᵀr − λ²x`; with a warm start they are `−λ(x − x₀)` and `Aᵀr − λ²(x − x₀)`. (3) `lsqr_damped_contract.rs:196` states the objective as `‖A·x − b‖₂ + λ·‖x‖₂`, unsquared, and the message at `:416` carries a run of 18 spaces. (4) `CHANGELOG.md:19-20` credits the two-rotation elimination to Paige and Saunders (1982) §2.1; §2.1 states the damped problem (eq. 2.4) and the projected problem (eq. 2.7) but no rotation (judge read of the paper, not re-read), so the SciPy recurrence is its source; the entry also names the `NormalEquations` criterion but not the `Converged` check and the observer, which see the augmented residual too. (5) Consumers: `kwavers-math` calls `Lsqr::solve_damped_into` (`crates/kwavers-math/src/linear_algebra/sparse/matfree.rs:215`) and stores `final_residual_norm`, and `crates/kwavers-diagnostics/src/reconstruction/sound_speed_shift/tests/lsqr.rs:118` documents the old rotation formula (both at kwavers `7e9ac5d97f`). The affected-consumer record belongs to the delivering PR body, and the kwavers-side check to the kwavers board, not to `CHANGELOG.md`.
- acceptance: `git grep -c -F 'λ·x]' -- crates/athena-core/src/solver/lsqr/execution.rs`, `git grep -c -F 'Aᵀr − λ²x' -- crates/athena-core/src/solver/lsqr/execution.rs`, `git grep -c -E 'non-finite .damping. ends the solve' -- crates/athena-core/src/solver/lsqr/algorithm.rs`, `git grep -c -F '‖A·x − b‖₂ + λ·‖x‖₂' -- crates/athena-leto/tests/lsqr_damped_contract.rs` and `git grep -c -E ' {10,}error' -- crates/athena-leto/tests/lsqr_damped_contract.rs` each exit 1 (each prints 1 at the basis); `git grep -c -F 'Termination::InitialResidual' -- crates/athena-core/src/solver/lsqr/algorithm.rs` prints 1 or more (exits 1 at the basis), so a reworded sentence that still names only `Breakdown` fails; the delivering PR body names the two kwavers sites of (5); `cargo doc -p athena-core --no-deps --locked` stays warning-clean.
- next step: correct the four code-side texts from the evidence above, then amend the changelog entry with the source split and the other two criteria; the consumer's own check runs on the kwavers board.
- basis: 170bf682f26f135ea15fd78e8d6ed9006b3443b2

<a id="athena-krylov-zero-reset-20261124"></a>
## ATHENA-KRYLOV-ZERO-RESET-20261124 — Hold the copy-then-scale-by-zero reset in one place [refactor] — todo

- priority: tightening; needs: none; milestone: M-device-least-squares
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

<a id="athena-minres-norm-scaling-20261127"></a>
## ATHENA-MINRES-NORM-SCALING-20261127 — Stop MINRES norms from underflowing or overflowing a finite solve [fix] — todo

- priority: correctness; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-leto/src/backend/krylov.rs`, `crates/athena-core/src/backend/krylov.rs` (`norm_l2` contract), `crates/athena-core/src/solver/minres/execution.rs`, `crates/athena-leto/tests/minres_contract.rs`, `crates/athena-leto/tests/minres_contract/workspace_reuse.rs`
- outcome: every MINRES norm is `norm_l2_prepared`, which Leto computes as `dot(v, v).sqrt()` (`athena-leto/src/backend/krylov.rs:141-146`), a native-precision sum of squares: the core contract says native precision (`athena-core/src/backend/krylov.rs:159`, `:170`) and leto-ops 0.52.0, the locked version, documents native accumulation for `dot` (`application/vector.rs:67-72`; its SIMD kernel `hermes_simd::dot` is unread). It underflows when every square is below the smallest subnormal and overflows when one exceeds `MAX`. Derived by reading (athena#199 review F8), not run on the Rust code; the first Lanczos step of (a) and the seed norm of (b) are transcribed in numpy at both precisions (`β₂ = 0`, seed norm `inf`), the steps after them read from `execution.rs`. (a) `diag(s, −s)`, `b = (1, 2)`, `x₀ = 0`, policy `(EPSILON, EPSILON, 3)`, `s = 2^-92` at `f32` and `2^-572` at `f64`: the squares of the second Lanczos image underflow, `β₂ = 0` (`execution.rs:369-375`), step 1 exits `Converged` with `x = −(1, 2)/(0.6s)` and a true residual ≈ 2.98 above `‖b‖` ≈ 2.24, though `x* = (1/s, −2/s)` is finite. (b) `diag(1, −1)`, `b = 2^64·(1, 2)` at `f32` (`2^512·(1, 2)` at `f64`): the seed norm is `inf` and the solve returns `NonFinite` at iteration 0 (`execution.rs:128-141`) though `x* = 2^64·(1, −2)` is finite. Hypothesis, unread: every solver whose seed or β uses `norm_l2` shares (b), and the device norm (a `DotOp` sum, hephaestus-wgpu 0.30.0 per the review) shares both. The reuse fixture `solve_with_overflowing_update_vector` (`workspace_reuse.rs:85-91`) and `overflowing_terminations` (`minres_contract.rs:259-274`) rely on the `β₂` underflow of (a), so a fix changes them.
- acceptance: one test in `minres_contract.rs`, generic over `f32` and `f64`, solves (a) and (b): (b) `Converged` with `x` within a bound derived from an itemized rounding count; (a) `Converged` within such a bound or a non-converged termination; at the basis it fails with (a) `Converged` at residual ≈ 2.98 and (b) `NonFinite` (derived; the delivering PR runs it at the basis); the reuse fixtures are re-derived so the non-finite-update tests still reach a non-finite `x`; `cargo nextest run -p athena-leto -E 'test(~minres)'` passes.
- next step: choose where the scaling lives (the backend norm, which changes the `norm_l2` contract and leaves the device norm to an upstream Hephaestus item, or MINRES alone), record it in an ADR if the trait contract changes, then write the failing test first.
- basis: 170bf682f26f135ea15fd78e8d6ed9006b3443b2

<a id="athena-solver-nonfinite-guess-20261127"></a>
## ATHENA-SOLVER-NONFINITE-GUESS-20261127 — Stop every solver reporting convergence with a non-finite solution [fix] — todo

- priority: correctness; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-core/src/solver/` (`cg`, `gmres`, `bicgstab`, `lsqr`, `chebyshev`, `deflation`, and one shared check beside `dimension.rs`), `crates/athena-core/src/report/solve.rs`, `docs/book/convergence_policy.md` (the `NonFinite` row), `crates/athena-leto/tests/`
- outcome: a NaN or infinity in `x₀` at a structurally empty column of the operator never enters `A·x₀`, so CG reports `InitialResidual` or `Converged` with that entry in `x`. By reading (athena#199 review F3), not run: the seed forms `b − A·x₀` from the apply alone and accepts a finite residual (`cg/algorithm.rs:156-220`, exit at `:197-201`), and the iterative exit recomputes `b − A·x` through the same apply (`measure_original_residual`, `:442-474`; `Converged` at `:353`); e.g. `[[1, 0], [0, 0]]`, `b = (1, 0)`, `x₀ = (1, NaN)` gives `r₀ = 0` and `InitialResidual`. MINRES closed this with a post-loop `x − x` test (`minres/algorithm.rs:112`, `execution.rs:212-233`; fixture `check_nullspace_component`, `minres_contract.rs:287-322`). Hypothesis, unread: GMRES, BiCGSTAB, LSQR, Chebyshev and `DeflatedCg` behave as CG. Budget and breakdown exits stay unchecked, as the MINRES Rustdoc states (`minres/algorithm.rs:37-42`).
- acceptance: one test, generic over `f32` and `f64`, runs the `check_nullspace_component` fixture (lifted out of `minres_contract.rs` into a module the solver test files share) through each solver and asserts `NonFinite` for NaN and `±∞` and `InitialResidual` or `Converged` for `MAX`; at the basis it fails for CG with `InitialResidual` (derived); the check is one shared `x − x` zero-norm test called at each solver's convergence exits, never a per-solver copy; `Termination::NonFinite` (`report/solve.rs:15`) and each solver's `# Termination` section name the exits that check, and the `NonFinite` row of `docs/book/convergence_policy.md` follows.
- next step: lift the fixture, run it against CG and read which of the other five fail, then move the MINRES check to the shared home and call it from each failing solver.
- basis: 170bf682f26f135ea15fd78e8d6ed9006b3443b2

<a id="athena-book-nonfinite-row-20261127"></a>
## ATHENA-BOOK-NONFINITE-ROW-20261127 — Name the checked solution entries in the NonFinite row of the policy chapter [docs] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `docs/book/convergence_policy.md`
- outcome: the row `NonFinite` (`:161`) reads "a residual or recurrence coefficient became non-finite", while `Termination::NonFinite` (`report/solve.rs:15`) also covers a checked solution entry and MINRES returns it with a finite residual at or below the threshold at its converged exits (`minres/algorithm.rs:108-114`; `minres_contract.rs:319` asserts `final_residual_norm == 0`). Read at the basis (athena#199 review F1). The row names the solvers that check; ATHENA-SOLVER-NONFINITE-GUESS-20261127 widens that set and updates the row in its PR.
- acceptance: `git grep -c -F 'solution entr' -- docs/book/convergence_policy.md` prints 1 or more (exits 1 at the basis, no match); `mdbook test docs/book` passes.
- next step: reword the row to name the checked solution entry and MINRES's converged exits.
- basis: 170bf682f26f135ea15fd78e8d6ed9006b3443b2

<a id="athena-minres-allocation-20261127"></a>
## ATHENA-MINRES-ALLOCATION-20261127 — Measure the MINRES allocation contract [test] — todo

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `crates/athena-leto/tests/allocation.rs`
- outcome: `allocation.rs` holds steady-state tests for CG (`:69`), GMRES (`:123`) and BiCGSTAB (`:178`) and none for MINRES (no line matches `minres` there), while the `MinresWorkspace` Rustdoc claims allocation-free reuse (`workspace.rs:3-10`). The converged-exit `x − x` check (`execution.rs:212-233`) writes into the seed buffer and reduces to a scalar, so its host path is allocation-free by reading, unmeasured (athena#199 review F4).
- acceptance: `repeated_minres_solves_allocate_nothing_after_initialization` in `allocation.rs`, with its siblings' `#[ignore]` reason, runs 16 converged MINRES solves (so the check runs) through `assert_steady_state` and passes under `cargo nextest run -p athena-leto --test allocation --run-ignored ignored-only`, the command of the Allocation instrument job (`ci.yml:102`); `git grep -c -i minres -- crates/athena-leto/tests/allocation.rs` prints 1 or more (exits 1 at the basis).
- next step: copy the CG test's structure, solve the `diag(1, -1)` oracle sixteen times, and read the delta.
- basis: 170bf682f26f135ea15fd78e8d6ed9006b3443b2

<a id="athena-judge-required-20261127"></a>
## ATHENA-JUDGE-REQUIRED-20261127 — Require the `judge` status on athena main [chore] — blocked

- priority: verification; needs: none; milestone: M-device-least-squares
- scope: `.github/workflows/ci.yml` (the caller), branch protection of `main` (hosting setting)
- outcome: `judge` is not a required check on athena main (`gh api repos/ryancinsight/athena/branches/main/protection/required_status_checks` lists `Lockfile integrity / Lockfile integrity`, `supply-chain` and `verify`, run at the basis), so athena#199 carried `judge-required` and merged on its head with no `judge` status (athena#199 review F7). No athena workflow posts `judge` (`git grep -i judge -- .github` prints nothing) and no workflow of the Atlas meta repository at origin/main 8a771bd3a has `statuses: write`.
- blocker: no stack workflow posts the `judge` status, and no Atlas board item for one exists at 8a771bd3a (searched `backlog.md`, `backlog/` and `gap_audit.md` for `judge`); re-open when such a workflow lands in the Atlas meta repository and athena's `ci.yml` can call it.
- acceptance: a ready unlabeled athena PR head carries a `judge` status posted by that workflow and `gh api repos/ryancinsight/athena/branches/main/protection/required_status_checks --jq .contexts` lists `judge` (it does not at the basis). The setting is executed through `gh api` under the merge-mechanics grant once the workflow exists, after open heads carry a status, since an unposted required context leaves a PR `UNSTABLE`; until then a `judge-required` PR merges only through collection with `--match-head-commit` after a judge pass.
- next step: read the stack's PR workflow when it lands, call it from `ci.yml`, post `judge` on open ready heads, then require it.
- basis: 170bf682f26f135ea15fd78e8d6ed9006b3443b2
