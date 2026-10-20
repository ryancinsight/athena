# Correctness backlog

Active patch target: **0.3.2**.

<a id="athena-gmres-rank-recovery-20261020"></a>
## ATHENA-GMRES-RANK-RECOVERY-20261020 — Reduced LS and checked telemetry [patch] — blocked

- Priority: correctness; scope: generic GMRES reduced solve/workspace and telemetry.
- [x] Reproduce lost rank-two recovery and impossible zero observer samples in
  native single/double precision before changing production.
- [x] Keep full-rank arithmetic; use preallocated column-pivoted QR only for
  numerical rank loss. Specify basic LS, not minimum coefficient/solution norm.
- [x] Check nonfinite inputs, rank zero, off-diagonal coupling, observer cadence
  and strict zero-allocation rank recovery with executable regressions.
- [x] Preserve #125's merged CG profiling/SpMV records and benchmark ownership.
- [ ] Complete final-head native/accelerator, allocation, book, API/MSRV,
  package/dependency and performance checks; record results in the draft PR.
- [ ] Obtain exact-head independent review and all applicable hosted gates
  before merge. Main's automatic four-package publication needs scope review.
- [x] Renew repaired Leto >=0.46.3 and current Hephaestus admission on the
  canonical graph after native provider source/packages/tags verify. Keep
  affected independent review and absolute/cohort performance assessment open.
- Basis: `docs/verification/gmres-rank-recovery.md`.
- Harmonia performance and all other release/service holds remain separate.

- [patch] Deflated CG coarse/complement coupling and true-residual reporting:
  implemented with executable CPU and mandatory accelerator regressions.
- [patch] Chebyshev singleton/narrow spectrum coefficients and reporting:
  implemented with executable CPU and mandatory accelerator regressions.
- Merge only the final head whose applicable repository gates pass. Package
  publishing is outside this task; no manifest version bump is included because
  this repository publishes bumps automatically on main.

<a id="athena-provider-range-20261019"></a>
## ATHENA-PROVIDER-RANGE-20261019 — Current and next provider admission [patch] — blocked

- Priority: correctness;target:0.3.1;scope:requirements,versions,lock and records only.
- [x] Preserve current lower bounds and all Rust/test/policy bytes.
- [x] Verify baseline resolver refusals and four real current/future typed consumer cohorts.
- [x] Pass current native,strict,docs,MSRV and explicit patch API gates.
- [ ] Complete standard-package,exact-head independent review and hosted gates.
- [ ] Coordinate final canonical provider ordering;retain actual publication/auth holds.
- Basis:`docs/verification/provider-range-bridge.md`;no older source pin or policy exception.

<a id="athena-cg-baseline-20261019"></a>
## ATHENA-CG-BASELINE-20261019 — CG criterion baseline on Leto CPU [perf] — todo

- Priority: tightening;scope:`crates/athena/benches/cg_poisson.rs`,criterion dev-dep,lock only.
- Outcome: pinned 5-point Dirichlet Poisson (grids 64/128, all-ones rhs, x0=0,
  unpreconditioned CG, reused workspace) with recorded medians before any
  kernel optimization (Atlas ADR 0002).
- Baseline 2026-10-19 (Windows host, release): grid/64 (4,096 unknowns, 119
  iters) 1.4284 ms [1.3813, 1.4726]; grid/128 (16,384 unknowns, 239 iters)
  13.475 ms [13.226, 13.720]. Both Converged; time ratio 9.43x = 2.0x iters x
  4.7x work/iter, coherent with O(N) CG on 2D Poisson.
- Acceptance: `cargo bench -p athena-krylov --bench cg_poisson --features cpu`
  green; fmt/clippy `--all-targets --all-features` clean; hosted CI green.

<a id="athena-cg-simd-slices-20261019"></a>
## ATHENA-CG-SIMD-SLICES-20261019 — Route vector updates through SIMD slice kernels [perf] — blocked

- Priority: tightening;scope:`crates/athena-leto/src/backend/krylov.rs` only.
- Blocked by ATHENA-GMRES-BREAKDOWN-TOLERANCE-20261019: SIMD `axpy` rounding
  (FMA single-rounding vs scalar mul-then-add) leaves a ~1ulp residue where the
  scalar loop cancels exactly, flipping the FGMRES oracle from `Breakdown` to
  `MaxIterations` (proven: base passes, SIMD fails, 2026-10-19).
- Corrected scope: CG's steady state calls `dot`/`norm` (already SIMD) plus
  `fused_cg_update`/`combine_direction`/`copy`, never `axpy`/`scale`; this item
  helps GMRES-family solvers only. CG work proceeds under
  ATHENA-CG-PROFILE-20261019.
- Needs: breakdown-tolerance fix merged; then same convergence/iterations on
  both grids, GMRES-oracle green, 136-test suite + strict gates green.

<a id="athena-gmres-breakdown-tolerance-20261019"></a>
## ATHENA-GMRES-BREAKDOWN-TOLERANCE-20261019 — Tolerance-based Arnoldi breakdown detection [correctness] — todo

- Priority: correctness;scope:`crates/athena-core/src/solver/gmres/` (`execution.rs` happy-breakdown check, `rotation.rs` zero-diagonal check).
- Finding: breakdown is detected by exact `== ZERO` comparison, which holds for
  one rounding order and fails for another equally valid one (scalar vs SIMD
  `axpy`; any GPU backend faces the same fragility). Rank loss is a rounding-
  level event and must be detected with an analytically derived tolerance
  scaled to the orthogonalization (e.g. restart-relative machine-epsilon bound
  against the pre-orthogonalization norm), never exact equality.
- Acceptance: FGMRES oracle green under both scalar and SIMD evaluation;
  full suite + strict gates green; tolerance derivation recorded in the
  deciding commit/ADR.

<a id="athena-cg-profile-20261019"></a>
## ATHENA-CG-PROFILE-20261019 — Per-kernel CG cost breakdown before optimizing [perf] — todo

- Priority: tightening;scope:measurement only, no production change.
- Outcome: attribute the grid/128 13.475 ms median across matvec, dots/norms,
  `fused_cg_update`, and `combine_direction` (targeted micro-benches or
  instrumentation). Analytic estimate says ~56 us/iter at ~27 GB/s traffic, i.e.
  plausibly memory-bound, which would make fusion (fewer traversals) the lever
  and per-kernel SIMD a non-lever; verify before cutting any optimization.
- Acceptance: recorded per-kernel shares + the derived next optimization with
  its stop condition; no production-byte change in this item.
- Measured 2026-10-19 (`kernels` bench, n=16384 medians): matvec_csr 39.8us,
  fused_cg_update 8.5us, combine_direction 4.4us, residual 4.2us, axpy 3.7us,
  scale 3.7us, copy 3.4us, dot 1.7us, norm_l2 1.4us. Per-iteration model
  (1 matvec + 2 dots + 1 norm + 1 fused + 1 combine) = 57.6us, matching the
  recorded 56.4us/iter baseline within 2%. Shares: matvec 69%, fused 15%,
  combine 8%, dots+norm 8%. Next optimization: CSR SpMV (scalar row loop);
  stop when it no longer dominates or a faster format is proven unnecessary.
- Method note: absolute us drift ~20% across sessions on this host; ratios
  within one run are the signal; A/B claims require back-to-back measurement
  (as done for the -8-11% SIMD routing win).
