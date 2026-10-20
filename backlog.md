# Correctness backlog

Active patch target: **0.3.2**.

<a id="athena-fused-update-20261020"></a>
## ATHENA-FUSED-UPDATE-20261020 — Next CG lever: fused vector update [perf] — todo

- Priority: tightening;scope:measurement + optimization of the update path.
- Evidence: the stencil cut (#130) moved grid/128 per-iteration cost from
  54.9us to ~27.1us; new shares are matvec 40%, fused_cg_update 31%
  (8.3us), combine_direction 15%, dots+norm 14%. The matvec no longer
  dominates alone — the fused update is the co-leader.
- Outcome: profile the fused update (memory traffic vs L3 ceiling,
  SIMD coverage), then cut the derived optimization back-to-back
  against the `kernels` bench.
- Acceptance: recorded diagnosis + measured win with the same
  convergence/iterations; tests + strict gates + hosted CI green.
- Stop: land once the update no longer co-leads or a faster form is
  proven unnecessary; file the next profiled leader in turn.

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

