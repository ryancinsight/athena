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

<a id="athena-cg-spmv-followup-20261020"></a>
## ATHENA-CG-SPMV-FOLLOWUP-20261020 — Next CG lever after the SpMV unroll [perf] — todo

- Priority: tightening;scope:measurement + routing decision only.
- Evidence: leto #375 unrolled the CSR gather-dot (4 accumulators,
  back-to-back -8-12% on banded-7); re-measured 2026-10-20 (`kernels`
  bench, n=16384 medians): matvec_csr 37.5us, fused 8.3us, combine 4.2us,
  residual 4.3us, axpy 3.7us, scale 3.4us, copy 3.4us, dot 1.7us,
  norm_l2 1.4us. Per-iteration model = 54.9us; matvec share 68%, still
  dominant. `cg_poisson` re-measured green: grid/64 1.297ms, grid/128
  11.224ms (both beat the recorded baseline).
- Outcome: route the next lever — leto LETO-CSR-INDEX-WIDTH-2026-09-27
  (u32 indices halve SpMV index traffic) is the standing candidate;
  confirm the memory-bound diagnosis first (traffic vs measured
  bandwidth), else derive the alternative.
- Acceptance: a measured bound diagnosis + the next optimization item
  with its stop condition; no production-byte change in this item.
