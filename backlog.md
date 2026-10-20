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

<a id="athena-matrixfree-eval-20261020"></a>
## ATHENA-MATRIXFREE-EVAL-20261020 — Matrix-free stencil matvec evaluation [perf] — todo

- Priority: tightening;scope:measurement spike + design note only.
- Evidence (bound diagnosis, leto `spmv/banded7` 2026-10-20):
  n=4096 10.0us (55 GB/s, cache), n=65536 168us (53 GB/s, L3),
  n=1M 6.20ms (20.6 GB/s effective on 128 MB traffic, DRAM). At DRAM
  sizes SpMV rides a large fraction of attainable single-thread
  bandwidth, so leto LETO-CSR-INDEX-WIDTH (u32 halves the 50% index
  share) is the correct DRAM-regime vehicle — no athena item needed.
  At grid/128 (1.7 MB « LLC) the matvec runs ~10 cycles/row against an
  L3 ceiling several times higher: latency-bound, and u32 indices are
  not its lever.
- Outcome: spike a matrix-free 5-point stencil matvec against the CSR
  path back-to-back (traffic drops ~5x: no values/indices/gather),
  with the backend-seam design (operator trait vs assembled matrix)
  recorded before any production cut.
- Acceptance: back-to-back numbers + the seam design decision with its
  stop condition; no production-byte change in this item.
