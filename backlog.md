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

<a id="athena-stencil-operator-20261020"></a>
## ATHENA-STENCIL-OPERATOR-20261020 — Production matrix-free stencil operator [perf] — todo

- Priority: tightening;scope:`crates/athena-leto/src/operator/stencil.rs`
  (new) + exports + tests; bench harness stays.
- Evidence (spike, `stencil_vs_csr` back-to-back 2026-10-20, grid/128):
  matvec_csr 31.4us vs matvec_stencil 12.1us (2.6x), equivalence within
  1e-12 on the deterministic fixture. Design: `LinearOperator<B>` needs
  no change — the stencil is a new implementor beside `CsrOperator`,
  generic over the backend scalar with the grid as constructor input.
- Outcome: land the operator with value-semantic tests (boundary rows,
  equivalence vs CSR within rounding), wire the `cg_poisson` bench to
  cover it, and re-measure the per-iteration model (expect ~-40%).
- Acceptance: tests + strict gates + hosted CI green; back-to-back
  re-measurement confirms the spike ratio; item carries the measured
  per-iteration win at close.
- Stop: land once the operator is covered and measured; further stencil
  micro-optimization (interior/boundary split, SIMD) only on a new
  profile showing it still dominates.
