# Correctness backlog

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
