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
## ATHENA-CG-SIMD-SLICES-20261019 — Route CG vector updates through SIMD slice kernels [perf] — todo

- Priority: tightening;scope:`crates/athena-leto/src/backend/krylov.rs` only.
- Outcome: `scale`/`axpy`/`residual` call `T::{scale_slice,axpy_slice,sub_slice}`
  (SIMD-backed in leto-ops) instead of scalar loops; `fused_cg_update` and
  `combine_direction` stay hand-rolled (no fused leto equivalent).
- Needs: ATHENA-CG-BASELINE-20261019 merged (baseline medians to beat).
- Acceptance: same convergence/iterations on both grids; median strictly below
  baseline with non-overlapping CI; 136-test suite + strict gates green.
