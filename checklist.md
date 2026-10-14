# Solver correctness checklist

Target: unreleased patch to 0.3.0; package publishing is excluded.

- [x] [patch] Reproduce both failures against main with executable assertions.
- [x] [patch] Implement Schur elimination and normalised Chebyshev recurrence.
- [x] [patch] Verify recovered solutions, residual reports, warm starts, budgets,
  full coarse spaces, equal/narrow bounds, and the existing analytic oracles.
- [x] [patch] Update Rustdoc, README, and changelog.
- [x] [patch] Local all-feature/default-feature checks, strict Clippy,
  formatting, 136 native tests plus four explicit allocation tests, three
  doctests, warning-free Rustdoc, supply-chain and lockfile checks.
- [ ] [patch] Complete hosted gates on the exact PR head before merging.
