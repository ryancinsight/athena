# Solver correctness audit

At main `6c8d81cc122c353be3239a38cdfee50d23dc6e14`, executable regressions
reproduce both source-derived failures: deflated CG returns `(0.5,0)` with
reported residual zero for the coupled two-by-two system, and Chebyshev
returns `NonFinite` with residual one after reaching the singleton-spectrum
solution.

The fixes retain the accepted input domains. Schur elimination restores the
missing coupling; an explicit residual checks the recovered iterate.
Normalising the Chebyshev ratio by its centre gives the finite singleton
limit without an empirical tolerance or separate approximate algorithm.
Both original analytic oracles remain intact. Hosted gate results, exact head,
and merge evidence are recorded in the PR; an unavailable accelerator is a
failure in the new device regressions, not skipped coverage.


## Rank-deficient GMRES recovery (2026-10-20)

At `03ed0ab0fe5d7ca87d56f304dad8e20199b18548`, an attainable three-step
Arnoldi factor loses an O(1) range direction when the original leading
coefficient is independently discarded. The same cycle emits an impossible
zero residual estimate although the checked original-system residual is
sqrt(1/2). Both failures reproduce in native single/double precision before
production changes. The independent minimum-norm claim is also false.

The repair uses column-pivoted QR only on numerical rank loss, defines a
basic reduced LS correction, and defers cycle-ending observer samples until
the true residual is recomputed. Full-rank arithmetic remains unchanged.
`docs/verification/gmres-rank-recovery.md` records the analytical oracle and
scope. Exact-head review, final hosted gates and publication-scope assessment
remain merge requirements; the separate Harmonia performance hold persists.

The canonical dependency renewal selects published Leto/LetoOps 0.46.3 and
current Hephaestus 0.25 without changing the reviewed Rust source. Native
provider acceptance does not clear Athena's affected graph, whole-cohort or
absolute application-performance review. Lifecycle remains held.
