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
