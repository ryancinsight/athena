# ADR 0004: Terminate on derived stagnation and divergence criteria

- Status: Accepted
- Date: 2026-08-14
- Class: [minor]

## Context

Restarted GMRES had two ways to stop short of the solution: the residual
threshold and the iteration cap. Neither describes a solve that has stopped
making progress. A stalled GMRES(m) run consumed its whole budget and reported
`Termination::MaxIterations`, which is indistinguishable from a solve that was
converging steadily and simply ran out of iterations. The two call for opposite
responses — raise the budget, versus change the preconditioner, the restart
width, or the method — and the report could not tell them apart.

Stagnation is not an edge case for the restarted method. GMRES(m) minimises the
residual over the Krylov space of one cycle; when that space yields no
reduction, the next cycle starts from the same residual and rebuilds
essentially the same space. Saad and Schultz (1986), *GMRES: A generalized
minimal residual algorithm for solving nonsymmetric linear systems*, SIAM J.
Sci. Stat. Comput. 7(3), 856-869, §3.2, records the stall for the restarted
method, and Greenbaum, Pták and Strakoš (1996), *Any nonincreasing convergence
curve is possible for GMRES*, SIAM J. Matrix Anal. Appl. 17(3), 465-469, show
the stalling family is not pathological.

Any progress criterion needs a scale to compare two residuals against. Adding a
tuned tolerance for it would put an unjustified magic number on the solver's
termination path, which is exactly what the repository's numerical contracts
forbid.

## Decision

Add `Termination::Stagnated` and `Termination::Diverged`, and one derived
quantity they are both measured against:

```text
residual_noise_floor(len, ||b||) = sqrt(len) * EPSILON * ||b||
```

This is the absolute accuracy of one explicitly recomputed `||b - Ax||_2`. It
is published as `athena_core::residual_noise_floor` with its full derivation
and its limits at the definition site, rather than inlined, because it is the
scale that makes any residual comparison meaningful and callers writing a
custom convergence test need the same number.

Two properties of the derivation matter:

- The floor is **absolute and scaled by `||b||`**, not relative to the current
  residual. The error of evaluating `b - Ax` is `O(u(||b|| + ||A|| ||x||))`,
  and `||A|| ||x|| ~ ||b||` for an iterate near the solution (Higham,
  *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §7.1). Near
  convergence `||r|| << ||b||`, so a floor relative to `||r||` would understate
  the uncertainty by the factor `||b|| / ||r||`.
- The `len` dependence is `sqrt(len)`, the statistical form (ibid. §3.1), not
  the worst-case `len * u` of Lemma 3.1. The worst case is attained only when
  every rounding error shares a sign, and the norm's summands are squares of
  one sign, which removes the cancellation it relies on. Using `len * u` would
  exceed 1 in `f32` at roughly 1.7e7 unknowns and declare every large
  single-precision system stagnant while its residual still carried several
  correct digits.

GMRES classifies a completed, unconverged restart cycle after the convergence,
breakdown, and non-finite tests:

- `Diverged` when `||r_k|| > ||r_0|| + noise`. Every cycle minimises over a
  space containing the zero correction, so the residual cannot exceed the
  initial one in exact arithmetic, and each cycle re-forms `b - Ax` explicitly
  rather than by recurrence, so evaluation error does not accumulate across
  cycles and the flat one-evaluation bound applies.
- `Stagnated` when the cycle reduced the residual by at most `noise`. The
  reduction is measured per cycle, not per iteration: the true residual is only
  re-formed at a restart, and within a cycle the residual is monotone
  non-increasing by construction, so an inner iteration carries no independent
  progress signal. The window is therefore the restart width itself, with no
  tuned window length.

`SolveReport` becomes `#[must_use]`.

## Rejected alternatives

### Return `Err` for every non-converged termination

Rejected. It reads as the stronger guarantee, but it is the wrong shape here
and would cost more than it buys.

`SolveError<E>` is parameterised by the backend error alone. Carrying a report
or a residual history in it would add a scalar parameter, `SolveError<T, E>`,
to a published crate, and would do so across all four solver families for a
change scoped to one. More substantially, the repository's control-flow rule
reserves `Result::Err` for contract failures and requires domain enums for
domain branching, and which terminal condition a legitimate solve reached —
converged, budget exhausted, stagnated, broke down — is domain branching. A
caller that asks for at most 32 iterations and receives 32 iterations' worth of
progress has not suffered a failed call.

The real defect the proposal targets is narrower: a caller could bind the
report and never inspect `termination`. `#[must_use]` on `SolveReport` closes
exactly that, at no API cost, and the three warm-up call sites it flagged were
strengthened to assert convergence rather than discard the report.

### Carry an owned residual history in the error or the report

Rejected. `IterationObserver` is already the residual-history seam, and its
contract is that Athena never allocates a history implicitly. Recording one
inside the solve would allocate on the iteration path in an `no_std + alloc`
crate and break the allocation-stability contract that warm solves touch no
heap. A caller who wants the history supplies an observer that stores it; a
caller who does not pays nothing. The stagnation test uses exactly this seam.

### Detect stagnation from a multi-cycle residual window

Rejected. A window longer than one cycle needs a length, and no derivation
fixes one. The single-cycle test is already sound for the restarted method: a
cycle that extracts nothing seeds the next cycle with the same residual, so the
first unproductive cycle is evidence, not noise.

### Tune the thresholds against a benchmark suite

Rejected as the tuned-tolerance defect. A number chosen so that a chosen set of
systems terminates as desired encodes those systems, not the method.

## Consequences

A GMRES solve that previously reported `MaxIterations` after stalling now
reports `Stagnated` at the first unproductive cycle, so it stops earlier and
with a different terminal value. This is a behavioural change to the public
contract even though it breaks no signature: a caller matching only on
`MaxIterations` to detect "did not converge" must use `converged()`, which
already covers every non-converged variant and needs no change as new ones
appear.

CG, BiCGSTAB, and LSQR keep their existing terminations. The two new variants
are declared on the shared `Termination` enum because the concepts are shared,
but only GMRES detects them; extending detection to the other families is
separate work, and each needs its own derivation — BiCGSTAB's residual is not
monotone, so the divergence argument above does not transfer to it.

## Verification

- A cyclic down-shift operator with `b = e_1` and restart width 1 stagnates
  exactly: `A b` is orthogonal to `b`, so the cycle minimiser is `alpha = 0`
  and the residual stays at `||b||`. The test asserts `Stagnated`, termination
  at the first cycle rather than at the 32-iteration budget, and a non-empty
  observed residual history pinned at 1.
- A companion test on a system GMRES(1) does reduce asserts `Converged`, so the
  detector is not a false positive that merely agrees with the stalling case.
- The pre-existing budget, breakdown, initial-residual, restart, preconditioned
  and `f32`/`f64` GMRES cases are unchanged and still pass, as is the
  allocation-stability suite.
- Unit tests pin the noise floor's `sqrt(len)` scaling, its linearity in
  `||b||`, and that it stays resolvable at 1e8 single-precision unknowns where
  the worst-case form would not.
