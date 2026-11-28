# ADR 0004: Terminate on derived stagnation and divergence criteria

- Status: Accepted
- Date: 2026-08-14
- Class: [minor]
- Revised 2026-11-16: CG detects `Stagnated` when its true residual stalls
  at the attainable accuracy after the recurrence residual met the threshold
  (athena#167). The change is [major] for callers that match `Breakdown`.
- Revised 2026-11-16: the noise floor's `||A|| ||x|| ~ ||b||` premise is
  restated as an assumption that fails when `Ax` cancels; the CG stall test
  does not use the floor.
- Revised 2026-11-27: MINRES recomputes `b - A x` before `Converged` and,
  with no stall test derived for it, ends `MaxIterations` at the budget or
  `Breakdown` at a closed Krylov space; the stalled case stays unseparated
  for MINRES.

## Context

A stalled GMRES(m) run consumed its whole budget and reported
`Termination::MaxIterations`, indistinguishable from a solve that was
converging steadily and ran out of iterations. The two call for opposite
responses — raise the budget, versus change the preconditioner, the restart
width, or the method.

Stagnation is not an edge case for the restarted method. GMRES(m) minimises the
residual over the Krylov space of one cycle; when that space yields no
reduction, the next cycle starts from the same residual and rebuilds
essentially the same space. Saad and Schultz (1986), *GMRES: A generalized
minimal residual algorithm for solving nonsymmetric linear systems*, SIAM J.
Sci. Stat. Comput. 7(3), 856-869, §3.2, records the stall for the restarted
method, and Greenbaum, Pták and Strakoš (1996), *Any nonincreasing convergence
curve is possible for GMRES*, SIAM J. Matrix Anal. Appl. 17(3), 465-469, show
the stalling family is not pathological.

Any progress criterion needs a scale to compare two residuals against, and a
tuned tolerance would put an unjustified number on the termination path.

## Decision

Add `Termination::Stagnated` and `Termination::Diverged`, and one derived
quantity they are both measured against:

```text
residual_noise_floor(len, ||b||) = sqrt(len) * EPSILON * ||b||
```

This estimates the absolute accuracy of one explicitly recomputed
`||b - Ax||_2`. It is published as `athena_core::residual_noise_floor` with its
derivation and limits, because callers writing a custom convergence test need
the same number.

Two properties of the derivation matter:

- The floor is **absolute and scaled by `||b||`**, not relative to the current
  residual. The error of evaluating `b - Ax` is of order
  `u(|b - Ax| + N_A |A| |x|)` (Carson and Demmel, UCB/EECS-2012-197,
  eq. (4.9)-(4.10)), so `||b||` is the scale only while
  `|| |A| |x| || ~ ||b||`. When `Ax` cancels, the ratio
  `|| |A| |x| || / ||b||` exceeds 1 and the floor understates the evaluation
  error by that ratio. For symmetric positive definite `A` the ratio is at
  most `|| |A| ||_2 / lambda_min`, because `|| |A| |x| || <= || |A| ||_2 ||x||`
  and `||Ax|| >= lambda_min ||x||`. That is the condition number for the
  order-62 stiffness matrix, whose `|A|` and `A` share the largest eigenvalue
  `2 + 2 cos(pi / 63)`, and the ratio approaches it only as `x` aligns with
  the lowest eigenvector. At `t_i = i / 63` the ratio is 22 for `x = 1`,
  18 for `t`, 14 for `t^2`, 15 for `cos(pi t)`, 1.46e3 for `t (1 - t)`, the
  solution of the CG stall test, and 1.61e3 for `sin(pi t)`, against a
  condition number of 1.61e3 (evaluated in `f64`).
  Near convergence `||r|| << ||b||`, so a floor relative to `||r||` would
  understate the uncertainty by the factor `||b|| / ||r||`.
- The `len` dependence is `sqrt(len)`, the statistical form (Higham,
  *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §3.1; locators
  recalled), not the worst-case `len * u` of Lemma 3.1, which is attained only
  when every rounding error shares a sign; the norm's summands are squares of
  one sign. `len * u` would exceed 1 in `f32` at roughly 1.7e7 unknowns and
  declare every large single-precision system stagnant.

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

Rejected. `SolveError<E>` is parameterised by the backend error alone;
carrying a report in it would add a scalar parameter to a published crate
across every solver family. The control-flow rule reserves `Result::Err` for
contract failures, and which terminal condition a legitimate solve reached is
domain branching. The real defect, a caller never inspecting `termination`,
is closed by `#[must_use]` on `SolveReport` at no API cost; the three warm-up
call sites it flagged now assert convergence.

### Carry an owned residual history in the error or the report

Rejected. `IterationObserver` is the residual-history seam, and Athena never
allocates a history implicitly: recording one would allocate on the iteration
path and break the contract that warm solves touch no heap. A caller who wants
the history supplies an observer that stores it, as the stagnation tests do.

### Detect stagnation from a multi-cycle residual window

Rejected. A longer window needs a length no derivation fixes, and a cycle that
extracts nothing seeds the next with the same residual, so the first
unproductive cycle is evidence, not noise.

### Tune the thresholds against a benchmark suite

Rejected as the tuned-tolerance defect. A number chosen so that a chosen set of
systems terminates as desired encodes those systems, not the method.

## Consequences

A stalled GMRES solve now reports `Stagnated` at the first unproductive cycle
instead of `MaxIterations` at the budget: a behavioural change without a
signature break. Callers detecting non-convergence use `converged()`, which
covers every non-converged variant as new ones appear.

BiCGSTAB and LSQR keep their existing terminations. The two variants are
declared on the shared `Termination` enum because the concepts are shared, and
each family that detects one needs its own derivation — BiCGSTAB's residual is
not monotone, so the divergence argument above does not transfer to it.

CG detects `Stagnated` with its own scale. Once the recurrence residual meets
the threshold `tau`, a measured `||b - Ax|| - ||r|| > tau` bounds the gap
`(b - Ax) - r` above `tau` (reverse triangle inequality), and the recurrence
has no term that reduces that gap (Cools et al., arXiv:1601.07068, (2.11) and
(2.15); Greenbaum (1997), SIAM J. Matrix Anal. Appl. 18(3), 535-551, read from
its abstract only). The test needs only `tau` and two measured norms, not the
noise floor, whose `||b||` scale understates the evaluation error when `Ax`
cancels. A CG solve that reported `Breakdown` after iterating until a dot
product underflowed, or `MaxIterations` after a stall found before the budget
ended, now reports `Stagnated` at the first measurement that satisfies the
test; a budget ending at that measurement still reports `MaxIterations`, as
`DeflatedCg` does. A caller that keeps the iterate on `Breakdown` or
`MaxIterations` must treat `Stagnated` alike. Stopping at the first measurement
with `||b - Ax|| > tau` was rejected: it cuts off a solve whose gap is below
`tau` while `||r||` still accounts for the excess.

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
- `cg_contract::residual_stall` solves the order-62 stiffness system with a
  uniform load at relative tolerance `4 EPSILON` in `f32` and `f64` and asserts
  `Stagnated` with one measurement of `b - Ax`, both at a scheduled check and
  at a zero `<r, z>` or `<p, A p>`, and `||b - Ax||` within a first-order gap
  bound; a budget ending at the stall check reports `MaxIterations`.
