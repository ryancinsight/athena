# 1. Convergence Policy

An iterative solver produces a sequence, not an answer. Deciding which element
of that sequence to hand back is a separate question from computing it, and
this chapter is about that question alone. It comes first because every solver
in Part II shares one stopping rule.

## What you can and cannot measure

What you want to be small is the **error** \\(e_k = x_\star - x_k\\). You cannot
compute it, because \\(x_\star\\) is what you are looking for. What you can
compute is the **residual** \\(r_k = b - A x_k\\), and the two are related by

\\[
e_k = A^{-1} r_k .
\\]

Taking norms and inserting \\(b = A x_\star\\) gives the bound that governs every
convergence test:

\\[
\frac{\\|e_k\\|}{\\|x_\star\\|} \le \kappa(A)\, \frac{\\|r_k\\|}{\\|b\\|},
\qquad \kappa(A) = \\|A\\|\\,\\|A^{-1}\\| .
\\]

A small residual therefore guarantees a small error only to the extent that
\\(A\\) is well conditioned. On a problem with \\(\kappa(A) = 10^8\\), a relative
residual of \\(10^{-10}\\) buys two correct digits, not ten. This is not a defect
of the solver; a residual is a *backward* error statement — \\(x_k\\) solves a
nearby problem exactly — and turning it into a forward error costs a factor of
\\(\kappa\\). Athena reports residuals, and this inequality is how you translate
one into a claim about your solution.

## The threshold

`ConvergencePolicy` combines an absolute and a relative tolerance:

\\[
\text{converged} \iff \\|r_k\\|_2 \le \max(\tau_{\text{abs}},\ \tau_{\text{rel}} \\|b\\|_2).
\\]

Both terms are needed and each covers the other's failure. A purely relative
test is scale invariant, which is what you normally want, but it is unreachable
when \\(b = 0\\): the threshold collapses to zero and no floating-point residual
attains it. A purely absolute test is a statement in the units of your problem,
which makes it meaningful for a physical residual but wrong the moment someone
rescales the system — the same solve in millimetres and metres would stop at
different accuracies. Taking the larger of the two gives a test that tracks the
problem's scale and still terminates on the degenerate case.

The policy is validated at construction: negative or non-finite tolerances, a
zero iteration budget, and a zero check interval are rejected with a specific
`InvalidConvergencePolicy` reason rather than producing a solver that cannot
stop. See [Example: Convergence Policy](examples/convergence_policy.md).

## Estimated and true residuals

Some methods carry a residual estimate that falls out of the recurrence for
free. GMRES is the clearest case: its Givens rotations leave the norm of the
minimised residual sitting in a scalar, at no extra cost. Trusting it is
tempting and wrong. The estimate is the residual of the *recurrence*, which
drifts from the residual of the *system* as rounding accumulates, and the drift
is in the optimistic direction — the estimate keeps falling after the true
residual has levelled off.

Athena therefore recomputes \\(b - A x\\) explicitly before declaring
convergence, in every solver. The estimate is still used, for the inner loop:
recomputing costs an extra operator application and, on a GPU, a
device-to-host synchronisation to read the scalar back, so it is done at
restart boundaries and at prospective convergence rather than every iteration.
The `check_interval` on the policy exposes that trade to the caller.

## When the residual stops falling

A threshold and an iteration cap describe two of the ways a solve can end.
There is a third, and it is common enough on hard problems to deserve its own
terminal value: the residual stops improving while remaining above the
threshold. Reporting that as "budget exhausted" would conflate it with a solve
that was converging steadily and merely needed more iterations, and the two
call for opposite responses — raise the cap, versus change the preconditioner
or the method.

Detecting it needs a scale. Two residuals differing by less than the accuracy
with which either was computed have not been shown to differ at all, so the
scale is the accuracy of the residual evaluation itself. Athena publishes it as
`residual_noise_floor`:

\\[
\eta = \sqrt{n}\, \varepsilon\, \\|b\\|_2 .
\\]

Two features of that expression are deliberate.

It is **absolute, and proportional to \\(\\|b\\|\\)** rather than to the current
residual. Forming \\(b - A x\\) in floating point incurs an error of order
\\(u(\\|b\\| + \\|A\\|\\|x\\|)\\), and \\(\\|A\\|\\|x\\| \approx \\|b\\|\\) once the
iterate is near the solution, so \\(\\|b\\|\\) is the scale (Higham, *Accuracy and
Stability of Numerical Algorithms*, 2nd ed., §7.1). Near convergence
\\(\\|r\\| \ll \\|b\\|\\), so a floor expressed relative to \\(\\|r\\|\\) would
understate the uncertainty by exactly the factor \\(\\|b\\| / \\|r\\|\\).

It uses \\(\sqrt{n}\\), not \\(n\\). The worst-case bound for summing \\(n\\)
terms sequentially is \\(nu\\) (ibid., Lemma 3.1), but that is attained only when
every rounding error carries the same sign, and the summands inside a Euclidean
norm are squares of one sign, which removes the cancellation the worst case
relies on. The statistical \\(\sqrt{n}u\\) of ibid. §3.1 is used instead. The
practical difference is decisive: at \\(n = 10^8\\) in single precision the
worst-case form gives \\(n\varepsilon \approx 12\\), a "floor" larger than any
residual, which would declare every large single-precision system stagnant
while its residual still carried three correct digits.

This is an estimate of the evaluation error, not a bound on it. An adversarial
rounding pattern can exceed it by up to \\(\sqrt{n}\\), and the derivation
assumes the backend accumulates in the scalar's own precision. A backend that
accumulates pairwise is bounded more tightly, so the floor stays conservative
for it.

## The terminal conditions

`Termination` is the complete list of ways a solve ends. `converged()` is true
for exactly the first three.

| Value | Meaning |
| --- | --- |
| `InitialResidual` | the initial guess already met the threshold |
| `Converged` | a recomputed residual met the threshold |
| `NormalEquations` | LSQR found the least-squares optimum of an inconsistent system |
| `MaxIterations` | the budget ran out while still making progress |
| `Stagnated` | a restart cycle reduced the residual by at most \\(\eta\\) |
| `Diverged` | the residual exceeded its initial value by more than \\(\eta\\) |
| `Breakdown` | a recurrence denominator was numerically zero |
| `NonPositiveCurvature` | CG met a direction violating its SPD contract |
| `NonFinite` | a residual or recurrence coefficient became non-finite |

None of these is an error in the `Result` sense. Which condition a legitimate
solve reached is domain information the caller branches on, so it is returned in
the `SolveReport` value; `SolveError` is reserved for a dimension mismatch or a
backend failure, which mean the call itself could not be carried out. The report
is `#[must_use]`, because discarding it is the one way a non-converged
termination gets accepted silently.

Branch on `converged()` rather than on a specific variant when the question is
"did this work". New terminal conditions get added — `Stagnated` and `Diverged`
are recent — and `converged()` classifies them correctly without a change at the
call site.

## References

- Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM,
  2002, §3.1 (summation error) and §7.1 (residual bounds for computed
  solutions).
- Barrett et al., *Templates for the Solution of Linear Systems*, §4.2, on
  stopping criteria and the condition-number factor between residual and error.
  <https://www.netlib.org/templates/templates.html>
- ADR 0004, `docs/adr/0004-derived-progress-termination.md`, for the stagnation
  and divergence decision and the alternatives rejected.
