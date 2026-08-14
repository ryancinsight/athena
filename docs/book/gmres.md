# 5. GMRES

GMRES is the other branch of the choice the previous chapter set up: keep
optimality for a general nonsymmetric \\(A\\), and accept that the cost of a
step grows with the step number. It is the most reliable method here, and the
only one with a parameter you have to choose.

## Minimal residual

GMRES picks the iterate minimising the ordinary Euclidean residual over the
Krylov subspace:

\\[
\\|b - A x_k\\|_2 = \min_{x \in x_0 + \mathcal{K}_k} \\|b - A x\\|_2 .
\\]

Because the subspaces nest, the minimum over a larger one cannot be worse:
\\(\\|r_{k+1}\\| \le \\|r_k\\|\\) always. GMRES residuals never increase. That
monotonicity is the property everything else in this chapter leans on, and it
is the property restarting preserves and rounding threatens.

## Arnoldi

Minimising over a subspace requires a basis for it, and the obvious basis
\\(\\{r_0, A r_0, A^2 r_0, \dots\\}\\) is useless numerically: the vectors all
tilt toward the dominant eigenvector, so the basis becomes numerically rank
deficient within a few steps. The Arnoldi process builds an orthonormal basis
instead. Starting from \\(v_1 = r_0 / \\|r_0\\|\\), each step applies the
operator and orthogonalises against everything so far:

```text
w = A v_j
for i in 1..=j:
    h[i,j] = ⟨w, v_i⟩
    w = w − h[i,j] v_i
h[j+1,j] = ‖w‖
v_{j+1} = w / h[j+1,j]
```

The coefficients form an upper Hessenberg matrix \\(\bar H_k\\) satisfying

\\[
A V_k = V_{k+1} \bar H_k ,
\\]

and with \\(x = x_0 + V_k y\\) and \\(\beta = \\|r_0\\|\\) the minimisation
becomes a small least-squares problem in \\(y\\):

\\[
\min_y \\| \beta e_1 - \bar H_k y \\|_2 .
\\]

Its size is \\((k+1) \times k\\), independent of \\(n\\), and Athena solves it by
applying a Givens rotation per column to reduce \\(\bar H_k\\) to triangular
form. The rotations are applied as each column arrives, so the factorisation is
maintained incrementally, and the last transformed right-hand side entry is
\\(\\|r_k\\|\\) for free — which is the residual estimate the
[Convergence Policy](convergence_policy.md) chapter warns against trusting on
its own.

Athena uses *modified* Gram--Schmidt in the loop above: each subtraction is
applied to the running \\(w\\), rather than all coefficients being computed
against the original \\(w\\). Classical Gram--Schmidt loses orthogonality at a
rate proportional to \\(\kappa^2\\), modified at a rate proportional to
\\(\kappa\\). The Givens construction is likewise the scaled form, which avoids
overflow that evaluating \\(\sqrt{a^2 + b^2}\\) directly would produce for large
\\(a\\) or \\(b\\).

There is one clean exit. If \\(h_{j+1,j}\\) is exactly zero, \\(w\\) lay entirely
in the space already built: the Krylov space is invariant, no further basis
vector exists, and the current iterate is the exact solution within it. This is
a *happy* breakdown, and Athena reports it as `Termination::Breakdown` with a
converged residual.

## Restarting

The cost of step \\(k\\) is \\(k\\) inner products and \\(k\\) vector updates, and
step \\(k\\) needs \\(k\\) stored basis vectors. Both grow without bound, so full
GMRES is unusable on a large problem that needs many iterations. The standard
remedy is to cap the cycle: run \\(m\\) steps, form the solution, discard the
basis, and start again from the new residual. That is GMRES(\\(m\\)), and \\(m\\)
is the restart width — in Athena a const generic, `Gmres<B, RESTART>`, so the
workspace's storage shape and index arithmetic are fixed at compile time.

Restarting bounds storage at \\(O(mn)\\) and per-cycle work at \\(O(m^2 n)\\),
and it keeps the residual monotone: each cycle minimises over a space
containing the zero correction, so a cycle can never make things worse.

What it destroys is the global optimality. Full GMRES minimises over the Krylov
space of *all* steps taken; GMRES(\\(m\\)) minimises over one cycle at a time and
throws away the accumulated space at each restart. The information discarded is
often exactly the information convergence depended on.

## Stagnation

The consequence has a name. GMRES(\\(m\\)) can reduce the residual by nothing at
all, indefinitely, and the smallest example is exact rather than approximate.
Take the cyclic down-shift \\(A e_i = e_{i+1}\\) with \\(b = e_1\\) and
\\(x_0 = 0\\). Then \\(A b = e_2\\) is orthogonal to \\(b = e_1\\), so GMRES(1)
minimises \\(\\|b - \alpha A b\\|\\) at \\(\alpha = 0\\): the iterate does not
move, the residual stays at \\(\\|b\\| = 1\\), and the next cycle starts from the
same residual and does the same thing. It never converges, for any number of
cycles.

That is the mechanism in general. A cycle that extracts no reduction seeds the
next cycle with the same residual, which rebuilds essentially the same Krylov
space and reproduces the same non-reduction. Greenbaum, Pták and Strakoš (1996)
showed the phenomenon is not pathological: *any* nonincreasing curve, including
a completely flat one, is the GMRES convergence curve of some matrix with any
prescribed spectrum. Eigenvalues alone do not predict GMRES convergence.

Because the first unproductive cycle is already evidence, Athena tests for it
directly rather than waiting for the iteration budget to drain. After each
completed, unconverged cycle it compares the reduction against the accuracy of
the residual's own evaluation, \\(\eta = \sqrt{n}\varepsilon\\|b\\|\\) from the
[Convergence Policy](convergence_policy.md) chapter:

- a reduction of at most \\(\eta\\) is `Termination::Stagnated`;
- a residual exceeding the initial one by more than \\(\eta\\) is
  `Termination::Diverged`, since the minimisation property says that cannot
  happen and its loss means the recurrence is no longer doing what it claims.

Neither threshold is tuned; both are the noise floor of the measurement. The
comparison is per *cycle*, not per iteration, because the true residual is only
re-formed at a restart and the residual is monotone within a cycle by
construction, so an inner iteration carries no independent progress signal. The
window is therefore the restart width itself, with no window length to pick.

Getting `Stagnated` back means: change the preconditioner, or raise the restart
width, or use a different method. It does not mean raise the iteration cap.

## Choosing the restart width

There is no formula. The trade is that larger \\(m\\) converges in fewer cycles
and is less prone to stagnation, while costing \\(O(m)\\) storage and
\\(O(m^2)\\) work per cycle. Values between 20 and 50 are common defaults.
Raising \\(m\\) is also the first thing to try against stagnation, and full
GMRES — \\(m\\) at least the iteration budget — never stagnates, if you can
afford it.

The better lever is usually the preconditioner. A preconditioner good enough to
converge in a handful of iterations makes the restart width nearly irrelevant.

## Right preconditioning

Athena preconditions GMRES on the right, iterating on \\(A M^{-1} y = b\\) with
\\(x = M^{-1} y\\). This keeps the Arnoldi residual estimate in the *original*
system's norm, so the cheap per-iteration estimate is comparable with the
convergence policy's threshold. Left preconditioning would put the estimate in
the preconditioned norm, and comparing that against a physical-residual policy
would be dimensionally misleading; recomputing the physical residual at every
inner iteration instead would add an operator application and, on a GPU, a
synchronisation per step.

The basis is built from the preconditioned vectors \\(z_j = M^{-1} v_j\\), so the
solution update \\(x = x_0 + \sum_j y_j z_j\\) needs them kept. That is why the
workspace holds two vector blocks rather than one, and why they are separate
blocks: the Arnoldi step holds one vector from each simultaneously, and separate
blocks make their disjointness a fact about two fields rather than a runtime
check.

## In Athena

```text
Gmres<B, RESTART>              zero-sized marker; RESTART is the cycle width
GmresWorkspace<B, RESTART>     two vector blocks, residual and work vectors,
                               a packed (RESTART+1) × RESTART Hessenberg
                               matrix, Givens rotations, transformed residual
                               coordinates, backsolve coefficients, and one
                               prepared dot plan per basis vector
```

The scalar arrays are flat `Vec`s with an index function, and the basis is a
`VectorBlock` for the reasons in the
[Krylov Backend](krylov_backend.md) chapter. Everything is allocated at
workspace construction; warm solves allocate nothing.

## References

- Saad and Schultz (1986), *GMRES: A generalized minimal residual algorithm for
  solving nonsymmetric linear systems*, SIAM J. Sci. Stat. Comput. 7(3),
  856--869. The original, including §3.2 on the restarted method's stall.
- Greenbaum, Pták and Strakoš (1996), *Any nonincreasing convergence curve is
  possible for GMRES*, SIAM J. Matrix Anal. Appl. 17(3), 465--469.
- Barrett et al., *Templates*, §2.3.4 for restarted GMRES and §3.1.2 for
  preconditioning. <https://www.netlib.org/templates/templates.html>
- ADR 0002 and ADR 0004 in `docs/adr/` for Athena's GMRES contract and its
  progress criteria.
