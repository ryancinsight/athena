# 4. BiCGSTAB

CG's short recurrence came from symmetry. Drop symmetry and you face a choice
that has no good answer: keep the optimality and pay storage that grows every
step, which is GMRES, or keep the fixed-cost recurrence and give up optimality,
which is this chapter. BiCGSTAB is the second branch, and it is usually the
first thing to try on a nonsymmetric system because it costs a constant amount
of memory.

## Where the short recurrence comes from

CG built \\(A\\)-orthogonal directions cheaply because symmetry made all but one
Gram--Schmidt coefficient vanish. Without symmetry there is no such inner
product. BiCG recovers a short recurrence by orthogonalising against a
*second*, shadow Krylov space built from \\(A^{\mathsf T}\\): it maintains two
sequences, \\(r_k \in \mathcal{K}_k(A, r_0)\\) and
\\(\tilde r_k \in \mathcal{K}_k(A^{\mathsf T}, \tilde r_0)\\), made mutually
orthogonal. Bi-orthogonality between two spaces is enough to give three-term
recurrences, so the cost per step is again fixed.

What it is not is a minimisation. Nothing about \\(x_k\\) is optimal over
\\(\mathcal{K}_k\\); it merely satisfies a Petrov--Galerkin condition. The
residual norm consequently does not decrease monotonically, and BiCG's
convergence curve is famously erratic.

BiCG also needs \\(A^{\mathsf T} v\\), which is a genuine practical problem: a
matrix-free operator that applies a stencil or a finite-element assembly often
has no transpose available at all.

## The stabilised variant

BiCGSTAB, due to van der Vorst (1992), removes both objections. It rewrites the
BiCG polynomial so the transpose is never needed, and it composes the BiCG step
with a local one-dimensional residual minimisation. Where BiCG produces a
residual \\(q_k(A) r_0\\), BiCGSTAB produces

\\[
r_k = \psi_k(A)\, q_k(A)\, r_0, \qquad \psi_k(t) = \prod_{j=1}^{k} (1 - \omega_j t),
\\]

with each \\(\omega_j\\) chosen to minimise the norm of the residual at that
step. Each factor is a steepest-descent step applied on top of the BiCG step,
which is where the "stabilised" comes from and why the convergence curve is
smoother than BiCG's. "Smoother" is the accurate word: it is still not monotone,
and BiCGSTAB residuals do go up.

The cost is two operator applications and two preconditioner applications per
iteration, against GMRES's one — but with storage that does not grow, and no
restart parameter to choose.

## The two breakdowns

The two-term structure has two denominators, and each can vanish.

\\(\rho_k = \langle \tilde r_0, r_k \rangle\\) is the bi-orthogonality
coefficient. It reaching zero means the shadow space has become orthogonal to
the residual and the bi-orthogonal basis cannot be continued. This is the
*serious* breakdown inherited from BiCG. It is rare with a random
\\(\tilde r_0\\) but structurally possible, and no amount of arithmetic care
prevents it.

\\(\omega_k\\) is the local minimisation coefficient. It vanishes when the
steepest-descent step has nothing to remove, which stalls the stabilising
factor.

Athena reports both as `Termination::Breakdown` rather than continuing with a
division by a numerically zero denominator. A breakdown is not necessarily a
failure of the problem: restarting from the current iterate with a different
shadow vector often continues, and switching to GMRES always does. What it is
not is something to paper over with a tolerance — the criterion is exact zero,
not a tuned threshold, and the report says which recurrence stopped.

## Preconditioning and the residual

Athena preconditions BiCGSTAB on the right, so the recurrence residual is the
residual of the original system and the convergence policy applies to it
directly. Left preconditioning would instead measure
\\(\\|M^{-1}(b - Ax)\\|\\), which differs from the true residual by up to
\\(\kappa(M)\\) — the policy would then be testing a quantity the caller did not
ask about.

Because the residual is non-monotone, a recurrence residual dipping below the
threshold is not sufficient evidence: it can dip and come back. Athena
recomputes \\(b - A x\\) explicitly before declaring convergence, as it does in
every solver, and only the recomputed norm can produce
`Termination::Converged`.

## When to use which

| Situation | Method |
| --- | --- |
| SPD operator | CG — nothing else is competitive |
| Nonsymmetric, memory tight | BiCGSTAB |
| Nonsymmetric, BiCGSTAB breaks down or oscillates badly | GMRES |
| Nonsymmetric, a good preconditioner makes convergence fast | GMRES, whose restart width can then be small |
| Rectangular, or square and inconsistent | LSQR |

The honest summary is that BiCGSTAB is the cheap first attempt and GMRES is the
reliable fallback. BiCGSTAB has no restart parameter to tune, which is a real
advantage; it can also fail on problems GMRES handles, which is a real cost.

## In Athena

```text
BiCgStab<B>              zero-sized algorithm marker
BiCgStabWorkspace<B>     the two-term recurrence's vectors and prepared
                         reductions; allocated once
```

The interface matches CG's: `solve_into`, `solve_with_observer`, a caller-owned
solution vector, and a reusable workspace whose warm solves allocate nothing.

## References

- van der Vorst (1992), *Bi-CGSTAB: a fast and smoothly converging variant of
  Bi-CG for the solution of nonsymmetric linear systems*, SIAM J. Sci. Stat.
  Comput. 13(2), 631--644.
- Barrett et al., *Templates*, §2.3.7, for the preconditioned algorithm.
  <https://www.netlib.org/templates/templates.html>
- Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., §7.4, for the
  BiCG family and its breakdown conditions.
