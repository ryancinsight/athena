# 3. Conjugate Gradient

CG is the method to use when \\(A\\) is symmetric positive definite, and it is
worth understanding first because it is the one case where everything works
out: the iterate is optimal, the work per step is constant, and the convergence
rate has a clean bound. The other three methods in this book are all attempts
to keep some of that when symmetry is lost.

## The energy norm

A symmetric positive definite \\(A\\) defines an inner product,
\\(\langle u, v \rangle_A = u^{\mathsf T} A v\\), and its norm
\\(\\|u\\|_A = \sqrt{u^{\mathsf T} A u}\\). CG chooses the iterate in
\\(x_0 + \mathcal{K}_k\\) minimising the error in *this* norm:

\\[
\\|x_\star - x_k\\|_A = \min_{x \in x_0 + \mathcal{K}_k} \\|x_\star - x\\|_A .
\\]

The energy norm is not an arbitrary choice made because it is convenient. For
\\(A x = b\\) arising from a discretised elliptic problem, \\(\\|e\\|_A^2\\) is
the energy of the error, and minimising it is the discrete form of the
variational principle the continuous problem satisfies. It is also the norm
that makes the algorithm possible, which is the next point.

Minimising over a growing subspace normally means solving a least-squares
problem of growing size — that is exactly what GMRES does, and what makes GMRES
expensive. CG avoids it. The minimiser is characterised by the error being
\\(A\\)-orthogonal to the subspace, so if the search directions
\\(p_0, \dots, p_{k-1}\\) are built \\(A\\)-orthogonal to each other, each step
is independent of the others and the minimisation decouples into one scalar
per direction. Building the next \\(A\\)-orthogonal direction would in general
require orthogonalising against all previous ones, but here \\(A\\) is symmetric,
so the Gram--Schmidt coefficients against everything except the immediately
previous direction vanish identically.

That is the whole trick, and it is why CG is short. It needs three vectors and
a fixed number of operations per iteration, regardless of how many iterations
have passed.

## The recurrence

With right preconditioner \\(M\\), one iteration is:

```text
z = M⁻¹ r                      preconditioner application
β = ⟨r, z⟩ / ⟨r_prev, z_prev⟩  (β = 0 on the first step)
p = z + β p                    direction recurrence
q = A p                        operator application
α = ⟨r, z⟩ / ⟨p, q⟩            step length
x = x + α p                    solution update
r = r − α q                    residual update
```

Athena fuses the last two into `fused_cg_update`, since they share a traversal
and a backend can dispatch them together. The direction recurrence is
`combine_direction`. Both are members of `KrylovBackend` for that reason and no
other.

The denominator \\(\langle p, q \rangle = p^{\mathsf T} A p\\) is the curvature of
the quadratic along the search direction. Positive definiteness says it is
positive for every nonzero \\(p\\); if the computed value is not, the operator
supplied is not what CG was promised. Athena stops there and reports
`Termination::NonPositiveCurvature` rather than continuing with a step length of
the wrong sign. This is the most common way a CG solve goes wrong in practice,
and it is almost always a real asymmetry or indefiniteness in the operator or
the preconditioner, not a rounding artefact. A preconditioner must be SPD too:
Jacobi is SPD only when the diagonal is positive, and an incomplete
factorisation of an indefinite matrix is not SPD at all.

## How fast

The polynomial picture from the introduction gives the classical bound. Writing
\\(\kappa = \kappa_2(A) = \lambda_{\max} / \lambda_{\min}\\),

\\[
\\|x_\star - x_k\\|_A \le 2 \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^{k} \\|x_\star - x_0\\|_A .
\\]

Two things to take from it. The rate depends on \\(\sqrt{\kappa}\\), not
\\(\kappa\\) — which is why CG beats simple iterative schemes on the same
problem — and reducing the error by a fixed factor takes
\\(O(\sqrt{\kappa})\\) iterations. For a second-order elliptic problem on a mesh
of spacing \\(h\\), \\(\kappa = O(h^{-2})\\), so unpreconditioned CG needs
\\(O(h^{-1})\\) iterations and refining the mesh makes the solve worse in two
ways at once. That is the whole motivation for preconditioning: an effective
\\(M\\) makes the iteration count nearly independent of \\(h\\).

The bound is also pessimistic in a specific and useful way. It uses only the
extremes of the spectrum, whereas CG actually responds to the full eigenvalue
distribution: a matrix with a few outliers and the rest tightly clustered
converges far faster than \\(\kappa\\) suggests, because a low-degree polynomial
can put a root at each outlier and be small on the cluster. Observed
convergence beating the bound is normal, not evidence of a bug.

## Finite precision

In exact arithmetic the search directions stay \\(A\\)-orthogonal and CG
terminates in at most \\(n\\) steps. In floating point that orthogonality is
lost, and the loss is not gentle: Paige showed for the closely related Lanczos
process that orthogonality is lost precisely as eigenvalues converge, and the
practical consequence is *delay* — CG takes more iterations than the bound
predicts, and may take more than \\(n\\). It still converges. Treating CG as a
direct method that stops at step \\(n\\) is the mistake; it is an iterative
method with a finite-termination property that finite precision removes.

This is also why Athena recomputes \\(b - A x\\) before declaring convergence
rather than trusting the recurrence residual \\(r\\), which drifts from the true
residual for the same reason.

## In Athena

```text
Cg<B>                    zero-sized algorithm marker
CgWorkspace<B>           residual, direction, image, preconditioned residual,
                         and prepared reductions; allocated once
```

`Cg::solve_into` takes the backend, operator, preconditioner, right-hand side,
a caller-owned solution vector, the workspace, and the policy, and returns a
`SolveReport`. `Cg::solve_with_observer` additionally reports each checked
residual, which is how you obtain a convergence history.

Terminal conditions specific to CG are `NonPositiveCurvature`, above, and
`Breakdown` when \\(\langle r, z \rangle\\) reaches exact zero without the
residual having met the threshold. The rest are shared and listed in the
[Convergence Policy](convergence_policy.md) chapter.

See [Example: CG Solver](examples/cg_solver.md) for a complete program.

## References

- Hestenes and Stiefel (1952), *Methods of Conjugate Gradients for Solving
  Linear Systems*, Journal of Research of the National Bureau of Standards
  49(6), 409--436. The original.
  <https://nvlpubs.nist.gov/nistpubs/jres/049/jresv49n6p409_a1b.pdf>
- Barrett et al., *Templates*, §2.3.1, for the preconditioned algorithm Athena
  implements. <https://www.netlib.org/templates/templates.html>
- Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., §6.7, for the
  derivation of the convergence bound.
- Paige (1976), *Error analysis of the Lanczos algorithm for tridiagonalizing a
  symmetric matrix*, IMA J. Applied Mathematics 18(3), 341--349, for the
  finite-precision loss of orthogonality.
