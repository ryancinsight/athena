# 7. MINRES, Chebyshev and Deflated CG

Three more recurrences share the `LinearOperator` contract of the earlier
chapters. Each drops something the others rely on, which is the reason to
choose it.

## MINRES: symmetric but not positive definite

CG needs \\(A\\) positive definite because its step length divides by the
curvature \\(p^{\mathsf T} A p\\), which can vanish or change sign. MINRES
needs only \\(A = A^{\mathsf T}\\). It builds the Lanczos basis of
\\(\mathcal{K}_j(A, r_0)\\) with the symmetric three-term recurrence and, like
GMRES, minimises \\(\\|b - A x\\|_2\\) over that space. The projected least-squares
problem is tridiagonal, so an incremental Givens QR solves it, and the solution
enters through a two-term update recurrence. No basis is stored: a
`MinresWorkspace` holds seven vectors whatever the iteration count.

`Minres` takes no preconditioner. A preconditioner that preserves the method's
theory must enter as a symmetric positive-definite factorisation
\\(M = L L^{\mathsf T}\\), which is a different seam from
`Preconditioner::apply`; that seam is not built until a consumer needs it.
A zero Lanczos coefficient or a zero QR diagonal before the policy is met is
reported as `Termination::Breakdown`. The recurrence is from Paige and Saunders
(1975), SIAM J. Numer. Anal. 12(4), 617--629.

## Chebyshev: no inner products

Chebyshev semi-iteration takes damped Richardson steps whose coefficients come
from Chebyshev polynomials of the interval \\([\lambda_{\min}, \lambda_{\max}]\\)
that contains the spectrum of an SPD operator. After \\(k\\) steps the error in
the operator's norm is at most \\(1/T_k(\mu)\\) of its start, with
\\(\mu = (\lambda_{\max} + \lambda_{\min})/(\lambda_{\max} - \lambda_{\min})\\).
The recurrence needs no inner product, so nothing in it forces a reduction
across processors; the residual norm is still computed each iteration for the
convergence test.

The cost is that the caller supplies the interval. `SpectralBounds::new`
rejects a non-finite, non-positive or inverted interval with a typed error, and
accepts equal bounds: then \\(A = \lambda I\\) and one step of scale
\\(1/\lambda\\) solves the system in exact arithmetic. The coefficients are
tracked through \\(\eta = 1/\mu \in [0, 1)\\) so that they stay bounded without
forming \\(T_k(\mu)\\). Storage is three vectors. The construction follows
Golub and Van Loan, *Matrix Computations*, section 11.3.

## Deflated CG: removing the slow directions

A few small eigenvalues make CG slow. If an orthonormal basis \\(U\\) spans
directions that carry them, CG can run on the complement and the coarse part
can be solved exactly. With \\(E = U^{\mathsf T} A U\\) and
\\(C = U E^{-1} U^{\mathsf T}\\), `DeflatedCg` solves
\\((A - A C A)\,y = (I - A C)\,r_0\\) by CG and recovers
\\(x = x_0 + y + C(r_0 - A y)\\). Because the Schur complement carries the coupling
between the coarse and complement coordinates, \\(U\\) need not be invariant under
\\(A\\). If it is an exact invariant subspace of \\(k\\) eigenvectors, the CG phase
terminates in at most \\(n - k\\) iterations.

`DeflationSubspace::from_vectors` orthonormalises the input vectors, caches \\(A U\\) and
factorises \\(E\\) once; the subspace must be used with the operator it was built
from. The final report checks the true residual \\(b - A x\\) against the original
right-hand-side policy, not the Schur residual.
