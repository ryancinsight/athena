# athena -- Iterative Solvers for Atlas

`athena` solves \\(A x = b\\) when \\(A\\) is too large to factor.

This book is written for someone meeting Krylov methods for the first time. It
builds the mathematics before the API: Part I covers what it means for an
iterative solve to be finished and what a solver needs from the machine it runs
on, Part II derives the four solver families Athena ships and says which
problem each one is for, and Part III places the crate in the Atlas stack.

## Why iterate at all

Gaussian elimination solves \\(A x = b\\) exactly in \\(O(n^3)\\) operations and
\\(O(n^2)\\) storage. For the matrices this crate targets that is not a slow
answer, it is no answer: a three-dimensional grid of \\(200^3\\) cells gives
\\(n = 8 \times 10^6\\), so the factors alone would need \\(6 \times 10^{13}\\)
numbers. The matrix itself is nowhere near that large, because a finite
difference or finite element discretisation couples each cell only to its
neighbours — a few nonzeros per row, tens of millions of nonzeros in total.
What kills direct methods is *fill-in*: elimination puts nonzeros where the
original matrix had structural zeros, and the factors of a sparse matrix are
generally dense.

Iterative methods never form a factorisation. They only ever ask the matrix one
question, and always the same one:

> given a vector \\(v\\), what is \\(A v\\)?

That question costs one pass over the nonzeros. Everything else a Krylov method
does — inner products, vector updates, norms — is linear in \\(n\\). So the
storage stays at the size of the matrix plus a handful of vectors, and the cost
becomes proportional to the number of iterations. The bargain is that you no
longer get the exact answer; you get a sequence of approximations and stop when
one is good enough. Deciding what "good enough" means, and detecting when the
sequence has stopped improving, is a large part of what this crate is about.

This is why the operator is a *trait* in Athena rather than a matrix type. A
solver needs a linear map, not a stored array: an operator that applies a
stencil, or a finite element assembly, or a matrix already resident in GPU
memory serves the recurrence equally well.

## The Krylov subspace

Start from a guess \\(x_0\\) and its residual \\(r_0 = b - A x_0\\). The only
things a solver can build from \\(A\\) and \\(r_0\\) using matrix-vector
products are

\\[
r_0,\quad A r_0,\quad A^2 r_0,\quad \dots
\\]

Their span after \\(k\\) products is the **Krylov subspace**

\\[
\mathcal{K}_k(A, r_0) = \operatorname{span}\\{ r_0, A r_0, \dots, A^{k-1} r_0 \\}.
\\]

Every method in this book draws its \\(k\\)-th iterate from the affine set
\\(x_0 + \mathcal{K}_k(A, r_0)\\). That is not a restriction the methods impose
on themselves; it is everything reachable with \\(k\\) applications of the
operator.

Two facts make the subspace a good place to look. First, it grows to contain
the answer: by the Cayley--Hamilton theorem \\(A^{-1}\\) is a polynomial in
\\(A\\) of degree less than \\(n\\), so the exact solution lies in
\\(x_0 + \mathcal{K}_n(A, r_0)\\), and a Krylov method is a direct method that
terminates in at most \\(n\\) steps in exact arithmetic. Second, and this is the
part that matters, it usually contains an excellent approximation long before
\\(k\\) reaches \\(n\\). Choosing \\(x_k \in x_0 + \mathcal{K}_k\\) is the same
as choosing a polynomial \\(p\\) of degree \\(< k\\) with
\\(x_k = x_0 + p(A) r_0\\), which makes the residual

\\[
r_k = (I - A p(A))\, r_0 = q(A)\, r_0, \qquad q(0) = 1 .
\\]

So the whole question is how small a degree-\\(k\\) polynomial with
\\(q(0) = 1\\) can be made on the spectrum of \\(A\\). When the eigenvalues are
clustered away from the origin, a low-degree polynomial is small on all of them
at once and convergence is fast. When they are spread over many orders of
magnitude, or straddle the origin, no low-degree polynomial is uniformly small
and convergence is slow. This single picture explains the condition-number
bounds in the CG chapter, why clustering the spectrum is exactly what a
preconditioner is for, and why the stagnation in the GMRES chapter is possible
at all.

The methods differ in *which* element of the subspace they pick and how they
find it cheaply:

| Method | Requires | Picks the iterate that | Cost per step |
| --- | --- | --- | --- |
| CG | symmetric positive definite \\(A\\) | minimises \\(\\|x - x_\star\\|_A\\) | fixed: three vectors |
| GMRES | any nonsingular \\(A\\) | minimises \\(\\|b - A x\\|_2\\) | grows: \\(k\\) vectors at step \\(k\\) |
| BiCGSTAB | any nonsingular \\(A\\) | satisfies a two-sided condition | fixed, but no minimisation |
| LSQR | any \\(A\\), including rectangular | minimises \\(\\|b - A x\\|_2\\) over all \\(x\\) | fixed |

CG gets both an optimal iterate and constant work because symmetry collapses
the orthogonalisation to three terms — that is the content of the short
recurrence in the CG chapter. GMRES keeps optimality for general \\(A\\) and
pays with storage that grows every step, which is why it must be restarted.
BiCGSTAB keeps the constant work and gives up optimality, which is why its
residual is not monotone. LSQR is the method for problems that have no solution
in the ordinary sense.

## Preconditioning

If convergence depends on where the eigenvalues sit, the practical lever is to
move them. A **preconditioner** is an approximation \\(M \approx A\\) that is
cheap to invert, applied so that the method iterates on a better-conditioned
operator. Athena uses *right* preconditioning throughout:

\\[
A M^{-1} y = b, \qquad x = M^{-1} y .
\\]

The alternative, left preconditioning, iterates on \\(M^{-1}A x = M^{-1}b\\) and
therefore measures \\(\\|M^{-1} r\\|\\) rather than \\(\\|r\\|\\). Those differ by
up to the condition number of \\(M\\), so a left-preconditioned solver's
convergence test is not a statement about the problem you asked about. Right
preconditioning leaves the residual of the *original* system in the recurrence,
which is why one convergence policy in this crate applies unchanged whether a
preconditioner is present or not.

The two extremes bracket the trade: \\(M = I\\) costs nothing and changes
nothing, \\(M = A\\) converges in one step and is the original problem. Useful
preconditioners live in between, and Athena ships several in the
[Krylov Backend](krylov_backend.md) chapter's provider crate: Jacobi (the
diagonal), incomplete LU, and successive over-relaxation.

## What the code looks like

The whole surface is four pieces: an **operator** supplying \\(A v\\), a
**preconditioner** supplying \\(M^{-1} v\\), a **convergence policy** deciding
when to stop, and a **workspace** holding the vectors so that a repeated solve
allocates nothing. A solver is a zero-sized marker type that drives them.

See [Example: CG Solver](examples/cg_solver.md) for the smallest complete
program, and [Example: Convergence Policy](examples/convergence_policy.md) for
the stopping rule on its own.

## References

The chapters cite specific results where they use them. Three general sources
cover the whole book:

- Yousef Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM,
  2003. The standard graduate text; freely available from the author.
- Richard Barrett et al., *Templates for the Solution of Linear Systems:
  Building Blocks for Iterative Methods*, SIAM, 1994.
  <https://www.netlib.org/templates/templates.html>. Athena's recurrences
  follow its algorithm statements, cited by section in the chapters below.
- Nicholas J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd
  ed., SIAM, 2002. The source of the rounding-error bounds the convergence
  chapter derives its thresholds from.
