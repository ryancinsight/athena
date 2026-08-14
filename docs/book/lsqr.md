# 6. LSQR

The first three methods all assume \\(A x = b\\) has a solution. LSQR is for
when it does not: \\(A\\) is rectangular, or square and inconsistent, and the
question is not "what solves this" but

\\[
\min_x \\|b - A x\\|_2 .
\\]

Overdetermined systems from data fitting, and regularised inverse problems, are
the usual sources.

## Why not the normal equations

The minimiser satisfies the normal equations
\\(A^{\mathsf T} A x = A^{\mathsf T} b\\), and \\(A^{\mathsf T}A\\) is symmetric
positive semidefinite, so one could apply CG to it directly. Do not. The
condition number squares:

\\[
\kappa_2(A^{\mathsf T} A) = \kappa_2(A)^2 .
\\]

By the CG bound of the [Conjugate Gradient](cg.md) chapter, the iteration count
then scales with \\(\sqrt{\kappa(A^{\mathsf T}A)} = \kappa(A)\\) rather than
\\(\sqrt{\kappa(A)}\\), and — worse — forming \\(A^{\mathsf T}A\\) explicitly
loses information that is irrecoverable in the working precision. A matrix with
\\(\kappa(A) = 10^8\\) is well within single precision's reach; its normal
equations are not.

LSQR is mathematically equivalent to CG on the normal equations but never forms
\\(A^{\mathsf T} A\\). It works with \\(A\\) and \\(A^{\mathsf T}\\) separately,
so the arithmetic sees only \\(\kappa(A)\\).

## Golub--Kahan bidiagonalisation

The basis-building step is the analogue of Arnoldi for a rectangular operator.
It generates two orthonormal sequences at once — \\(u_i\\) in the row space's
codomain and \\(v_i\\) in its domain — by alternating applications of \\(A\\) and
\\(A^{\mathsf T}\\):

```text
β₁u₁ = b                    normalise
α₁v₁ = Aᵀu₁                 normalise
for i in 1..:
    β_{i+1} u_{i+1} = A v_i    − α_i u_i        normalise
    α_{i+1} v_{i+1} = Aᵀu_{i+1} − β_{i+1} v_i   normalise
```

The scalars form a lower bidiagonal \\(B_k\\), and the least-squares problem
projects onto it exactly as GMRES projects onto its Hessenberg matrix — a small
problem, size independent of \\(A\\), solved by Givens rotations maintained
incrementally. Two matrix applications per step, one by \\(A\\) and one by
\\(A^{\mathsf T}\\), and a fixed number of vectors: five in Athena, two of length
`rows` and three of length `columns`.

The requirement to apply \\(A^{\mathsf T}\\) is why LSQR's operator seam is
`RectangularOperator` rather than the `LinearOperator` the square solvers use.
It is a real constraint on matrix-free operators, and unlike BiCG's transpose
requirement there is no way around it: the problem itself is stated in terms of
\\(A^{\mathsf T}\\).

## Two stopping criteria, because there are two problems

This is the part of LSQR that differs most from the other three solvers, and
getting it wrong makes correct solves look like failures.

A **consistent** system has \\(b\\) in the range of \\(A\\), so the residual goes
to zero and the ordinary test \\(\\|r\\| \le \max(\tau_{\text{abs}},
\tau_{\text{rel}}\\|b\\|)\\) applies. Athena reports `Termination::Converged`.

An **inconsistent** system — the genuine least-squares case — has a residual
bounded away from zero by construction. Testing it against a small threshold
would never succeed, and the solve would run to the iteration cap despite having
found the exact minimiser at some earlier step. The right criterion there is the
normal-equation residual: the optimum is characterised by
\\(A^{\mathsf T} r = 0\\), so the test is \\(\\|A^{\mathsf T} r\\|\\) relative to
\\(\\|r\\|\\), and Athena reports meeting it as
`Termination::NormalEquations`.

Both are checked. Which one fires tells you which kind of problem you had.
`converged()` is true for both, so code asking only "did this work" needs no
special handling.

## In Athena

```text
Lsqr<B>                  zero-sized algorithm marker
LsqrWorkspace<B>         the bidiagonalisation's five vectors and prepared
                         reductions; allocated once
```

LSQR is backend-neutral like the rest, but `RectangularOperator` is implemented
for Leto only, so its shipped execution is CPU. Nothing in the recurrence
prevents an accelerator implementation; the operator seam is simply not
implemented for Hephaestus yet.

Note that LSQR takes no preconditioner in Athena's surface. Preconditioning a
least-squares problem means finding \\(M\\) with \\(A M^{-1}\\) better
conditioned, which is a different construction from the square case and is not
provided.

## References

- Paige and Saunders (1982), *LSQR: An algorithm for sparse linear equations
  and sparse least squares*, ACM Transactions on Mathematical Software 8(1),
  43--71. The original, including the stopping criteria above.
- Golub and Kahan (1965), *Calculating the singular values and pseudo-inverse
  of a matrix*, SIAM J. Numerical Analysis Ser. B 2(2), 205--224, for the
  bidiagonalisation.
- Björck, *Numerical Methods for Least Squares Problems*, SIAM, 1996, for the
  conditioning argument against the normal equations.
