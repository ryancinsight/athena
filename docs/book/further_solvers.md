# 7. MINRES, Chebyshev and Deflated CG

Three more recurrences share the `LinearOperator` contract of the earlier
chapters. Each drops something the others rely on, which is the reason to
choose it: MINRES drops positive definiteness, Chebyshev drops inner products,
and deflated CG drops the slow part of the spectrum.

## MINRES: symmetric but not positive definite

CG needs \\(A\\) positive definite because its step length divides by the
curvature \\(p^{\mathsf T} A p\\), which can vanish or change sign. MINRES
needs only \\(A = A^{\mathsf T}\\). Writing \\(r_0 = b - A x_0\\), it picks the
iterate that solves

\\[
\min_{x \in x_0 + \mathcal{K}_j(A, r_0)} \\|b - A x\\|_2 ,
\\]

the same problem GMRES solves, but exploits symmetry so that the basis needs
no storage.

### The Lanczos recurrence

Set \\(\beta_1 = \\|r_0\\|\\), \\(v_0 = 0\\) and \\(v_1 = r_0 / \beta_1\\). For
\\(j = 1, 2, \dots\\) the symmetric Lanczos process computes

\\[
w_j = A v_j - \beta_j v_{j-1}, \qquad
\alpha_j = \langle v_j, w_j \rangle, \qquad
\beta_{j+1} = \\|w_j - \alpha_j v_j\\|_2, \qquad
v_{j+1} = \frac{w_j - \alpha_j v_j}{\beta_{j+1}} .
\\]

Each step applies the operator once and keeps only the last two \\(v\\). In
exact arithmetic the \\(v_i\\) are orthonormal and
\\(A V_j = V_{j+1} \underline{T}_j\\), where \\(\underline{T}_j\\) is the
\\((j+1) \times j\\) tridiagonal matrix with \\(\alpha_i\\) on the diagonal and
\\(\beta_{i+1}\\) on the sub- and superdiagonals. For \\(x = x_0 + V_j y\\) the
residual is \\(V_{j+1}(\beta_1 e_1 - \underline{T}_j y)\\), whose norm is
\\(\\|\beta_1 e_1 - \underline{T}_j y\\|_2\\): a least-squares problem of size
\\(j\\), independent of the dimension of \\(A\\).

### Givens QR of the tridiagonal

A plane rotation with cosine \\(c_i = \delta_i / \rho_i\\) and sine
\\(s_i = \beta_{i+1} / \rho_i\\), where
\\(\rho_i = \sqrt{\delta_i^2 + \beta_{i+1}^2}\\), removes each subdiagonal entry
as the column arrives. The new column \\((\beta_j, \alpha_j, \beta_{j+1})\\) has
already been rotated by the previous two rotations, which gives

\\[
\epsilon_j = s_{j-2} \beta_j, \qquad
\lambda_j = c_{j-1} c_{j-2} \beta_j + s_{j-1} \alpha_j, \qquad
\delta_j = c_{j-1} \alpha_j - s_{j-1} c_{j-2} \beta_j ,
\\]

where \\(\epsilon_j\\) and \\(\lambda_j\\) are entries of the triangular factor in
rows \\(j-2\\) and \\(j-1\\), and \\(\delta_j\\) is the diagonal entry that the
new rotation turns into \\(\rho_j\\). Both pending rotations start as the identity (\\(c_0 = c_{-1} = 1\\),
\\(s_0 = s_{-1} = 0\\)), so the first two columns need no special case. The
right-hand side rotates alongside:

\\[
\bar\varphi_1 = \beta_1, \qquad
\varphi_j = c_j \bar\varphi_j, \qquad
\bar\varphi_{j+1} = -s_j \bar\varphi_j .
\\]

The residual norm of the current iterate is \\(|\bar\varphi_{j+1}|\\), so the
convergence test needs no extra operator application. Because
\\(|s_j| \le 1\\), it never increases.

### The solution update

With \\(R_j\\) the triangular factor, \\(x_j = x_0 + V_j R_j^{-1} \varphi\\).
Define \\(W_j = V_j R_j^{-1}\\), so \\(W_j R_j = V_j\\). Reading off column
\\(j\\) of that identity gives the two-term update recurrence

\\[
w_j = \frac{v_j - \lambda_j w_{j-1} - \epsilon_j w_{j-2}}{\rho_j}, \qquad
x_j = x_{j-1} + \varphi_j w_j , \qquad w_0 = w_{-1} = 0 .
\\]

The seed zeroes \\(w_0\\) and \\(w_{-1}\\) at every solve, so a reused workspace
gives the same result as a fresh one under IEEE value equality. No basis is
stored: \\(v_j, v_{j-1}\\), the work vector \\(A v_j\\), the three \\(w\\), and
the seed residual make seven vectors whatever the iteration count.

### What the iteration achieves

Every iterate lies in \\(x_0 + \mathcal{K}_j\\), so its residual is
\\(r_j = q(A) r_0\\) for a polynomial \\(q\\) of degree at most \\(j\\) with
\\(q(0) = 1\\), and MINRES picks the one that minimises \\(\\|r_j\\|_2\\):

\\[
\\|r_j\\|_2 \le \min_{q}\, \max_{\lambda \in \sigma(A)} |q(\lambda)| \; \\|r_0\\|_2 .
\\]

In exact arithmetic, a nonsingular operator with \\(m\\) distinct eigenvalues is
solved in at most \\(m\\) steps, because its minimal polynomial, scaled to equal
one at zero, gives \\(q(A) = 0\\). The example below has two. These statements
hold for symmetric \\(A\\) in exact arithmetic; in floating point the recurrence
\\(|\bar\varphi_{j+1}|\\) drifts from the true residual, and Athena reports the
recurrence value without recomputing \\(b - A x\\).

## Chebyshev: no inner products

Chebyshev semi-iteration solves an SPD system when the caller knows an
interval \\([\lambda_{\min}, \lambda_{\max}]\\) containing the spectrum. It
chooses the polynomial that is as small as possible on the whole interval
while equal to one at zero:

\\[
p_k(t) = \frac{T_k(\varphi(t))}{T_k(\mu)}, \qquad
\varphi(t) = \frac{\lambda_{\max} + \lambda_{\min} - 2t}{\lambda_{\max} - \lambda_{\min}}, \qquad
\mu = \varphi(0) = \frac{\lambda_{\max} + \lambda_{\min}}{\lambda_{\max} - \lambda_{\min}} ,
\\]

where \\(T_k\\) is the Chebyshev polynomial of the first kind. The error after
\\(k\\) steps is \\(e_k = p_k(A) e_0\\).

### The recurrence

Evaluating \\(T_k(\mu)\\) directly overflows for narrow intervals, where
\\(\mu\\) is huge. Athena instead tracks the ratio
\\(v_k = T_k(\mu) / (\mu T_{k-1}(\mu))\\), which stays in \\([1, 2]\\). With
\\(\eta = 1/\mu\\) and \\(\omega = 2/(\lambda_{\max} + \lambda_{\min})\\),

\\[
v_1 = 1, \quad v_{k+1} = 2 - \frac{\eta^2}{v_k}, \qquad
\tau_1 = 1, \quad \tau_{k+1} = \frac{2}{v_{k+1}}, \qquad
s_k = \tau_k \omega\, r_{k-1} + (\tau_k - 1)\, s_{k-1}, \qquad
x_k = x_{k-1} + s_k ,
\\]

where \\(\tau_1 = 1\\) makes the \\(s_0\\) term vanish. Each step applies the
operator once, to recompute \\(r_k = b - A x_k\\). The only reduction is the
residual norm used by the convergence test; the step coefficients require no additional reduction.

When \\(\lambda_{\min} = \lambda_{\max} = \lambda\\) the interval is a point and
\\(A = \lambda I\\): \\(\eta = 0\\), \\(\tau_k = 1\\) and \\(\omega = 1/\lambda\\), a
Richardson step that is exact in one iteration. Athena computes \\(\eta\\) as
\\((1 - \lambda_{\min}/\lambda_{\max}) / (1 + \lambda_{\min}/\lambda_{\max})\\),
so \\(\mu\\) is never formed and this limit needs no special case.

### Convergence and its conditions

If \\(\sigma(A) \subset [\lambda_{\min}, \lambda_{\max}]\\) then
\\(|p_k(\lambda)| \le 1 / T_k(\mu)\\) on the spectrum, and because \\(A\\) is
symmetric,

\\[
\\|e_k\\|_A \le \frac{\\|e_0\\|_A}{T_k(\mu)}
\le 2 \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^{k} \\|e_0\\|_A,
\qquad \kappa = \frac{\lambda_{\max}}{\lambda_{\min}} .
\\]

The second inequality follows from
\\(T_k(\mu) = \tfrac12 (z^k + z^{-k}) \ge \tfrac12 z^k\\) with
\\(z = \mu + \sqrt{\mu^2 - 1} = (\sqrt\kappa + 1)/(\sqrt\kappa - 1)\\); it is the
rate of the [Conjugate Gradient](cg.md) bound. The price of avoiding inner
products is that the interval must be right. An eigenvalue \\(\lambda\\) below
\\(\lambda_{\min}\\) or above \\(\lambda_{\max}\\) still decays while
\\(0 < \lambda < \lambda_{\min} + \lambda_{\max}\\), because
\\(|T_k(\varphi(\lambda))| / T_k(\mu)\\) stays below one exactly then, but more
slowly. At or beyond \\(\lambda_{\min} + \lambda_{\max}\\), or at a
non-positive eigenvalue, \\(|p_k(\lambda)|\\) does not decay and the iteration
fails to converge. Nothing in the solve checks the interval against the
operator.

`SpectralBounds::new` validates the numbers only: both finite, the lower bound
positive, the upper at least the lower. A violation is a typed
`SpectralBoundsError`.

## Deflated CG: removing the slow directions

A few small eigenvalues make CG slow, because the polynomial must be small near
zero. Suppose the directions that carry them span a \\(k\\)-dimensional
subspace with orthonormal basis \\(U\\). Deflation solves the coarse part of
the problem exactly and runs CG on what is left.

### The projector and the Schur system

Let \\(E = U^{\mathsf T} A U\\), \\(C = U E^{-1} U^{\mathsf T}\\),
\\(Q = I - U U^{\mathsf T}\\) and \\(r_0 = b - A x_0\\). Since
\\(C A C = C\\), the matrix \\(P = I - A C\\) is a projector, and
\\(U^{\mathsf T} P = U^{\mathsf T} - E E^{-1} U^{\mathsf T} = 0\\). Write
\\(x = x_0 + U a + W c\\) with \\(W\\) an orthonormal basis of the complement
of \\(U\\), so that

\\[
[U\;W]^{\mathsf T} A [U\;W] = \begin{pmatrix} E & F \\\\ F^{\mathsf T} & G \end{pmatrix}, \qquad
F = U^{\mathsf T} A W, \quad G = W^{\mathsf T} A W .
\\]

Eliminating the coarse unknown \\(a\\) from \\(A(U a + W c) = r_0\\) leaves the
Schur complement \\(G - F^{\mathsf T} E^{-1} F\\), which is positive definite when
\\(A\\) is. In operator form, with \\(y = W c\\),

\\[
(A - A C A)\, y = (I - A C)\, r_0 , \qquad
x = x_0 + y + C (r_0 - A y) ,
\\]

where the second equation recovers the eliminated coordinates
\\(U a = C(r_0 - A y)\\). Substituting gives the residual of the recovered
iterate,

\\[
b - A x = (I - A C)(r_0 - A y) = (I - A C) r_0 - (A - A C A)\, y ,
\\]

so a solution of the Schur system is a solution of the original system.
Nothing here requires \\(U\\) to be invariant under \\(A\\): the Schur
complement carries the coupling between the coarse and complement coordinates.
Athena runs CG on \\(Q (A - A C A) Q\\) with right-hand side
\\(Q (I - A C) r_0\\), starting from \\(y = 0\\).

### Convergence and its conditions

If \\(U\\) spans an exact invariant subspace of \\(k\\) eigenvectors of an SPD
operator, then \\(F = U^{\mathsf T} A W = 0\\) and the Schur complement is
\\(G\\), whose spectrum is the remaining \\(n - k\\) eigenvalues. CG on it
terminates in at most \\(n - k\\) iterations in exact arithmetic. The example
below deflates the two smallest eigenvalues of a five-by-five system and CG
needs at most three steps. For a basis that is only approximately invariant
the Schur system is still correct, and the iteration count depends on the
spectrum of \\(G - F^{\mathsf T} E^{-1} F\\).

The subspace must be built from the same operator used for the solve: the
cached images \\(A U\\) and the factorisation of \\(E\\) belong to that
operator.

The CG phase measures the Schur residual, which equals \\(b - A x\\) in exact
arithmetic because \\(U^{\mathsf T}(I - A C) = 0\\) puts both in the complement.
The final report recomputes \\(b - A x\\) and tests it against the threshold of
the original right-hand side. A converged CG phase whose recovered iterate
misses that threshold is reported as `Stagnated`, or as `MaxIterations` when
the CG phase used the whole budget, never as converged.

## In Athena

```text
Minres<B>                zero-sized algorithm marker
MinresWorkspace<B>       seven vectors and prepared reductions; allocated once
Chebyshev<B>             zero-sized algorithm marker
ChebyshevWorkspace<B>    residual, step and image; allocated once
SpectralBounds<T>        the validated interval [λ_min, λ_max]
DeflatedCg<B>            zero-sized algorithm marker
DeflationSubspace<B>     orthonormal basis U, cached images A U and the
                         factorised coarse matrix E; built once per operator
CgWorkspace<B>           the CG phase's workspace, shared with plain Cg
```

None of the three takes a `Preconditioner`. `Minres` waits for a symmetric
positive-definite factorisation seam \\(M = L L^{\mathsf T}\\), which differs from
`Preconditioner::apply` and is not built until a consumer needs it.

### MINRES

`Minres::solve_into` takes the backend, operator, right-hand side, solution,
workspace and policy. It applies the operator once to seed the recurrence and
once per iteration. It reports `Converged` when \\(|\bar\varphi_{j+1}|\\) meets
the threshold and `Breakdown` when the Lanczos space closes
(\\(\beta_{j+1} = 0\\)) or a rotation radius is not positive before the policy
is met. Its threshold is built from \\(\\|r_0\\|\\) rather than \\(\\|b\\|\\), so
the two agree when \\(x_0 = 0\\). Before returning `Converged` or
`InitialResidual`, MINRES checks every solution component. A nonfinite
component changes the termination to `NonFinite` and preserves the reported
residual. Device tests exercise both the indefinite solve and this component
check in `athena-hephaestus/tests/solver_correctness.rs`.

### Chebyshev

`Chebyshev::solve_into` takes the same arguments plus the `SpectralBounds`.
It applies the operator once to seed and once per iteration, recomputes the
residual every iteration, and measures the threshold against \\(\\|b\\|\\).
Storage is three vectors.

```rust
{{#rustdoc_include ../../crates/athena-leto/examples/book_chebyshev.rs:60:85}}
```

The loop compares each component with the closed-form solution \\((1, 2, 3)\\)
within `BOUND`, derived in the sample from the stopping test.

### Deflated CG

`DeflationSubspace::from_vectors` orthonormalises the candidate vectors by
modified Gram--Schmidt, applies the operator once per vector, and factorises
\\(E\\) with partial pivoting. It rejects an empty or oversized set
(`Invalid`), a dependent vector (`Degenerate`) and a singular \\(E\\)
(`SingularCoarse`) with typed errors. `DeflatedCg::solve_into` takes the
backend, operator, subspace, right-hand side, solution, a `CgWorkspace`
and the policy. Besides the operator applications of the CG phase it applies the
operator three times: for the initial residual, for the recovery, and for the
final true residual. A residual that already meets the policy returns
`InitialResidual` after the first of them.

Storage is the four vectors of the `CgWorkspace`; per solve, six vectors and
three host arrays of length \\(k\\) allocated by the solve itself, or two vectors
and no arrays when the initial residual already meets the policy; and the basis
and its images, two blocks of \\(k\\) vectors. A `DeflatedCg` solve therefore
allocates, unlike a solve with any of the other six solvers.

```rust
{{#rustdoc_include ../../crates/athena-leto/examples/book_deflated_cg.rs:64:94}}
```

The asserts check termination within \\(n - k = 3\\) iterations and each component
against \\((1, 2, 3, 4, 5)\\) within `BOUND`, derived in the sample from the stopping test.

## References

- Paige and Saunders (1975), *Solution of sparse indefinite systems of linear
  equations*, SIAM Journal on Numerical Analysis 12(4), 617--629,
  doi:10.1137/0712047. Read from the bibliographic record only; Athena's
  source cites it as the origin of the MINRES recurrence, and the derivation
  above is checked against that source instead.
- Barrett et al., *Templates for the Solution of Linear Systems: Building
  Blocks for Iterative Methods*, SIAM, 1994, §2.3.2 (MINRES and SYMMLQ) and
  §2.3.9 (Chebyshev Iteration). <https://www.netlib.org/templates/templates.html>.
  Section titles read from the table of contents; the algorithms in them were
  not compared with the recurrences above, derived from Athena's source.
- Nicolaides (1987), *Deflation of conjugate gradients with applications to
  boundary value problems*, SIAM Journal on Numerical Analysis 24(2), 355--365,
  doi:10.1137/0724027, and Saad, Yeung, Erhel and Guyomarc'h (2000), *A
  deflated version of the conjugate gradient algorithm*, SIAM Journal on
  Scientific Computing 21(5), 1909--1926, doi:10.1137/S1064829598339761. Read
  from the bibliographic records only; no theorem is cited, and the Schur
  derivation above is checked against Athena's source instead.
