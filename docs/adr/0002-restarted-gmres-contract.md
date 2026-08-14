# ADR 0002: Add restarted right-preconditioned GMRES

- Status: Accepted
- Date: 2026-07-19
- Class: [arch] [minor]

## Context

Athena's first vertical solver contract moved conjugate-gradient orchestration
above Leto and Hephaestus. Leto still exports a restarted GMRES recurrence,
leaving solver policy split between the array provider and Athena.

GMRES targets general nonsymmetric linear systems and must retain its Arnoldi
basis. Unbounded iteration therefore implies unbounded storage. The complete
replacement needs bounded, reusable basis storage on both CPU and GPU without
copying device vectors through the host.

The algorithm follows *Templates for the Solution of Linear Systems*:

- §2.3.4 defines restarted GMRES(m), modified Gram--Schmidt Arnoldi
  orthogonalization, the upper-Hessenberg least-squares system, Givens
  rotations, and delayed solution updates; and
- §3.1.2 defines left/right preconditioning and the transformed operator.

<https://www.netlib.org/templates/templates.html>

## Decision

Add one `Gmres<B, const RESTART: usize>` zero-sized algorithm marker and one
`GmresWorkspace<B, RESTART>` allocation boundary.

The recurrence uses right preconditioning:

```text
z_j = M^-1 v_j
w   = A z_j
x   = x_0 + sum_j y_j z_j
```

This keeps the Arnoldi residual estimate in the original system's Euclidean
norm. Every restart or prospective convergence recomputes `b - A x`; only the
recomputed norm can produce `Termination::Converged`.

`GmresWorkspace` owns:

- `RESTART + 1` backend-native Arnoldi vectors;
- `RESTART` backend-native preconditioned basis vectors;
- one residual and one work vector;
- a packed `(RESTART + 1) × RESTART` Hessenberg matrix;
- Givens rotations, transformed residual coordinates, and backsolve
  coefficients.

Construction performs all vector and host-scalar allocations. Repeated solves
reuse the same storage. The restart width is a structural const generic, so
each admitted width has one statically dispatched recurrence and an exact
bounded storage shape.

Extend `KrylovBackend` with only the two vector operations GMRES additionally
requires:

- in-place scale; and
- in-place AXPY.

Leto implements both through contiguous array views. Hephaestus implements
both through prepared Athena WGSL kernels. Full WGPU basis vectors remain
device-resident. The workspace also owns one prepared Hephaestus dot plan per
Arnoldi basis vector and prepared residual/work norm plans; scalar readback
remains the convergence control-flow boundary.

## Numerical contracts

- Modified Gram--Schmidt constructs the Arnoldi basis.
- Scaled Givens construction avoids avoidable overflow from directly
  evaluating `sqrt(a*a + b*b)`.
- Exact zero on the new subdiagonal is a happy-breakdown candidate. A singular
  triangular factor or a non-converged true residual terminates as
  `Breakdown`.
- Any non-finite Arnoldi, rotation, or residual scalar terminates as
  `NonFinite`.
- The convergence threshold remains
  `max(abs_tol, rel_tol * ||b||_2)`.
- No tolerance is used to redefine algebraic zero.
- A completed cycle that reduces the residual by no more than the accuracy of
  the residual's own evaluation terminates as `Stagnated`, and a residual
  exceeding the initial one by more than that accuracy terminates as
  `Diverged`. Both scales are derived, not tuned; see
  [ADR 0004](0004-derived-progress-termination.md).

## Rejected alternatives

### Keep GMRES in Leto

Rejected because Leto would remain a second solver-policy owner and could not
execute the same recurrence over Hephaestus buffers.

### Use left preconditioning

Rejected because the inexpensive rotated residual would then measure the
preconditioned system. Comparing it directly with Athena's physical-residual
policy would be dimensionally misleading; recomputing the physical residual
at every inner iteration would add an operator application and GPU
synchronization.

### Allocate basis vectors per solve or restart

Rejected because restart width fixes the maximum storage analytically.
Allocation belongs in workspace construction and reuse is part of the public
contract.

### Download WGPU basis vectors

Rejected because it would make the GPU path a host algorithm with device SpMV.
Only reduction scalars and the explicitly requested final result cross the
boundary.

## Migration

After CPU and WGPU conformance passed:

1. Leto's `gmres`, `GmresResult`, and solver module were deleted;
2. Leto retained CSR, SpMV, dot, arrays, and views;
3. Atlas and provider documentation and residue scans were synchronized; and
4. the Leto public removal was classified as a major SemVer change.

No re-export, forwarding helper, or compatibility module remains.

## Verification

- one generic CPU suite instantiates `f32` and `f64`;
- a manufactured nonsymmetric system verifies the known solution and direct
  `A x = b` residual;
- a restart width smaller than the system dimension exercises multiple cycles;
- identity and Jacobi right preconditioners share the same recurrence;
- repeated CPU solves allocate nothing after workspace construction;
- a real Hephaestus WGPU solve matches the manufactured solution;
- prepared WGPU reductions reuse fixed device allocations and reject identity
  mismatches before dispatch;
- ZST and const-restart layout claims are asserted; and
- residue scans prove Leto exports no iterative-solver recurrence.
