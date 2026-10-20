# Changelog

All notable changes to Athena are documented in this file.

## [0.3.2] - Unreleased

### Fixed

- [patch] Recover rank-deficient GMRES reduced least squares with column
  pivoting instead of discarding coefficients in an unpivoted factor.
  Return a basic correction; minimum coefficient/physical solution norm is
  not promised. Preserve the full-rank triangular path and preallocate all
  rank-recovery scratch with the workspace.
- [patch] Report the recomputed residual at terminal GMRES cycle checks;
  rank-deficient factors cannot use the final transformed component alone.
  Keep rounding floors explicit as statistical numerical-rank estimates,
  not deterministic worst-case bounds or certificates of exact rank.

### Changed

- Refresh the compatible canonical graph and require the tested published
  Leto/LetoOps0.46.1, Hephaestus0.22.2, Mnemosyne memory0.10.1/arena0.7.1
  patch minima. Retain future provider admission ceilings and synchronize
  internal0.3.2 minima; source/publication acceptance remain gated.

## [0.3.1] - Unreleased

### Changed

- [patch] Admit Eunomia>=0.10,<0.12 and Leto/LetoOps>=0.46,<0.48,preserving current-provider support and all Rust
  implementation/test/feature bytes. Current canonical lock and real typed
  current/future consumer probes verify the bridge; final provider ordering,
  exact-head review and hosted/publication gates remain required.

## [Unreleased]

### Fixed

- Deflated CG now eliminates coarse/complement coupling through the Schur
  operator `A - A U (Uᵀ A U)⁻¹ Uᵀ A`, supports noninvariant bases, preserves
  the incoming initial guess and original RHS threshold, and reports the
  recovered iterate's true residual. Cached `A U` images retain one base
  operator application per deflated application.
- Chebyshev semi-iteration supports equal spectral bounds through its
  singleton-spectrum Richardson limit. Normalised coefficients remain finite
  for narrow intervals; every updated iterate is checked before advancing
  coefficients. Relative tolerance uses the original RHS norm.


### Changed

- Advance the locked Atlas provider graph to Eunomia 0.10.0, Mnemosyne 0.10.0
  (arena 0.7.0), Leto/Leto-ops 0.46.0, and Hephaestus 0.22.1, following the
  same pins Apollo's workspace resolves. The Hephaestus 0.22 seam genericized
  `WgpuVectorOps<T>` and retired the `RetainedReductions` implementation, so
  the Athena backend binds the one-shot `dot`/`norm_l2` reductions through
  zero-sized prepared policies (the Leto backend's established form) instead
  of owned retained handles; kernels still compile once in the operations
  bundle. All-feature check, strict Clippy, 136/136 Nextest, doctests,
  Rustdoc, and the ignored allocation suite pass.
- Refresh the locked Atlas provider graph to Leto `2beb4f17`, Hermes
  `eb1a2f87`, Hephaestus `dc7b72c6`, Eunomia `88c685f2`, and their current
  Mnemosyne, Moirai, and Themis dependencies. The update is dependency-ordered
  so Hermes 0.7 is consumed through the provider graph without a compatibility
  path; all-feature check, strict Clippy, 63/63 Nextest, doctests, Rustdoc,
  audit, and cargo-deny pass.

- **Breaking.** `KrylovBackend` gains the `VectorBlock` associated type and
  the `allocate_block`, `block_view`, and `block_view_mut` methods, so a
  backend owns the residency of a fixed-count vector set instead of
  `athena-core` fixing it as `Vec<Vector>`. `GmresWorkspace` holds its Arnoldi
  basis and preconditioned basis as blocks: the Leto backend places each set
  in one contiguous allocation lent out as offset subviews, while the
  Hephaestus backend keeps independent device buffers, which is what its
  per-buffer allocation and binding model requires. Downstream implementors of
  `KrylovBackend` must add the three methods and bind `VectorBlock`;
  `Vec<Self::Vector>` with indexed borrows reproduces the previous behaviour
  exactly. Consumers that only call the seam are unaffected. See
  [ADR 0003](docs/adr/0003-krylov-vector-block-seam.md).

- Restarted GMRES classifies a completed, unconverged cycle by the progress it
  made, so a stalled solve reports `Termination::Stagnated` at the first
  unproductive cycle instead of consuming its whole budget and reporting
  `Termination::MaxIterations`. A caller matching only on `MaxIterations` to
  detect non-convergence should use `SolveReport::converged()`, which already
  covers every non-converged variant. No signature changes; CG, BiCGSTAB, and
  LSQR keep their existing terminations. See
  [ADR 0004](docs/adr/0004-derived-progress-termination.md).
- `SolveReport` is `#[must_use]`. Discarding it is how a non-converged
  termination gets silently accepted, so ignoring one is now a warning.

### Added

- `Termination::Stagnated` and `Termination::Diverged`, and
  `residual_noise_floor`, the derived absolute accuracy of one recomputed
  residual norm that both criteria are measured against. The floor is
  `sqrt(len) * EPSILON * ||b||`, published with its derivation and its limits
  so a caller writing a custom convergence test can use the same scale rather
  than a tuned tolerance.
- `LetoVectorBlock`, the contiguous host storage backing the Leto
  `VectorBlock`, and an `athena-leto` backend-seam conformance suite that pins
  block contiguity, zero-initialization, non-aliasing between neighbouring
  vectors, and out-of-range rejection.
- `LetoBackendError::BlockExtentOverflow` for a vector block whose
  `count * len` extent exceeds the addressable range.
- Backend-neutral PCG recurrence with GAT-based provider views, reusable
  workspace, validated convergence policy, and allocation-free reports.
- Restarted right-preconditioned GMRES with a const-generic restart width,
  reusable basis workspace, modified Gram--Schmidt Arnoldi construction,
  scaled Givens rotations, and true-residual convergence checks.
- Leto CPU backend, CSR and borrowed dense operators, and Jacobi
  preconditioning for `f32` and `f64`.
- Hephaestus WGPU backend with resident vectors, GPU CSR SpMV and reductions,
  fused PCG kernels, and prepared scale/AXPY kernels for GMRES.
- CPU and WGPU manufactured SPD and nonsymmetric-system examples and
  conformance tests.
- Public construction of non-exhaustive iteration telemetry for external
  convergence orchestrators.
- Prepared Hephaestus dot and L2-norm plans owned by reusable PCG and GMRES
  workspaces, with fixed-allocation validation and provider benchmark evidence.

### Changed

- Move public CG and restarted GMRES orchestration from Leto to Athena after
  CPU and WGPU conformance. Leto remains the host-array, CSR, SpMV, and
  reduction provider.
- Align Leto, Hephaestus, Aequitas, Hermes, and Eunomia revisions so provider
  numeric types retain one Git source identity.
