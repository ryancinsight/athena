# Changelog

All notable changes to Athena are documented in this file.

## Athena adapters 0.5.0

### Breaking

- Use Leto 0.52 for public CPU array views and Hephaestus 0.30 for the
  accelerator bridge. Update the array and device providers together.
  The solver and preconditioner contracts retain the published
  athena-core 0.4.0 package identity.

## 0.4.0

### Breaking

- Use registry Eunomia 0.12.1 in the Krylov scalar contract and Leto 0.51
  in the CPU adapter's public array views. Update applications that
  implement Athena operators to these dependency versions together.
- Use the public constructors or defaults for solver state, workspaces,
  operators, and preconditioners. Their private fields are initialized
  through those APIs.

### Fixed

- Verify standalone registry lockfiles by matching source-less packages to
  declared Athena workspace members. The staged check reads the index,
  and the full check resolves the lockfile outside the Atlas overlay.
- [major] CG reports `Termination::Stagnated`, not `Breakdown`, when `‖b − Ax‖`
  exceeds its converged recurrence residual by more than the threshold, and
  stops there instead of iterating until a dot product underflows; a stall found
  before the budget ends the solve no longer reports `MaxIterations` (ADR 0004).
  Migration: treat `Stagnated` like `Breakdown` and `MaxIterations`. Affected:
  CFDrs `cfd-math` `interpret`, which maps `Stagnated` to an error.
- [patch] Classify exact CG convergence between scheduled residual checks as
  convergence and report current original-system residuals on curvature and
  budget termination. Recompute `b - A x` before accepting convergence so
  recurrence drift cannot produce a false success. Reuse existing vector
  scratch and prepared resources; each verification operator application is
  counted in the report.

### Changed

- [patch] Admit Leto 0.50.0 and widen the `leto` and `leto-ops`
  workspace requirements from `>=0.46.3, <0.49.0` to `>=0.46.3, <0.51.0`.
  The lock moves Leto and Leto-ops (0.48.0 to 0.50.0, source `39d65120` to
  `6483eac3`) and the Hephaestus core and WGPU 0.28.0 source to `742cdb84`,
  the revision whose Leto requirement admits 0.50; no other row changes.
  Leto 0.50 removes the unsafe `VecStorage::uninit`, which Athena does not
  call, and zero-fills five allocating Leto-ops wrappers. No Athena
  signature changes.

- [patch] Advance every first-party lock row onto its repository's current
  default head. The lock is regenerated outside the Atlas dependency overlay,
  so the `source = "git+..."` rows survive and the registry rows are
  unchanged. The Leto rows move to the 0.48 line at `39d65120`, the Themis
  row to 0.11.5 at `99980f09`, and the Moirai rows to 0.12.0 at `48bacb85`.
  Moirai 0.12.0 removes the required `StateWord::get_mut` method, which
  Athena neither implements nor calls, and Athena re-exports no Moirai or
  Eunomia type, so the public Athena surface is unchanged.

## [0.3.3] - 2026-10-22

### Changed

- [patch] Advance the Leto/Hermes/Eunomia pins to the merged
  trait-hierarchy tips. No API or numerical change; the advanced
  Leto is the baseline for the recorded
  `ATHENA-FUSED-UPDATE-20261020` diagnosis.

- [patch] Redundant `RealField + FloatElement` bounds collapse to
  `RealField` across `athena-core` and `athena-leto` (source and tests);
  `RealField: FloatElement` already, so every signature accepts and
  promises exactly what it did. Orphaned `FloatElement` imports removed.

## [0.3.2] - 2026-10-20

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

- Require Leto/LetoOps >=0.46.3,<0.48 to exclude the 0.46.2 finite-range
  regression. The canonical lock selects merged/published Leto #376 at
  5051cd9; native 0.46.3 registry source and fresh consumers are verified.
  Athena dependency-delta review and lifecycle gates remain required.
- Admit Hephaestus >=0.22.2,<0.26 and test its current 0.25 public default.
  The consumed core/WGPU crate files are byte-identical across 0.24/0.25;
  #506 adds embedding ports outside Athena's dependency surface.
  Preserve Mnemosyne memory0.10.1/arena0.7.1 and synchronized internal0.3.2
  minima. No Rust implementation/test/policy bytes change in this renewal.

## [0.3.1] - 2026-10-19

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

- [patch] Admit Eunomia>=0.10,<0.12 and Leto/LetoOps>=0.46,<0.48,preserving current-provider support and all Rust
  implementation/test/feature bytes. Current canonical lock and real typed
  current/future consumer probes verify the bridge; final provider ordering,
  exact-head review and hosted/publication gates remain required.

- Advance the locked Atlas provider graph to Eunomia 0.10.0, Mnemosyne 0.10.0
  (arena 0.7.0), Leto/Leto-ops 0.46.0, and Hephaestus 0.22.1, following the
  same pins Apollo's workspace resolves. The Hephaestus 0.22 seam genericized
  `WgpuVectorOps<T>` and retired the `RetainedReductions` implementation, so
  the Athena backend binds the one-shot `dot`/`norm_l2` reductions through
  zero-sized prepared policies (the Leto backend's established form) instead
  of owned retained handles; kernels still compile once in the operations
  bundle. All-feature check, strict Clippy, 136/136 Nextest, doctests,
  Rustdoc, and the ignored allocation suite pass.

## [0.2.0] - 2026-10-05

### Changed

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

## [0.1.1] - 2026-08-12

### Added

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
