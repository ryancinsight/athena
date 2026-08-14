# Changelog

All notable changes to Athena are documented in this file.

## [Unreleased]

### Changed

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

### Added

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
