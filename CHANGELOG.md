# Changelog

All notable changes to Athena are documented in this file.

## [Unreleased]

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
