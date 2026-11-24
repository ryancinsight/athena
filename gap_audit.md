# Solver correctness audit

## Incremental release baseline

Reviewed [athena-krylov 0.5.0](crates/athena/Cargo.toml) at [0ad161c90b07](https://github.com/ryancinsight/athena/tree/0ad161c90b0747c4694f17b665ba9ea0e344e9d8). The exact version appears in the [registry](https://crates.io/crates/athena-krylov/0.5.0).

The next [patch] checkpoint uses [lsqr contract](crates/athena-leto/tests/lsqr_contract.rs). [Target version and acceptance criteria](backlog.md#athena-stable-qualification).

## GMRES performance hold

The rank-deficient GMRES recovery (`c9ddce7`) is pinned by
`crates/athena-leto/tests/gmres_rank_deficient.rs`; the separate Harmonia
performance hold persists.

The canonical dependency renewal selects published Leto/LetoOps 0.46.3 and
current Hephaestus 0.25 without changing the reviewed Rust source. Native
provider acceptance does not clear Athena's affected graph, whole-cohort or
absolute application-performance review. Lifecycle remains held.
