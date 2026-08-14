# ADR 0003: Move Krylov vector-set residency onto the backend seam

- Status: Accepted
- Date: 2026-08-14
- Class: [arch] [major]

## Context

`GmresWorkspace` held its two Arnoldi vector sets as `Vec<B::Vector>`. On the
Leto backend `B::Vector = Array1<T>`, so a workspace at restart width `m` was
`2m + 1` independently allocated vectors whose addresses the host allocator
chose, with no relationship between them.

Every other array in that same struct is already flat. `hessenberg` is one
`Vec<B::Scalar>` addressed by `hessenberg_index(row, column)`; `cosine`,
`sine`, `transformed_residual`, and `coefficients` are likewise single
allocations. The vector sets were the one pointer-scattered structure in the
crate, and the contiguous-arena rule exists for exactly that shape.

Two facts constrain the fix rather than excusing the defect:

1. The vectors are allocated once at workspace construction, not per
   iteration. This is a placement defect, not allocation churn — the
   allocation-stability tests already passed and still do.
2. Per-buffer storage is correct on an accelerator. Hephaestus allocates,
   binds, and dispatches per buffer, so one flat device allocation would have
   to be re-split into per-vector bindings at every use. Flattening is a
   host-memory win with no device analogue.

A fix that flattened `B::Vector` sets in `athena-core` would therefore impose
a host layout on a backend where it is wrong.

## Decision

Add `KrylovBackend::VectorBlock` — an owned, backend-chosen storage for a
fixed count of equal-length vectors — with `allocate_block`, `block_view`, and
`block_view_mut`. `GmresWorkspace` holds two `B::VectorBlock` values instead of
two `Vec<B::Vector>`.

The residency decision moves to the backend that owns the memory:

- `LetoBackend` binds `VectorBlock = LetoVectorBlock<T>`, one Leto array
  holding the whole set. Vector `i` occupies `[i·len, (i+1)·len)` and is lent
  as a dense rank-one view over that subslice.
- `HephaestusBackend` binds `VectorBlock = Vec<D::Buffer<T>>`, unchanged
  per-buffer residency.

The recurrence sees indexed views either way and contains no layout
assumption.

Blocks lend one vector at a time. The Arnoldi step needs an immutable basis
vector and a mutable preconditioned-basis vector simultaneously, so those stay
two separate blocks and their disjointness remains a field-level type fact —
no runtime aliasing check, no split-borrow API.

### Rejected alternatives

- **Flatten inside `athena-core` over `B::Vector`.** Requires core to know the
  host layout, and imposes it on the accelerator. Rejected on the ownership
  boundary.
- **A block type parameterized by a layout policy.** No second host layout
  exists to justify the dimension; speculative generality per justified
  constructs.
- **Pad each vector to a vector-unit boundary.** Restores per-vector
  alignment at the cost of reintroducing dead space between vectors — the
  thing flattening removes. Deferred until a measurement demands it; the
  trade is recorded on `LetoVectorBlock`.
- **Seal `KrylovBackend` to avoid the major bump.** The seam exists to be
  implemented from sibling crates and from future device backends, so sealing
  it would break the contract it exists to provide. Rejected; the major
  classification is accepted instead.

## Consequences

`cargo semver-checks` classifies this as major: a non-sealed public trait
gained an associated type and three methods without defaults, which breaks any
downstream `KrylovBackend` implementor. Associated-type defaults are unstable,
so the methods cannot carry meaningful defaults and the break is inherent to
placing the residency decision on the seam.

No known consumer is affected. CFDrs and Harmonia consume `athena-core` but
implement `Preconditioner` and name `KrylovBackend` associated types; neither
implements `KrylovBackend`. The migration for any implementor is to add the
three methods and bind `VectorBlock`, for which `Vec<Self::Vector>` with
indexed borrows reproduces the previous behaviour exactly.

Contiguity is pinned by `athena-leto/tests/backend_contract.rs`, which asserts
that consecutive block vectors sit at their flat offsets. Without it the
layout would be unverified, because the recurrence only ever asks for indexed
views and would pass identically against independently allocated vectors.

No benchmark evidence accompanies this change: the repository ships no
Criterion instrument, and none was authored for it. The justification is
structural — one extent instead of `2m + 1` allocator-chosen ones — not a
measured speedup, and it is not claimed as one.
