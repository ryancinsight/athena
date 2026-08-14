# 2. Krylov Backend

The previous chapter said a Krylov method only ever asks the matrix for
\\(A v\\). This chapter is about the other half of that claim: what a solver
asks the *machine* for, and why that list is short enough to be a trait.

## The operation set

Read any Krylov recurrence and the vector operations it performs are the same
handful:

| Operation | Used for |
| --- | --- |
| `axpy`: \\(y \mathrel{+}= \alpha x\\) | every recurrence update |
| `scale`: \\(x \mathrel{*}= \alpha\\) | normalising a basis vector |
| `copy` | seeding a cycle |
| `dot`: \\(x^{\mathsf T} y\\) | orthogonalisation coefficients |
| `norm_l2`: \\(\\|x\\|_2\\) | convergence tests |
| `residual`: \\(b - v\\) | recomputing the true residual |

Everything above that line is arithmetic on the coefficients, which is scalar
work on the host. So a solver can be written once against this set and run
anywhere the set can be implemented. That is what `KrylovBackend` is: not an
abstraction over arrays, but the exact interface a Krylov recurrence needs.

Two entries are less obvious. `residual` exists as its own operation rather
than as a copy followed by an `axpy` because a backend can fuse it into one
pass, and on a GPU the difference is a kernel launch. `fused_cg_update`, which
applies \\(x \mathrel{+}= \alpha p\\) and \\(r \mathrel{-}= \alpha A p\\)
together, is there for the same reason: the two updates share a traversal, and
CG performs them every iteration.

## Static dispatch

`KrylovBackend` is a generic parameter, never a `dyn` object. A solver is
declared as `Gmres<B, RESTART>` and monomorphises to the backend it is
instantiated with, so an `axpy` inside an inner loop compiles to the backend's
own code with no vtable and no call through a function pointer. On a host
backend that inner loop is a few nanoseconds of work; an indirect call would be
a measurable fraction of it.

The same reasoning shapes the algorithm markers. `Gmres<B, RESTART>` is
zero-sized — it holds a `PhantomData`, carries no state, and exists only to
give the recurrence a place to live and the restart width a place to be a
compile-time constant. The restart width being a const generic rather than a
field is what lets the workspace's storage shape be exact and its index
arithmetic be resolved during compilation.

## Views, and why they are associated types

The operations above take *views*, not owned vectors:

```text
type View<'a>: Copy where Self: 'a;
type ViewMut<'a> where Self: 'a;
```

These are generic associated types — associated types parameterised by a
lifetime — and the lifetime is the point. A view borrows storage the backend
owns for exactly as long as the operation needs it, so passing a vector to
`axpy` copies nothing. On the Leto host backend a view is an array view over a
slice; on the Hephaestus device backend it is a borrowed typed GPU buffer. The
recurrence names neither type. It says `Self::View<'_>` and the compiler
resolves it.

An ordinary associated type could not express this, because the borrow's
lifetime differs at every call site. Without the GAT the trait would have to
hand back owned vectors, and every operation would copy.

## Vector blocks

One member of the trait is not a vector operation at all:

```text
type VectorBlock;
fn allocate_block(&self, count: usize, len: usize) -> Result<Self::VectorBlock, Self::Error>;
fn block_view<'a>(&'a self, block: &'a Self::VectorBlock, index: usize) -> Self::View<'a>;
fn block_view_mut<'a>(&'a self, block: &'a mut Self::VectorBlock, index: usize) -> Self::ViewMut<'a>;
```

A `VectorBlock` is a fixed number of equal-length vectors allocated together.
GMRES needs one: its Arnoldi basis is \\(m + 1\\) vectors, allocated once and
then written one at a time and read repeatedly. The obvious representation is
`Vec<Vector>`, and it is the wrong one to *fix in the solver*, because the right
answer differs by backend.

On a host, the basis should be one contiguous allocation with vector \\(i\\) at
offset \\(i \cdot n\\). The orthogonalisation sweep in the GMRES chapter walks
every basis vector in order on every step, and a contiguous extent is a walk the
hardware prefetcher can follow. A `Vec` of separate allocations is the same
elements at addresses the allocator chose, with pointer chasing between them.

On a GPU, per-buffer storage is not a compromise, it is the model: allocation
and binding are per-buffer, and a device backend keeping independent buffers is
doing the natural thing.

So the residency decision belongs to whoever owns the memory. `VectorBlock` is
an associated type; the solver allocates a block, indexes it, and contains no
assumption about how it is laid out. Leto binds it to a single array lending
offset subviews. Hephaestus binds it to independent device buffers. Both
monomorphise, so neither pays for the other's choice. The decision and its
alternatives are recorded in `docs/adr/0003-krylov-vector-block-seam.md`.

Blocks lend one vector at a time. That is a deliberate restriction: the Arnoldi
step needs to hold a basis vector and a preconditioned basis vector
simultaneously, and it does so by keeping them in *separate* blocks. Their
disjointness is then a fact about two fields rather than a runtime check on two
indices, and the borrow checker enforces it.

## Prepared reductions

`prepare_dot` and `prepare_norm_l2` return a `PreparedDot` or `PreparedNorm`
bound to specific vector allocations, which `dot_prepared` and
`norm_l2_prepared` then execute. On the host these are nearly free. On a GPU
they are the point: a reduction needs intermediate buffers, a bind group, and a
pipeline, and building them per call would dominate the reduction. Preparing
them once at workspace construction means a solve dispatches only the work.

This is why a GMRES workspace holds one prepared dot plan per basis vector: the
orthogonalisation sweep does a dot against each, and each has its own fixed
allocation to be prepared against.

## The allocation contract

Putting the pieces together gives the contract the workspace types exist to
enforce: **construction allocates, solving does not**. A `GmresWorkspace`
allocates its blocks, its residual and work vectors, its prepared reductions,
and its scalar arrays once, sized by the vector length and the const-generic
restart width. Repeated solves at the same size reuse all of it.

That is testable, and it is tested: `crates/athena-leto/tests/allocation.rs`
runs sixteen warm solves for CG, GMRES, and BiCGSTAB under an instrumented
allocator and asserts the measured region performed zero allocations, zero
reallocations, and zero deallocations.

## The provider crates

`athena-core` is `no_std + alloc` and depends on no infrastructure; it holds the
traits and the recurrences. Two crates implement the seam:

- **`athena-leto`** — the CPU backend, over Leto arrays. Supplies CSR and dense
  operators, square and rectangular, and Jacobi, incomplete LU, triangular, and
  SOR preconditioners.
- **`athena-hephaestus`** — the GPU backend, over Hephaestus device buffers.
  Basis vectors stay resident on the device; only reduction scalars cross back,
  which is what makes the convergence test the synchronisation boundary.

A third implementation adds itself by writing the trait. No solver code changes.
