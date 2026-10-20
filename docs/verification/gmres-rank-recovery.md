# Rank-deficient GMRES recovery

## Analytical regression

For Aδ = [[δ,0,1],[δ,0,1],[0,1,0]], b = e1, x_initial = 0,
Identity preconditioning and restart width 3,

    ||b - Aδ x||² = 1/2 + 2(δ x0 + x2 - 1/2)² + x1².

The Arnoldi basis is e1,e2,e3 for nonzero δ. The reduced triangular factor is
[[sqrt(2)δ,0,sqrt(2)],[0,1,0],[0,0,0]], with transformed RHS
[1/sqrt(2),0,1/sqrt(2)]. The two nonzero singular values of Aδ are
sqrt(2(1+δ²)) and 1. Thus tiny original diagonals do not justify discarding
original coefficients independently: the final column retains an O(1)
range direction. With δ = ±EPSILON², the old path returns zero, residual²1,
although [0,0,1/2] reaches residual²1/2. The terminal inner-cycle estimate
zero cannot represent the residual of any solution of this system.

The new public integration tests preserve these assertions, exercising
signed δ, unit δ, native f32/f64 and observer intervals 1,2,4. The same LS
and telemetry assertions fail against the original production source before
the repair. Additional private tests exercise off-diagonal rank-one coupling,
rank zero, nonfinite data and unchanged full-rank inputs/arithmetic.

## Contract and cost

Column-pivoted QR refactors the small reduced factor only when an unusable
unpivoted diagonal is encountered. For R P = Q [U V; 0 E], the numerical-rank
policy neglects E, solves U z = (Qᵀg)[..rank], sets free permuted coordinates
to zero and maps through P. This is a basic reduced least-squares correction.
It need not minimize coefficient norm or physical solution norm after the
flexible Z basis is applied. R=[[1,1],[0,0]], g=[1,0] deliberately returns
[1,0], although [1/2,1/2] has smaller coefficient norm.

The existing square-root accumulation floors are retained as numerical
policy heuristics. Their statistical assumptions and factor-two margins do
not establish deterministic worst-case bounds or exact rank certificates.
The original-system residual is still explicitly recomputed after every
correction. Cycle-ending samples now use that checked residual; ordinary
inner-cycle samples retain their documented recurrence estimate semantics.

Full-rank subtraction/division order is unchanged. Recovery uses native
scalar arithmetic, preallocated O(RESTART) indices and the existing factor/
RHS scratch, with no per-solve allocation. Its rare pivoted-QR fallback has
O(RESTART³) work. The allocation instrument executes the new strict test
alongside all existing strict contracts; no gate, limit or old test changes.

## Verification and integration

The draft PR must associate native/actual-accelerator tests, strict lint,
minimal features, doctests/Rustdoc, executable book/examples, actual Rust1.95,
explicit patch-class API checks against published0.3.1, clean standard
packaging, supply-chain/lock/conformance and unchanged full-rank benchmark
results with its final head. The benchmark compares identical current
provider graphs/compiler and the unchanged 64/128-grid, 100-sample workload;
full-rank bitwise comparison covers both native precisions and multiple
restart widths. Final command logs and regression receipts are retained in
the isolated review packet; hosted gates and independent review remain
required before merge.

Version 0.3.2 is synchronized across four packages and internal minima.
The canonical requirements are Leto/LetoOps >=0.46.3,<0.48, excluding the
published 0.46.2 finite-range failure. The normal Git lock selects merged
Leto #376 `5051cd9577650a5712aa6a7effcd8955c9d8bcc7`; no product path patch,
manifest revision pin or downgrade is used. Native 0.46.3 source/package/tag
verification and a fresh registry-only consumer pass before this renewal.
The provider's exact numerical/matched-performance acceptance is scoped to
its native change. Historical cross-window absolute application performance,
the unexplained slow run, and the private PyO3 advisory failures remain explicit.

Hephaestus admission is >=0.22.2,<0.26 and the canonical lock selects its
current 0.25 default `36965bf2d775f09f4e5a76daae57d2a64b145e1c`. Athena uses
DenseVectorOps/SparseOperatorOps rather than #505's removed fusion traits.
Other provider defaults remain current. All 132 Athena Rust implementation,
test and benchmark files match reviewed numerical source C9 byte for byte;
no timeout, policy, feature, assertion or benchmark input changes. This
material dependency delta needs affected independent review. Earlier matched
Leto application timings use Hephaestus 0.23 and do not accept later cohorts.

Normal final-head native/mandatory-accelerator, allocation, API, package,
MSRV, supply-chain and source gates must pass; publication is not inferred
from local Git builds. #125's CG profiling/benchmark ownership is preserved.
Athena main automatically publishes four pending packages and may deploy
book updates; assess that exact scope and affected review before its lifecycle.
No Athena merge/publication or other owner's release/service hold is cleared.

Hephaestus #506 adds CUDA/ROCm/Metal embedding ports. Every file of the two
consumed core/WGPU crates (556 files, including all Rust and own manifests)
is byte-identical across 0.24 and 0.25; only inherited versions advance.
Actual 0.25 registry archive checksums/source identities verify before the
minimal admission changes. The old bound rejects a fresh normal update;
the renewed normal lock advances only these two nodes, without downgrades.
Athena's ComputeDevice/DeviceBuffer, DenseVectorOps and SparseOperatorOps
interfaces remain unchanged and require native/actual Metal, API and package
verification. No qualification of the new backend embedding ports is inferred.
The frozen 0.24 graph has independent source/numerical/API/package review and
matched benchmark evidence with small point differences and greater baseline
drift. No causal cost, zero-overhead claim or absolute threshold follows. The
new final graph still needs affected independent review before lifecycle.
