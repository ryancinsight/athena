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

Version0.3.2 is synchronized across four packages and internal minima.
The local renewal requires Leto/LetoOps >=0.46.3,<0.48, excluding the published
0.46.2 finite-range failure. Cargo generates a normal canonical Git lock at
unmerged Leto #376; no path patch, manifest revision pin or downgrade is used.
This is provisional compatibility evidence, not accepted/released dependency
proof. Independent numerical/performance review, upstream publication and
final default/published graph renewal remain prerequisites. Normal packaging
must report an unavailable required version rather than use an older provider.

Hephaestus admission expands to >=0.22.2,<0.24 for its newly advanced public
0.23 default. Other provider defaults remain current. The renewal preserves
all Athena Rust implementation, test, benchmark and policy bytes relative to
the reviewed numerical repair. Its material dependency delta needs renewed
independent assessment; the earlier source review does not accept this graph.
#125's CG profiling and benchmark ownership remain unchanged. Main publishes
four pending packages and deploys book updates: assess that exact scope before
merge; the isolated local preparation updates no owner branch and publishes
nothing. Harmonia and all other release/service holds remain in force.
