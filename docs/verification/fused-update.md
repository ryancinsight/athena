# Fused CG update diagnosis and optimization (ATHENA-FUSED-UPDATE-20261020)

Basis: athena `5847236` (main after #131), leto `5051cd9` (0.46.3),
hermes `23b031a` (0.9.1). Host: shared Windows workstation, generic
x86-64 build (no `target-cpu`), `kernels`/`cg_poisson` benches at
`--warm-up-time 1 --measurement-time 3`, sample sizes 30/20.

## Diagnosis

`athena-leto`'s `fused_cg_update` (31% of grid/128 per-iteration cost
after the stencil cut) was a hand scalar zip loop, while every sibling
backend operation routes through leto `Scalar::*_slice` kernels backed
by Hermes runtime-dispatched SIMD. In a generic build the hand loop
compiles at SSE2 width with separate multiply/add (two roundings per
element); the Hermes kernels execute AVX2/AVX-512/NEON `fmadd` under
runtime dispatch.

Refuted shortcuts, measured on the `kernels` bench (medians, same run):

- Splitting into two `axpy_slice` calls: `axpy` row 10.44 µs, so the
  split costs ~20.9 µs against the fused row's 18.35 µs — one fused
  pass beats two half passes.
- Width is not the only limiter: `copy` (8.38 µs, 2 streams) and
  `scale` (8.23 µs) show the mutating rows are largely stream-bound,
  but a `target-cpu=native` probe moved the fused row −18%
  (p = 0.03) while `copy` did not change (p = 0.48), isolating a real
  SIMD-width component worth roughly a fifth of the row cost.

## Change

One generic kernel per layer, monomorphized end to end:

- hermes `axpy_pair` (hermes#265, 0.9.2): one runtime-dispatched SIMD
  pass `out_a += alpha_a * x_a; out_b += alpha_b * x_b`, bit-identical
  to two `axpy` calls, judge-verified at `74cc839`.
- leto `Scalar::axpy_pair_slice` (0.46.4): SIMD-routed for
  f32/f64/F16/Bf16 through `SimdStrategy`; scalar default for plain
  and complex types.
- athena-leto `fused_cg_update` delegates to `axpy_pair_slice` with
  `(alpha, direction, solution, -alpha, image, residual)`. Each
  element becomes one `fmadd` — the same rounding this backend's
  dot and axpy kernels already use.

## Evidence

Value identity: hermes tests assert `axpy_pair` against a `mul_add`
reference across 13 lengths (f64) and tail order (f32); leto tests
assert `axpy_pair_slice` equals two `axpy_slice` calls at
f32/f64/i64; athena backend-contract tests assert both updates at an
odd length (1031) and typed length rejection before mutation. All
athena-leto + athena-core suites pass; CG contracts unchanged.

Convergence invariance (probe over the bench-pinned Poisson cases,
both operator forms): iteration counts are identical under the old
scalar loop and the pair kernel — 119 at grid/64 and 239 at
grid/128. Final residual norms differ only at ~1e-13 relative.

Performance (`kernels_cpu/fused_cg_update`, back-to-back A/B in one
session, medians with 95% CI): pair 13.89 µs [13.22, 14.46] vs
scalar 19.61 µs [17.32, 20.59] — a 29% reduction with disjoint
intervals. Against the pre-change saved baseline (18.35 µs) the pair
row is −24%. End-to-end `cg_poisson` rows on this shared host are
noise-dominated (an unchanged control row, `axpy`, moved −30%
between runs; grid/128 intervals span ±10%+): stencil-grid/64 moved
−3.3% with disjoint intervals, the 128 rows are inconclusive and are
not claimed.

## Residual leaders

With the fused update at SIMD width, the recorded per-iteration
shares point next at `combine_direction` (15%, still a scalar loop;
an `axpby`-form kernel is the corresponding upstream shape) and the
dots/norm pass (14%). Filed as follow-up items on the board.
