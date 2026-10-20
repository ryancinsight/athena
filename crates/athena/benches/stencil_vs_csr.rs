//! Matrix-free 5-point stencil matvec versus the CSR path, back-to-back.
//!
//! Spike for ATHENA-MATRIXFREE-EVAL-20261020: the grid/128 CG profile
//! attributes 68% of per-iteration cost to `matvec_csr`, and the bound
//! diagnosis shows that size is latency-bound (~10 cycles/row against a
//! several-times-higher L3 ceiling), so narrowing CSR indices is not its
//! lever. This bench measures a direct stencil application — no values,
//! no index arrays, no gather — against the identical CSR operator on the
//! identical fixture, in one Criterion run so session drift cancels.
//!
//! The stencil operator lives in this bench file: the spike records
//! numbers plus the seam-design decision, and production code is
//! untouched. Summation order differs from CSR's unrolled gather-dot,
//! so equivalence is asserted within rounding-level tolerance, not
//! bitwise.
//!
//! Run with `cargo bench -p athena-krylov --bench stencil_vs_csr
//! --features cpu` on a quiet host.

use athena::{
    KrylovBackend, LinearOperator,
    cpu::{CsrOperator, LetoBackend, LetoBackendError},
};
use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use eunomia::FloatElement;
use leto::Array1;
use leto::Storage;
use leto_ops::CsrMatrix;

/// Benchmark scale: the grid/128 Poisson system (16,384 unknowns).
const GRID: usize = 128;
const UNKNOWNS: usize = GRID * GRID;

/// Rounding-level equivalence bound: five-term sums of unit-magnitude
/// values in different orders differ by ~1e-14; 1e-12 keeps three
/// orders of headroom while catching any structural mismatch.
const EQUIVALENCE_TOL: f64 = 1e-12;

/// Deterministic dense input: golden-ratio hash, sine-scattered.
/// Identical to the `kernels` bench fixture.
fn dense() -> Array1<f64> {
    Array1::from_shape_fn([UNKNOWNS], |[i]| {
        (f64::from_count(i) * 0.618_033_988_749_895).sin()
    })
}

/// Five-point Dirichlet Laplacian in CSR row-major order with ascending
/// column indices: up, left, diagonal, right, down.
/// Identical to the `kernels` bench fixture.
fn poisson_2d(grid: usize) -> CsrMatrix<f64> {
    let unknowns = grid * grid;
    let mut values = Vec::with_capacity(5 * unknowns);
    let mut columns = Vec::with_capacity(5 * unknowns);
    let mut row_ptr = Vec::with_capacity(unknowns + 1);
    row_ptr.push(0);
    for row in 0..grid {
        for column in 0..grid {
            let index = row * grid + column;
            if row > 0 {
                values.push(-1.0);
                columns.push(index - grid);
            }
            if column > 0 {
                values.push(-1.0);
                columns.push(index - 1);
            }
            values.push(4.0);
            columns.push(index);
            if column + 1 < grid {
                values.push(-1.0);
                columns.push(index + 1);
            }
            if row + 1 < grid {
                values.push(-1.0);
                columns.push(index + grid);
            }
            row_ptr.push(values.len());
        }
    }
    match CsrMatrix::from_parts(values, columns, row_ptr, unknowns, unknowns) {
        Ok(matrix) => matrix,
        Err(error) => panic!("pinned Poisson fixture is malformed: {error:?}"),
    }
}

/// Matrix-free five-point Dirichlet Laplacian on a `grid x grid` lattice.
///
/// Row-major unknown order, matching [`poisson_2d`]: `y[i] = 4x[i]`
/// minus each in-bounds orthogonal neighbor.
struct StencilOperator {
    grid: usize,
}

impl LinearOperator<LetoBackend<f64>> for StencilOperator {
    fn dimension(&self) -> usize {
        self.grid * self.grid
    }

    fn apply(
        &self,
        _backend: &LetoBackend<f64>,
        input: <LetoBackend<f64> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<f64> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let x = input
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let y = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let grid = self.grid;
        for row in 0..grid {
            for column in 0..grid {
                let index = row * grid + column;
                let mut acc = 4.0 * x[index];
                if row > 0 {
                    acc -= x[index - grid];
                }
                if column > 0 {
                    acc -= x[index - 1];
                }
                if column + 1 < grid {
                    acc -= x[index + 1];
                }
                if row + 1 < grid {
                    acc -= x[index + grid];
                }
                y[index] = acc;
            }
        }
        Ok(())
    }
}

/// Unwrap a kernel result with its name attached; a failing kernel must
/// fail the benchmark, never record a number.
fn kernel<T, E: core::fmt::Debug>(name: &str, result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("kernel {name} failed: {error:?}"),
    }
}

fn stencil_vs_csr(criterion: &mut Criterion) {
    let backend = LetoBackend::<f64>::default();
    let input = dense();
    let matrix = poisson_2d(GRID);
    let csr = kernel("csr_build", CsrOperator::new(matrix));
    let stencil = StencilOperator { grid: GRID };
    assert_eq!(csr.dimension(), stencil.dimension());

    // Equivalence gate: both operators must agree within rounding
    // before any timing comparison is meaningful.
    let view = backend.view(&input);
    let mut y_csr = Array1::from_shape_fn([UNKNOWNS], |_| 0.0);
    let mut y_stencil = Array1::from_shape_fn([UNKNOWNS], |_| 0.0);
    kernel(
        "csr_reference",
        csr.apply(&backend, view, backend.view_mut(&mut y_csr)),
    );
    kernel(
        "stencil_reference",
        stencil.apply(&backend, view, backend.view_mut(&mut y_stencil)),
    );
    let worst = y_csr
        .storage()
        .as_slice()
        .iter()
        .zip(y_stencil.storage().as_slice())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst <= EQUIVALENCE_TOL,
        "stencil/CSR mismatch {worst} exceeds {EQUIVALENCE_TOL}"
    );

    let elements = match u64::try_from(UNKNOWNS) {
        Ok(elements) => elements,
        Err(error) => panic!("benchmark scale exceeds the throughput counter: {error:?}"),
    };
    let mut group = criterion.benchmark_group("stencil_vs_csr_cpu");
    group.throughput(Throughput::Elements(elements));

    let view_lhs = backend.view(&input);
    group.bench_function("matvec_csr", |bencher| {
        bencher.iter_batched(
            || input.clone(),
            |mut output| {
                kernel(
                    "matvec",
                    csr.apply(
                        &backend,
                        black_box(view_lhs),
                        backend.view_mut(black_box(&mut output)),
                    ),
                );
                black_box(output);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("matvec_stencil", |bencher| {
        bencher.iter_batched(
            || input.clone(),
            |mut output| {
                kernel(
                    "matvec",
                    stencil.apply(
                        &backend,
                        black_box(view_lhs),
                        backend.view_mut(black_box(&mut output)),
                    ),
                );
                black_box(output);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, stencil_vs_csr);
criterion_main!(benches);
