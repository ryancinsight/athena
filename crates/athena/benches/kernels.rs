//! Per-kernel cost breakdown for the CG Poisson solve over the Leto CPU backend.
//!
//! Times each [`KrylovBackend`] operation CG invokes per iteration at the
//! grid/128 scale (16,384 unknowns) with deterministic dense inputs, plus the
//! Poisson CSR matvec. Summing medians by the per-iteration call census
//! attributes the `cg_poisson` solve time without a sampling profiler; the
//! dominant term names the next optimization (item
//! ATHENA-CG-PROFILE-20261019).
//!
//! Inputs are fixed across samples; mutating kernels receive a pristine
//! target per sample via untimed setup, with views built inside the timed
//! region (pointer-width construction, negligible against microsecond
//! kernels). Every fallible call asserts success so a failing kernel fails
//! loudly instead of recording a number.
//!
//! Run with `cargo bench -p athena-krylov --bench kernels --features cpu`
//! on a quiet host and report median + CI from Criterion's standard output.
//! [`KrylovBackend`]: `athena::KrylovBackend`

use athena::{
    KrylovBackend, LinearOperator,
    cpu::{CsrOperator, LetoBackend},
};
use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use eunomia::FloatElement;
use leto::Array1;
use leto_ops::CsrMatrix;

/// Benchmark scale: the grid/128 Poisson system (16,384 unknowns).
const UNKNOWNS: usize = 16384;
/// Representative scalar factors, matching mid-solve CG magnitudes.
const ALPHA: f64 = 0.25;
const BETA: f64 = 0.75;

/// Deterministic dense input: golden-ratio hash, sine-scattered.
fn dense() -> Array1<f64> {
    Array1::from_shape_fn([UNKNOWNS], |[i]| {
        (f64::from_count(i) * 0.618_033_988_749_895).sin()
    })
}

/// Five-point Dirichlet Laplacian in CSR row-major order with ascending
/// column indices: up, left, diagonal, right, down.
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

/// Unwrap a kernel result with its name attached; a failing kernel must fail
/// the benchmark, never record a number.
fn kernel<T, E: core::fmt::Debug>(name: &str, result: Result<T, E>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("kernel {name} failed: {error:?}"),
    }
}

/// Shared fixture: backend, deterministic inputs, and the Poisson operator.
/// Views are built per benchmark from these arrays.
struct Fixture {
    backend: LetoBackend<f64>,
    lhs: Array1<f64>,
    rhs: Array1<f64>,
    operator: CsrOperator<f64>,
}

type WallGroup<'a> = criterion::BenchmarkGroup<'a, criterion::measurement::WallTime>;

fn bench_reductions(group: &mut WallGroup<'_>, fixture: &Fixture) {
    let backend = &fixture.backend;
    let view_lhs = backend.view(&fixture.lhs);
    let view_rhs = backend.view(&fixture.rhs);

    group.bench_function("dot", |bencher| {
        let prepared = kernel("prepare_dot", backend.prepare_dot(view_lhs, view_rhs));
        bencher.iter(|| {
            black_box(kernel(
                "dot",
                backend.dot_prepared(&prepared, black_box(view_lhs), black_box(view_rhs)),
            ));
        });
    });

    group.bench_function("norm_l2", |bencher| {
        let prepared = kernel("prepare_norm", backend.prepare_norm_l2(view_lhs));
        bencher.iter(|| {
            black_box(kernel(
                "norm",
                backend.norm_l2_prepared(&prepared, black_box(view_lhs)),
            ));
        });
    });
}

fn bench_updates(group: &mut WallGroup<'_>, fixture: &Fixture) {
    let backend = &fixture.backend;
    let lhs = &fixture.lhs;
    let view_lhs = backend.view(lhs);
    let view_rhs = backend.view(&fixture.rhs);

    group.bench_function("axpy", |bencher| {
        bencher.iter_batched(
            || fixture.lhs.clone(),
            |mut target| {
                kernel(
                    "axpy",
                    backend.axpy(
                        backend.view_mut(black_box(&mut target)),
                        black_box(view_rhs),
                        ALPHA,
                    ),
                );
                black_box(target);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("scale", |bencher| {
        bencher.iter_batched(
            || fixture.lhs.clone(),
            |mut target| {
                kernel(
                    "scale",
                    backend.scale(backend.view_mut(black_box(&mut target)), BETA),
                );
                black_box(target);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("copy", |bencher| {
        bencher.iter_batched(
            || fixture.lhs.clone(),
            |mut target| {
                kernel(
                    "copy",
                    backend.copy(
                        black_box(view_rhs),
                        backend.view_mut(black_box(&mut target)),
                    ),
                );
                black_box(target);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("residual", |bencher| {
        bencher.iter_batched(
            || fixture.lhs.clone(),
            |mut target| {
                kernel(
                    "residual",
                    backend.residual(
                        black_box(view_rhs),
                        black_box(view_lhs),
                        backend.view_mut(black_box(&mut target)),
                    ),
                );
                black_box(target);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_fused(group: &mut WallGroup<'_>, fixture: &Fixture) {
    let backend = &fixture.backend;
    let lhs = &fixture.lhs;
    let rhs = &fixture.rhs;
    let view_lhs = backend.view(lhs);
    let view_rhs = backend.view(rhs);

    group.bench_function("fused_cg_update", |bencher| {
        bencher.iter_batched(
            || (fixture.lhs.clone(), fixture.rhs.clone()),
            |(mut solution, mut residual)| {
                kernel(
                    "fused",
                    backend.fused_cg_update(
                        backend.view_mut(black_box(&mut solution)),
                        black_box(view_lhs),
                        backend.view_mut(black_box(&mut residual)),
                        black_box(view_rhs),
                        ALPHA,
                    ),
                );
                black_box((solution, residual));
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("combine_direction", |bencher| {
        bencher.iter_batched(
            || fixture.lhs.clone(),
            |mut direction| {
                kernel(
                    "combine",
                    backend.combine_direction(
                        backend.view_mut(black_box(&mut direction)),
                        black_box(view_rhs),
                        BETA,
                    ),
                );
                black_box(direction);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_matvec(group: &mut WallGroup<'_>, fixture: &Fixture) {
    let backend = &fixture.backend;
    let view_lhs = backend.view(&fixture.lhs);

    group.bench_function("matvec_csr", |bencher| {
        bencher.iter_batched(
            || fixture.lhs.clone(),
            |mut output| {
                kernel(
                    "matvec",
                    fixture.operator.apply(
                        backend,
                        black_box(view_lhs),
                        backend.view_mut(black_box(&mut output)),
                    ),
                );
                black_box(output);
            },
            BatchSize::SmallInput,
        );
    });
}

fn kernels(criterion: &mut Criterion) {
    let backend = LetoBackend::<f64>::default();
    let elements = match u64::try_from(UNKNOWNS) {
        Ok(elements) => elements,
        Err(error) => panic!("benchmark scale exceeds the throughput counter: {error:?}"),
    };
    let matrix = poisson_2d(128);
    let fixture = Fixture {
        backend,
        lhs: dense(),
        rhs: dense(),
        operator: kernel("csr_build", CsrOperator::new(matrix)),
    };
    let mut group = criterion.benchmark_group("kernels_cpu");
    group.throughput(Throughput::Elements(elements));
    bench_reductions(&mut group, &fixture);
    bench_updates(&mut group, &fixture);
    bench_fused(&mut group, &fixture);
    bench_matvec(&mut group, &fixture);
    group.finish();
}

criterion_group!(benches, kernels);
criterion_main!(benches);
