//! Criterion baseline: restarted GMRES on a 2D Poisson system over the Leto
//! CPU backend.
//!
//! Same pinned fixture as `cg_poisson`: a five-point Dirichlet Laplacian on a
//! square grid with an all-ones right-hand side, solved from a zero initial
//! guess with unpreconditioned `Gmres<LetoBackend<f64>, RESTART>`. GMRES
//! exercises the `axpy`/`scale` vector kernels in its Arnoldi
//! orthogonalization, which CG's fused updates bypass, so this benchmark is
//! the oracle for vector-kernel routing changes while `cg_poisson` guards the
//! fused path. Each sample asserts convergence; the workspace is reused
//! across samples exactly as production reuses it.
//!
//! Run with `cargo bench -p athena-krylov --bench gmres_poisson --features cpu`
//! on a quiet host and report median + CI from Criterion's standard output.

use athena::{
    ConvergencePolicy, Gmres, GmresWorkspace, Identity, Termination,
    cpu::{CsrOperator, LetoBackend},
};
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use leto::Array1;
use leto_ops::CsrMatrix;

/// Grids benchmarked: 64 (4,096 unknowns) and 128 (16,384 unknowns).
const GRIDS: [usize; 2] = [64, 128];
/// Representative restart width for an unpreconditioned Poisson solve.
const RESTART: usize = 50;

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

/// Everything a sample needs except its fresh solution vector.
struct Case {
    backend: LetoBackend<f64>,
    operator: CsrOperator<f64>,
    right_hand_side: Array1<f64>,
    workspace: GmresWorkspace<LetoBackend<f64>, RESTART>,
    policy: ConvergencePolicy<f64>,
    unknowns: usize,
}

fn build_case(grid: usize) -> Case {
    let unknowns = grid * grid;
    let backend = LetoBackend::<f64>::default();
    let matrix = poisson_2d(grid);
    let operator = match CsrOperator::new(matrix) {
        Ok(operator) => operator,
        Err(error) => panic!("pinned Poisson operator rejected: {error:?}"),
    };
    let right_hand_side = match Array1::from_shape_vec([unknowns], vec![1.0; unknowns]) {
        Ok(rhs) => rhs,
        Err(error) => panic!("pinned right-hand side rejected: {error:?}"),
    };
    let workspace = match GmresWorkspace::new(&backend, unknowns) {
        Ok(workspace) => workspace,
        Err(error) => panic!("GMRES workspace allocation failed: {error:?}"),
    };
    let policy = match ConvergencePolicy::new(1.0e-10, 1.0e-8, 4 * unknowns) {
        Ok(policy) => policy,
        Err(error) => panic!("pinned convergence policy rejected: {error:?}"),
    };
    Case {
        backend,
        operator,
        right_hand_side,
        workspace,
        policy,
        unknowns,
    }
}

fn gmres_poisson(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("gmres_poisson_cpu");
    for grid in GRIDS {
        let mut case = build_case(grid);
        let elements = match u64::try_from(case.unknowns) {
            Ok(elements) => elements,
            Err(error) => panic!("grid {grid} exceeds the throughput counter: {error:?}"),
        };
        group.throughput(Throughput::Elements(elements));
        group.bench_with_input(BenchmarkId::new("grid", grid), &grid, |bencher, _| {
            bencher.iter_batched(
                || Array1::zeros([case.unknowns]),
                |mut solution| {
                    let report = match Gmres::<LetoBackend<f64>, RESTART>::solve_into(
                        &case.backend,
                        &case.operator,
                        &Identity,
                        &case.right_hand_side,
                        &mut solution,
                        &mut case.workspace,
                        case.policy,
                    ) {
                        Ok(report) => report,
                        Err(error) => panic!("benchmarked solve failed: {error:?}"),
                    };
                    assert_eq!(report.termination, Termination::Converged);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, gmres_poisson);
criterion_main!(benches);
