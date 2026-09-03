//! Allocation contract for initialized CPU solves.
//!
//! The global allocator is [`mnemosyne::Mnemosyne`] wrapped in
//! [`stats_alloc::StatsAlloc`] so that all solver allocations route through
//! Mnemosyne (the project-wide memory back-end) and are simultaneously
//! visible to the instrumentation. Each test calls
//! [`mnemosyne::warm_current_thread`] before it starts measuring to flush
//! thread-local-allocator initialization traffic (options parsing, arena
//! segment acquisition) out of the measurement window.
//!
//! `stats_alloc::Region` measures the process-global allocator, so these
//! cases are only meaningful when each runs in its own process. `cargo
//! nextest run` provides that isolation; under the threaded `cargo test`
//! harness one case observes the setup allocations of another and fails
//! spuriously.

use athena_core::{
    BiCgStab, BiCgStabWorkspace, Cg, CgWorkspace, ConvergencePolicy, Gmres, GmresWorkspace,
    Identity,
};
use athena_leto::{CsrOperator, LetoBackend};
use leto::Array1;
use leto_ops::CsrMatrix;
use mnemosyne::Mnemosyne;
use mnemosyne::scratch::ScratchPool;
use stats_alloc::{Region, StatsAlloc};

static INSTRUMENTED_MNEMOSYNE: StatsAlloc<Mnemosyne> = StatsAlloc::new(Mnemosyne);

#[global_allocator]
static GLOBAL: &StatsAlloc<Mnemosyne> = &INSTRUMENTED_MNEMOSYNE;

#[test]
fn mnemosyne_scratch_pool_is_used_for_safe_arena_backed_temporary_storage() {
    mnemosyne::warm_current_thread();
    let pool = ScratchPool::<f64>::new();

    pool.with_scratch(16, |scratch| {
        assert_eq!(scratch.len(), 16);
        scratch.fill(1.25);
        assert!(
            scratch
                .iter()
                .all(|value| (*value - 1.25).abs() <= f64::EPSILON)
        );
    });
}

#[test]
fn repeated_cpu_solves_allocate_nothing_after_initialization() {
    mnemosyne::warm_current_thread();
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(
        vec![4.0_f64, 1.0, 1.0, 3.0],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([2], vec![6.0, 7.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(64.0 * f64::EPSILON, 64.0 * f64::EPSILON, 4)
        .expect("invariant: valid policy");

    let warm_up = Cg::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("warm-up solve must succeed");
    assert!(warm_up.converged());
    solution.fill(0.0);

    let region = Region::new(GLOBAL);
    for _ in 0..16 {
        let report = Cg::<LetoBackend<f64>>::solve_into(
            &backend,
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            &mut workspace,
            policy,
        )
        .expect("measured solve must succeed");
        assert!(report.converged());
        solution.fill(0.0);
    }
    let change = region.change();

    assert_steady_state(change);
}

#[test]
fn repeated_gmres_solves_allocate_nothing_after_initialization() {
    mnemosyne::warm_current_thread();
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(
        vec![4.0_f64, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![0, 2, 5, 7],
        3,
        3,
    )
    .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([3], vec![2.0, -1.0, 4.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        GmresWorkspace::<_, 3>::new(&backend, 3).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(4096.0 * f64::EPSILON, 4096.0 * f64::EPSILON, 6)
        .expect("invariant: valid policy");

    let warm_up = Gmres::<LetoBackend<f64>, 3>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("warm-up solve must succeed");
    assert!(warm_up.converged());
    solution.fill(0.0);

    let region = Region::new(GLOBAL);
    for _ in 0..16 {
        let report = Gmres::<LetoBackend<f64>, 3>::solve_into(
            &backend,
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            &mut workspace,
            policy,
        )
        .expect("measured solve must succeed");
        assert!(report.converged());
        solution.fill(0.0);
    }
    let change = region.change();

    assert_steady_state(change);
}

#[test]
fn repeated_bicgstab_solves_allocate_nothing_after_initialization() {
    mnemosyne::warm_current_thread();
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(
        vec![4.0_f64, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![0, 2, 5, 7],
        3,
        3,
    )
    .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([3], vec![2.0, -1.0, 4.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        BiCgStabWorkspace::new(&backend, 3).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(4096.0 * f64::EPSILON, 4096.0 * f64::EPSILON, 32)
        .expect("invariant: valid policy");

    let warm_up = BiCgStab::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("warm-up solve must succeed");
    assert!(warm_up.converged());
    solution.fill(0.0);

    let region = Region::new(GLOBAL);
    for _ in 0..16 {
        let report = BiCgStab::<LetoBackend<f64>>::solve_into(
            &backend,
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            &mut workspace,
            policy,
        )
        .expect("measured solve must succeed");
        assert!(report.converged());
        solution.fill(0.0);
    }
    let change = region.change();

    assert_steady_state(change);
}

/// Assert the measured region performed no heap traffic at all.
///
/// Reports the entire `Stats` on failure. `assert_eq!` per field stops at
/// the first mismatch, which tells you a count moved but not its shape --
/// and shape is what identifies the culprit. Bytes separate one large
/// buffer from several small ones, and a matching allocation/deallocation
/// pair points at a temporary rather than retained state.
#[track_caller]
fn assert_steady_state(change: stats_alloc::Stats) {
    assert_eq!(
        (
            change.allocations,
            change.reallocations,
            change.deallocations
        ),
        (0, 0, 0),
        "warm solves must not touch the heap; observed {change:?}"
    );
}

/// Classify warm-solve heap traffic on hosts where it is nonzero.
///
/// The strict contract above asserts zero traffic; on some hosted Linux
/// runners the GMRES case observes a small fixed burst instead (4 allocs,
/// ~900 B) that Windows does not reproduce and that no inspected solver
/// path can produce — every buffer between `initialize` and the terminal
/// report lives in the caller-owned workspace. The signature of allocator
/// or runtime environment noise rather than solver leakage is:
///
/// 1. **Fixed size.** The traffic count does not grow when the number of
///    measured solves doubles. A solve-path allocation would scale.
/// 2. **Balanced retention.** Bytes allocated ≈ bytes deallocated; the
///    process returns what it borrowed. A leak retains.
///
/// This test measures 16 and then 32 warm solves in two separate regions
/// and asserts those two properties, failing only when traffic scales with
/// repetitions or memory is retained. It runs alongside the strict test
/// under `--ignored` from the hosted instrument job, which reports the
/// observed shape either way: green here means "environment noise,
/// bounded", red means "solve-path defect".
#[test]
#[ignore = "companion to repeated_gmres_solves_allocate_nothing_after_initialization; run under --ignored by the hosted allocation-instrument job"]
fn warm_solve_heap_traffic_is_bounded_and_not_retained() {
    mnemosyne::warm_current_thread();
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(
        vec![4.0_f64, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![0, 2, 5, 7],
        3,
        3,
    )
    .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([3], vec![2.0, -1.0, 4.0]).expect("invariant: exact shape");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        GmresWorkspace::<_, 3>::new(&backend, 3).expect("invariant: host allocation succeeds");
    let policy = ConvergencePolicy::new(4096.0 * f64::EPSILON, 4096.0 * f64::EPSILON, 6)
        .expect("invariant: valid policy");

    let mut measure = |solves: usize| -> stats_alloc::Stats {
        let warm_up = Gmres::<LetoBackend<f64>, 3>::solve_into(
            &backend,
            &operator,
            &Identity,
            &right_hand_side,
            &mut solution,
            &mut workspace,
            policy,
        )
        .expect("warm-up solve must succeed");
        assert!(warm_up.converged());
        solution.fill(0.0);

        let region = Region::new(GLOBAL);
        for _ in 0..solves {
            let report = Gmres::<LetoBackend<f64>, 3>::solve_into(
                &backend,
                &operator,
                &Identity,
                &right_hand_side,
                &mut solution,
                &mut workspace,
                policy,
            )
            .expect("measured solve must succeed");
            assert!(report.converged());
            solution.fill(0.0);
        }
        region.change()
    };

    let single = measure(16);
    let doubled = measure(32);

    // Property 1: no per-solve growth. Whatever fixed burst the environment
    // produces at region entry must repeat identically, not double.
    let grew =
        doubled.allocations > single.allocations || doubled.reallocations > single.reallocations;
    assert!(
        !grew,
        "warm-solve heap traffic scaled with repetitions, so a solve path \
         allocates: 16 solves {single:?}, 32 solves {doubled:?}"
    );

    // Property 2: nothing retained. Allocated bytes minus deallocated bytes
    // is the net heap growth of the measured window; a leak retains. The
    // subtraction is on `i128` because a wrapping `usize` difference would
    // read as "nothing retained" exactly when deallocated exceeds allocated.
    let net = |s: &stats_alloc::Stats| -> i128 {
        i128::try_from(s.bytes_allocated).unwrap_or(i128::MAX)
            - i128::try_from(s.bytes_deallocated).unwrap_or(i128::MAX)
    };
    let single_net = net(&single);
    let doubled_net = net(&doubled);
    assert!(
        single_net <= 0 && doubled_net <= 0,
        "warm solves retained heap memory: net bytes after 16 solves \
         {single_net}, after 32 solves {doubled_net} (16 solves {single:?}, \
         32 solves {doubled:?})"
    );
}
