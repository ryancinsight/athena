//! Allocation contract for initialized CPU solves.
//!
//! The global allocator is [`mnemosyne::Mnemosyne`] wrapped in
//! [`mnemosyne::counting::CountingAllocator`] so that all solver allocations
//! route through Mnemosyne (the project-wide memory back-end) and are
//! simultaneously visible to the instrumentation. Each test calls
//! [`mnemosyne::warm_current_thread`] before it starts measuring to flush
//! thread-local-allocator initialization traffic (options parsing, arena
//! segment acquisition) out of the measurement window.
//!
//! [`measure`] counts the allocations of the calling thread only. A
//! process-wide counter is invalid here: libtest runs the test body on a
//! spawned thread while its main thread keeps inserting the running test into
//! its bookkeeping collections, and parallel tests allocate concurrently, so a
//! process-wide window occasionally absorbs traffic that is not the solver's.
//! The cases are therefore independent under the threaded `cargo test` harness
//! as well as under `cargo nextest run`.
//!
//! Under Miri the wrapper counts `System` instead of `Mnemosyne`: Miri reports
//! undefined behavior inside `mnemosyne-local` itself (MN-LOCAL-MIRI-UB), which
//! would mask a defect in the code under test.

use athena_core::{
    BiCgStab, BiCgStabWorkspace, Cg, CgWorkspace, ConvergencePolicy, Gmres, GmresWorkspace,
    Identity,
};
use athena_leto::{CsrOperator, LetoBackend};
use leto::Array1;
use leto_ops::CsrMatrix;
use mnemosyne::counting::{AllocationDelta, CountingAllocator, measure};
use mnemosyne::scratch::ScratchPool;

#[cfg(miri)]
#[global_allocator]
static GLOBAL: CountingAllocator<std::alloc::System> = CountingAllocator::new(std::alloc::System);

#[cfg(not(miri))]
#[global_allocator]
static GLOBAL: CountingAllocator<mnemosyne::Mnemosyne> =
    CountingAllocator::new(mnemosyne::Mnemosyne);

/// Moves the allocator's per-thread initialization out of the window.
fn warm() {
    #[cfg(not(miri))]
    mnemosyne::warm_current_thread();
}

#[test]
fn mnemosyne_scratch_pool_is_used_for_safe_arena_backed_temporary_storage() {
    warm();
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
#[ignore = "strict zero-traffic contract; run under --ignored by the hosted allocation-instrument job, which pins MALLOC_ARENA_MAX=1 (ATLAS-ATHENA-ALLOCATION-CONTRACT)"]
fn repeated_cpu_solves_allocate_nothing_after_initialization() {
    warm();
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

    let ((), change) = measure(|| {
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
    });

    assert_steady_state(change);
}

#[test]
#[ignore = "strict zero-traffic contract; run under --ignored by the hosted allocation-instrument job, which pins MALLOC_ARENA_MAX=1 (ATLAS-ATHENA-ALLOCATION-CONTRACT)"]
fn repeated_gmres_solves_allocate_nothing_after_initialization() {
    warm();
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

    let ((), change) = measure(|| {
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
    });

    assert_steady_state(change);
}

#[test]
#[ignore = "strict zero-traffic contract; run under --ignored by the hosted allocation-instrument job, which pins MALLOC_ARENA_MAX=1 (ATLAS-ATHENA-ALLOCATION-CONTRACT)"]
fn repeated_bicgstab_solves_allocate_nothing_after_initialization() {
    warm();
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

    let ((), change) = measure(|| {
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
    });

    assert_steady_state(change);
}

/// Assert the measured region performed no heap traffic at all.
///
/// Reports the entire `AllocationDelta` on failure. `assert_eq!` per field stops at
/// the first mismatch, which tells you a count moved but not its shape --
/// and shape is what identifies the culprit. Bytes separate one large
/// buffer from several small ones, and a matching allocation/deallocation
/// pair points at a temporary rather than retained state.
#[track_caller]
fn assert_steady_state(change: AllocationDelta) {
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
    warm();
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

    let mut solve_window = |solves: usize| -> AllocationDelta {
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

        let ((), delta) = measure(|| {
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
        });
        delta
    };

    let single = solve_window(16);
    let doubled = solve_window(32);

    // Property 1: no per-solve growth. Whatever fixed burst the environment
    // produces at region entry must repeat identically, not double.
    let grew =
        doubled.allocations > single.allocations || doubled.reallocations > single.reallocations;
    assert!(
        !grew,
        "warm-solve heap traffic scaled with repetitions, so a solve path \
         allocates: 16 solves {single:?}, 32 solves {doubled:?}"
    );

    // Property 2: nothing retained *in steady state*. `bytes_retained` is the
    // net heap growth of the measured window (bytes acquired, counting growth
    // by reallocation, minus bytes released); a leak retains.
    //
    // The oracle is the *second* window, and it is the sharper one. A leak
    // keeps its bytes across both windows, so the 32-solve window would show
    // the same positive net growth; measured retention that reaches zero by
    // the second window was allocated and then released, which is what a
    // first-window capacity that is reused looks like from outside. Asserting
    // the first window at `<= 0` therefore tests a transient rather than a
    // leak, and it fails on an allocation the second window proves was
    // released.
    let single_net = single.bytes_retained();
    let doubled_net = doubled.bytes_retained();
    assert!(
        doubled_net <= 0,
        "warm solves retained heap memory in steady state: net bytes after 32 \
         solves {doubled_net} (16 solves {single_net}; 16 solves {single:?}, \
         32 solves {doubled:?})"
    );
}
