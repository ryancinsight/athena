//! Contract for the accelerator Jacobi preconditioner.
//!
//! # Device coverage is not allowed to vanish silently
//!
//! The construction contract — square check, structural validation, zero
//! diagonal rejection, and the inverse values themselves — is host arithmetic
//! and runs unconditionally with no device involved.
//!
//! The remaining cases need a device. When no adapter is available they *fail*
//! rather than pass, so this suite cannot report success having executed zero
//! device cases. A host that genuinely has no accelerator sets
//! `ATHENA_HEPHAESTUS_DEVICE_OPTIONAL=1`, which downgrades those failures to a
//! printed `DEVICE CASE SKIPPED` line — an acknowledged, greppable skip rather
//! than a silent green.
//!
//! The backend and the preconditioner are both generic over the device; these
//! cases instantiate them at WGPU because that is the backend testable on this
//! host. Nothing under test names a device API.

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, KrylovBackend, Preconditioner, SolveReport,
};
use athena_hephaestus::{CsrOperator, HephaestusBackend, Jacobi, inverse_diagonal_from_csr};
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_wgpu::{GpuCsrMatrix, WgpuDevice, WgpuSparseOps, WgpuVectorOps};

type Backend = HephaestusBackend<WgpuDevice, WgpuVectorOps, f32>;
type Operator = CsrOperator<WgpuSparseOps, GpuCsrMatrix<f32>>;

/// Symmetric positive definite `[[4, 1], [1, 3]]`. Its diagonal is `[4, 3]`, so
/// the inverse diagonal is `[0.25, 1/3]` — one exactly representable entry and
/// one that is not, which is what makes the rounding contract observable.
const SPD_VALUES: [f32; 4] = [4.0, 1.0, 1.0, 3.0];
const SPD_COLUMNS: [usize; 4] = [0, 1, 0, 1];
const SPD_ROW_PTR: [usize; 3] = [0, 2, 4];

/// Dimension of the graded system. Small enough that unpreconditioned CG is
/// guaranteed to terminate well inside the iteration budget, large enough that
/// the conditioning, not exact termination, governs how fast it gets there.
const GRADED_DIMENSION: usize = 32;

/// Assert `got` matches `expected` within `ulps` units in the last place.
///
/// A correctly rounded floating-point operation carries at most half an ulp of
/// relative error, so each rounding step in a derivation contributes `0.5` here.
fn assert_within_ulps(got: f32, expected: f32, ulps: f32) {
    let bound = ulps * f32::EPSILON * expected.abs();
    assert!(
        (got - expected).abs() <= bound,
        "got {got}, expected {expected}, bound {bound}"
    );
}

// ---------------------------------------------------------------------------
// Host-side construction contract: no device, no skip, always executed.
// ---------------------------------------------------------------------------

#[test]
fn the_inverse_diagonal_is_the_reciprocal_of_each_diagonal_entry() {
    let inverse = inverse_diagonal_from_csr(&SPD_VALUES, &SPD_COLUMNS, &SPD_ROW_PTR, 2, 2)
        .expect("invariant: the matrix is square with a nonzero diagonal");

    assert_eq!(inverse.len(), 2);
    // One correctly rounded division per entry.
    assert_within_ulps(inverse[0], 0.25, 0.5);
    assert_within_ulps(inverse[1], 1.0 / 3.0, 0.5);
}

#[test]
fn the_diagonal_is_found_wherever_it_sits_within_its_row() {
    // Row 0 stores its diagonal second, row 1 stores it first, so a scan that
    // assumed either position would read the off-diagonal instead.
    let inverse =
        inverse_diagonal_from_csr(&[1.0_f32, 4.0, 8.0, 1.0], &[1, 0, 1, 0], &[0, 2, 4], 2, 2)
            .expect("invariant: the matrix is square with a nonzero diagonal");

    assert_within_ulps(inverse[0], 0.25, 0.5);
    assert_within_ulps(inverse[1], 0.125, 0.5);
}

#[test]
fn an_empty_system_yields_an_empty_inverse_diagonal() {
    let inverse = inverse_diagonal_from_csr::<f32>(&[], &[], &[0], 0, 0)
        .expect("invariant: a zero-dimensional system is square and has no diagonal");
    assert!(inverse.is_empty());
}

#[test]
fn a_rectangular_matrix_is_rejected_as_non_square() {
    let outcome = inverse_diagonal_from_csr(&[1.0_f32, 2.0], &[0, 1], &[0, 2], 1, 2);
    let Err(HephaestusError::DispatchFailed { message }) = outcome else {
        panic!("a rectangular matrix must be rejected as non-square");
    };
    assert!(message.contains("1 x 2"), "{message}");
}

#[test]
fn a_stored_zero_diagonal_entry_is_rejected_with_its_index() {
    // Diagonal [4, 0]: the second row would divide by zero.
    let outcome = inverse_diagonal_from_csr(&[4.0_f32, 0.0], &[0, 1], &[0, 1, 2], 2, 2);
    let Err(HephaestusError::InvalidConfiguration { message }) = outcome else {
        panic!("a zero diagonal entry must be rejected");
    };
    assert!(message.contains("entry 1"), "{message}");
}

#[test]
fn a_structurally_absent_diagonal_entry_is_rejected_with_its_index() {
    // Row 0 stores only column 1, so its diagonal is an implicit zero and is
    // rejected on the same terms as a stored one.
    let outcome = inverse_diagonal_from_csr(&[1.0_f32, 1.0], &[1, 0], &[0, 1, 2], 2, 2);
    let Err(HephaestusError::InvalidConfiguration { message }) = outcome else {
        panic!("an absent diagonal entry must be rejected");
    };
    assert!(message.contains("entry 0"), "{message}");
}

#[test]
fn a_row_pointer_of_the_wrong_length_is_rejected() {
    let outcome = inverse_diagonal_from_csr(&SPD_VALUES, &SPD_COLUMNS, &[0, 2], 2, 2);
    assert!(matches!(
        outcome,
        Err(HephaestusError::InvalidConfiguration { .. })
    ));
}

#[test]
fn column_indices_disagreeing_with_the_values_are_rejected() {
    let outcome = inverse_diagonal_from_csr(&SPD_VALUES, &[0, 1, 0], &SPD_ROW_PTR, 2, 2);
    assert!(matches!(
        outcome,
        Err(HephaestusError::InvalidConfiguration { .. })
    ));
}

#[test]
fn a_row_spanning_past_the_stored_entries_is_rejected() {
    let outcome = inverse_diagonal_from_csr(&[1.0_f32], &[0], &[0, 5], 1, 1);
    let Err(HephaestusError::InvalidConfiguration { message }) = outcome else {
        panic!("a row span outside the stored entries must be rejected");
    };
    assert!(message.contains("row 0"), "{message}");
}

// ---------------------------------------------------------------------------
// Device cases.
// ---------------------------------------------------------------------------

/// Acquire the device backend, or account for its absence loudly.
///
/// A missing adapter is a hard failure unless `ATHENA_HEPHAESTUS_DEVICE_OPTIONAL`
/// is set, so device coverage cannot silently drop to zero.
fn device_backend(case: &str) -> Option<Backend> {
    match WgpuDevice::try_default("athena-hephaestus-preconditioner") {
        Ok(device) => {
            eprintln!("DEVICE CASE EXECUTED ({case}) on {}", device.backend_name());
            let operations = WgpuVectorOps::new(&device).expect("invariant: kernels compile");
            Some(HephaestusBackend::new(device, operations))
        }
        Err(error) => {
            assert!(
                std::env::var_os("ATHENA_HEPHAESTUS_DEVICE_OPTIONAL").is_some(),
                "no accelerator device for case {case}: {error}. Device coverage would be zero; \
                 set ATHENA_HEPHAESTUS_DEVICE_OPTIONAL=1 to accept host-only coverage."
            );
            eprintln!("DEVICE CASE SKIPPED ({case}): {error}");
            None
        }
    }
}

fn upload(backend: &Backend, values: &[f32]) -> <Backend as KrylovBackend>::Vector {
    backend
        .device()
        .upload(values)
        .expect("invariant: upload succeeds")
}

fn download(backend: &Backend, vector: &<Backend as KrylovBackend>::Vector) -> Vec<f32> {
    let mut host = vec![0.0_f32; backend.vector_len(vector)];
    backend
        .device()
        .download(vector, &mut host)
        .expect("invariant: readback succeeds");
    host
}

#[test]
fn the_device_resident_inverse_diagonal_matches_the_host_computation() {
    let Some(backend) = device_backend("inverse diagonal upload") else {
        return;
    };
    let jacobi = Jacobi::from_csr_parts(
        backend.device(),
        &SPD_VALUES,
        &SPD_COLUMNS,
        &SPD_ROW_PTR,
        2,
        2,
    )
    .expect("invariant: the matrix is square with a nonzero diagonal");

    assert_eq!(jacobi.dimension(), 2);
    let mut host = [0.0_f32; 2];
    backend
        .device()
        .download(jacobi.inverse_diagonal(), &mut host)
        .expect("invariant: readback succeeds");
    assert_within_ulps(host[0], 0.25, 0.5);
    assert_within_ulps(host[1], 1.0 / 3.0, 0.5);
}

#[test]
fn applying_the_preconditioner_scales_each_component_by_the_inverse_diagonal() {
    let Some(backend) = device_backend("apply") else {
        return;
    };
    let jacobi = Jacobi::from_csr_parts(
        backend.device(),
        &SPD_VALUES,
        &SPD_COLUMNS,
        &SPD_ROW_PTR,
        2,
        2,
    )
    .expect("invariant: the matrix is square with a nonzero diagonal");
    let residual = upload(&backend, &[6.0, 7.0]);
    let mut output = backend.allocate(2).expect("invariant: allocation");

    jacobi
        .apply(
            &backend,
            backend.view(&residual),
            backend.view_mut(&mut output),
        )
        .expect("invariant: matched operand lengths");

    // Two correctly rounded operations per component: the reciprocal and the
    // product, so at most one ulp of relative error.
    let host = download(&backend, &output);
    assert_within_ulps(host[0], 6.0 / 4.0, 1.0);
    assert_within_ulps(host[1], 7.0 / 3.0, 1.0);
}

#[test]
fn a_mismatched_operand_length_is_rejected_by_the_seam() {
    let Some(backend) = device_backend("length mismatch") else {
        return;
    };
    let jacobi = Jacobi::from_csr_parts(
        backend.device(),
        &SPD_VALUES,
        &SPD_COLUMNS,
        &SPD_ROW_PTR,
        2,
        2,
    )
    .expect("invariant: the matrix is square with a nonzero diagonal");
    let residual = upload(&backend, &[6.0, 7.0, 8.0]);
    let mut output = backend.allocate(3).expect("invariant: allocation");

    let outcome = jacobi.apply(
        &backend,
        backend.view(&residual),
        backend.view_mut(&mut output),
    );

    let Err(error) = outcome else {
        panic!("a length mismatch must not be dispatched");
    };
    let message = error.to_string();
    assert!(
        message.contains('3') && message.contains('2'),
        "the error must name the mismatched lengths (3 against 2): {message}"
    );
}

/// Symmetric tridiagonal system whose diagonal is graded `1, 2, ... n`.
///
/// Every off-diagonal is `-0.4` times the smaller of the two diagonal entries it
/// couples, so each row is strictly diagonally dominant and the matrix is
/// therefore symmetric positive definite. Gershgorin puts the spectrum of `A` in
/// `[0.2, 1.8n]` and the spectrum of `D⁻¹A` in `[0.2, 1.8]`, so Jacobi drops the
/// condition number from about `9n` to about `9`.
struct GradedSystem {
    values: Vec<f32>,
    columns: Vec<usize>,
    row_ptr: Vec<usize>,
    right_hand_side: Vec<f32>,
    solution: Vec<f32>,
}

fn graded_system() -> GradedSystem {
    let mut diagonal = Vec::with_capacity(GRADED_DIMENSION);
    let mut entry = 1.0_f32;
    for _ in 0..GRADED_DIMENSION {
        diagonal.push(entry);
        entry += 1.0;
    }
    let coupling: Vec<f32> = diagonal
        .windows(2)
        .map(|pair| -0.4 * pair[0].min(pair[1]))
        .collect();

    let mut values = Vec::new();
    let mut columns = Vec::new();
    let mut row_ptr = vec![0_usize];
    for (row, &center) in diagonal.iter().enumerate() {
        if row > 0 {
            let left = row - 1;
            values.push(coupling[left]);
            columns.push(left);
        }
        values.push(center);
        columns.push(row);
        if let Some(&right) = coupling.get(row) {
            values.push(right);
            columns.push(row + 1);
        }
        row_ptr.push(values.len());
    }

    // A solution varying across components, so a permutation or an off-by-one
    // in the assembly cannot pass unnoticed.
    let mut component = 1.0_f32;
    let mut solution = Vec::with_capacity(GRADED_DIMENSION);
    for _ in 0..GRADED_DIMENSION {
        solution.push(component);
        component += 0.25;
    }

    let right_hand_side = row_ptr
        .iter()
        .zip(row_ptr.iter().skip(1))
        .map(|(&start, &end)| {
            values[start..end]
                .iter()
                .zip(&columns[start..end])
                .map(|(&value, &column)| value * solution[column])
                .sum()
        })
        .collect();

    GradedSystem {
        values,
        columns,
        row_ptr,
        right_hand_side,
        solution,
    }
}

fn policy() -> ConvergencePolicy<f32> {
    ConvergencePolicy::new(1e-5, 1e-5, 200).expect("invariant: tolerance is finite and positive")
}

/// Solve `system` with `preconditioner`, check the answer, and return the report.
fn solve_graded<P: Preconditioner<Backend>>(
    backend: &Backend,
    operator: &Operator,
    preconditioner: &P,
    right_hand_side: &<Backend as KrylovBackend>::Vector,
    system: &GradedSystem,
) -> SolveReport<f32> {
    let mut solution = backend
        .allocate(GRADED_DIMENSION)
        .expect("invariant: allocation");
    let mut workspace = CgWorkspace::new(backend, GRADED_DIMENSION).expect("invariant: workspace");
    let report = Cg::<Backend>::solve_into(
        backend,
        operator,
        preconditioner,
        right_hand_side,
        &mut solution,
        &mut workspace,
        policy(),
    )
    .expect("invariant: valid dimensions");
    assert!(report.converged(), "got {:?}", report.termination);

    // The recurrence stops at a relative residual of 1e-5. Propagating that
    // through the condition number of A — at most 9 * GRADED_DIMENSION by the
    // Gershgorin bounds above — leaves a norm-wise relative solution error
    // below 3e-3; 5e-3 covers the f32 rounding on top of it.
    let computed = download(backend, &solution);
    let error: f32 = computed
        .iter()
        .zip(&system.solution)
        .map(|(&got, &expected)| (got - expected) * (got - expected))
        .sum::<f32>()
        .sqrt();
    let reference: f32 = system
        .solution
        .iter()
        .map(|&value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(
        error <= 5e-3 * reference,
        "relative solution error {}",
        error / reference
    );

    report
}

#[test]
fn jacobi_reduces_the_iteration_count_on_a_graded_system() {
    // The property that distinguishes a preconditioner from a no-op: the same
    // recurrence, the same operator, the same right-hand side, fewer
    // iterations to the same answer.
    let Some(backend) = device_backend("preconditioned CG") else {
        return;
    };
    let system = graded_system();
    let operator = Operator::from_parts(
        WgpuSparseOps,
        backend.device(),
        &system.values,
        &system.columns,
        &system.row_ptr,
        GRADED_DIMENSION,
        GRADED_DIMENSION,
    )
    .expect("invariant: the assembled CSR parts are valid");
    let jacobi = Jacobi::from_csr_parts(
        backend.device(),
        &system.values,
        &system.columns,
        &system.row_ptr,
        GRADED_DIMENSION,
        GRADED_DIMENSION,
    )
    .expect("invariant: the assembled matrix has a nonzero diagonal");
    let right_hand_side = upload(&backend, &system.right_hand_side);

    let unpreconditioned = solve_graded(&backend, &operator, &Identity, &right_hand_side, &system);
    let preconditioned = solve_graded(&backend, &operator, &jacobi, &right_hand_side, &system);

    assert!(
        preconditioned.preconditioner_applications > 0,
        "the preconditioner was never applied"
    );
    assert!(
        preconditioned.iterations < unpreconditioned.iterations,
        "preconditioned {} iterations, unpreconditioned {}",
        preconditioned.iterations,
        unpreconditioned.iterations
    );
}
