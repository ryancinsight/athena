//! Smoother sweep fixtures and the owned-adapter equivalence oracle.

use super::*;

fn assert_vector_eq<const N: usize>(vector: &[f64], expected: [f64; N]) {
    assert_eq!(vector.len(), N);
    for (idx, expected_value) in expected.into_iter().enumerate() {
        assert_eq!(vector[idx], expected_value);
    }
}

fn create_test_matrix() -> CsrMatrix<f64> {
    let n = 5;
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for i in 0..n {
        if i > 0 {
            col_indices.push(i - 1);
            values.push(-1.0);
        }
        col_indices.push(i);
        values.push(2.0);
        if i < n - 1 {
            col_indices.push(i + 1);
            values.push(-1.0);
        }
        row_ptr.push(col_indices.len());
    }
    CsrMatrix::from_parts(values, col_indices, row_ptr, n, n)
        .expect("invariant: tridiagonal fixture CSR structure is valid")
}

const TEST_RHS: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

/// The fixture's right-hand side as an owned operator operand.
fn rhs_array() -> Array1<f64> {
    Array1::from_shape_vec([TEST_RHS.len()], TEST_RHS.to_vec())
        .expect("invariant: fixture rhs shape matches")
}

#[test]
fn test_gauss_seidel_smoother() {
    let matrix = create_test_matrix();
    let b = rhs_array();
    let mut x = Array1::zeros([5]);

    GaussSeidelSmoother::new(1.0).apply(&matrix, &mut x, &b, 1);

    assert_vector_eq(
        x.as_slice().expect("contiguous"),
        [0.5, 1.25, 2.125, 3.0625, 4.031_25],
    );
}

#[test]
fn test_jacobi_smoother() {
    let matrix = create_test_matrix();
    let b = rhs_array();
    let mut x = Array1::zeros([5]);

    JacobiSmoother::new(1.0).apply(&matrix, &mut x, &b, 1);

    assert_vector_eq(x.as_slice().expect("contiguous"), [0.5, 1.0, 1.5, 2.0, 2.5]);
}

#[test]
fn test_symmetric_gauss_seidel() {
    let matrix = create_test_matrix();
    let b = rhs_array();
    let mut x = Array1::zeros([5]);

    SymmetricGaussSeidelSmoother::new(1.0).apply(&matrix, &mut x, &b, 1);

    assert_vector_eq(
        x.as_slice().expect("contiguous"),
        [
            2.291_015_625,
            3.582_031_25,
            4.664_062_5,
            5.078_125,
            4.031_25,
        ],
    );
}

#[test]
fn test_sor_smoother() {
    let matrix = create_test_matrix();
    let b = rhs_array();
    let mut x = Array1::zeros([5]);

    SORSmoother::new(1.0).apply(&matrix, &mut x, &b, 1);

    assert_vector_eq(
        x.as_slice().expect("contiguous"),
        [0.5, 1.25, 2.125, 3.0625, 4.031_25],
    );
}

#[test]
fn test_chebyshev_smoother() {
    let matrix = create_test_matrix();
    let b = TEST_RHS;
    let mut x = [0.0_f64; 5];

    let (eigen_min, eigen_max) = ChebyshevSmoother::estimate_eigenvalues(&matrix);
    // Both bounds are the exact literals the estimator clamps through:
    // 0.1 is the Gershgorin floor fallback, 4.0 is the dyadic row bound.
    assert_eq!(eigen_min.to_bits(), 0.1_f64.to_bits());
    assert_eq!(eigen_max.to_bits(), 4.0_f64.to_bits());

    let smoother = ChebyshevSmoother::new(eigen_min, eigen_max, 3);
    smoother.apply_slices(&matrix, &mut x, &b, 1);

    let theta = f64::midpoint(eigen_max, eigen_min);
    let delta = (eigen_max - eigen_min) / 2.0;
    let sigma = theta / delta;
    let rho_old = 1.0 / sigma;
    let rho_new = 1.0 / (2.0 * sigma - rho_old);
    let alpha = rho_new / rho_old;
    let expected_first = alpha * b[0] / 2.0;
    assert!((x[0] - expected_first).abs() <= f64::EPSILON * 16.0);
}

/// The slice kernel and the owned adapter agree exactly on one sweep.
///
/// The kernel is the single implementation; the owned method is an adapter,
/// so any divergence between them is a copy or index defect.
#[test]
fn owned_adapter_matches_slice_kernel_bitwise() {
    let matrix = create_test_matrix();
    let b = rhs_array();
    let mut through_owned = Array1::zeros([5]);
    GaussSeidelSmoother::new(1.0).apply(&matrix, &mut through_owned, &b, 1);

    let mut through_slice = [0.0_f64; 5];
    GaussSeidelSmoother::new(1.0).apply_slices(&matrix, &mut through_slice, &TEST_RHS, 1);

    for (index, (owned, slice)) in through_owned
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(through_slice.iter())
        .enumerate()
    {
        assert_eq!(
            owned.to_bits(),
            slice.to_bits(),
            "owned adapter and slice kernel disagree at index {index}"
        );
    }
}
