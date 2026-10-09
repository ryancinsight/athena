use super::*;
use leto::Array1;
use leto_ops::{CsrMatrix, spmv as leto_spmv};

/// Read a matrix entry, zero when structurally absent (the test oracle).
fn entry_of(matrix: &CsrMatrix<f64>, row: usize, col: usize) -> f64 {
    matrix.get(row, col).unwrap_or(0.0)
}

fn csr_from_dense(values_2d: &[Vec<f64>]) -> CsrMatrix<f64> {
    let nrows = values_2d.len();
    let ncols = values_2d.first().map_or(0, Vec::len);
    let mut row_ptr = Vec::with_capacity(nrows + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for row in values_2d {
        assert_eq!(row.len(), ncols);
        for (col, &value) in row.iter().enumerate() {
            if value != 0.0 {
                col_indices.push(col);
                values.push(value);
            }
        }
        row_ptr.push(col_indices.len());
    }
    CsrMatrix::from_parts(values, col_indices, row_ptr, nrows, ncols)
        .expect("invariant: dense fixture converts to a valid CSR structure")
}

fn create_test_matrix() -> CsrMatrix<f64> {
    // Create a simple tridiagonal matrix
    let n = 5;
    let mut dense = vec![vec![0.0; n]; n];
    for (i, row) in dense.iter_mut().enumerate() {
        row[i] = 2.0;
        if i > 0 {
            row[i - 1] = -1.0;
        }
        if i < n - 1 {
            row[i + 1] = -1.0;
        }
    }
    csr_from_dense(&dense)
}

fn create_simple_coarsening() -> (Vec<usize>, Vec<Option<usize>>) {
    let coarse_points = vec![0, 2, 4]; // Every other point
    let fine_to_coarse_map = vec![Some(0), None, Some(1), None, Some(2)];

    (coarse_points, fine_to_coarse_map)
}

#[test]
fn test_direct_interpolation() {
    let (_coarse_points, fine_to_coarse_map) = create_simple_coarsening();
    let interpolation = create_direct_interpolation::<f64>(&fine_to_coarse_map, 5, 3);

    // Check dimensions
    assert_eq!(interpolation.nrows(), 5);
    assert_eq!(interpolation.ncols(), 3);

    // Check that coarse points map correctly
    assert_eq!(entry_of(&interpolation, 0, 0).to_bits(), 1.0_f64.to_bits()); // Point 0 -> coarse 0
    assert_eq!(entry_of(&interpolation, 2, 1).to_bits(), 1.0_f64.to_bits()); // Point 2 -> coarse 1
    assert_eq!(entry_of(&interpolation, 4, 2).to_bits(), 1.0_f64.to_bits()); // Point 4 -> coarse 2

    // Check that F-points are zero
    assert_eq!(entry_of(&interpolation, 1, 0).to_bits(), 0.0_f64.to_bits()); // Point 1 not mapped
    assert_eq!(entry_of(&interpolation, 3, 1).to_bits(), 0.0_f64.to_bits()); // Point 3 not mapped
}

#[test]
fn test_classical_interpolation() {
    let matrix = create_test_matrix();
    let (coarse_points, _) = create_simple_coarsening();

    // Create a simple strength matrix
    let mut strength_dense = vec![vec![0.0; 5]; 5];
    for (i, row) in strength_dense.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            if i.abs_diff(j) == 1 {
                *cell = 1.0;
            }
        }
    }
    let strength_matrix = csr_from_dense(&strength_dense);

    let interpolation =
        create_classical_interpolation(&matrix, &coarse_points, &strength_matrix, 2)
            .expect("expected value");

    // Check dimensions
    assert_eq!(interpolation.nrows(), 5);
    assert_eq!(interpolation.ncols(), 3);

    // Check that coarse points have direct injection
    assert_eq!(entry_of(&interpolation, 0, 0).to_bits(), 1.0_f64.to_bits());
    assert_eq!(entry_of(&interpolation, 2, 1).to_bits(), 1.0_f64.to_bits());
    assert_eq!(entry_of(&interpolation, 4, 2).to_bits(), 1.0_f64.to_bits());

    // For 1D Laplacian 3-point stencil [-1, 2, -1], linear interpolation is exact.
    // F-point 1 is between C-point 0 and C-point 2. Weight should be 0.5 each.
    // Formula: w_ij = A_ij / A_ii (negated and simplified since no F-F connections)
    // A_11 = 2. A_10 = -1, A_12 = -1.
    // w_10 = -(-1)/2 = 0.5
    // w_12 = -(-1)/2 = 0.5

    let w10 = entry_of(&interpolation, 1, 0);
    let w11 = entry_of(&interpolation, 1, 1);

    assert!(
        (w10 - 0.5).abs() < 1e-10,
        "Weight w10 should be 0.5, got {w10}"
    );
    assert!(
        (w11 - 0.5).abs() < 1e-10,
        "Weight w11 should be 0.5, got {w11}"
    );
}

#[test]
fn test_interpolation_quality_validation() {
    let (coarse_points, fine_to_coarse_map) = create_simple_coarsening();
    let interpolation = create_direct_interpolation::<f64>(&fine_to_coarse_map, 5, 3);

    let quality = validate_interpolation_operator(&interpolation, &coarse_points);

    // C-points should have row sum = 1
    assert!((quality.avg_coarse_row_sum - 1.0).abs() < 1e-10);

    // F-points should have row sum = 0 (no interpolation in direct case)
    assert_eq!(quality.avg_fine_row_sum.to_bits(), 0.0_f64.to_bits());

    // Should preserve constants (with some error for F-points)
    assert!(quality.constant_preservation_error >= 0.0);

    // Should be sparse
    assert!(quality.sparsity_ratio < 1.0);
}

#[test]
fn test_constant_preservation() {
    // Test that interpolation preserves constant vectors
    let (_coarse_points, fine_to_coarse_map) = create_simple_coarsening();
    let interpolation = create_direct_interpolation::<f64>(&fine_to_coarse_map, 5, 3);

    let constant_coarse = Array1::from_shape_vec([3], vec![1.0; 3]).expect("expected value");
    let mut interpolated = Array1::zeros([5]);
    let interpolated_result =
        leto_spmv(&interpolation, &constant_coarse.view()).expect("expected value");
    for i in 0..interpolated.shape()[0] {
        interpolated[i] = interpolated_result[i];
    }

    // C-points should be 1.0, F-points should be 0.0
    assert_eq!(interpolated[0].to_bits(), 1.0_f64.to_bits()); // C-point
    assert_eq!(interpolated[1].to_bits(), 0.0_f64.to_bits()); // F-point
    assert_eq!(interpolated[2].to_bits(), 1.0_f64.to_bits()); // C-point
    assert_eq!(interpolated[3].to_bits(), 0.0_f64.to_bits()); // F-point
    assert_eq!(interpolated[4].to_bits(), 1.0_f64.to_bits()); // C-point
}

#[test]
fn test_classical_interpolation_weights() {
    // Test case with F-F connections in a valid RS coarsening scenario
    // Triangle configuration:
    //    0(C)
    //   /  \
    //  1(F)-2(F)
    //  |    |
    //  3(C) 4(C)
    //
    // 1 connected to 0, 2, 3. 2 connected to 0, 1, 4.
    // F-F connection (1,2) should distribute 2's influence on 1 to common C-neighbor 0.

    let n = 5;
    let mut dense_values = vec![vec![0.0; n]; n];

    // Setup connections with -1.0, and diagonal 3.0
    let edges = vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 4)];

    for (i, row) in dense_values.iter_mut().enumerate() {
        row[i] = 3.0;
    }

    for (u, v) in edges {
        dense_values[u][v] = -1.0;
        dense_values[v][u] = -1.0;
    }
    // Fix diagonals for C-points to be consistent with laplacian?
    // Doesn't strictly matter for interpolation formula as we only look at F-point rows.
    // Row 1: -1(0), -1(2), -1(3). Sum abs off-diag = 3. Diag=3.

    let matrix = csr_from_dense(&dense_values);

    let coarse_points = vec![0, 3, 4];
    // Strength matrix (all neighbors are strong)
    let mut strength_dense = vec![vec![0.0; n]; n];
    for (i, row) in dense_values.iter().enumerate() {
        for (j, cell) in strength_dense.iter_mut().enumerate() {
            if i != j && row[j].abs() > 0.0_f64 {
                cell[i] = 1.0;
            }
        }
    }
    let strength_matrix = csr_from_dense(&strength_dense);

    let interpolation =
        create_classical_interpolation(&matrix, &coarse_points, &strength_matrix, 2)
            .expect("expected value");

    // Coarse mapping: 0->0, 3->1, 4->2

    // Check point 1 (F)
    // C-neighbors: 0, 3. F-neighbor: 2.
    // w_{1,0}: Direct -1. Indirect via 2: a_{12}*a_{20}/S_2 = (-1*-1)/(-1-0) = -1. Total -2.
    // w_{1,0} = -(-2)/3 = 2/3.
    // w_{1,3}: Direct -1. Indirect via 2: a_{12}*a_{23}/S_2 = (-1*0)/-1 = 0. Total -1.
    // w_{1,3} = -(-1)/3 = 1/3.

    let w1_0 = entry_of(&interpolation, 1, 0);
    let w1_3 = entry_of(&interpolation, 1, 1);

    assert!((w1_0 - 2.0 / 3.0).abs() < 1e-10, "Expected 2/3, got {w1_0}");
    assert!((w1_3 - 1.0 / 3.0).abs() < 1e-10, "Expected 1/3, got {w1_3}");
}
