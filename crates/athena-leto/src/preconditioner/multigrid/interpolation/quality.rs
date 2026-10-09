//! Interpolation quality metrics.
//!
//! Row-sum and constant-preservation diagnostics for a prolongation
//! operator: the Galerkin contract needs C-point rows that inject exactly
//! and F-point rows that preserve constants.

use eunomia::{FloatElement, NumericElement, RealField};
use leto_ops::{CsrMatrix, RealScalar, spmv as leto_spmv};

use super::super::scalars::ratio_of_counts;
use super::builders::mean_of_values;

/// Validate interpolation operator properties
///
/// # Panics
///
/// Panics when the constant-vector probe shape disagrees with the
/// interpolation's column count - the caller passes the operator itself.
pub fn validate_interpolation_operator<T: RealField + Copy + FloatElement + RealScalar>(
    interpolation: &CsrMatrix<T>,
    coarse_points: &[usize],
) -> InterpolationQuality {
    let fine_n = interpolation.nrows();
    let coarse_n = interpolation.ncols();

    // Check row sums (should be 1 for F-points, 1 for C-points)
    let row_ptr = interpolation.row_ptr();
    let row_sums: Vec<T> = (0..fine_n)
        .map(|i| {
            let mut row_sum = <T as NumericElement>::ZERO;
            for &value in &interpolation.values()[row_ptr[i]..row_ptr[i + 1]] {
                row_sum += value;
            }
            row_sum
        })
        .collect();

    // Check for C-points (should have row sum = 1 and single non-zero entry)
    let mut coarse_row_sums = Vec::new();
    let mut fine_row_sums = Vec::new();

    for &cp in coarse_points {
        if cp < row_sums.len() {
            coarse_row_sums.push(row_sums[cp]);
        }
    }

    // F-points are all points not in coarse_points
    for (i, sum) in row_sums.iter().enumerate() {
        if !coarse_points.contains(&i) {
            fine_row_sums.push(*sum);
        }
    }

    // Calculate quality metrics
    let avg_coarse_sum = if coarse_row_sums.is_empty() {
        0.0
    } else {
        mean_of_values(
            coarse_row_sums
                .iter()
                .map(|&s| NumericElement::to_f64(s))
                .sum::<f64>(),
            coarse_row_sums.len(),
        )
    };

    let avg_fine_sum = if fine_row_sums.is_empty() {
        0.0
    } else {
        mean_of_values(
            fine_row_sums
                .iter()
                .map(|&s| NumericElement::to_f64(s))
                .sum::<f64>(),
            fine_row_sums.len(),
        )
    };

    // Check conservation property (interpolation should preserve constants)
    let constant_vector =
        leto::Array1::from_shape_vec([coarse_n], vec![<T as NumericElement>::ONE; coarse_n])
            .expect("invariant: constant vector shape is valid");
    let interpolated = leto_spmv(interpolation, &constant_vector.view())
        .expect("invariant: interpolation dimensions are valid");

    let constant_error: f64 = mean_of_values(
        (0..fine_n)
            .map(|idx| {
                NumericElement::to_f64(NumericElement::abs(
                    interpolated[idx] - <T as NumericElement>::ONE,
                ))
            })
            .sum::<f64>(),
        fine_n,
    );

    // Sparsity metrics
    let total_entries = fine_n * coarse_n;
    let non_zero_entries = interpolation.nnz();
    let sparsity_ratio = if total_entries > 0 {
        ratio_of_counts(non_zero_entries, total_entries)
    } else {
        0.0
    };

    InterpolationQuality {
        avg_coarse_row_sum: avg_coarse_sum,
        avg_fine_row_sum: avg_fine_sum,
        constant_preservation_error: constant_error,
        sparsity_ratio,
        non_zero_entries,
        total_entries,
    }
}

/// Quality metrics for interpolation operators
#[derive(Debug, Clone)]
pub struct InterpolationQuality {
    /// Average row sum for C-points (should be 1.0)
    pub avg_coarse_row_sum: f64,
    /// Average row sum for F-points (should be 1.0)
    pub avg_fine_row_sum: f64,
    /// Error in preserving constants (should be ~0)
    pub constant_preservation_error: f64,
    /// Ratio of non-zero entries to total entries
    pub sparsity_ratio: f64,
    /// Number of non-zero entries
    pub non_zero_entries: usize,
    /// Total number of entries
    pub total_entries: usize,
}

impl InterpolationQuality {
    /// Check if interpolation quality is acceptable
    #[must_use]
    pub fn is_acceptable(&self) -> bool {
        (self.avg_coarse_row_sum - 1.0).abs() < 1e-10
            && (self.avg_fine_row_sum - 1.0).abs() < 1e-10
            && self.constant_preservation_error < 1e-6
            && self.sparsity_ratio < 0.1 // Less than 10% non-zeros
    }
}
