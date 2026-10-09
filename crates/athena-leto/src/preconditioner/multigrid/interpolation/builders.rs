//! AMG interpolation builders.
//!
//! Classical (Ruge-Stüben), direct, and standard (distance-weighted)
//! prolongation operators assembled into caller-owned CSR vectors.

use super::super::scalars::count_to_scalar;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Result;
use leto_ops::{CsrMatrix, RealScalar};

#[inline]
pub(super) fn mean_of_values(sum: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum / count_to_scalar::<f64>(count)
    }
}

/// Create classical interpolation operator (Ruge-Stüben)
///
/// # Errors
///
/// Returns the leto CSR validation error when the assembled weights do not
/// form a valid CSR structure.
///
/// # Panics
///
/// Panics when a listed coarse point carries no mapped local index - the
/// caller passes a `coarse_map` built from that list.
pub fn create_classical_interpolation<T: RealField + Copy + FloatElement + RealScalar>(
    fine_matrix: &CsrMatrix<T>,
    coarse_points: &[usize],
    strength_matrix: &CsrMatrix<T>,
    _max_interpolation_points: usize,
) -> Result<CsrMatrix<T>> {
    let fine_n = fine_matrix.nrows();
    let coarse_n = coarse_points.len();

    let mut row_offsets = vec![0; fine_n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    // Fast lookup for coarse points: mapping global index -> local coarse index
    let mut coarse_map = vec![None; fine_n];
    for (local_idx, &global_idx) in coarse_points.iter().enumerate() {
        if global_idx < fine_n {
            coarse_map[global_idx] = Some(local_idx);
        }
    }

    for fine_i in 0..fine_n {
        if let Some(coarse_idx) = coarse_map[fine_i] {
            // Point is a C-point: Direct injection
            col_indices.push(coarse_idx);
            values.push(<T as NumericElement>::ONE);
        } else {
            // Point is an F-point: Interpolate from C-neighbors using Ruge-Stüben formula
            // w_ij = - ( A_ij + sum_{k in F_i^s} ( A_ik * A_kj / sum_{m in C_i} A_km ) ) / ( A_ii + sum_{n in D_i^w} A_in )

            let row_start = fine_matrix.row_ptr()[fine_i];
            let row_end = fine_matrix.row_ptr()[fine_i + 1];
            let fine_row_cols = &fine_matrix.col_indices()[row_start..row_end];
            let fine_row_vals = &fine_matrix.values()[row_start..row_end];

            let strength_start = strength_matrix.row_ptr()[fine_i];
            let strength_end = strength_matrix.row_ptr()[fine_i + 1];
            let strength_cols = &strength_matrix.col_indices()[strength_start..strength_end];

            // Identify sets C_i (strong C-points) and F_i^s (strong F-points)
            let mut c_i = Vec::new();
            let mut f_i_s = Vec::new();

            for &neighbor_idx in strength_cols {
                if coarse_map[neighbor_idx].is_some() {
                    c_i.push(neighbor_idx);
                } else if neighbor_idx != fine_i {
                    f_i_s.push(neighbor_idx);
                }
            }

            if c_i.is_empty() {
                // If no strong C-points, we can't interpolate strongly.
                // Fallback to zero or weak interpolation?
                // Usually implies poor coarsening or isolated F-point.
                // Leaving row empty effectively means zero value (Dirichlet-like).
            } else {
                // Identify diagonal A_ii and sum of weak connections
                let mut a_ii = <T as NumericElement>::ZERO;
                let mut sum_weak = <T as NumericElement>::ZERO;

                for (k, &neighbor_idx) in fine_row_cols.iter().enumerate() {
                    let val = fine_row_vals[k];
                    if neighbor_idx == fine_i {
                        a_ii = val;
                    } else {
                        // Check if connection is strong
                        if strength_cols.binary_search(&neighbor_idx).is_err() {
                            // Weak connection, add to diagonal sum
                            sum_weak += val;
                        }
                    }
                }

                let diagonal = a_ii + sum_weak;

                if NumericElement::abs(diagonal) > <T as RealField>::EPSILON {
                    // Precompute denominators for k in F_i^s: sum_{m in C_i} A_km
                    let mut k_denoms = Vec::with_capacity(f_i_s.len());
                    for &k in &f_i_s {
                        let mut denom = <T as NumericElement>::ZERO;
                        for &m in &c_i {
                            denom += fine_matrix.get(k, m).unwrap_or(<T as NumericElement>::ZERO);
                        }
                        k_denoms.push(denom);
                    }

                    // Compute weights for each j in C_i
                    let mut weights = Vec::new();
                    let neg_diag_inv = <T as NumericElement>::ONE / -diagonal;

                    for &j in &c_i {
                        let coarse_local_idx = coarse_map[j]
                            .expect("invariant: every classical interpolation C-point is mapped");

                        // A_ij (direct connection)
                        let mut direct_value = <T as NumericElement>::ZERO;
                        if let Ok(found) = fine_row_cols.binary_search(&j) {
                            direct_value = fine_row_vals[found];
                        }

                        let mut indirect_sum = <T as NumericElement>::ZERO;

                        // Sum over k in F_i^s
                        for (idx, &k) in f_i_s.iter().enumerate() {
                            let denom = k_denoms[idx];
                            if NumericElement::abs(denom) > <T as RealField>::EPSILON {
                                // A_ik
                                let mut neighbour_value = <T as NumericElement>::ZERO;
                                if let Ok(k_idx) = fine_row_cols.binary_search(&k) {
                                    neighbour_value = fine_row_vals[k_idx];
                                }

                                // A_kj
                                let a_kj =
                                    fine_matrix.get(k, j).unwrap_or(<T as NumericElement>::ZERO);

                                indirect_sum += neighbour_value * a_kj / denom;
                            }
                        }

                        let weight = (direct_value + indirect_sum) * neg_diag_inv;
                        weights.push((coarse_local_idx, weight));
                    }

                    // Sort by index for CSR format
                    weights.sort_by_key(|w| w.0);

                    for (idx, val) in weights {
                        col_indices.push(idx);
                        values.push(val);
                    }
                }
            }
        }
        row_offsets[fine_i + 1] = col_indices.len();
    }

    CsrMatrix::from_parts(values, col_indices, row_offsets, fine_n, coarse_n)
}

/// Create direct interpolation operator///
/// # Panics
///
/// Panics when the assembled injection weights do not form a valid CSR
/// structure - the caller's map is index-bounded by construction.
pub fn create_direct_interpolation<T: RealField + Copy + FloatElement + RealScalar>(
    fine_to_coarse_map: &[Option<usize>],
    fine_n: usize,
    coarse_n: usize,
) -> CsrMatrix<T> {
    let mut row_offsets = vec![0; fine_n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for (fine_i, &coarse_opt) in fine_to_coarse_map.iter().enumerate() {
        if let Some(coarse_i) = coarse_opt {
            col_indices.push(coarse_i);
            values.push(<T as NumericElement>::ONE);
        }
        row_offsets[fine_i + 1] = col_indices.len();
    }

    CsrMatrix::from_parts(values, col_indices, row_offsets, fine_n, coarse_n)
        .expect("invariant: an injection operator is a valid CSR structure")
}

/// Create standard interpolation operator///
/// # Errors
///
/// Returns the leto CSR validation error when the assembled weights do not
/// form a valid CSR structure.
///
/// # Panics
///
/// Panics when a listed coarse point carries no mapped local index - the
/// caller passes a `coarse_map` built from that list.
pub fn create_standard_interpolation<T: RealField + Copy + FloatElement + RealScalar>(
    fine_matrix: &CsrMatrix<T>,
    coarse_points: &[usize],
    strength_matrix: &CsrMatrix<T>,
) -> Result<CsrMatrix<T>> {
    let fine_n = fine_matrix.nrows();
    let coarse_n = coarse_points.len();

    let mut row_offsets = vec![0; fine_n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for fine_i in 0..fine_n {
        if coarse_points.contains(&fine_i) {
            // Direct injection for coarse points
            let coarse_idx = coarse_points
                .iter()
                .position(|&x| x == fine_i)
                .expect("invariant: a listed coarse point has a local index");
            col_indices.push(coarse_idx);
            values.push(<T as NumericElement>::ONE);
        } else {
            // Distance-weighted interpolation for F-points
            let mut weights = Vec::new();
            let mut total_weight = <T as NumericElement>::ZERO;

            // Find neighboring coarse points
            for (coarse_local_idx, &coarse_global_idx) in coarse_points.iter().enumerate() {
                let distance = fine_i.abs_diff(coarse_global_idx);

                // Weight by inverse distance and connection strength
                if distance > 0 {
                    let strength = {
                        let stored = strength_matrix
                            .get(fine_i, coarse_global_idx)
                            .unwrap_or(<T as NumericElement>::ZERO);
                        if stored == <T as NumericElement>::ZERO {
                            <T as FloatElement>::from_f64(1e-6)
                        } else {
                            stored
                        }
                    };
                    let weight =
                        strength / (count_to_scalar::<T>(distance) + <T as NumericElement>::ONE);
                    weights.push((coarse_local_idx, weight));
                    total_weight += weight;
                }
            }

            for &(coarse_idx, weight) in &weights {
                let normalized_weight = if total_weight > <T as NumericElement>::ZERO {
                    weight / total_weight
                } else {
                    <T as NumericElement>::ONE / count_to_scalar(weights.len())
                };
                col_indices.push(coarse_idx);
                values.push(normalized_weight);
            }
        }
        row_offsets[fine_i + 1] = col_indices.len();
    }

    CsrMatrix::from_parts(values, col_indices, row_offsets, fine_n, coarse_n)
}
