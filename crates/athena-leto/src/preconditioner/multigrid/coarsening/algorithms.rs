//! Coarsening algorithms for AMG hierarchy construction.
//!
//! Provides Ruge-Stüben, aggregation, Falgout (CLJP), PMIS, HMIS, and hybrid
//! coarsening strategies.

use super::super::scalars::{count_to_scalar, ratio_of_counts};
use super::CoarseningResult;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Result;
use leto_ops::{CsrMatrix, RealScalar, Xorshift64};

/// Ruge-Stüben coarsening algorithm
///
/// # Errors
/// Returns the leto CSR validation error when the strength-of-connection
/// /// matrix cannot be assembled from the operator's stored parts.
pub fn ruge_stueben_coarsening<T: RealField + Copy + FloatElement + RealScalar>(
    matrix: &CsrMatrix<T>,
    strength_threshold: T,
) -> Result<CoarseningResult<T>> {
    let n = matrix.nrows();
    let mut coarse_points = Vec::new();
    let mut fine_to_coarse_map = vec![None; n];

    // Step 1: Compute strength of connection matrix
    let strength_matrix = compute_strength_matrix(matrix, strength_threshold)?;

    // Transpose strength matrix to get S^T (influence graph)
    let strength_transpose = strength_matrix.transpose();

    // Step 2: Initialize lambda (measure of importance)
    let mut lambda = vec![0; n];
    let transposed_offsets = strength_transpose.row_ptr();
    for i in 0..n {
        lambda[i] = transposed_offsets[i + 1] - transposed_offsets[i];
    }

    // Status: 0 = Undecided, 1 = C-point, 2 = F-point
    let mut status = vec![0; n];
    let mut undecided_count = n;

    let strength_offsets = strength_matrix.row_ptr();
    let strength_indices = strength_matrix.col_indices();
    let transposed_indices = strength_transpose.col_indices();

    // Step 3: C/F Splitting (Standard Ruge-Stueben First Pass)
    while undecided_count > 0 {
        // Pick the undecided point carrying the strongest influence. A
        // maximum of zero means no undecided point influences anything, so
        // the split is complete.
        let mut best_lambda = 0;
        let mut i_opt = None;

        for k in 0..n {
            if status[k] == 0 && lambda[k] > best_lambda {
                best_lambda = lambda[k];
                i_opt = Some(k);
            }
        }

        let Some(i) = i_opt else {
            break;
        };
        if best_lambda == 0 {
            break;
        }
        {
            // Make i a C-point
            status[i] = 1;
            undecided_count -= 1;
            coarse_points.push(i);
            fine_to_coarse_map[i] = Some(coarse_points.len() - 1);

            // For all undecided j that are strongly influenced by i (j in S_i^T)
            for &j in &transposed_indices[transposed_offsets[i]..transposed_offsets[i + 1]] {
                if status[j] == 0 {
                    // Make j an F-point
                    status[j] = 2;
                    undecided_count -= 1;

                    // For all undecided k that strongly influence j (k in S_j)
                    for &kk in &strength_indices[strength_offsets[j]..strength_offsets[j + 1]] {
                        if status[kk] == 0 {
                            lambda[kk] += 1;
                        }
                    }
                }
            }
        }
    }

    // Second Pass - Classify remaining undecided points as C-points
    for i in 0..n {
        if status[i] == 0 {
            status[i] = 1;
            coarse_points.push(i);
            fine_to_coarse_map[i] = Some(coarse_points.len() - 1);
        }
    }

    // Step 4: Map F-points to their strongest connected C-point
    assign_closest_coarse_points(&mut fine_to_coarse_map, &status, &strength_matrix);

    Ok(CoarseningResult {
        coarse_points,
        fine_to_coarse_map,
        strength_matrix,
    })
}

/// Aggregation-based coarsening
///
/// # Errors
/// Returns the leto CSR validation error when the strength-of-connection
/// /// matrix cannot be assembled from the operator's stored parts.
pub fn aggregation_coarsening<T: RealField + Copy + FloatElement + RealScalar>(
    matrix: &CsrMatrix<T>,
    max_aggregate_size: usize,
) -> Result<CoarseningResult<T>> {
    let n = matrix.nrows();
    let mut coarse_points = Vec::new();
    let mut fine_to_coarse_map = vec![None; n];
    let mut aggregated = vec![false; n];

    let strength_matrix = compute_strength_matrix(matrix, <T as FloatElement>::from_f64(0.5))?;
    let s_offsets = strength_matrix.row_ptr();
    let s_indices = strength_matrix.col_indices();

    let mut aggregate_id = 0;

    for i in 0..n {
        if !aggregated[i] {
            let mut aggregate = vec![i];
            aggregated[i] = true;

            for &j in &s_indices[s_offsets[i]..s_offsets[i + 1]] {
                if !aggregated[j] && aggregate.len() < max_aggregate_size {
                    aggregate.push(j);
                    aggregated[j] = true;
                }
            }

            coarse_points.push(aggregate[0]);

            for &point in &aggregate {
                fine_to_coarse_map[point] = Some(aggregate_id);
            }

            aggregate_id += 1;
        }
    }

    Ok(CoarseningResult {
        coarse_points,
        fine_to_coarse_map,
        strength_matrix,
    })
}

/// Hybrid coarsening (Ruge-Stüben with aggregation fallback)
///
/// # Errors
/// Propagates the strategy error: the strength-of-connection assembly
/// /// failure, or the aggregation fallback's own assembly failure.
pub fn hybrid_coarsening<T: RealField + Copy + FloatElement + RealScalar>(
    matrix: &CsrMatrix<T>,
    strength_threshold: T,
    max_aggregate_size: usize,
) -> Result<CoarseningResult<T>> {
    match ruge_stueben_coarsening(matrix, strength_threshold) {
        Ok(result) => {
            let assigned_points = result
                .fine_to_coarse_map
                .iter()
                .filter(|x| x.is_some())
                .count();
            let assignment_ratio =
                count_to_scalar::<f64>(assigned_points) / count_to_scalar::<f64>(matrix.nrows());

            if assignment_ratio > 0.8 {
                Ok(result)
            } else {
                aggregation_coarsening(matrix, max_aggregate_size)
            }
        }
        Err(_) => aggregation_coarsening(matrix, max_aggregate_size),
    }
}

/// Assign unmapped F-points to their strongest connected C-point
fn assign_closest_coarse_points<T: RealField + Copy + FloatElement + RealScalar>(
    fine_to_coarse_map: &mut [Option<usize>],
    status: &[i32],
    strength_matrix: &CsrMatrix<T>,
) {
    let n = strength_matrix.nrows();
    let s_offsets = strength_matrix.row_ptr();
    let s_indices = strength_matrix.col_indices();

    for i in 0..n {
        if fine_to_coarse_map[i].is_none() {
            let mut max_strength = <T as NumericElement>::ZERO;
            let mut best_coarse_idx = None;

            for (&j, &strength) in s_indices[s_offsets[i]..s_offsets[i + 1]]
                .iter()
                .zip(strength_matrix.values()[s_offsets[i]..s_offsets[i + 1]].iter())
            {
                if status[j] == 1 && (best_coarse_idx.is_none() || strength > max_strength) {
                    max_strength = strength;
                    best_coarse_idx = fine_to_coarse_map[j];
                }
            }

            if let Some(idx) = best_coarse_idx {
                fine_to_coarse_map[i] = Some(idx);
            }
        }
    }
}

/// Falgout coarsening algorithm (CLJP method)
///
/// Reference: Falgout, R. D. (2006). An introduction to algebraic multigrid
///
/// # Errors
/// Returns the leto CSR validation error when the strength-of-connection
/// /// matrix cannot be assembled from the operator's stored parts.
pub fn falgout_coarsening<T: RealField + Copy + FloatElement + RealScalar>(
    matrix: &CsrMatrix<T>,
    strength_threshold: T,
) -> Result<CoarseningResult<T>> {
    let n = matrix.nrows();
    let mut coarse_points = Vec::new();
    let mut fine_to_coarse_map = vec![None; n];

    let strength_matrix = compute_strength_matrix(matrix, strength_threshold)?;
    let s_offsets = strength_matrix.row_ptr();
    let s_indices = strength_matrix.col_indices();

    let mut measures = vec![0.0; n];
    for i in 0..n {
        measures[i] = count_to_scalar::<f64>(s_offsets[i + 1] - s_offsets[i]);
    }

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| measures[b].total_cmp(&measures[a]));

    let lambda = 4.0 / 3.0;
    let mut status = vec![0; n];

    for &i in &sorted_indices {
        if status[i] != 0 {
            continue;
        }

        let mut should_be_coarse = true;
        let mut coarse_neighbors = 0usize;
        let total_strong_connections = s_offsets[i + 1] - s_offsets[i];

        for &j in &s_indices[s_offsets[i]..s_offsets[i + 1]] {
            if status[j] == 1 {
                coarse_neighbors += 1;
            }
        }

        if total_strong_connections > 0 {
            let ratio = ratio_of_counts(coarse_neighbors, total_strong_connections);
            if ratio >= lambda {
                should_be_coarse = false;
            }
        }

        if should_be_coarse {
            status[i] = 1;
            coarse_points.push(i);
            fine_to_coarse_map[i] = Some(coarse_points.len() - 1);

            for &j in &s_indices[s_offsets[i]..s_offsets[i + 1]] {
                if status[j] == 0 {
                    status[j] = 2;
                    fine_to_coarse_map[j] = Some(coarse_points.len() - 1);
                }
            }
        } else {
            status[i] = 2;
        }
    }

    assign_closest_coarse_points(&mut fine_to_coarse_map, &status, &strength_matrix);

    Ok(CoarseningResult {
        coarse_points,
        fine_to_coarse_map,
        strength_matrix,
    })
}

/// PMIS (Parallel Modified Independent Set) coarsening
///
/// Reference: Luby's algorithm adapted for parallel coarsening
///
/// # Errors
/// Returns the leto CSR validation error when the strength-of-connection
/// /// matrix cannot be assembled from the operator's stored parts.
pub fn pmis_coarsening<T: RealField + Copy + FloatElement + RealScalar>(
    matrix: &CsrMatrix<T>,
    strength_threshold: T,
) -> Result<CoarseningResult<T>> {
    let n = matrix.nrows();
    let mut coarse_points = Vec::new();
    let mut fine_to_coarse_map = vec![None; n];

    let strength_matrix = compute_strength_matrix(matrix, strength_threshold)?;
    let s_offsets = strength_matrix.row_ptr();
    let s_indices = strength_matrix.col_indices();

    let mut status = vec![0; n];

    // A fixed seed makes a given matrix coarsen identically on every run
    // (the stack's deterministic-execution preference); Xorshift64 provides
    // the uniform [0, 1) stream `rand::thread_rng` produced.
    let mut rng = Xorshift64::new(0x5DEE_CE66_D15E_A5E5);
    let priorities: Vec<f64> = (0..n).map(|_| rng.next_unit_f64()).collect();

    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| priorities[b].total_cmp(&priorities[a]));

    for &i in &sorted_indices {
        if status[i] != 0 {
            continue;
        }

        status[i] = 1;
        coarse_points.push(i);
        fine_to_coarse_map[i] = Some(coarse_points.len() - 1);

        for &j in &s_indices[s_offsets[i]..s_offsets[i + 1]] {
            if status[j] == 0 {
                status[j] = 2;
            }
        }
    }

    assign_closest_coarse_points(&mut fine_to_coarse_map, &status, &strength_matrix);

    Ok(CoarseningResult {
        coarse_points,
        fine_to_coarse_map,
        strength_matrix,
    })
}

/// HMIS (Hybrid Modified Independent Set) coarsening
///
/// Combines PMIS with aggressive coarsening for better parallel performance.
///
/// # Errors
/// Propagates the PMIS or aggressive-coarsening error: the
/// /// strength-of-connection assembly failure.
pub fn hmis_coarsening<T: RealField + Copy + FloatElement + RealScalar>(
    matrix: &CsrMatrix<T>,
    strength_threshold: T,
    aggressive_threshold: T,
) -> Result<CoarseningResult<T>> {
    let n = matrix.nrows();

    match pmis_coarsening(matrix, strength_threshold) {
        Ok(result) => {
            let coarsening_ratio = if n == 0 {
                <T as NumericElement>::ZERO
            } else {
                count_to_scalar::<T>(result.coarse_points.len()) / count_to_scalar::<T>(n)
            };

            if coarsening_ratio < aggressive_threshold {
                aggressive_coarsening(matrix, strength_threshold, aggressive_threshold)
            } else {
                Ok(result)
            }
        }
        Err(_) => aggressive_coarsening(matrix, strength_threshold, aggressive_threshold),
    }
}

/// Aggressive coarsening strategy for difficult matrices
///
/// # Errors
/// Propagates the aggregation fallback's strength-of-connection assembly
/// /// failure.
fn aggressive_coarsening<T: RealField + Copy + FloatElement + RealScalar>(
    matrix: &CsrMatrix<T>,
    _strength_threshold: T,
    _target_ratio: T,
) -> Result<CoarseningResult<T>> {
    let max_aggregate_size = 8;
    aggregation_coarsening(matrix, max_aggregate_size)
}

/// Compute strength of connection matrix
///
/// # Errors
/// Returns the leto CSR validation error when the filtered entries do not
/// /// form a valid CSR structure.
pub(super) fn compute_strength_matrix<T: RealField + Copy + FloatElement + RealScalar>(
    matrix: &CsrMatrix<T>,
    strength_threshold: T,
) -> Result<CsrMatrix<T>> {
    let n = matrix.nrows();
    let mut row_offsets = vec![0; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    let m_offsets = matrix.row_ptr();
    let m_indices = matrix.col_indices();
    let m_values = matrix.values();

    for i in 0..n {
        let mut max_off_diag: T = <T as NumericElement>::ZERO;
        for k in m_offsets[i]..m_offsets[i + 1] {
            let j = m_indices[k];
            if i != j {
                let value = NumericElement::abs(m_values[k]);
                max_off_diag = if max_off_diag > value {
                    max_off_diag
                } else {
                    value
                };
            }
        }

        for k in m_offsets[i]..m_offsets[i + 1] {
            let j = m_indices[k];
            if i != j {
                let strength = NumericElement::abs(m_values[k]);
                if strength >= strength_threshold * max_off_diag {
                    col_indices.push(j);
                    values.push(strength);
                }
            }
        }
        row_offsets[i + 1] = col_indices.len();
    }

    CsrMatrix::from_parts(values, col_indices, row_offsets, n, n)
}
