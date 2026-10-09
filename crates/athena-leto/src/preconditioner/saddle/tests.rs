//! Saddle-point preconditioner tests, from the recorded `CFDrs` fixtures.
//!
//! Every value test runs at both shipped precisions with a bound derived from
//! the arithmetic depth of the `SIMPLE` recurrence: one reciprocal plus a few
//! O(1) products and sums per entry, so `16·ε_T·max(|expected|, 1)` bounds
//! each side of the comparison. The fixtures are dyadic, so the entries
//! themselves are exact at `f32` and `f64`.

use athena_core::Preconditioner;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::{Array1, LetoError};
use leto_ops::CsrMatrix;

use super::diagonal::DiagonalPreconditioner;
use super::{BlockDiagonalPreconditioner, ComponentBlockPreconditioner, SimplePreconditioner};
use crate::LetoBackend;

macro_rules! csr {
    ($t:ty, $values:expr, $cols:expr, $row_ptr:expr, $rows:expr, $cols_count:expr) => {
        CsrMatrix::from_parts(
            $values
                .iter()
                .map(|&v| <$t as FloatElement>::from_f64(v))
                .collect(),
            $cols.to_vec(),
            $row_ptr.to_vec(),
            $rows,
            $cols_count,
        )
        .expect("invariant: fixture CSR structure is valid")
    };
}

macro_rules! saddle_point_matrix {
    ($t:ty) => {
        csr!(
            $t,
            [4.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0usize, 2, 1, 3, 0, 2, 1, 3],
            [0usize, 2, 4, 6, 8],
            4,
            4
        )
    };
}

macro_rules! normalized_saddle_point_matrix {
    ($t:ty) => {
        csr!(
            $t,
            [4.0, -1.0, 4.0, -1.0, -1.0, -1.0],
            [0usize, 2, 1, 3, 0, 1],
            [0usize, 2, 4, 5, 6],
            4,
            4
        )
    };
}

macro_rules! component_saddle_point_matrix {
    ($t:ty) => {
        csr!(
            $t,
            [4.0, -1.0, 4.0, -1.0, 4.0, 4.0, 4.0, 4.0, -1.0, -1.0],
            [0usize, 6, 1, 7, 2, 3, 4, 5, 0, 1],
            [0usize, 2, 4, 5, 6, 7, 8, 9, 10],
            8,
            8
        )
    };
}

/// Saddle system with several couplings per pressure row and column.
///
/// Six component-major velocity DOFs (component size 2, no cross-component
/// momentum entries) and two pressure DOFs. Every pressure row couples three
/// or four velocities, pressure columns share velocity 5, and the pressure
/// block carries an off-diagonal entry, so the coupling stores hold
/// multi-entry rows and columns in both directions. All entries are dyadic,
/// so every product and sum before the Schur reciprocals is exact.
macro_rules! coupled_saddle_point_matrix {
    ($t:ty) => {
        csr!(
            $t,
            [
                4.0, -1.0, 1.0, -1.0, 4.0, -0.5, 4.0, -1.0, 1.0, -0.5, 4.0, 0.25, 4.0, -1.0, 4.0,
                2.0, 0.5, 1.0, -1.0, 0.5, 1.0, 2.0, 0.5, 1.0, -0.5, 0.25, 3.0
            ],
            [
                0usize, 1, 6, 0, 1, 6, 2, 3, 7, 2, 3, 6, 4, 7, 5, 6, 7, 0, 1, 3, 5, 6, 7, 2, 4, 5,
                7
            ],
            [0usize, 3, 6, 9, 12, 14, 17, 23, 27],
            8,
            8
        )
    };
}

macro_rules! close {
    ($t:ty, $actual:expr, $expected:expr) => {{
        let expected = <$t as FloatElement>::from_f64($expected);
        let bound = <$t as FloatElement>::from_f64(16.0)
            * <$t as RealField>::EPSILON
            * expected.abs().max(<$t as NumericElement>::ONE);
        assert!(
            ($actual - expected).abs() <= bound,
            "value {} deviates from {expected} beyond {bound}",
            $actual
        );
    }};
}

macro_rules! diagonal_values_invert_the_matrix_diagonal_case {
    ($t:ty) => {{
        let matrix = csr!($t, [2.0, 4.0, 8.0], [0usize, 1, 2], [0usize, 1, 2, 3], 3, 3);
        let precond = DiagonalPreconditioner::new(&matrix);
        close!($t, precond.diag_inv()[0], 0.5);
        close!($t, precond.diag_inv()[1], 0.25);
        close!($t, precond.diag_inv()[2], 0.125);
    }};
}

#[test]
fn diagonal_values_invert_the_matrix_diagonal() {
    diagonal_values_invert_the_matrix_diagonal_case!(f32);
    diagonal_values_invert_the_matrix_diagonal_case!(f64);
}

macro_rules! block_diagonal_scales_both_blocks_case {
    ($t:ty) => {{
        let matrix = saddle_point_matrix!($t);
        let precond = BlockDiagonalPreconditioner::new(&matrix, 2, 2)
            .expect("invariant: matrix order matches the block split");
        let backend = LetoBackend::<$t>::default();
        let b = Array1::from_shape_vec(
            [4],
            vec![
                <$t as FloatElement>::from_f64(4.0),
                <$t as FloatElement>::from_f64(4.0),
                <$t as FloatElement>::from_f64(1.0),
                <$t as FloatElement>::from_f64(1.0),
            ],
        )
        .expect("invariant: rhs shape");
        let mut x = Array1::zeros([4]);
        Preconditioner::apply(&precond, &backend, b.view(), x.view_mut())
            .expect("invariant: rhs length matches");
        // Momentum block: u = [4/4, 4/4]; pressure block: p = [1/1, 1/1].
        for index in 0..4 {
            close!($t, x[index], 1.0);
        }
    }};
}

#[test]
fn block_diagonal_scales_both_blocks() {
    block_diagonal_scales_both_blocks_case!(f32);
    block_diagonal_scales_both_blocks_case!(f64);
}

macro_rules! simple_coupled_correction_case {
    ($t:ty) => {{
        let matrix = saddle_point_matrix!($t);
        let precond = SimplePreconditioner::new(&matrix, 2, 2)
            .expect("invariant: matrix order matches the block split");
        let backend = LetoBackend::<$t>::default();
        let b = Array1::from_shape_vec(
            [4],
            vec![
                <$t as FloatElement>::from_f64(4.0),
                <$t as FloatElement>::from_f64(4.0),
                <$t as FloatElement>::from_f64(2.0),
                <$t as FloatElement>::from_f64(2.0),
            ],
        )
        .expect("invariant: rhs shape");
        let mut x = Array1::zeros([4]);
        Preconditioner::apply(&precond, &backend, b.view(), x.view_mut())
            .expect("invariant: rhs length matches");
        // C - D diag(A)^-1 G = 3/4, yielding the exact solution.
        close!($t, x[0], 2.0 / 3.0);
        close!($t, x[1], 2.0 / 3.0);
        close!($t, x[2], 4.0 / 3.0);
        close!($t, x[3], 4.0 / 3.0);
    }};
}

#[test]
fn simple_coupled_correction_is_exact() {
    simple_coupled_correction_case!(f32);
    simple_coupled_correction_case!(f64);
}

macro_rules! simple_preserves_normalized_continuity_sign_case {
    ($t:ty) => {{
        let matrix = normalized_saddle_point_matrix!($t);
        let precond = SimplePreconditioner::new(&matrix, 2, 2)
            .expect("invariant: matrix order matches the block split");
        let backend = LetoBackend::<$t>::default();
        let b = Array1::from_shape_vec(
            [4],
            vec![
                <$t as FloatElement>::from_f64(4.0),
                <$t as FloatElement>::from_f64(4.0),
                <$t as FloatElement>::from_f64(2.0),
                <$t as FloatElement>::from_f64(2.0),
            ],
        )
        .expect("invariant: rhs shape");
        let mut x = Array1::zeros([4]);
        Preconditioner::apply(&precond, &backend, b.view(), x.view_mut())
            .expect("invariant: rhs length matches");
        close!($t, x[0], -2.0);
        close!($t, x[1], -2.0);
        close!($t, x[2], -12.0);
        close!($t, x[3], -12.0);
    }};
}

#[test]
fn simple_preserves_normalized_continuity_sign() {
    simple_preserves_normalized_continuity_sign_case!(f32);
    simple_preserves_normalized_continuity_sign_case!(f64);
}

macro_rules! component_factors_velocity_blocks_case {
    ($t:ty) => {{
        let matrix = component_saddle_point_matrix!($t);
        let precond = ComponentBlockPreconditioner::new(&matrix, 6, 2)
            .expect("invariant: no cross-component coupling");
        let backend = LetoBackend::<$t>::default();
        let b = Array1::from_shape_vec(
            [8],
            vec![
                <$t as FloatElement>::from_f64(4.0),
                <$t as FloatElement>::from_f64(4.0),
                <$t as FloatElement>::from_f64(0.0),
                <$t as FloatElement>::from_f64(0.0),
                <$t as FloatElement>::from_f64(0.0),
                <$t as FloatElement>::from_f64(0.0),
                <$t as FloatElement>::from_f64(2.0),
                <$t as FloatElement>::from_f64(2.0),
            ],
        )
        .expect("invariant: rhs shape");
        let mut x = Array1::zeros([8]);
        Preconditioner::apply(&precond, &backend, b.view(), x.view_mut())
            .expect("invariant: rhs length matches");
        close!($t, x[0], -2.0);
        close!($t, x[1], -2.0);
        close!($t, x[2], 0.0);
        close!($t, x[3], 0.0);
        close!($t, x[4], 0.0);
        close!($t, x[5], 0.0);
        close!($t, x[6], -12.0);
        close!($t, x[7], -12.0);
    }};
}

#[test]
fn component_factors_velocity_blocks() {
    component_factors_velocity_blocks_case!(f32);
    component_factors_velocity_blocks_case!(f64);
}

macro_rules! simple_matches_dense_saddle_formula_case {
    ($t:ty) => {{
        let matrix = coupled_saddle_point_matrix!($t);
        let precond = SimplePreconditioner::new(&matrix, 6, 2)
            .expect("invariant: matrix order matches the block split");
        let backend = LetoBackend::<$t>::default();
        let rhs_values: [f64; 8] = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 1.5, -0.5];
        let b = Array1::from_shape_vec(
            [8],
            rhs_values
                .iter()
                .map(|&v| <$t as FloatElement>::from_f64(v))
                .collect(),
        )
        .expect("invariant: rhs shape");
        let mut x = Array1::zeros([8]);
        Preconditioner::apply(&precond, &backend, b.view(), x.view_mut())
            .expect("invariant: rhs length matches");

        // Dense reference: u* = f / diag(A);
        // S_ii = C_ii - sum_v D_iv G_vi / A_vv; p = (g - D u*) / S_ii;
        // u = u* - diag(A)^-1 G p.
        let entry = |row: usize, column: usize| {
            matrix
                .get(row, column)
                .map(<$t as NumericElement>::to_f64)
                .unwrap_or(0.0)
        };
        let u_star: Vec<f64> = (0..6)
            .map(|velocity| rhs_values[velocity] / entry(velocity, velocity))
            .collect();
        let pressure: Vec<f64> = (0..2)
            .map(|i| {
                let schur = entry(6 + i, 6 + i)
                    - (0..6)
                        .map(|v| entry(6 + i, v) * entry(v, 6 + i) / entry(v, v))
                        .sum::<f64>();
                let divergence_u: f64 = (0..6).map(|v| entry(6 + i, v) * u_star[v]).sum();
                (rhs_values[6 + i] - divergence_u) / schur
            })
            .collect();
        let expected: Vec<f64> = (0..6)
            .map(|v| {
                u_star[v] - (0..2).map(|i| entry(v, 6 + i) * pressure[i]).sum::<f64>() / entry(v, v)
            })
            .chain(pressure.iter().copied())
            .collect();

        // The fixture is dyadic and the reference computes the same
        // recurrence, so both sides lie within 16 eps of each other at the
        // widest precision.
        let bound = 16.0 * <$t as NumericElement>::to_f64(<$t as RealField>::EPSILON);
        for (index, reference) in expected.iter().enumerate() {
            let actual = <$t as NumericElement>::to_f64(x[index]);
            let tolerance = bound * reference.abs().max(1.0);
            assert!(
                (actual - reference).abs() <= tolerance,
                "SIMPLE entry {index}: {actual} vs dense {reference} (tolerance {tolerance})"
            );
        }
    }};
}

#[test]
fn simple_matches_dense_saddle_formula() {
    simple_matches_dense_saddle_formula_case!(f32);
    simple_matches_dense_saddle_formula_case!(f64);
}

#[test]
fn mismatched_lengths_reject_with_length_mismatch() {
    let matrix = saddle_point_matrix!(f64);
    let block = BlockDiagonalPreconditioner::new(&matrix, 2, 2)
        .expect("invariant: matrix order matches the block split");
    let simple = SimplePreconditioner::new(&matrix, 2, 2)
        .expect("invariant: matrix order matches the block split");
    let backend = LetoBackend::<f64>::default();
    let wrong = Array1::zeros([3]);

    match Preconditioner::apply(
        &block,
        &backend,
        wrong.view(),
        Array1::zeros([4]).view_mut(),
    ) {
        Err(crate::LetoBackendError::LengthMismatch { left, .. }) => {
            assert_eq!(left, 3, "the rejection names the received length");
        }
        Err(crate::LetoBackendError::Leto(LetoError::InvalidInput(reason))) => {
            panic!("unexpected variant: {reason}");
        }
        Err(other) => panic!("a length-3 rhs must reject as a length mismatch, got {other}"),
        Ok(()) => panic!("a length-3 rhs must be rejected"),
    }
    match Preconditioner::apply(
        &simple,
        &backend,
        wrong.view(),
        Array1::zeros([4]).view_mut(),
    ) {
        Err(crate::LetoBackendError::LengthMismatch { left, .. }) => {
            assert_eq!(left, 3, "the rejection names the received length");
        }
        Err(other) => panic!("a length-3 rhs must reject as a length mismatch, got {other}"),
        Ok(()) => panic!("a length-3 rhs must be rejected"),
    }
}
