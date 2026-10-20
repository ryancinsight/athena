//! Column-pivoted QR recovery for the small reduced GMRES factor.
//!
//! For `R P = Q [U V; 0 E]`, treating the trailing block `E` as negligible
//! gives a basic least-squares solution: solve `U z = (Qᵀg)[..rank]`, set the
//! other permuted coordinates to zero, and map through `P`. Orthogonality
//! preserves the residual norm. The coupled block `V` is retained during
//! factorization; dropping original coordinates independently is incorrect.
//!
//! This is numerical-rank truncation, not an exact-rank certificate. A basic
//! solution need not minimize coefficient norm. All storage is borrowed from
//! the prepared workspace; the usual full-rank triangular path never enters.
use super::{ScalarFailure, givens, singular_pivot_floor};
use eunomia::RealField;

struct Factor<'a, T, const RESTART: usize> {
    entries: &'a mut [T],
    rhs: &'a mut [T],
    count: usize,
}
impl<T: RealField, const RESTART: usize> Factor<'_, T, RESTART> {
    const fn index(row: usize, column: usize) -> usize {
        column * (RESTART + 1) + row
    }
    /// Scaled sum of squares avoids squaring an extreme column magnitude.
    fn column_norm(&self, row: usize, column: usize) -> Result<T, ScalarFailure> {
        let mut scale = T::ZERO;
        let mut sum = T::ONE;
        for index in row..self.count {
            let value = self.entries[Self::index(index, column)].abs();
            if !value.is_finite() {
                return Err(ScalarFailure::NonFinite);
            }
            if value != T::ZERO {
                if value > scale {
                    let ratio = scale / value;
                    sum = T::ONE + sum * ratio * ratio;
                    scale = value;
                } else {
                    let ratio = value / scale;
                    sum += ratio * ratio;
                }
            }
        }
        let norm = scale * sum.sqrt();
        if norm.is_finite() {
            Ok(norm)
        } else {
            Err(ScalarFailure::NonFinite)
        }
    }
    fn pivot(&self, step: usize) -> Result<(usize, T), ScalarFailure> {
        let mut pivot = step;
        let mut maximum = T::ZERO;
        for column in step..self.count {
            let norm = self.column_norm(step, column)?;
            if norm > maximum {
                maximum = norm;
                pivot = column;
            }
        }
        Ok((pivot, maximum))
    }
    fn swap_columns(&mut self, first: usize, second: usize) {
        for row in 0..self.count {
            self.entries
                .swap(Self::index(row, first), Self::index(row, second));
        }
    }
    fn eliminate(&mut self, step: usize, row: usize) -> Result<(), ScalarFailure> {
        let lower_index = Self::index(row, step);
        if self.entries[lower_index] == T::ZERO {
            return Ok(());
        }
        let (cosine, sine) = givens(
            self.entries[Self::index(step, step)],
            self.entries[lower_index],
        )?;
        for column in step..self.count {
            let upper_index = Self::index(step, column);
            let lower_index = Self::index(row, column);
            let upper = self.entries[upper_index];
            let lower = self.entries[lower_index];
            let next_upper = cosine * upper + sine * lower;
            let next_lower = -sine * upper + cosine * lower;
            if !next_upper.is_finite() || !next_lower.is_finite() {
                return Err(ScalarFailure::NonFinite);
            }
            self.entries[upper_index] = next_upper;
            self.entries[lower_index] = next_lower;
        }
        self.entries[Self::index(row, step)] = T::ZERO;
        let upper = self.rhs[step];
        let lower = self.rhs[row];
        self.rhs[step] = cosine * upper + sine * lower;
        self.rhs[row] = -sine * upper + cosine * lower;
        if self.rhs[step].is_finite() && self.rhs[row].is_finite() {
            Ok(())
        } else {
            Err(ScalarFailure::NonFinite)
        }
    }
}

#[cold]
pub(super) fn solve<T: RealField, const RESTART: usize>(
    entries: &mut [T],
    rhs: &mut [T],
    coefficients: &mut [T],
    permutation: &mut [usize],
    count: usize,
) -> Result<(), ScalarFailure> {
    if rhs[..count].iter().any(|value| !value.is_finite()) {
        return Err(ScalarFailure::NonFinite);
    }
    for (index, slot) in permutation[..count].iter_mut().enumerate() {
        *slot = index;
    }
    let mut factor = Factor::<T, RESTART> {
        entries,
        rhs,
        count,
    };
    let mut rank = 0;
    let mut floor = T::ZERO;
    for step in 0..count {
        let (pivot, norm) = factor.pivot(step)?;
        if step == 0 {
            floor = singular_pivot_floor(count, norm);
        }
        if norm <= floor {
            break;
        }
        factor.swap_columns(step, pivot);
        permutation.swap(step, pivot);
        for row in (step + 1)..count {
            factor.eliminate(step, row)?;
        }
        rank += 1;
    }
    coefficients[..count].fill(T::ZERO);
    for row in (0..rank).rev() {
        let mut value = factor.rhs[row];
        for column in (row + 1)..rank {
            value -= factor.entries[Factor::<T, RESTART>::index(row, column)]
                * coefficients[permutation[column]];
        }
        let result = value / factor.entries[Factor::<T, RESTART>::index(row, row)];
        if !result.is_finite() {
            return Err(ScalarFailure::NonFinite);
        }
        coefficients[permutation[row]] = result;
    }
    Ok(())
}
