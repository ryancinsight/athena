use athena_core::{KrylovBackend, LinearOperator};
use eunomia::RealField;
use leto_ops::RealScalar;

use crate::{LetoBackend, LetoBackendError};

/// Matrix-free five-point Dirichlet Laplacian on a `grid × grid` lattice.
///
/// Row-major unknown order: `y[i] = 4·x[i]` minus each in-bounds
/// orthogonal neighbor, matching the [`super::CsrOperator`] assembly of
/// the same stencil. Carries no values or index arrays: application
/// streams the input once and writes the output once, which is why the
/// cache-resident Poisson matvec runs several times faster through this
/// operator than through CSR (spike: 12.1µs vs 31.4µs at grid/128).
#[derive(Clone, Debug, PartialEq)]
pub struct PoissonStencilOperator<T> {
    grid: usize,
    dimension: usize,
    _scalar: core::marker::PhantomData<fn() -> T>,
}

impl<T> PoissonStencilOperator<T> {
    /// Validate and construct a `grid × grid` stencil operator.
    ///
    /// # Errors
    ///
    /// Returns [`LetoBackendError::GridExtentOverflow`] when `grid²`
    /// exceeds the addressable range.
    pub fn new(grid: usize) -> Result<Self, LetoBackendError> {
        let dimension = grid
            .checked_mul(grid)
            .ok_or(LetoBackendError::GridExtentOverflow { grid })?;
        Ok(Self {
            grid,
            dimension,
            _scalar: core::marker::PhantomData,
        })
    }

    /// Return the lattice width; the operator dimension is its square.
    #[must_use]
    pub const fn grid(&self) -> usize {
        self.grid
    }
}

impl<T: RealScalar + RealField> LinearOperator<LetoBackend<T>> for PoissonStencilOperator<T> {
    #[inline]
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(
        &self,
        _backend: &LetoBackend<T>,
        input: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let input = input
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        if input.len() != self.dimension {
            return Err(LetoBackendError::LengthMismatch {
                left: input.len(),
                right: self.dimension,
            });
        }
        if output.len() != self.dimension {
            return Err(LetoBackendError::LengthMismatch {
                left: output.len(),
                right: self.dimension,
            });
        }
        let grid = self.grid;
        let diagonal = T::from_count(4);
        for row in 0..grid {
            for column in 0..grid {
                let index = row * grid + column;
                let mut acc = diagonal * input[index];
                if row > 0 {
                    acc -= input[index - grid];
                }
                if column > 0 {
                    acc -= input[index - 1];
                }
                if column + 1 < grid {
                    acc -= input[index + 1];
                }
                if row + 1 < grid {
                    acc -= input[index + grid];
                }
                output[index] = acc;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eunomia::FloatElement;
    use leto::{Array1, Storage};

    use crate::operator::CsrOperator;

    /// Exact-integer 3×3 case: corners carry 3 points, edges 4, the
    /// center 5, so every boundary shape is pinned. All arithmetic is
    /// exact, hence `assert_eq!` regardless of summation order.
    fn closed_form_3x3<T: RealScalar + RealField>() {
        let backend = LetoBackend::<T>::default();
        let operator =
            PoissonStencilOperator::<T>::new(3).expect("invariant: 3x3 grid is addressable");
        assert_eq!(operator.grid(), 3);
        assert_eq!(operator.dimension(), 9);
        let input = Array1::from_shape_fn([9], |[i]| T::from_count(i + 1));
        let mut output = Array1::from_shape_fn([9], |_| T::ZERO);
        operator
            .apply(
                &backend,
                backend.view(&input),
                backend.view_mut(&mut output),
            )
            .expect("invariant: matching contiguous views apply");
        let expected = [
            T::from_f64(-2.0),
            T::from_f64(-1.0),
            T::from_f64(4.0),
            T::from_f64(3.0),
            T::from_f64(0.0),
            T::from_f64(7.0),
            T::from_f64(16.0),
            T::from_f64(11.0),
            T::from_f64(22.0),
        ];
        assert_eq!(output.storage().as_slice(), expected.as_slice());
    }

    #[test]
    fn grid_3x3_matches_closed_form() {
        closed_form_3x3::<f32>();
        closed_form_3x3::<f64>();
    }

    /// Five-point Dirichlet Laplacian in CSR, mirroring the bench
    /// fixture, as the differential oracle for the stencil.
    fn poisson_csr<T: RealScalar + RealField>(grid: usize) -> leto_ops::CsrMatrix<T> {
        let unknowns = grid * grid;
        let mut values = Vec::with_capacity(5 * unknowns);
        let mut columns = Vec::with_capacity(5 * unknowns);
        let mut row_ptr = Vec::with_capacity(unknowns + 1);
        row_ptr.push(0);
        for row in 0..grid {
            for column in 0..grid {
                let index = row * grid + column;
                if row > 0 {
                    values.push(-T::ONE);
                    columns.push(index - grid);
                }
                if column > 0 {
                    values.push(-T::ONE);
                    columns.push(index - 1);
                }
                values.push(T::from_count(4));
                columns.push(index);
                if column + 1 < grid {
                    values.push(-T::ONE);
                    columns.push(index + 1);
                }
                if row + 1 < grid {
                    values.push(-T::ONE);
                    columns.push(index + grid);
                }
                row_ptr.push(values.len());
            }
        }
        leto_ops::CsrMatrix::from_parts(values, columns, row_ptr, unknowns, unknowns)
            .expect("invariant: test Poisson fixture is well-formed")
    }

    /// Stencil and CSR apply the same operator: non-integer input makes
    /// summation order observable, so the bound is rounding-derived
    /// (100·eps·peak), not exact.
    fn matches_csr<T: RealScalar + RealField>() {
        let grid = 8;
        let unknowns = grid * grid;
        let backend = LetoBackend::<T>::default();
        let stencil =
            PoissonStencilOperator::<T>::new(grid).expect("invariant: 8x8 grid is addressable");
        let csr = CsrOperator::new(poisson_csr::<T>(grid))
            .expect("invariant: test Poisson matrix is square");
        let input = Array1::from_shape_fn([unknowns], |[i]| {
            T::from_f64(0.5 + f64::from_count(i) * 0.125)
        });
        let view = backend.view(&input);
        let mut y_stencil = Array1::from_shape_fn([unknowns], |_| T::ZERO);
        let mut y_csr = Array1::from_shape_fn([unknowns], |_| T::ZERO);
        stencil
            .apply(&backend, view, backend.view_mut(&mut y_stencil))
            .expect("invariant: matching contiguous views apply");
        csr.apply(&backend, view, backend.view_mut(&mut y_csr))
            .expect("invariant: matching contiguous views apply");
        let peak = y_csr.storage().as_slice().iter().fold(T::ZERO, |acc, &v| {
            let magnitude = T::abs(v);
            if magnitude > acc { magnitude } else { acc }
        });
        let bound = T::from_count(100) * T::EPSILON * peak;
        for (index, (&got, &want)) in y_stencil
            .storage()
            .as_slice()
            .iter()
            .zip(y_csr.storage().as_slice())
            .enumerate()
        {
            assert!(
                T::abs(got - want) <= bound,
                "entry {index}: {got:?} vs CSR {want:?} exceeds the rounding bound {bound:?}"
            );
        }
    }

    #[test]
    fn stencil_matches_csr_within_rounding() {
        matches_csr::<f32>();
        matches_csr::<f64>();
    }

    #[test]
    fn constructor_guards_the_addressable_range() {
        assert!(PoissonStencilOperator::<f64>::new(0).is_ok());
        assert_eq!(
            PoissonStencilOperator::<f64>::new(0)
                .expect("invariant: empty grid constructs")
                .dimension(),
            0
        );
        assert!(matches!(
            PoissonStencilOperator::<f64>::new(usize::MAX),
            Err(LetoBackendError::GridExtentOverflow { .. })
        ));
    }
}
