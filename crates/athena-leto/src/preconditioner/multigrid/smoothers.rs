//! Multigrid smoothers for AMG preconditioning.
//!
//! Each smoother is a sweep kernel over a CSR system: it damps the
//! high-frequency residual components the coarse-grid correction cannot see.
//! All five dispatch exhaustively from the level's [`Smoother`](super::level::Smoother)
//! enum; none allocates per sweep except Jacobi's explicit previous-iterate
//! snapshot, which its update contract requires.

use super::scalars::diagonal_epsilon;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar, spmv as leto_spmv};

fn residual<T>(matrix: &CsrMatrix<T>, b: &Array1<T>, x: &Array1<T>) -> Array1<T>
where
    T: RealField + Copy + RealScalar,
{
    let applied =
        leto_spmv(matrix, &x.view()).expect("invariant: smoother SpMV dimensions are valid");
    let mut out = Array1::zeros([b.shape()[0]]);
    for i in 0..b.shape()[0] {
        out[i] = b[i] - applied[i];
    }
    out
}

/// Gauss-Seidel smoother
#[derive(Debug, Clone)]
pub struct GaussSeidelSmoother<T: RealField + Copy> {
    relaxation_factor: T,
}

impl<T: RealField + Copy> GaussSeidelSmoother<T> {
    /// Create a new Gauss-Seidel smoother
    pub fn new(relaxation_factor: T) -> Self {
        Self { relaxation_factor }
    }
}

impl<T: RealField + Copy + FloatElement + RealScalar> GaussSeidelSmoother<T> {
    /// Apply the smoother to the system `Ax = b` for `iterations` sweeps.
    pub fn apply(
        &self,
        matrix: &CsrMatrix<T>,
        x: &mut Array1<T>,
        b: &Array1<T>,
        iterations: usize,
    ) {
        let offsets = matrix.row_ptr();
        let indices = matrix.col_indices();
        let values = matrix.values();

        for _ in 0..iterations {
            for i in 0..matrix.nrows() {
                let mut sum = <T as NumericElement>::ZERO;
                let mut diag = <T as NumericElement>::ZERO;

                for k in offsets[i]..offsets[i + 1] {
                    let j = indices[k];
                    let val = values[k];
                    if i == j {
                        diag = val;
                    } else {
                        sum += val * x[j];
                    }
                }

                if NumericElement::abs(diag) > diagonal_epsilon() {
                    let new_value = (b[i] - sum) / diag;
                    x[i] = x[i] + self.relaxation_factor * (new_value - x[i]);
                }
            }
        }
    }
}

/// Symmetric Gauss-Seidel smoother
#[derive(Debug, Clone)]
pub struct SymmetricGaussSeidelSmoother<T: RealField + Copy> {
    relaxation_factor: T,
}

impl<T: RealField + Copy> SymmetricGaussSeidelSmoother<T> {
    /// Create a new symmetric Gauss-Seidel smoother
    pub fn new(relaxation_factor: T) -> Self {
        Self { relaxation_factor }
    }
}

impl<T: RealField + Copy + FloatElement + RealScalar> SymmetricGaussSeidelSmoother<T> {
    /// Apply the smoother to the system `Ax = b` for `iterations` sweeps.
    pub fn apply(
        &self,
        matrix: &CsrMatrix<T>,
        x: &mut Array1<T>,
        b: &Array1<T>,
        iterations: usize,
    ) {
        let offsets = matrix.row_ptr();
        let indices = matrix.col_indices();
        let values = matrix.values();

        // Forward sweep
        let gs_forward = GaussSeidelSmoother::new(self.relaxation_factor);
        gs_forward.apply(matrix, x, b, iterations);

        // Backward sweep
        for _ in 0..iterations {
            for i in (0..matrix.nrows()).rev() {
                let mut sum = <T as NumericElement>::ZERO;
                let mut diag = <T as NumericElement>::ZERO;

                for k in offsets[i]..offsets[i + 1] {
                    let j = indices[k];
                    let val = values[k];
                    if i == j {
                        diag = val;
                    } else {
                        sum += val * x[j];
                    }
                }

                if NumericElement::abs(diag) > diagonal_epsilon() {
                    let new_value = (b[i] - sum) / diag;
                    x[i] = x[i] + self.relaxation_factor * (new_value - x[i]);
                }
            }
        }
    }
}

/// Jacobi smoother
#[derive(Debug, Clone)]
pub struct JacobiSmoother<T: RealField + Copy> {
    relaxation_factor: T,
}

impl<T: RealField + Copy> JacobiSmoother<T> {
    /// Create a new Jacobi smoother
    pub fn new(relaxation_factor: T) -> Self {
        Self { relaxation_factor }
    }
}

impl<T: RealField + Copy + FloatElement + RealScalar> JacobiSmoother<T> {
    /// Apply the smoother to the system `Ax = b` for `iterations` sweeps.
    pub fn apply(
        &self,
        matrix: &CsrMatrix<T>,
        x: &mut Array1<T>,
        b: &Array1<T>,
        iterations: usize,
    ) {
        let offsets = matrix.row_ptr();
        let indices = matrix.col_indices();
        let values = matrix.values();

        for _ in 0..iterations {
            let x_old = x.clone();

            for i in 0..matrix.nrows() {
                let mut sum = <T as NumericElement>::ZERO;
                let mut diag = <T as NumericElement>::ZERO;

                for k in offsets[i]..offsets[i + 1] {
                    let j = indices[k];
                    let val = values[k];
                    if i == j {
                        diag = val;
                    } else {
                        sum += val * x_old[j];
                    }
                }

                if NumericElement::abs(diag) > diagonal_epsilon() {
                    let new_value = (b[i] - sum) / diag;
                    x[i] = x_old[i] + self.relaxation_factor * (new_value - x_old[i]);
                }
            }
        }
    }
}

/// SOR (Successive Over-Relaxation) smoother
#[derive(Debug, Clone)]
pub struct SORSmoother<T: RealField + Copy> {
    relaxation_factor: T,
}

impl<T: RealField + Copy> SORSmoother<T> {
    /// Create a new SOR smoother
    pub fn new(relaxation_factor: T) -> Self {
        Self { relaxation_factor }
    }
}

impl<T: RealField + Copy + FloatElement + RealScalar> SORSmoother<T> {
    /// Apply the smoother to the system `Ax = b` for `iterations` sweeps.
    pub fn apply(
        &self,
        matrix: &CsrMatrix<T>,
        x: &mut Array1<T>,
        b: &Array1<T>,
        iterations: usize,
    ) {
        let offsets = matrix.row_ptr();
        let indices = matrix.col_indices();
        let values = matrix.values();

        for _ in 0..iterations {
            for i in 0..matrix.nrows() {
                let mut sum = <T as NumericElement>::ZERO;
                let mut diag = <T as NumericElement>::ZERO;

                for k in offsets[i]..offsets[i + 1] {
                    let j = indices[k];
                    let val = values[k];
                    if i == j {
                        diag = val;
                    } else {
                        sum += val * x[j];
                    }
                }

                if NumericElement::abs(diag) > diagonal_epsilon() {
                    let new_value = (b[i] - sum) / diag;
                    x[i] = x[i] + self.relaxation_factor * (new_value - x[i]);
                }
            }
        }
    }
}

/// Chebyshev polynomial smoother for high-frequency error components
#[derive(Debug, Clone)]
pub struct ChebyshevSmoother<T: RealField + Copy> {
    eigenvalues_min: T,
    eigenvalues_max: T,
    degree: usize,
}

impl<T: RealField + Copy + FloatElement + RealScalar> ChebyshevSmoother<T> {
    /// Create a new Chebyshev smoother
    pub fn new(eigenvalues_min: T, eigenvalues_max: T, degree: usize) -> Self {
        Self {
            eigenvalues_min,
            eigenvalues_max,
            degree,
        }
    }

    /// Estimate eigenvalue bounds using Gershgorin circle theorem
    #[must_use]
    pub fn estimate_eigenvalues(matrix: &CsrMatrix<T>) -> (T, T) {
        let mut min_eigen: T = <T as FloatElement>::from_f64(1e6);
        let mut max_eigen = <T as NumericElement>::ZERO;

        let offsets = matrix.row_ptr();
        let indices = matrix.col_indices();
        let values = matrix.values();

        for i in 0..matrix.nrows().min(100) {
            // Check more rows for better estimate
            let mut row_sum = <T as NumericElement>::ZERO;
            let mut diag = <T as NumericElement>::ZERO;

            for k in offsets[i]..offsets[i + 1] {
                let j = indices[k];
                let val = NumericElement::abs(values[k]);
                if i == j {
                    diag = val;
                } else {
                    row_sum += val;
                }
            }

            // Gershgorin discs: [diag - sum, diag + sum]
            let lower = if diag > row_sum {
                diag - row_sum
            } else {
                <T as FloatElement>::from_f64(0.1)
            };
            let upper = diag + row_sum;

            min_eigen = min_eigen.min_scalar(lower);
            max_eigen = max_eigen.max_scalar(upper);
        }

        (
            min_eigen.max_scalar(<T as FloatElement>::from_f64(0.1)),
            max_eigen.max_scalar(<T as FloatElement>::from_f64(1.0)),
        )
    }
}

impl<T: RealField + Copy + FloatElement + RealScalar> ChebyshevSmoother<T> {
    /// Apply the smoother to the system `Ax = b` for `iterations` sweeps.
    pub fn apply(
        &self,
        matrix: &CsrMatrix<T>,
        x: &mut Array1<T>,
        b: &Array1<T>,
        iterations: usize,
    ) {
        let theta =
            (self.eigenvalues_max + self.eigenvalues_min) / <T as FloatElement>::from_f64(2.0);
        let delta =
            (self.eigenvalues_max - self.eigenvalues_min) / <T as FloatElement>::from_f64(2.0);
        let sigma = if NumericElement::abs(delta) < <T as FloatElement>::from_f64(1e-10) {
            <T as NumericElement>::ZERO
        } else {
            theta / delta
        };

        let rho_old = if sigma == <T as NumericElement>::ZERO {
            <T as NumericElement>::ZERO
        } else {
            <T as NumericElement>::ONE / sigma
        };

        // Compute initial residual
        let mut r = residual(matrix, b, x);

        let alpha = if sigma == <T as NumericElement>::ZERO {
            <T as NumericElement>::ONE / self.eigenvalues_max
        } else {
            let rho_new =
                <T as NumericElement>::ONE / (<T as FloatElement>::from_f64(2.0) * sigma - rho_old);
            rho_new / rho_old
        };

        let offsets = matrix.row_ptr();
        let indices = matrix.col_indices();
        let values = matrix.values();

        // Chebyshev iterations
        for _ in 0..iterations.min(self.degree) {
            for i in 0..matrix.nrows() {
                let mut diag = <T as NumericElement>::ZERO;
                for k in offsets[i]..offsets[i + 1] {
                    if indices[k] == i {
                        diag = values[k];
                        break;
                    }
                }

                if NumericElement::abs(diag) > diagonal_epsilon() {
                    x[i] += alpha * r[i] / diag;
                }
            }
            r = residual(matrix, b, x);
        }
    }
}

#[cfg(test)]
mod tests;
