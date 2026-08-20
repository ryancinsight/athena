use athena_core::{KrylovBackend, Preconditioner};
use bytemuck::Pod;
use eunomia::{NumericElement, RealField};
use hephaestus_core::{
    ComputeDevice, DenseVectorOps, DeviceBuffer, HephaestusError, Result, RetainedReductions,
};

use crate::HephaestusBackend;

/// Jacobi preconditioner holding the inverse matrix diagonal in device memory.
///
/// The inverse is stored rather than the diagonal itself, so an application is
/// an elementwise multiply rather than a divide. The multiply goes through
/// [`DenseVectorOps`], so this preconditioner carries no device API of its own
/// and serves every backend implementing that seam.
pub struct Jacobi<D: ComputeDevice, T: Pod> {
    inverse_diagonal: D::Buffer<T>,
}

impl<D: ComputeDevice, T: Pod> Jacobi<D, T> {
    /// Borrow the device-resident inverse diagonal.
    #[must_use]
    pub const fn inverse_diagonal(&self) -> &D::Buffer<T> {
        &self.inverse_diagonal
    }

    /// Dimension of the square system this preconditioner was built for.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.inverse_diagonal.len()
    }
}

impl<D: ComputeDevice, T: RealField + Pod> Jacobi<D, T> {
    /// Invert the diagonal of a square matrix in canonical CSR form and upload
    /// it.
    ///
    /// The parts are the same ones [`crate::CsrOperator::from_parts`] takes, so
    /// one description of a system builds both the operator and its
    /// preconditioner. The diagonal is extracted and inverted on the host —
    /// [`inverse_diagonal_from_csr`] is that step on its own — and only the
    /// result crosses to the device.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::DispatchFailed`] for a rectangular matrix,
    /// [`HephaestusError::InvalidConfiguration`] for malformed CSR structure or
    /// a zero diagonal entry, and any transfer failure the device reports.
    pub fn from_csr_parts(
        device: &D,
        values: &[T],
        col_indices: &[usize],
        row_ptr: &[usize],
        rows: usize,
        columns: usize,
    ) -> Result<Self> {
        let inverse = inverse_diagonal_from_csr(values, col_indices, row_ptr, rows, columns)?;
        Ok(Self {
            inverse_diagonal: device.upload(&inverse)?,
        })
    }
}

impl<D, V, T> Preconditioner<HephaestusBackend<D, V, T>> for Jacobi<D, T>
where
    D: ComputeDevice + 'static,
    V: DenseVectorOps<D, T> + RetainedReductions<D, T> + 'static,
    T: RealField + Pod,
{
    /// `output = diagonal⁻¹ ⊙ residual`, one seam dispatch, entirely on device.
    ///
    /// A mismatched or aliased operand is rejected by the seam rather than
    /// re-checked here; the lengths this preconditioner must agree with are the
    /// solver's, which only the seam sees.
    fn apply(
        &self,
        backend: &HephaestusBackend<D, V, T>,
        residual: <HephaestusBackend<D, V, T> as KrylovBackend>::View<'_>,
        output: <HephaestusBackend<D, V, T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<()> {
        backend.operations().multiply_into(
            backend.device(),
            &self.inverse_diagonal,
            residual,
            output,
        )
    }
}

/// Extract and invert the diagonal of a square matrix in canonical CSR form.
///
/// Host-side and device-free: this is the whole construction contract of
/// [`Jacobi`], separated so a caller that already holds an inverse diagonal, or
/// that wants to validate a system before acquiring a device, need not go
/// through an upload.
///
/// A diagonal entry absent from the sparsity structure is zero, and is rejected
/// on the same terms as a stored zero: the preconditioner would divide by it.
///
/// # Errors
///
/// Returns [`HephaestusError::DispatchFailed`] for a rectangular matrix, and
/// [`HephaestusError::InvalidConfiguration`] when the CSR parts are structurally
/// inconsistent or a diagonal entry is zero — the message names the offending
/// row.
pub fn inverse_diagonal_from_csr<T: RealField>(
    values: &[T],
    col_indices: &[usize],
    row_ptr: &[usize],
    rows: usize,
    columns: usize,
) -> Result<Vec<T>> {
    if rows != columns {
        return Err(HephaestusError::DispatchFailed {
            message: format!("Jacobi requires a square matrix: got {rows} x {columns}"),
        });
    }
    if col_indices.len() != values.len() {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "CSR column indices ({}) and values ({}) must agree in length",
                col_indices.len(),
                values.len()
            ),
        });
    }
    if row_ptr.len().checked_sub(1) != Some(rows) {
        return Err(HephaestusError::InvalidConfiguration {
            message: format!(
                "CSR row_ptr must hold one entry more than the {rows} rows: got {}",
                row_ptr.len()
            ),
        });
    }

    let mut inverse = Vec::with_capacity(rows);
    for (row, (&start, &end)) in row_ptr.iter().zip(row_ptr.iter().skip(1)).enumerate() {
        let (Some(row_columns), Some(row_values)) =
            (col_indices.get(start..end), values.get(start..end))
        else {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!(
                    "CSR row {row} spans {start}..{end}, outside the {} stored entries",
                    values.len()
                ),
            });
        };
        let diagonal = row_columns
            .iter()
            .zip(row_values)
            .find_map(|(&column, &value)| (column == row).then_some(value))
            .unwrap_or(<T as NumericElement>::ZERO);
        if diagonal == <T as NumericElement>::ZERO {
            return Err(HephaestusError::InvalidConfiguration {
                message: format!("Jacobi requires a nonzero diagonal: entry {row} is zero"),
            });
        }
        inverse.push(<T as NumericElement>::ONE / diagonal);
    }
    Ok(inverse)
}
