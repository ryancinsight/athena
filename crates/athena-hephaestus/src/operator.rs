use athena_core::{KrylovBackend, LinearOperator};
use bytemuck::Pod;
use eunomia::RealField;
use hephaestus_core::{
    ComputeDevice, DenseVectorOps, HephaestusError, Result, RetainedReductions, SparseOperatorOps,
};

use crate::HephaestusBackend;

/// Square sparse operator resident in device memory, over any Hephaestus
/// backend.
///
/// The matrix and its application both go through
/// [`SparseOperatorOps`], so this operator carries no device API of its own and
/// serves every backend implementing that seam.
pub struct CsrOperator<S, M> {
    operations: S,
    matrix: M,
    dimension: usize,
}

impl<S, M> CsrOperator<S, M> {
    /// Borrow the device-resident matrix.
    pub const fn matrix(&self) -> &M {
        &self.matrix
    }

    /// Dimension of the square system.
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
}

impl<S, M> CsrOperator<S, M> {
    /// Upload canonical CSR parts and require the result to be square.
    ///
    /// # Errors
    ///
    /// Returns [`HephaestusError::DispatchFailed`] for a rectangular matrix,
    /// plus any structural, index-width, or transfer failure the seam reports.
    pub fn from_parts<D, T>(
        operations: S,
        device: &D,
        values: &[T],
        col_indices: &[usize],
        row_ptr: &[usize],
        rows: usize,
        columns: usize,
    ) -> Result<Self>
    where
        D: ComputeDevice,
        T: Pod,
        S: SparseOperatorOps<D, T, Matrix = M>,
    {
        if rows != columns {
            return Err(HephaestusError::DispatchFailed {
                message: format!("Krylov operators must be square: got {rows} x {columns}"),
            });
        }
        let matrix = operations.upload_csr(device, values, col_indices, row_ptr, rows, columns)?;
        Ok(Self {
            operations,
            matrix,
            dimension: rows,
        })
    }
}

impl<D, V, S, T, M> LinearOperator<HephaestusBackend<D, V, T>> for CsrOperator<S, M>
where
    D: ComputeDevice + 'static,
    V: DenseVectorOps<D, T> + RetainedReductions<D, T> + 'static,
    S: SparseOperatorOps<D, T, Matrix = M>,
    T: RealField + Pod,
{
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(
        &self,
        backend: &HephaestusBackend<D, V, T>,
        input: <HephaestusBackend<D, V, T> as KrylovBackend>::View<'_>,
        output: <HephaestusBackend<D, V, T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<()> {
        self.operations
            .apply(backend.device(), &self.matrix, input, output)
    }
}
