use athena_core::{KrylovBackend, LinearOperator};
use hephaestus_core::{HephaestusError, Result};
use hephaestus_wgpu::{GpuCsrMatrix, spmv_into};
use leto_ops::CsrMatrix;

use crate::WgpuBackend;

/// Square CSR operator resident in Hephaestus WGPU buffers.
#[derive(Clone, Debug)]
pub struct WgpuCsrOperator {
    matrix: GpuCsrMatrix<f32>,
    dimension: usize,
}

impl WgpuCsrOperator {
    /// Upload and validate Leto's canonical CSR matrix.
    ///
    /// # Errors
    ///
    /// Returns a shape, conversion, allocation, or transfer failure.
    pub fn from_cpu(backend: &WgpuBackend, matrix: &CsrMatrix<f32>) -> Result<Self> {
        let (rows, columns) = matrix.shape();
        if rows != columns {
            return Err(HephaestusError::DispatchFailed {
                message: format!("Athena operator must be square: got {rows} x {columns}"),
            });
        }
        Ok(Self {
            matrix: GpuCsrMatrix::from_cpu(backend.device(), matrix)?,
            dimension: rows,
        })
    }

    /// Borrow the GPU-resident CSR matrix.
    #[must_use]
    pub const fn matrix(&self) -> &GpuCsrMatrix<f32> {
        &self.matrix
    }
}

impl LinearOperator<WgpuBackend> for WgpuCsrOperator {
    #[inline]
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(
        &self,
        backend: &WgpuBackend,
        input: <WgpuBackend as KrylovBackend>::View<'_>,
        output: <WgpuBackend as KrylovBackend>::ViewMut<'_>,
    ) -> Result<()> {
        spmv_into(backend.device(), &self.matrix, input, output)
    }
}
