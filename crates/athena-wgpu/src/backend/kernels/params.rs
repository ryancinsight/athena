use bytemuck::{Pod, Zeroable};
use hephaestus_core::{HephaestusError, Result};

/// Uniform parameters shared by Athena's vector kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct VectorParams {
    scalar: f32,
    len: u32,
    padding: [u32; 2],
}

impl VectorParams {
    pub(crate) fn new(scalar: f32, len: usize) -> Result<Self> {
        Ok(Self {
            scalar,
            len: u32::try_from(len).map_err(|_| HephaestusError::DispatchFailed {
                message: format!("Athena vector length {len} exceeds u32::MAX"),
            })?,
            padding: [0; 2],
        })
    }
}
