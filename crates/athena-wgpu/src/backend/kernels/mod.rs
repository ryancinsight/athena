//! Solver-specific fused WGPU kernels.

mod axpy;
mod direction;
mod params;
mod residual;
mod scale;
mod update;

pub(crate) use axpy::AxpyKernel;
pub(crate) use direction::DirectionKernel;
pub(crate) use params::VectorParams;
pub(crate) use residual::ResidualKernel;
pub(crate) use scale::ScaleKernel;
pub(crate) use update::UpdateKernel;
