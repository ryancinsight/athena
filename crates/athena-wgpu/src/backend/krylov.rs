use athena_core::KrylovBackend;
use hephaestus_core::{
    Binding, CommandStream, ComputeDevice, DeviceBuffer, DispatchGrid, HephaestusError,
    KernelDevice, Result,
};
use hephaestus_wgpu::{
    PreparedDot, PreparedL2Norm, StridedOperand, WgpuBuffer, WgpuDevice, WgpuPrepared,
    prepare_dot as prepare_gpu_dot, prepare_norm_l2 as prepare_gpu_norm_l2,
};
use leto::Layout;

use super::kernels::{
    AxpyKernel, DirectionKernel, ResidualKernel, ScaleKernel, UpdateKernel, VectorParams,
};

/// Hephaestus WGPU backend with prepared Athena vector kernels.
///
/// Full vectors remain device-resident across iterations. Convergence
/// reductions transfer only one scalar to the host.
#[derive(Clone, Debug)]
pub struct WgpuBackend {
    device: WgpuDevice,
    residual: WgpuPrepared<ResidualKernel>,
    update: WgpuPrepared<UpdateKernel>,
    direction: WgpuPrepared<DirectionKernel>,
    scale: WgpuPrepared<ScaleKernel>,
    axpy: WgpuPrepared<AxpyKernel>,
}

/// Prepared Hephaestus dot product with its fixed Athena input identities.
pub struct WgpuPreparedDot {
    operation: PreparedDot<f32>,
    left: WgpuBuffer<f32>,
    right: WgpuBuffer<f32>,
}

/// Prepared Hephaestus Euclidean norm with its fixed Athena input identity.
pub struct WgpuPreparedNorm {
    operation: PreparedL2Norm<f32>,
    input: WgpuBuffer<f32>,
}

impl WgpuBackend {
    /// Prepare solver-specific kernels on an acquired Hephaestus device.
    ///
    /// # Errors
    ///
    /// Returns a shader preparation or device failure.
    pub fn new(device: WgpuDevice) -> Result<Self> {
        Ok(Self {
            residual: device.prepare(&ResidualKernel)?,
            update: device.prepare(&UpdateKernel)?,
            direction: device.prepare(&DirectionKernel)?,
            scale: device.prepare(&ScaleKernel)?,
            axpy: device.prepare(&AxpyKernel)?,
            device,
        })
    }

    /// Borrow the Hephaestus device used for allocation and dispatch.
    #[must_use]
    pub const fn device(&self) -> &WgpuDevice {
        &self.device
    }

    fn grid(len: usize) -> Result<DispatchGrid> {
        DispatchGrid::covering_domain([len, 1, 1], [256, 1, 1])
    }

    fn layout(len: usize) -> Result<Layout<1>> {
        Layout::c_contiguous([len]).map_err(|error| HephaestusError::DispatchFailed {
            message: format!("Athena vector layout failed: {error}"),
        })
    }

    fn validate_lengths(left: usize, right: usize) -> Result<()> {
        if left == right {
            Ok(())
        } else {
            Err(HephaestusError::LengthMismatch {
                host_len: left,
                device_len: right,
            })
        }
    }

    fn download_scalar(&self, scalar: &WgpuBuffer<f32>) -> Result<f32> {
        let mut host = [0.0_f32];
        self.device.download(scalar, &mut host)?;
        Ok(host[0])
    }

    fn validate_prepared_input(
        role: &'static str,
        expected: &WgpuBuffer<f32>,
        actual: &WgpuBuffer<f32>,
    ) -> Result<()> {
        if expected.raw() == actual.raw() {
            Ok(())
        } else {
            Err(HephaestusError::DispatchFailed {
                message: format!("prepared {role} received a different device allocation"),
            })
        }
    }
}

impl KrylovBackend for WgpuBackend {
    type Scalar = f32;
    type Error = HephaestusError;
    type Vector = WgpuBuffer<f32>;
    type PreparedDot = WgpuPreparedDot;
    type PreparedNorm = WgpuPreparedNorm;
    type View<'a>
        = &'a WgpuBuffer<f32>
    where
        Self: 'a;
    type ViewMut<'a>
        = &'a mut WgpuBuffer<f32>
    where
        Self: 'a;

    #[inline]
    fn allocate(&self, len: usize) -> Result<Self::Vector> {
        self.device.alloc_zeroed(len)
    }

    #[inline]
    fn view<'a>(&'a self, vector: &'a Self::Vector) -> Self::View<'a> {
        vector
    }

    #[inline]
    fn view_mut<'a>(&'a self, vector: &'a mut Self::Vector) -> Self::ViewMut<'a> {
        vector
    }

    #[inline]
    fn vector_len(&self, vector: &Self::Vector) -> usize {
        vector.len()
    }

    fn copy(&self, source: Self::View<'_>, target: Self::ViewMut<'_>) -> Result<()> {
        Self::validate_lengths(source.len(), target.len())?;
        let mut stream = self.device.stream()?;
        stream.copy(source, target)?;
        stream.submit()
    }

    fn scale(&self, target: Self::ViewMut<'_>, factor: f32) -> Result<()> {
        if target.is_empty() {
            return Ok(());
        }
        self.device.dispatch(
            &self.scale,
            &[Binding::read_write(target)],
            &VectorParams::new(factor, target.len())?,
            Self::grid(target.len())?,
        )
    }

    fn axpy(&self, target: Self::ViewMut<'_>, source: Self::View<'_>, factor: f32) -> Result<()> {
        Self::validate_lengths(target.len(), source.len())?;
        if target.is_empty() {
            return Ok(());
        }
        self.device.dispatch(
            &self.axpy,
            &[Binding::read_write(target), Binding::read(source)],
            &VectorParams::new(factor, target.len())?,
            Self::grid(target.len())?,
        )
    }

    fn prepare_dot(
        &self,
        left: Self::View<'_>,
        right: Self::View<'_>,
    ) -> Result<Self::PreparedDot> {
        Self::validate_lengths(left.len(), right.len())?;
        let layout = Self::layout(left.len())?;
        let operation = prepare_gpu_dot(
            &self.device,
            StridedOperand {
                buffer: left,
                layout: &layout,
            },
            StridedOperand {
                buffer: right,
                layout: &layout,
            },
        )?;
        Ok(WgpuPreparedDot {
            operation,
            left: left.clone(),
            right: right.clone(),
        })
    }

    fn dot_prepared(
        &self,
        prepared: &Self::PreparedDot,
        left: Self::View<'_>,
        right: Self::View<'_>,
    ) -> Result<f32> {
        Self::validate_lengths(left.len(), right.len())?;
        Self::validate_prepared_input("dot left operand", &prepared.left, left)?;
        Self::validate_prepared_input("dot right operand", &prepared.right, right)?;
        prepared.operation.dispatch(&self.device)?;
        self.download_scalar(prepared.operation.output())
    }

    fn prepare_norm_l2(&self, vector: Self::View<'_>) -> Result<Self::PreparedNorm> {
        let layout = Self::layout(vector.len())?;
        let operation = prepare_gpu_norm_l2(
            &self.device,
            StridedOperand {
                buffer: vector,
                layout: &layout,
            },
        )?;
        Ok(WgpuPreparedNorm {
            operation,
            input: vector.clone(),
        })
    }

    fn norm_l2_prepared(
        &self,
        prepared: &Self::PreparedNorm,
        vector: Self::View<'_>,
    ) -> Result<f32> {
        Self::validate_prepared_input("L2 norm operand", &prepared.input, vector)?;
        prepared.operation.dispatch(&self.device)?;
        self.download_scalar(prepared.operation.output())
    }

    fn residual(
        &self,
        right_hand_side: Self::View<'_>,
        image: Self::View<'_>,
        residual: Self::ViewMut<'_>,
    ) -> Result<()> {
        Self::validate_lengths(right_hand_side.len(), image.len())?;
        Self::validate_lengths(right_hand_side.len(), residual.len())?;
        if residual.is_empty() {
            return Ok(());
        }
        self.device.dispatch(
            &self.residual,
            &[
                Binding::read(right_hand_side),
                Binding::read(image),
                Binding::read_write(residual),
            ],
            &VectorParams::new(0.0, residual.len())?,
            Self::grid(residual.len())?,
        )
    }

    fn fused_cg_update(
        &self,
        solution: Self::ViewMut<'_>,
        direction: Self::View<'_>,
        residual: Self::ViewMut<'_>,
        image: Self::View<'_>,
        alpha: f32,
    ) -> Result<()> {
        Self::validate_lengths(solution.len(), direction.len())?;
        Self::validate_lengths(solution.len(), residual.len())?;
        Self::validate_lengths(solution.len(), image.len())?;
        if solution.is_empty() {
            return Ok(());
        }
        self.device.dispatch(
            &self.update,
            &[
                Binding::read_write(solution),
                Binding::read(direction),
                Binding::read_write(residual),
                Binding::read(image),
            ],
            &VectorParams::new(alpha, solution.len())?,
            Self::grid(solution.len())?,
        )
    }

    fn combine_direction(
        &self,
        direction: Self::ViewMut<'_>,
        preconditioned_residual: Self::View<'_>,
        beta: f32,
    ) -> Result<()> {
        Self::validate_lengths(direction.len(), preconditioned_residual.len())?;
        if direction.is_empty() {
            return Ok(());
        }
        self.device.dispatch(
            &self.direction,
            &[
                Binding::read_write(direction),
                Binding::read(preconditioned_residual),
            ],
            &VectorParams::new(beta, direction.len())?,
            Self::grid(direction.len())?,
        )
    }
}
