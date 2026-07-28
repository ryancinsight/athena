use core::marker::PhantomData;

use athena_core::KrylovBackend;
use bytemuck::Pod;
use eunomia::RealField;
use hephaestus_core::{
    ComputeDevice, DenseVectorOps, DeviceBuffer, HephaestusError, Result, RetainedReductions,
};

/// Athena Krylov backend over any Hephaestus device.
///
/// The backend pairs a device with the [`DenseVectorOps`] bundle prepared
/// against it. Every vector operation an Athena recurrence performs resolves to
/// that seam, so this one type serves WGPU, CUDA, Metal, `ROCm`, and any future
/// Hephaestus backend — adding a device adds no code here.
///
/// Full vectors stay device-resident across iterations. Only the convergence
/// reductions cross to the host, one scalar at a time.
pub struct HephaestusBackend<D, V, T> {
    device: D,
    operations: V,
    scalar: PhantomData<fn() -> T>,
}

impl<D, V, T> HephaestusBackend<D, V, T>
where
    D: ComputeDevice + 'static,
    V: DenseVectorOps<D, T> + RetainedReductions<D, T> + 'static,
    T: RealField + Pod,
{
    /// Pair a device with vector operations prepared against it.
    ///
    /// The caller constructs `operations` from `device`, which is what binds
    /// the compiled kernels to the device they will dispatch on.
    pub const fn new(device: D, operations: V) -> Self {
        Self {
            device,
            operations,
            scalar: PhantomData,
        }
    }

    /// Borrow the underlying device for allocation, transfer, and operator
    /// construction.
    pub const fn device(&self) -> &D {
        &self.device
    }

    /// Borrow the prepared vector operations.
    pub const fn operations(&self) -> &V {
        &self.operations
    }
}

impl<D, V, T> KrylovBackend for HephaestusBackend<D, V, T>
where
    D: ComputeDevice + 'static,
    V: DenseVectorOps<D, T> + RetainedReductions<D, T> + 'static,
    T: RealField + Pod,
{
    type Scalar = T;
    type Error = HephaestusError;
    type Vector = D::Buffer<T>;
    // Athena workspaces retain their reductions beside the vectors those
    // reductions measure, which is what keeps a solve allocation-free, so
    // this backend binds the retained rather than the borrowing form.
    type PreparedDot = V::RetainedDot;
    type PreparedNorm = V::RetainedNorm;
    type View<'a>
        = &'a D::Buffer<T>
    where
        Self: 'a;
    type ViewMut<'a>
        = &'a mut D::Buffer<T>
    where
        Self: 'a;

    fn allocate(&self, len: usize) -> Result<Self::Vector> {
        self.device.alloc_zeroed(len)
    }

    fn view<'a>(&'a self, vector: &'a Self::Vector) -> Self::View<'a> {
        vector
    }

    /// A writable view is a unique borrow: Hephaestus operations that write a
    /// buffer take `&mut`, so the shared handle a device buffer offers is not
    /// sufficient at this boundary.
    fn view_mut<'a>(&'a self, vector: &'a mut Self::Vector) -> Self::ViewMut<'a> {
        vector
    }

    fn vector_len(&self, vector: &Self::Vector) -> usize {
        vector.len()
    }

    fn copy(&self, source: Self::View<'_>, target: Self::ViewMut<'_>) -> Result<()> {
        self.operations.copy_vector(&self.device, source, target)
    }

    fn scale(&self, target: Self::ViewMut<'_>, factor: Self::Scalar) -> Result<()> {
        self.operations.scale_vector(&self.device, target, factor)
    }

    fn axpy(
        &self,
        target: Self::ViewMut<'_>,
        source: Self::View<'_>,
        factor: Self::Scalar,
    ) -> Result<()> {
        self.operations.axpy(&self.device, target, source, factor)
    }

    fn prepare_dot(
        &self,
        left: Self::View<'_>,
        right: Self::View<'_>,
    ) -> Result<Self::PreparedDot> {
        self.operations.retain_dot(&self.device, left, right)
    }

    fn dot_prepared(
        &self,
        prepared: &Self::PreparedDot,
        left: Self::View<'_>,
        right: Self::View<'_>,
    ) -> Result<Self::Scalar> {
        self.operations
            .dot_retained(&self.device, prepared, left, right)
    }

    fn prepare_norm_l2(&self, vector: Self::View<'_>) -> Result<Self::PreparedNorm> {
        self.operations.retain_norm_l2(&self.device, vector)
    }

    fn norm_l2_prepared(
        &self,
        prepared: &Self::PreparedNorm,
        vector: Self::View<'_>,
    ) -> Result<Self::Scalar> {
        self.operations
            .norm_l2_retained(&self.device, prepared, vector)
    }

    fn residual(
        &self,
        right_hand_side: Self::View<'_>,
        image: Self::View<'_>,
        residual: Self::ViewMut<'_>,
    ) -> Result<()> {
        self.operations
            .subtract_into(&self.device, right_hand_side, image, residual)
    }

    /// `x += αp` and `r -= αAp`.
    ///
    /// Expressed as two seam updates rather than one fused kernel. Fusing the
    /// pair saves one dispatch per CG iteration, but a fused form would put a
    /// solver-shaped operation into the substrate's vector contract; the pair
    /// stays composed here until a measurement justifies the trade.
    fn fused_cg_update(
        &self,
        solution: Self::ViewMut<'_>,
        direction: Self::View<'_>,
        residual: Self::ViewMut<'_>,
        image: Self::View<'_>,
        alpha: Self::Scalar,
    ) -> Result<()> {
        self.operations
            .axpy(&self.device, solution, direction, alpha)?;
        self.operations.axpy(&self.device, residual, image, -alpha)
    }

    /// `p = z + βp`, the accumulator-scaled update the seam names `xpay`.
    fn combine_direction(
        &self,
        direction: Self::ViewMut<'_>,
        preconditioned_residual: Self::View<'_>,
        beta: Self::Scalar,
    ) -> Result<()> {
        self.operations
            .xpay(&self.device, direction, preconditioned_residual, beta)
    }
}
