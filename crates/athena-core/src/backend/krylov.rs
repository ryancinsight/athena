use eunomia::RealField;

/// Vector arithmetic required by backend-neutral Krylov recurrences.
///
/// The generic associated view types preserve backend-native borrowing:
/// Leto maps them to zero-copy array views, while Hephaestus maps them to
/// borrowed typed device buffers. Solver code is statically dispatched and
/// never boxes a backend operation.
pub trait KrylovBackend {
    /// Real scalar used by the backend.
    type Scalar: RealField;
    /// Backend-specific failure.
    type Error;
    /// Owned, reusable vector storage.
    type Vector;
    /// Owned, reusable storage for a fixed count of equal-length vectors.
    ///
    /// A Krylov basis is a set of vectors allocated together once, written one
    /// at a time, and read repeatedly within an iteration. Naming that set a
    /// backend type instead of a `Vec<Self::Vector>` moves its residency
    /// decision to the backend that owns the memory: a host backend places the
    /// whole set in one contiguous extent and lends offset subviews, while a
    /// device backend keeps independent buffers because allocation and binding
    /// there are per-buffer. Recurrences see indexed views either way and
    /// contain no layout assumption.
    type VectorBlock;
    /// Backend-owned fixed-buffer dot-product resources.
    type PreparedDot;
    /// Backend-owned fixed-buffer Euclidean-norm resources.
    type PreparedNorm;
    /// Immutable zero-copy vector view.
    type View<'a>: Copy
    where
        Self: 'a;
    /// Mutable or interior-mutable zero-copy vector view.
    type ViewMut<'a>
    where
        Self: 'a;

    /// Allocate a zero-initialized vector.
    ///
    /// # Errors
    ///
    /// Returns the backend allocation failure.
    fn allocate(&self, len: usize) -> Result<Self::Vector, Self::Error>;

    /// Allocate `count` zero-initialized vectors of length `len` as one block.
    ///
    /// # Errors
    ///
    /// Returns the backend allocation failure, including a `count * len`
    /// extent the backend cannot represent.
    fn allocate_block(&self, count: usize, len: usize) -> Result<Self::VectorBlock, Self::Error>;

    /// Borrow an immutable vector view.
    fn view<'a>(&'a self, vector: &'a Self::Vector) -> Self::View<'a>;

    /// Borrow a writable vector view.
    fn view_mut<'a>(&'a self, vector: &'a mut Self::Vector) -> Self::ViewMut<'a>;

    /// Borrow vector `index` of `block` immutably.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not less than the `count` the block was
    /// allocated with. Every caller in this crate indexes a const-bounded
    /// basis, so the bound is structural rather than input-dependent.
    fn block_view<'a>(&'a self, block: &'a Self::VectorBlock, index: usize) -> Self::View<'a>;

    /// Borrow vector `index` of `block` for writing.
    ///
    /// Blocks lend one vector at a time. A recurrence that must hold two
    /// vectors at once keeps them in separate blocks, which is what makes the
    /// disjointness a type fact rather than a runtime check.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not less than the `count` the block was
    /// allocated with.
    fn block_view_mut<'a>(
        &'a self,
        block: &'a mut Self::VectorBlock,
        index: usize,
    ) -> Self::ViewMut<'a>;

    /// Return an owned vector's logical length.
    fn vector_len(&self, vector: &Self::Vector) -> usize;

    /// Copy `source` into caller-owned `target`.
    ///
    /// # Errors
    ///
    /// Returns a backend error, including a length mismatch.
    fn copy(&self, source: Self::View<'_>, target: Self::ViewMut<'_>) -> Result<(), Self::Error>;

    /// Scale `target` in place.
    ///
    /// # Errors
    ///
    /// Returns a backend dispatch failure.
    fn scale(&self, target: Self::ViewMut<'_>, factor: Self::Scalar) -> Result<(), Self::Error>;

    /// Apply `target += factor * source` in place.
    ///
    /// # Errors
    ///
    /// Returns a backend dispatch or shape failure.
    fn axpy(
        &self,
        target: Self::ViewMut<'_>,
        source: Self::View<'_>,
        factor: Self::Scalar,
    ) -> Result<(), Self::Error>;

    /// Prepare a dot product over fixed vector allocations.
    ///
    /// # Errors
    ///
    /// Returns a backend allocation, preparation, or shape error.
    fn prepare_dot(
        &self,
        left: Self::View<'_>,
        right: Self::View<'_>,
    ) -> Result<Self::PreparedDot, Self::Error>;

    /// Execute a prepared native-precision dot product.
    ///
    /// # Errors
    ///
    /// Returns a backend dispatch, shape, or prepared-input mismatch error.
    fn dot_prepared(
        &self,
        prepared: &Self::PreparedDot,
        left: Self::View<'_>,
        right: Self::View<'_>,
    ) -> Result<Self::Scalar, Self::Error>;

    /// Compute a one-shot native-precision dot product.
    ///
    /// # Errors
    ///
    /// Returns a backend preparation or dispatch error.
    fn dot(
        &self,
        left: Self::View<'_>,
        right: Self::View<'_>,
    ) -> Result<Self::Scalar, Self::Error> {
        let prepared = self.prepare_dot(left, right)?;
        self.dot_prepared(&prepared, left, right)
    }

    /// Prepare a Euclidean norm over a fixed vector allocation.
    ///
    /// # Errors
    ///
    /// Returns a backend allocation, preparation, or shape error.
    fn prepare_norm_l2(&self, vector: Self::View<'_>) -> Result<Self::PreparedNorm, Self::Error>;

    /// Execute a prepared native-precision Euclidean norm.
    ///
    /// # Errors
    ///
    /// Returns a backend dispatch or prepared-input mismatch error.
    fn norm_l2_prepared(
        &self,
        prepared: &Self::PreparedNorm,
        vector: Self::View<'_>,
    ) -> Result<Self::Scalar, Self::Error>;

    /// Compute a one-shot native-precision Euclidean norm.
    ///
    /// # Errors
    ///
    /// Returns a backend preparation or dispatch error.
    fn norm_l2(&self, vector: Self::View<'_>) -> Result<Self::Scalar, Self::Error> {
        let prepared = self.prepare_norm_l2(vector)?;
        self.norm_l2_prepared(&prepared, vector)
    }

    /// Compute `residual = right_hand_side - image`.
    ///
    /// # Errors
    ///
    /// Returns a backend dispatch or shape error.
    fn residual(
        &self,
        right_hand_side: Self::View<'_>,
        image: Self::View<'_>,
        residual: Self::ViewMut<'_>,
    ) -> Result<(), Self::Error>;

    /// Apply the fused CG update `x += αp; r -= αAp`.
    ///
    /// Backends may fuse both updates in one traversal or device dispatch.
    ///
    /// # Errors
    ///
    /// Returns a backend dispatch or shape error.
    fn fused_cg_update(
        &self,
        solution: Self::ViewMut<'_>,
        direction: Self::View<'_>,
        residual: Self::ViewMut<'_>,
        image: Self::View<'_>,
        alpha: Self::Scalar,
    ) -> Result<(), Self::Error>;

    /// Apply the CG direction recurrence `p = z + βp`.
    ///
    /// # Errors
    ///
    /// Returns a backend dispatch or shape error.
    fn combine_direction(
        &self,
        direction: Self::ViewMut<'_>,
        preconditioned_residual: Self::View<'_>,
        beta: Self::Scalar,
    ) -> Result<(), Self::Error>;
}
