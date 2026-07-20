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

    /// Borrow an immutable vector view.
    fn view<'a>(&'a self, vector: &'a Self::Vector) -> Self::View<'a>;

    /// Borrow a writable vector view.
    fn view_mut<'a>(&'a self, vector: &'a mut Self::Vector) -> Self::ViewMut<'a>;

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

    /// Compute the native-precision dot product.
    ///
    /// # Errors
    ///
    /// Returns a backend dispatch or shape error.
    fn dot(&self, left: Self::View<'_>, right: Self::View<'_>)
    -> Result<Self::Scalar, Self::Error>;

    /// Compute the native-precision Euclidean norm.
    ///
    /// # Errors
    ///
    /// Returns a backend dispatch error.
    fn norm_l2(&self, vector: Self::View<'_>) -> Result<Self::Scalar, Self::Error>;

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
