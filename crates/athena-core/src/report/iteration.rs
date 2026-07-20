/// Scalar iteration telemetry.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterationState<T> {
    /// One-based iteration number.
    pub iteration: usize,
    /// Current Euclidean residual norm.
    pub residual_norm: T,
    /// Effective convergence threshold.
    pub threshold: T,
}

/// Observer for allocation-free residual telemetry.
///
/// Applications choose whether to store, stream, or discard samples; Athena
/// never allocates a residual-history vector implicitly.
pub trait IterationObserver<T> {
    /// Observe one configured residual check.
    fn observe(&mut self, state: IterationState<T>);
}

/// Zero-sized observer that discards telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoObserver;

impl<T> IterationObserver<T> for NoObserver {
    #[inline]
    fn observe(&mut self, _state: IterationState<T>) {}
}
