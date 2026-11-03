/// Scalar iteration telemetry.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IterationState<T> {
    /// One-based iteration number.
    pub iteration: usize,
    /// Current residual norm. CG and GMRES can report recurrence estimates
    /// during iteration and explicitly recomputed norms at terminal checks.
    pub residual_norm: T,
    /// Effective convergence threshold.
    pub threshold: T,
}

impl<T> IterationState<T> {
    /// Construct one solver residual sample.
    ///
    /// # Examples
    ///
    /// ```
    /// use athena_core::IterationState;
    ///
    /// let sample = IterationState::new(3, 0.25_f64, 0.5);
    ///
    /// assert_eq!(sample.iteration, 3);
    /// assert!(sample.residual_norm <= sample.threshold);
    /// ```
    #[must_use]
    pub const fn new(iteration: usize, residual_norm: T, threshold: T) -> Self {
        Self {
            iteration,
            residual_norm,
            threshold,
        }
    }
}

/// Observer for allocation-free residual telemetry.
///
/// Applications choose whether to store, stream, or discard samples; Athena
/// never allocates a residual-history vector implicitly.
pub trait IterationObserver<T> {
    /// Observe one scheduled or algorithm-required terminal residual check.
    fn observe(&mut self, state: IterationState<T>);
}

/// Zero-sized observer that discards telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoObserver;

impl<T> IterationObserver<T> for NoObserver {
    #[inline]
    fn observe(&mut self, _state: IterationState<T>) {}
}
