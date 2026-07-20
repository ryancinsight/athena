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

impl<T> IterationState<T> {
    /// Construct one checked residual sample.
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

#[cfg(test)]
mod tests {
    use super::IterationState;

    #[test]
    fn constructs_external_observer_sample() {
        let sample = IterationState::new(3, 0.25_f64, 0.5);

        assert_eq!(sample.iteration, 3);
        assert_eq!(sample.residual_norm.to_bits(), 0.25_f64.to_bits());
        assert_eq!(sample.threshold.to_bits(), 0.5_f64.to_bits());
    }
}
