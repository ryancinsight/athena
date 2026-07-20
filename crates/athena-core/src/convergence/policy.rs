use core::fmt;
use eunomia::{NumericElement, RealField};

/// Invalid convergence-policy reason.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvalidConvergencePolicy {
    /// An absolute or relative tolerance was negative or non-finite.
    InvalidTolerance,
    /// No iterations were permitted.
    ZeroIterations,
    /// Residual checks were disabled by a zero interval.
    ZeroCheckInterval,
}

impl fmt::Display for InvalidConvergencePolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTolerance => formatter
                .write_str("absolute and relative tolerances must be finite and non-negative"),
            Self::ZeroIterations => formatter.write_str("maximum iterations must be positive"),
            Self::ZeroCheckInterval => {
                formatter.write_str("residual check interval must be positive")
            }
        }
    }
}

impl core::error::Error for InvalidConvergencePolicy {}

/// Absolute-plus-relative convergence policy.
///
/// A residual `r` is converged when
/// `‖r‖₂ <= max(absolute_tolerance, relative_tolerance * ‖b‖₂)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvergencePolicy<T> {
    absolute_tolerance: T,
    relative_tolerance: T,
    max_iterations: usize,
    check_interval: usize,
}

impl<T: RealField> ConvergencePolicy<T> {
    /// Construct a validated policy that checks every iteration.
    ///
    /// # Errors
    ///
    /// Returns the exact invalid-policy reason.
    pub fn new(
        absolute_tolerance: T,
        relative_tolerance: T,
        max_iterations: usize,
    ) -> Result<Self, InvalidConvergencePolicy> {
        Self::with_check_interval(absolute_tolerance, relative_tolerance, max_iterations, 1)
    }

    /// Construct a policy with a residual check interval.
    ///
    /// # Errors
    ///
    /// Returns the exact invalid-policy reason.
    pub fn with_check_interval(
        absolute_tolerance: T,
        relative_tolerance: T,
        max_iterations: usize,
        check_interval: usize,
    ) -> Result<Self, InvalidConvergencePolicy> {
        let zero = <T as NumericElement>::ZERO;
        if !absolute_tolerance.is_finite()
            || !relative_tolerance.is_finite()
            || absolute_tolerance < zero
            || relative_tolerance < zero
        {
            return Err(InvalidConvergencePolicy::InvalidTolerance);
        }
        if max_iterations == 0 {
            return Err(InvalidConvergencePolicy::ZeroIterations);
        }
        if check_interval == 0 {
            return Err(InvalidConvergencePolicy::ZeroCheckInterval);
        }
        Ok(Self {
            absolute_tolerance,
            relative_tolerance,
            max_iterations,
            check_interval,
        })
    }

    /// Absolute residual tolerance.
    #[must_use]
    pub const fn absolute_tolerance(&self) -> T {
        self.absolute_tolerance
    }

    /// Relative residual tolerance.
    #[must_use]
    pub const fn relative_tolerance(&self) -> T {
        self.relative_tolerance
    }

    /// Maximum Krylov iterations.
    #[must_use]
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Iteration interval between explicit convergence checks.
    #[must_use]
    pub const fn check_interval(&self) -> usize {
        self.check_interval
    }

    /// Compute the effective residual threshold for `right_hand_side_norm`.
    #[must_use]
    pub fn threshold(&self, right_hand_side_norm: T) -> T {
        let relative = self.relative_tolerance * right_hand_side_norm;
        if relative > self.absolute_tolerance {
            relative
        } else {
            self.absolute_tolerance
        }
    }

    /// Return whether iteration `iteration` is a configured check point.
    #[must_use]
    pub const fn should_check(&self, iteration: usize) -> bool {
        iteration.is_multiple_of(self.check_interval) || iteration == self.max_iterations
    }
}

#[cfg(test)]
mod tests {
    use super::{ConvergencePolicy, InvalidConvergencePolicy};

    #[test]
    fn validates_every_policy_boundary() {
        assert_eq!(
            ConvergencePolicy::new(f64::NAN, 0.0, 1),
            Err(InvalidConvergencePolicy::InvalidTolerance)
        );
        assert_eq!(
            ConvergencePolicy::new(-f64::EPSILON, 0.0, 1),
            Err(InvalidConvergencePolicy::InvalidTolerance)
        );
        assert_eq!(
            ConvergencePolicy::new(0.0, 0.0, 0),
            Err(InvalidConvergencePolicy::ZeroIterations)
        );
        assert_eq!(
            ConvergencePolicy::with_check_interval(0.0, 0.0, 1, 0),
            Err(InvalidConvergencePolicy::ZeroCheckInterval)
        );
    }

    #[test]
    fn combines_absolute_and_relative_thresholds() {
        let policy = ConvergencePolicy::new(0.25_f64, 0.1, 8).expect("invariant: valid policy");
        assert!((policy.threshold(1.0) - 0.25).abs() <= f64::EPSILON);
        assert!((policy.threshold(10.0) - 1.0).abs() <= f64::EPSILON);
    }
}
