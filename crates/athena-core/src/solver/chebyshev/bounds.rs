//! Validated spectral bounds for Chebyshev semi-iteration.

use eunomia::{NumericElement, RealField};

/// Eigenvalue interval `[λ_min, λ_max]` of a symmetric positive-definite
/// operator, the input the Chebyshev recurrence needs.
///
/// Construction validates the interval: finite, strictly positive, and
/// ordered. The strict positivity carries the SPD contract — a
/// semidefinite or indefinite operator has no damping guarantee. Equal
/// bounds are supported: for an SPD operator the singleton spectrum means
/// `A = λ I`, and one Richardson step with scale `1/λ` solves the system
/// in exact arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpectralBounds<T> {
    minimum: T,
    maximum: T,
}

impl<T> SpectralBounds<T>
where
    T: RealField + NumericElement,
{
    /// Build the bounds, validating the interval.
    ///
    /// # Errors
    ///
    /// Returns a tuple of the failure and the offending value: a non-finite
    /// or non-positive `minimum`, a non-finite `maximum`, or a `maximum`
    /// below `minimum`.
    pub fn new(minimum: T, maximum: T) -> Result<Self, SpectralBoundsError<T>> {
        if !minimum.is_finite() {
            return Err(SpectralBoundsError::NotFinite { value: minimum });
        }
        if minimum <= T::ZERO {
            return Err(SpectralBoundsError::NotPositiveDefinite { value: minimum });
        }
        if !maximum.is_finite() {
            return Err(SpectralBoundsError::NotFinite { value: maximum });
        }
        if maximum < minimum {
            return Err(SpectralBoundsError::Inverted { minimum, maximum });
        }
        Ok(Self { minimum, maximum })
    }

    /// The interval's lower end `λ_min`.
    #[must_use]
    pub const fn minimum(&self) -> T {
        self.minimum
    }

    /// The interval's upper end `λ_max`.
    #[must_use]
    pub const fn maximum(&self) -> T {
        self.maximum
    }
}

/// Why a spectral-bounds interval was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralBoundsError<T> {
    /// A bound was NaN or infinite.
    NotFinite {
        /// The offending bound.
        value: T,
    },
    /// The lower bound was zero or negative: no SPD guarantee.
    NotPositiveDefinite {
        /// The offending lower bound.
        value: T,
    },
    /// The upper bound lay below the lower bound.
    Inverted {
        /// The interval's lower end.
        minimum: T,
        /// The interval's upper end.
        maximum: T,
    },
}
