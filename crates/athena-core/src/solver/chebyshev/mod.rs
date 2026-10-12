//! Chebyshev semi-iteration for SPD operators with spectral bounds.

mod algorithm;
mod bounds;
mod execution;
mod workspace;

pub use algorithm::Chebyshev;
pub use bounds::{SpectralBounds, SpectralBoundsError};
pub use workspace::ChebyshevWorkspace;
