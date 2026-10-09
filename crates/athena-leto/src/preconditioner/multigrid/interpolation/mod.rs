//! AMG interpolation: prolongation operators and their quality metrics.
//!
//! [`builders`] assembles the operators, [`quality`] measures them, and
//! [`tests`] pins both to the recorded fixtures.

mod builders;
mod quality;

#[cfg(test)]
mod tests;

pub use builders::{
    create_classical_interpolation, create_direct_interpolation, create_standard_interpolation,
};
pub use quality::{InterpolationQuality, validate_interpolation_operator};
