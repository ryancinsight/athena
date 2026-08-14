//! Convergence policy.

mod policy;
mod residual_noise;

pub use policy::{ConvergencePolicy, InvalidConvergencePolicy};
pub use residual_noise::residual_noise_floor;
