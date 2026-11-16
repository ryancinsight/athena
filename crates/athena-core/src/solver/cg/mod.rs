//! Preconditioned conjugate gradient.

mod algorithm;
mod state;
mod workspace;

pub use algorithm::Cg;
pub use workspace::CgWorkspace;
