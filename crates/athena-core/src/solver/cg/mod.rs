//! Preconditioned conjugate gradient.

mod algorithm;
mod workspace;

pub use algorithm::Cg;
pub use workspace::CgWorkspace;
