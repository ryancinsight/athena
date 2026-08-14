//! Restarted right-preconditioned GMRES.

mod algorithm;
mod cycle;
mod rotation;
mod workspace;

pub use algorithm::Gmres;
pub use workspace::GmresWorkspace;
