//! Tiered linear-solver fallback chain for saddle-point systems.
//!
//! # Algorithm — Tiered Linear Solver Fallback
//!
//! Attempts solvers in priority order; returns the first successful solution.
//! Saddle-point systems from mixed FEM (Taylor-Hood P2-P1) are non-SPD; the
//! best preconditioner depends on problem size and conditioning.
//!
//! ## Solve Priority Order (cold path)
//!
//! ```text
//! 1. Direct sparse LU             — exact, used when n < threshold
//! 2. GMRES + component blocks    — provider sparse-LU momentum solves
//! 3. GMRES + SIMPLE              — coupled saddle-point preconditioner
//! 4. GMRES + Algebraic Multigrid — elliptic fallback
//! 5. GMRES + BlockDiagonal       — diagonal saddle-point fallback
//! 6. GMRES (unpreconditioned)    — only when no block preconditioner built
//! 7. GMRES + ILU                  — incomplete-LU fallback
//! 8. BiCGSTAB (unpreconditioned) — last resort for non-symmetry
//! ```
//!
//! The warm path (`solve_with_guess_state`) leads with weighted-Jacobi/SOR
//! (inexpensive for the ordered, diagonally dominant Venturi pattern) and
//! reuses cached AMG hierarchy and component-block symbolic factors when the
//! CSR sparsity pattern is unchanged.
//!
//! **Rationale.** Direct LU is exact and preferred for small systems; for
//! large saddle-point systems from mixed FEM the block diagonal preconditioner
//! (Elman, Silvester & Wathen 2014, §3.4) exploits the 2×2 block structure to
//! give mesh-independent GMRES convergence. The ILU fallback addresses highly
//! anisotropic or nearly singular systems where block preconditioning fails.
//!
//! # References
//!
//! - Elman, H., Silvester, D. & Wathen, A. (2014). *Finite Elements and Fast
//!   Iterative Solvers.* Oxford University Press, §3.4.
//! - Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.).
//!   SIAM, Chapter 6.
//! - Benzi, M., Golub, G.H. & Liesen, J. (2005). "Numerical solution of
//!   saddle point problems." *Acta Numerica* 14:1–137.

mod cold;
mod direct;
mod helpers;
mod state;
mod warm;

pub use state::{LinearSolverChain, LinearSolverState};
