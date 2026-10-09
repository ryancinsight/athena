//! Algebraic Multigrid (AMG) preconditioner.
//!
//! - Ruge, J. W. and Stueben, K. (1987). Algebraic multigrid. In the
//!
//! For an SPD M-matrix `A` arising from the discretisation of a second-order
//! elliptic PDE, the AMG V-cycle preconditioner achieves a convergence factor
//! bounded independently of the mesh size `h`:
//!
//! ```text
//! ρ_AMG ≤ γ < 1     (independent of h)
//! ```
//!
//! provided the coarsening strategy satisfies the strong-connection heuristic
//! and the interpolation operator reproduces constant vectors exactly.
//!
//! **Proof sketch.** The two-grid convergence analysis requires two properties:
//! (1) a *smoothing property* — the smoother (e.g. Gauss-Seidel) reduces
//! high-frequency error components by a factor `η < 1` per sweep; and
//! (2) an *approximation property* — the coarse-grid correction captures
//! low-frequency error, i.e. `‖e_h − P_h e_{2h}‖_A ≤ C ‖A_h e_h‖`.
//! Combining these gives `ρ_2G ≤ η + C(1−η)² / (1 + C(1−η)) < 1`.
//! The multi-level extension follows by recursive application (Bramble 1993,
//! - Ruge, J. W. and Stueben, K. (1987). Algebraic multigrid. In the
//!   guarantees the approximation property holds uniformly in `h`.
//!
//! # References
//!
//! - Ruge, J. W. and Stueben, K. (1987). Algebraic multigrid. In the
//!   volume edited by `McCormick`, SIAM, pp. 73-130.
//! - Stüben, K. (2001). "A review of algebraic multigrid." *J. Comput. Appl.
//!   Math.* 128:281–309.
//! - Bramble, J. H. (1993). *Multigrid Methods.* Pitman Research Notes in
//!   Mathematics, Longman.

mod cycle;
mod setup;
mod state;

pub use state::AlgebraicMultigrid;
