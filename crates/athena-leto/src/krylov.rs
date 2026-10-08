//! Reusable restarted-GMRES workspaces over the Leto backend.
//!
//! Athena fixes the GMRES restart width at compile time through a const
//! generic, so the Arnoldi basis and the Hessenberg workspace are sized at
//! compile time and every recurrence step monomorphizes. Callers select the
//! width at runtime, so this module is the one home for that bridge: a fixed
//! geometric ladder of instantiations and a dispatch that picks the smallest
//! width covering the request.
//!
//! Widening a restart never costs correctness — GMRES(m) minimises the
//! residual over the Krylov subspace `K_m(A, r₀)`, which contains `K_m'(A, r₀)`
//! for every `m' <= m` (Saad & Schultz 1986, §2) — so rounding a request up
//! the ladder is safe, and only trades basis memory for subspace depth.

use athena_core::{
    ConvergencePolicy, Gmres, GmresWorkspace, IterationObserver, LinearOperator, NoObserver,
    Preconditioner, SolveError, SolveReport,
};
use core::fmt;
use eunomia::RealField;
use leto::Array1;
use leto_ops::RealScalar;

use crate::{LetoBackend, LetoBackendError};

/// Reusable restarted-GMRES workspace for a fixed dimension and restart width.
///
/// Construction performs every allocation and prepares Athena's reductions, so
/// repeated solves at the same dimension — successive nonlinear iterations,
/// successive coupled time steps — reuse the Krylov basis instead of
/// reallocating it. The requested restart is rounded through the ladder to the
/// smallest covering width; [`Self::width`] reports the rung actually held.
pub struct KrylovWorkspace<T: RealScalar + RealField> {
    inner: Ladder<T>,
    dimension: usize,
}

/// The restart widths `Gmres` is instantiated at.
///
/// The ladder is geometric so a request of any size lands within a factor of
/// two of its width, bounding both the memory a solve reserves and the number
/// of monomorphisations the backend carries.
enum Ladder<T: RealScalar + RealField> {
    W8(GmresWorkspace<LetoBackend<T>, 8>),
    W16(GmresWorkspace<LetoBackend<T>, 16>),
    W32(GmresWorkspace<LetoBackend<T>, 32>),
    W64(GmresWorkspace<LetoBackend<T>, 64>),
    W128(GmresWorkspace<LetoBackend<T>, 128>),
    W256(GmresWorkspace<LetoBackend<T>, 256>),
}

/// Ladder rung selected for a requested restart width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RestartWidth {
    W8,
    W16,
    W32,
    W64,
    W128,
    W256,
}

impl RestartWidth {
    /// Smallest ladder width covering `requested`, saturating at the largest.
    ///
    /// A request above the ceiling is served by the ceiling rather than
    /// rejected: the restart is a tuning parameter, and capping it costs
    /// convergence depth per cycle rather than correctness.
    const fn covering(requested: usize) -> Self {
        if requested <= 8 {
            Self::W8
        } else if requested <= 16 {
            Self::W16
        } else if requested <= 32 {
            Self::W32
        } else if requested <= 64 {
            Self::W64
        } else if requested <= 128 {
            Self::W128
        } else {
            Self::W256
        }
    }
}

impl<T> KrylovWorkspace<T>
where
    T: RealScalar + RealField,
{
    /// Allocate a workspace for `dimension` unknowns and a requested restart.
    ///
    /// # Errors
    ///
    /// Returns the first backend allocation or reduction-preparation failure.
    pub fn new(restart: usize, dimension: usize) -> Result<Self, LetoBackendError> {
        let backend = LetoBackend::<T>::default();
        let inner = match RestartWidth::covering(restart) {
            RestartWidth::W8 => {
                GmresWorkspace::<LetoBackend<T>, 8>::new(&backend, dimension).map(Ladder::W8)
            }
            RestartWidth::W16 => {
                GmresWorkspace::<LetoBackend<T>, 16>::new(&backend, dimension).map(Ladder::W16)
            }
            RestartWidth::W32 => {
                GmresWorkspace::<LetoBackend<T>, 32>::new(&backend, dimension).map(Ladder::W32)
            }
            RestartWidth::W64 => {
                GmresWorkspace::<LetoBackend<T>, 64>::new(&backend, dimension).map(Ladder::W64)
            }
            RestartWidth::W128 => {
                GmresWorkspace::<LetoBackend<T>, 128>::new(&backend, dimension).map(Ladder::W128)
            }
            RestartWidth::W256 => {
                GmresWorkspace::<LetoBackend<T>, 256>::new(&backend, dimension).map(Ladder::W256)
            }
        }?;
        Ok(Self { inner, dimension })
    }

    /// Unknowns this workspace was allocated for.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Restart width this workspace solves at: the ladder rung the requested
    /// restart rounded to, not necessarily the requested value.
    #[must_use]
    pub const fn width(&self) -> usize {
        match &self.inner {
            Ladder::W8(_) => 8,
            Ladder::W16(_) => 16,
            Ladder::W32(_) => 32,
            Ladder::W64(_) => 64,
            Ladder::W128(_) => 128,
            Ladder::W256(_) => 256,
        }
    }

    /// Solve `A·x = b` with restarted GMRES and an explicit preconditioner.
    ///
    /// `solution` carries the initial iterate in and the final iterate out. A
    /// solve that stalls, stagnates, or breaks down is reported
    /// value-semantically in the returned [`SolveReport`] rather than as an
    /// error: the last iterate is still present, and whether it is usable is
    /// the caller's judgement.
    ///
    /// # Errors
    ///
    /// Returns a dimension or backend failure.
    pub fn solve<O, P>(
        &mut self,
        operator: &O,
        preconditioner: &P,
        right_hand_side: &Array1<T>,
        solution: &mut Array1<T>,
        policy: ConvergencePolicy<T>,
    ) -> Result<SolveReport<T>, SolveError<LetoBackendError>>
    where
        O: LinearOperator<LetoBackend<T>>,
        P: Preconditioner<LetoBackend<T>>,
    {
        self.solve_with_observer(
            operator,
            preconditioner,
            right_hand_side,
            solution,
            policy,
            &mut NoObserver,
        )
    }

    /// Solve `A·x = b` while reporting every checked residual to `observer`.
    ///
    /// Athena accumulates no residual history itself; a caller that wants one
    /// supplies the observer that records it. [`Self::solve`] is this call with
    /// the discarding observer.
    ///
    /// # Errors
    ///
    /// See [`Self::solve`].
    pub fn solve_with_observer<O, P, Obs>(
        &mut self,
        operator: &O,
        preconditioner: &P,
        right_hand_side: &Array1<T>,
        solution: &mut Array1<T>,
        policy: ConvergencePolicy<T>,
        observer: &mut Obs,
    ) -> Result<SolveReport<T>, SolveError<LetoBackendError>>
    where
        O: LinearOperator<LetoBackend<T>>,
        P: Preconditioner<LetoBackend<T>>,
        Obs: IterationObserver<T>,
    {
        let backend = LetoBackend::<T>::default();
        macro_rules! run {
            ($width:literal, $workspace:expr) => {
                Gmres::<LetoBackend<T>, $width>::solve_with_observer(
                    &backend,
                    operator,
                    preconditioner,
                    right_hand_side,
                    solution,
                    $workspace,
                    policy,
                    observer,
                )
            };
        }

        match &mut self.inner {
            Ladder::W8(workspace) => run!(8, workspace),
            Ladder::W16(workspace) => run!(16, workspace),
            Ladder::W32(workspace) => run!(32, workspace),
            Ladder::W64(workspace) => run!(64, workspace),
            Ladder::W128(workspace) => run!(128, workspace),
            Ladder::W256(workspace) => run!(256, workspace),
        }
    }
}

impl<T> fmt::Debug for KrylovWorkspace<T>
where
    T: RealScalar + RealField,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("KrylovWorkspace")
            .field("dimension", &self.dimension)
            .field("restart", &self.width())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::{KrylovWorkspace, RestartWidth};
    use crate::operator::BorrowedCsrOperator;
    use athena_core::{ConvergencePolicy, Identity};
    use eunomia::{FloatElement, NumericElement, RealField};
    use leto::{Array1, Storage};
    use leto_ops::CsrMatrix;

    #[test]
    fn the_ladder_covers_every_request() {
        // Each request must land on the smallest width that is at least as
        // large, so a widened restart never searches a smaller subspace than
        // the caller asked for.
        for (requested, expected) in [
            (1, RestartWidth::W8),
            (8, RestartWidth::W8),
            (9, RestartWidth::W16),
            (30, RestartWidth::W32),
            (100, RestartWidth::W128),
            (200, RestartWidth::W256),
        ] {
            assert_eq!(
                RestartWidth::covering(requested),
                expected,
                "request {requested}"
            );
        }
    }

    #[test]
    fn a_request_above_the_ceiling_saturates() {
        assert_eq!(RestartWidth::covering(10_000), RestartWidth::W256);
    }

    /// Solve a diagonal two-point system twice through one workspace.
    ///
    /// The fixture reproduces the diagonal-solve case `CFDrs` recorded before
    /// this bridge moved here (ADR 0062 Phase 1 oracle), instantiated at every
    /// scalar the backend ships. For a 2×2 SPD diagonal system the Krylov
    /// space spans ℝ², so one restart cycle carries the exact solution and
    /// the observed error is the recurrence's rounding: O(m) Givens
    /// rotations, each contributing `O(ε_T)` against the operand scale ‖b‖₂ =
    /// √40, bounded by `κ·ε_T·‖b‖₂` with κ = 100. The tolerance is that bound
    /// floored at the recorded 1e-12, so f64 converges on the recorded
    /// fixture tolerance and f32 converges on its own rounding floor.
    macro_rules! diagonal_case {
        ($t:ty) => {{
            let matrix = CsrMatrix::from_parts(
                vec![
                    <$t as FloatElement>::from_f64(2.0),
                    <$t as FloatElement>::from_f64(3.0),
                ],
                vec![0, 1],
                vec![0, 1, 2],
                2,
                2,
            )
            .expect("invariant: diagonal CSR structure is valid");
            let right_hand_side = Array1::from_shape_vec(
                [2],
                vec![
                    <$t as FloatElement>::from_f64(2.0),
                    <$t as FloatElement>::from_f64(6.0),
                ],
            )
            .expect("invariant: RHS shape matches diagonal system");
            let rhs_norm = <$t as FloatElement>::from_f64((2.0f64 * 2.0 + 6.0f64 * 6.0).sqrt());
            let rounding =
                <$t as FloatElement>::from_f64(100.0) * <$t as RealField>::EPSILON * rhs_norm;
            let recorded_floor = <$t as FloatElement>::from_f64(1e-12);
            let tolerance = if rounding > recorded_floor {
                rounding
            } else {
                recorded_floor
            };
            let policy = ConvergencePolicy::new(tolerance, <$t as NumericElement>::ZERO, 20)
                .expect("invariant: derived tolerance and budget are valid");
            let mut workspace =
                KrylovWorkspace::new(30, 2).expect("invariant: small workspace allocates");

            for _ in 0..2 {
                let mut solution = Array1::from_elem([2], <$t as NumericElement>::ZERO);
                let report = workspace
                    .solve(
                        &BorrowedCsrOperator::new(&matrix)
                            .expect("invariant: diagonal system is square"),
                        &Identity,
                        &right_hand_side,
                        &mut solution,
                        policy,
                    )
                    .expect("invariant: diagonal system is solvable");
                assert!(
                    report.final_residual_norm <= report.threshold,
                    "residual {:?} exceeds threshold {:?}",
                    report.final_residual_norm,
                    report.threshold
                );
                let values = solution.storage().as_slice();
                let bound = rounding;
                assert!(
                    (<$t as FloatElement>::from_f64(1.0) - values[0]).abs() <= bound,
                    "x[0] {:?} deviates from 1 beyond the rounding bound",
                    values[0]
                );
                assert!(
                    (<$t as FloatElement>::from_f64(2.0) - values[1]).abs() <= bound,
                    "x[1] {:?} deviates from 2 beyond the rounding bound",
                    values[1]
                );
            }
        }};
    }

    #[test]
    fn reused_workspace_preserves_diagonal_solve_values() {
        diagonal_case!(f32);
        diagonal_case!(f64);
    }
}
