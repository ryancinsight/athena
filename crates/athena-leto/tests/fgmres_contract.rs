//! Value-semantic conformance for the flexible-GMRES property: the solver's
//! stored preconditioned basis makes it FGMRES, so a *varying* preconditioner
//! stays inside its theory.
//!
//! The oracle is ADR 0062's Phase-2 table under ADR 0056's new-construction
//! rule: FGMRES with a varying preconditioner converges where restarted GMRES
//! with a fixed preconditioner stalls, on a constructed
//! varying-preconditioner case, at `f32` and `f64`.

use athena_core::{
    ConvergencePolicy, Gmres, GmresWorkspace, KrylovBackend, Preconditioner, Termination,
};
use athena_leto::{CsrOperator, LetoBackend, LetoBackendError};
use core::cell::Cell;
use core::marker::PhantomData;
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

/// Two-unknown diagonal operator `diag(2, 3)`: symmetric positive definite
/// with the closed-form solution `x* = (1/2, 1/3)` for `b = (1, 1)`.
fn diagonal_matrix<T>() -> CsrMatrix<T>
where
    T: RealScalar + FloatElement,
{
    CsrMatrix::from_parts(
        vec![T::from_f64(2.0), T::from_f64(3.0)],
        vec![0, 1],
        vec![0, 1, 2],
        2,
        2,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

/// Projection onto the first coordinate: `z = (r₀, 0)`. Fixed, it can never
/// move the solution's second component, so the residual's second entry is
/// unreachable — the stall half of the oracle.
struct FirstCoordinateProjection<T>(PhantomData<fn() -> T>);

impl<T> FirstCoordinateProjection<T> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Projection that alternates between the two coordinates per application:
/// `z = (r₀, 0)` on even applications, `z = (0, r₁)` on odd ones. Neither
/// half alone reaches the full solution, but the alternating sequence does:
/// the varying preconditioner of the flexible oracle.
struct AlternatingProjection<T> {
    phase: Cell<usize>,
    scalar: PhantomData<fn() -> T>,
}

impl<T> AlternatingProjection<T> {
    const fn new() -> Self {
        Self {
            phase: Cell::new(0),
            scalar: PhantomData,
        }
    }
}

impl<T> Preconditioner<LetoBackend<T>> for FirstCoordinateProjection<T>
where
    T: RealScalar + RealField,
{
    fn apply(
        &self,
        _backend: &LetoBackend<T>,
        residual: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let residual = residual
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        output[0] = residual[0];
        output[1] = <T as NumericElement>::ZERO;
        Ok(())
    }
}

impl<T> Preconditioner<LetoBackend<T>> for AlternatingProjection<T>
where
    T: RealScalar + RealField,
{
    fn apply(
        &self,
        _backend: &LetoBackend<T>,
        residual: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let residual = residual
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let first = self.phase.get().is_multiple_of(2);
        self.phase.set(self.phase.get() + 1);
        let zero = <T as NumericElement>::ZERO;
        if first {
            output[0] = residual[0];
            output[1] = zero;
        } else {
            output[0] = zero;
            output[1] = residual[1];
        }
        Ok(())
    }
}

fn assert_closed_form<T>(solution: &Array1<T>, bound: T)
where
    T: RealField + core::fmt::Debug,
{
    let values = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    assert_eq!(values.len(), 2, "manufactured system dimension");
    assert!(
        (values[0] - T::from_f64(0.5)).abs() <= bound,
        "first closed-form component: {:?}",
        values[0]
    );
    assert!(
        (values[1] - T::from_f64(1.0 / 3.0)).abs() <= bound,
        "second closed-form component: {:?}",
        values[1]
    );
}

fn varying_preconditioner_converges<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let operator = CsrOperator::new(diagonal_matrix()).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec([2], vec![T::from_f64(1.0), T::from_f64(1.0)])
        .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        GmresWorkspace::<_, 2>::new(&backend, 2).expect("invariant: host allocation succeeds");
    // The alternating sequence needs one cycle of two steps to place both
    // coordinate directions; the budget allows a full second cycle so the
    // Krylov completion, not the cap, terminates the solve.
    let policy = ConvergencePolicy::new(<T as RealField>::EPSILON, <T as RealField>::EPSILON, 4)
        .expect("invariant: tolerance is finite and positive");

    // The flexible half: the varying projection keeps the solver inside its
    // theory, and the solve reaches the closed form.
    let preconditioner = AlternatingProjection::<T>::new();
    let report = Gmres::<LetoBackend<T>, 2>::solve_into(
        &backend,
        &operator,
        &preconditioner,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert_eq!(report.termination, Termination::Converged);
    // Two flexible Arnoldi steps over length-two vectors perform fewer than
    // 64 rounded scalar operations with unit amplification (the operator's
    // diagonal entries are O(1)); 256 machine epsilons is a factor-of-four
    // margin over that first-order count.
    let bound = T::from_f64(256.0) * <T as RealField>::EPSILON;
    assert_closed_form(&solution, bound);

    // The stall half: the same solver with the fixed first-coordinate
    // projection can build only first-coordinate corrections. Its first cycle
    // resolves that component and then degenerates — the next preconditioned
    // vector is zero — so the run must report the rank loss value-semantically
    // with the second component unresolved, never the closed form.
    let mut stalled = Array1::zeros([2]);
    let mut stalled_workspace =
        GmresWorkspace::<_, 2>::new(&backend, 2).expect("invariant: host allocation succeeds");
    let report = Gmres::<LetoBackend<T>, 2>::solve_into(
        &backend,
        &operator,
        &FirstCoordinateProjection::<T>::new(),
        &right_hand_side,
        &mut stalled,
        &mut stalled_workspace,
        policy,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert_eq!(
        report.termination,
        Termination::Breakdown,
        "the fixed projection stalls on the unresolved component"
    );
    let values = stalled.as_slice().expect("invariant: Array1 is contiguous");
    assert!(
        (values[0] - T::from_f64(0.5)).abs() <= bound,
        "the stalled solve resolves its reachable component: {:?}",
        values[0]
    );
    assert!(
        values[1].abs() <= bound,
        "the unreachable component stays at zero: {:?}",
        values[1]
    );
    assert!(
        report.final_residual_norm >= T::from_f64(0.9),
        "the stalled residual keeps the unreachable magnitude: {:?}",
        report.final_residual_norm
    );
}

#[test]
fn fgmres_oracle_converges_varying_and_stalls_fixed() {
    varying_preconditioner_converges::<f32>();
    varying_preconditioner_converges::<f64>();
}
