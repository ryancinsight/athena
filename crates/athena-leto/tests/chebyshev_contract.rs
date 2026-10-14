//! Value-semantic CPU solver conformance for Chebyshev semi-iteration.
//!
//! The oracle is ADR 0062's Phase-2 table under ADR 0056's new-construction
//! rule: the smoothing property on a high-frequency residual against the
//! analytic damping bound `1/T_k(μ)`, at `f32` and `f64`.

use athena_core::{
    Chebyshev, ChebyshevWorkspace, ConvergencePolicy, SpectralBounds, SpectralBoundsError,
    Termination,
};
use athena_leto::{CsrOperator, LetoBackend};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

/// SPD diagonal operator `diag(2, 3, 5)` with the exact spectral interval
/// `[2, 5]`.
fn spd_matrix<T>() -> CsrMatrix<T>
where
    T: RealScalar + FloatElement,
{
    CsrMatrix::from_parts(
        vec![T::from_f64(2.0), T::from_f64(3.0), T::from_f64(5.0)],
        vec![0, 1, 2],
        vec![0, 1, 2, 3],
        3,
        3,
    )
    .expect("invariant: manufactured CSR parts are valid")
}

/// `T_k(μ)` by the three-term recurrence, in the same scalar type the solver
/// runs — the analytic damping bound the oracle asserts against.
fn chebyshev_damping<T>(mu: T, k: usize) -> T
where
    T: RealField,
{
    let two = <T as NumericElement>::ONE + <T as NumericElement>::ONE;
    let mut previous = <T as NumericElement>::ONE;
    let mut current = mu;
    for _ in 1..k {
        let next = two * mu * current - previous;
        previous = current;
        current = next;
    }
    <T as NumericElement>::ONE / current
}

fn solves_the_manufactured_system<T>()
where
    T: RealScalar + RealField + FloatElement + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let operator = CsrOperator::new(spd_matrix()).expect("invariant: matrix is square");
    let right_hand_side = Array1::from_shape_vec(
        [3],
        vec![T::from_f64(2.0), T::from_f64(6.0), T::from_f64(15.0)],
    )
    .expect("invariant: vector shape is exact");
    let mut solution = Array1::zeros([3]);
    let mut workspace =
        ChebyshevWorkspace::new(&backend, 3).expect("invariant: host allocation succeeds");
    let bounds = SpectralBounds::new(T::from_f64(2.0), T::from_f64(5.0))
        .expect("invariant: exact spectral interval is valid");
    // The budget follows the damping bound: reaching an epsilon-scaled
    // threshold needs T_k(mu) above 1/eps, i.e. k*acosh(mu) ~ 33 for
    // mu = 7/3, so k ~ 23; the budget carries that with margin.
    let policy = ConvergencePolicy::new(<T as RealField>::EPSILON, <T as RealField>::EPSILON, 28)
        .expect("invariant: tolerance is finite and positive");

    let report = Chebyshev::<LetoBackend<T>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
        bounds,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert_eq!(report.termination, Termination::Converged);
    // Each step is one length-three recurrence plus a residual
    // recomputation: under 40 rounded operations per entry over the whole
    // run, with the damping coefficients bounded by two. 512 machine
    // epsilons covers the first-order count with margin.
    let bound = T::from_f64(512.0) * <T as RealField>::EPSILON;
    let values = solution
        .as_slice()
        .expect("invariant: Array1 is contiguous");
    for (component, expected) in values.iter().zip([1.0, 2.0, 3.0]) {
        let delta = *component - T::from_f64(expected);
        assert!(
            delta.abs() <= bound,
            "manufactured component {component:?} vs {expected}"
        );
    }
}

#[test]
fn chebyshev_solves_spd_system_for_every_cpu_scalar() {
    solves_the_manufactured_system::<f32>();
    solves_the_manufactured_system::<f64>();
}

#[test]
fn chebyshev_damps_the_high_frequency_residual_to_the_analytic_bound() {
    // The smoothing oracle: a residual concentrated on the interval's upper
    // end (the high-frequency component) is damped by exactly the analytic
    // Chebyshev factor — `p_k(λ_max) = T_k(−1)/T_k(μ) = ±1/T_k(μ)` — so the
    // run at a fixed step count pins the bound rather than merely meeting it.
    let backend = LetoBackend::<f64>::default();
    let matrix = CsrMatrix::from_parts(vec![1.0, 8.0], vec![0, 1], vec![0, 1, 2], 2, 2)
        .expect("invariant: manufactured CSR parts are valid");
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let right_hand_side =
        Array1::from_shape_vec([2], vec![0.0, 4.0]).expect("invariant: vector shape is exact");
    let steps = 6;
    let mut solution = Array1::zeros([2]);
    let mut workspace =
        ChebyshevWorkspace::new(&backend, 2).expect("invariant: host allocation succeeds");
    let bounds =
        SpectralBounds::new(1.0, 8.0).expect("invariant: exact spectral interval is valid");
    let policy = ConvergencePolicy::new(0.0, 0.0, steps).expect("invariant: exact policy is valid");

    let report = Chebyshev::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
        bounds,
    )
    .expect("invariant: manufactured solve has valid dimensions");

    assert_eq!(report.termination, Termination::MaxIterations);
    assert_eq!(report.iterations, steps);
    let mu = 9.0 / 7.0;
    let ratio = report.final_residual_norm / report.initial_residual_norm;
    let damping = chebyshev_damping(mu, steps);
    // The residual ratio and the bound are the same quantity computed by
    // two rounding paths (the vector recurrence and the scalar recurrence):
    // their agreement carries both paths' accumulated first-order error,
    // under 64 machine epsilons over six steps, and the ratio can never
    // exceed the bound by more than that same agreement window.
    let agreement = 64.0 * f64::EPSILON;
    assert!(
        (ratio - damping).abs() <= agreement,
        "damping ratio {ratio:?} vs analytic bound {damping:?}"
    );
    assert!(
        ratio <= damping + agreement,
        "the smoothing property holds to the analytic bound: {ratio:?} vs {damping:?}"
    );
}

#[test]
fn spectral_bounds_reject_invalid_intervals() {
    assert_eq!(
        SpectralBounds::<f64>::new(0.0, 1.0),
        Err(SpectralBoundsError::NotPositiveDefinite { value: 0.0 })
    );
    assert_eq!(
        SpectralBounds::<f64>::new(2.0, 1.0),
        Err(SpectralBoundsError::Inverted {
            minimum: 2.0,
            maximum: 1.0
        })
    );
    assert!(matches!(
        SpectralBounds::<f64>::new(f64::NAN, 1.0),
        Err(SpectralBoundsError::NotFinite { .. })
    ));
    let bounds =
        SpectralBounds::<f64>::new(2.0, 5.0).expect("invariant: valid interval is accepted");
    // The accessors return the validated values bit-exactly; the tuple
    // comparison is the roundtrip assertion itself.
    assert_eq!((bounds.minimum(), bounds.maximum()), (2.0, 5.0));
}

/// A singleton spectrum means A=lambda I: one Richardson update with
/// omega=1/lambda annihilates the error. No infinite coefficient is needed.
fn solves_singleton_spectrum<T>()
where
    T: RealScalar + RealField + FloatElement + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    for dimension in [1, 3] {
        let matrix = CsrMatrix::from_parts(
            vec![T::ONE; dimension],
            (0..dimension).collect(),
            (0..=dimension).collect(),
            dimension,
            dimension,
        )
        .expect("invariant: identity");
        let operator = CsrOperator::new(matrix).expect("invariant: square");
        let rhs = Array1::from_shape_vec([dimension], vec![T::ONE; dimension])
            .expect("invariant: exact shape");
        let mut solution = Array1::zeros([dimension]);
        let mut workspace =
            ChebyshevWorkspace::new(&backend, dimension).expect("invariant: allocation");
        let bounds =
            SpectralBounds::new(T::ONE, T::ONE).expect("invariant: singleton interval is accepted");
        let policy =
            ConvergencePolicy::new(T::EPSILON, T::ZERO, 1).expect("invariant: one exact step");
        let report = Chebyshev::<LetoBackend<T>>::solve_into(
            &backend,
            &operator,
            &rhs,
            &mut solution,
            &mut workspace,
            policy,
            bounds,
        )
        .expect("invariant: valid dimensions");
        assert_eq!(
            solution.as_slice().expect("invariant: contiguous"),
            vec![T::ONE; dimension]
        );
        assert_eq!(report.termination, Termination::Converged, "{report:?}");
        assert_eq!(report.iterations, 1);
        assert_eq!(report.operator_applications, 2);
        assert_eq!(report.final_residual_norm, T::ZERO);
        assert!(report.initial_residual_norm >= T::ONE);
    }
}

#[test]
fn chebyshev_solves_equal_bounds_for_every_cpu_scalar() {
    solves_singleton_spectrum::<f32>();
    solves_singleton_spectrum::<f64>();
}

fn checks_narrow_intervals_and_warm_starts<T>()
where
    T: RealScalar + RealField + FloatElement + core::fmt::Debug,
{
    let backend = LetoBackend::<T>::default();
    let matrix = CsrMatrix::from_parts(vec![T::ONE], vec![0], vec![0, 1], 1, 1)
        .expect("invariant: identity");
    let operator = CsrOperator::new(matrix).expect("invariant: square");
    let rhs = Array1::from_shape_vec([1], vec![T::ONE]).expect("invariant: shape");
    let mut workspace = ChebyshevWorkspace::new(&backend, 1).expect("invariant: allocation");
    let policy = ConvergencePolicy::new(T::ZERO, T::EPSILON, 3).expect("invariant: valid");
    for maximum in [T::ONE, T::ONE + T::EPSILON] {
        for initial in [T::ZERO, T::from_f64(0.5), T::ONE] {
            let mut solution =
                Array1::from_shape_vec([1], vec![initial]).expect("invariant: shape");
            let bounds = SpectralBounds::new(T::ONE, maximum).expect("invariant: ordered");
            let report = Chebyshev::<LetoBackend<T>>::solve_into(
                &backend,
                &operator,
                &rhs,
                &mut solution,
                &mut workspace,
                policy,
                bounds,
            )
            .expect("invariant: dimensions");
            let actual = (T::ONE - solution.as_slice().expect("invariant: contiguous")[0]).abs();
            assert!(report.converged(), "{report:?}");
            assert_eq!(
                report.threshold,
                T::EPSILON,
                "policy is relative to b, not r0"
            );
            assert_eq!(report.initial_residual_norm, T::ONE - initial);
            assert_eq!(report.final_residual_norm, actual);
            assert!(actual <= report.threshold);
            assert_eq!(report.iterations, usize::from(initial != T::ONE));
        }
    }
}

#[test]
fn chebyshev_handles_narrow_bounds_and_workspace_reuse_for_every_cpu_scalar() {
    checks_narrow_intervals_and_warm_starts::<f32>();
    checks_narrow_intervals_and_warm_starts::<f64>();
}
