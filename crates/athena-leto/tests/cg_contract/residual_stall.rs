//! CG stops when `b - A x` stalls at its attainable accuracy.
//!
//! `A = tridiag(-1, 2, -1)` of order 62, load `b_i = 2 / 63²`: the solution
//! is a parabola, `A x` cancels, `‖|A| |x|‖ / ‖b‖ ≈ 1.46e3`, and a relative
//! tolerance of `4 EPSILON` lies below the attainable `‖b - A x‖`.

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, IterationObserver, IterationState, SolveReport,
    Termination,
};
use athena_leto::{CsrOperator, LetoBackend};
use eunomia::{FloatElement, NumericElement, RealField};
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar};

const UNKNOWNS: usize = 62;
/// Nonzeros per row: `N_A` of the matrix-vector product rounding model.
const ROW_NONZEROS: usize = 3;
/// Above the iteration bound and the iterations `<r, r>` needs to
/// underflow, so only the termination rule ends the solve.
const BUDGET: usize = 1 << 16;

struct Trace<T>(Vec<IterationState<T>>);

impl<T> IterationObserver<T> for Trace<T> {
    fn observe(&mut self, state: IterationState<T>) {
        self.0.push(state);
    }
}

fn load<T: RealField>() -> T {
    let cells = T::from_count(UNKNOWNS + 1);
    (<T as NumericElement>::ONE + <T as NumericElement>::ONE) / (cells * cells)
}

fn norm<T: RealField>(values: &[T]) -> T {
    let zero = <T as NumericElement>::ZERO;
    values.iter().fold(zero, |sum, &v| sum + v * v).sqrt()
}

fn solve<T: RealScalar + RealField>(
    budget: usize,
    interval: usize,
) -> (SolveReport<T>, Vec<T>, Vec<IterationState<T>>) {
    let one = <T as NumericElement>::ONE;
    let (mut values, mut columns, mut offsets) = (Vec::new(), Vec::new(), vec![0]);
    for row in 0..UNKNOWNS {
        for column in row.saturating_sub(1)..(row + 2).min(UNKNOWNS) {
            values.push(if column == row { one + one } else { -one });
            columns.push(column);
        }
        offsets.push(values.len());
    }
    let matrix = CsrMatrix::from_parts(values, columns, offsets, UNKNOWNS, UNKNOWNS)
        .expect("invariant: rows are sorted and in range");
    let backend = LetoBackend::<T>::default();
    let operator = CsrOperator::new(matrix).expect("invariant: matrix is square");
    let rhs = Array1::from_shape_vec([UNKNOWNS], vec![load::<T>(); UNKNOWNS])
        .expect("invariant: exact vector shape");
    let mut solution = Array1::zeros([UNKNOWNS]);
    let mut workspace = CgWorkspace::new(&backend, UNKNOWNS).expect("valid host workspace");
    let relative = T::from_count(4) * <T as RealField>::EPSILON;
    let policy = ConvergencePolicy::with_check_interval(
        <T as NumericElement>::ZERO,
        relative,
        budget,
        interval,
    )
    .expect("valid policy");
    let mut trace = Trace(Vec::new());
    let report = Cg::solve_with_observer(
        &backend,
        &operator,
        &Identity,
        &rhs,
        &mut solution,
        &mut workspace,
        policy,
        &mut trace,
    )
    .expect("matching dimensions and real CPU operations");
    let solution = solution.as_slice().expect("contiguous").to_vec();
    (report, solution, trace.0)
}

fn stall_terminates_at_the_first_measurement<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let _span = tracing::info_span!("cg_residual_stall").entered();
    let one = <T as NumericElement>::ONE;
    let (report, solution, trace) = solve::<T>(BUDGET, 1);
    let iterations = report.iterations;
    assert_eq!(report.termination, Termination::Stagnated, "{report:?}");
    // Initial residual, one product per iteration and exactly one measurement
    // of `b - A x`: the first measurement past the recurrence crossing
    // classifies the stall. Every earlier sample is an unconverged
    // recurrence residual.
    assert_eq!(report.operator_applications, iterations + 2, "{report:?}");
    assert_eq!(trace.len(), iterations);
    assert_eq!(
        trace[iterations - 1].residual_norm,
        report.final_residual_norm
    );
    let recurrence = &trace[..iterations - 1];
    assert!(
        recurrence
            .iter()
            .all(|s| s.residual_norm > report.threshold)
    );

    // Exact-arithmetic CG has `‖r_k‖ ≤ 2 √κ q^k ‖r_0‖`, `q = (√κ - 1) /
    // (√κ + 1)` (the A-norm bound of docs/book/cg.md with
    // `λ_min ‖e‖_A² ≤ ‖r‖² ≤ λ_max ‖e‖_A²`), and `κ = cot²(π / 126)` here,
    // so the recurrence crosses `tol` within `ln(2 √κ / tol) / ln(1 / q)`
    // iterations. Finite precision can delay CG; the operator count above,
    // not this bound, separates the stall rule from the previous behaviour.
    let sqrt_kappa = T::from_f64(1.0 / (core::f64::consts::PI / 126.0).tan());
    let ratio = (sqrt_kappa + one) / (sqrt_kappa - one);
    let tolerance = T::from_count(4) * <T as RealField>::EPSILON;
    let steps = ((one + one) * sqrt_kappa / tolerance).ln() / ratio.ln();
    assert!(
        T::from_count(iterations) <= steps,
        "{iterations} > {steps:?}"
    );

    // The gap `(b - A x_k) - r_k` starts at zero (`x_0 = 0`, `r_0 = b`) and
    // changes only by local rounding errors of the updates (Cools et al.,
    // arXiv:1601.07068, (2.11) and (2.15)). To first order in
    // `u = EPSILON / 2`, with `|δ(A p)| ≤ N_A u |A| |p|` (Carson and Demmel,
    // UCB/EECS-2012-197, §3), one step adds at most
    // `u (‖|A|‖ ((N_A + 1) ‖α p‖ + ‖x_{j+1}‖) + ‖α A p‖ + ‖r_{j+1}‖)`.
    // With `‖α p‖ ≤ 2X`, `‖α A p‖ ≤ 2R`, `X = max ‖x_j‖`, `R = max ‖r_j‖`,
    // `‖|A|‖₂ ≤ 4` (row sums) and the evaluation error of `b - A x_k`:
    //   ‖b - A x_k‖ ≤ tau + k u ‖|A|‖ (2 N_A + 3) X + 3 k u R
    //                 + (N_A + 1) u (‖b‖ + ‖|A|‖ X).
    // X is measured by re-running the deterministic solve with each budget
    // `j < k`; R is the largest observed recurrence residual or `‖b‖`.
    let reported = report.final_residual_norm;
    assert!(reported > report.threshold);
    let iterate = (1..iterations)
        .map(|budget| norm(&solve::<T>(budget, 1).1))
        .fold(norm(&solution), FloatElement::max);
    let rhs = report.initial_residual_norm;
    let residual = recurrence.iter().fold(rhs, |m, s| m.max(s.residual_norm));
    let unit = <T as RealField>::EPSILON / (one + one);
    let absolute_norm = T::from_count(4);
    let nonzeros = T::from_count(ROW_NONZEROS);
    let steps_taken = T::from_count(iterations);
    let gap = steps_taken
        * unit
        * (absolute_norm * (nonzeros + nonzeros + T::from_count(3)) * iterate
            + T::from_count(3) * residual);
    let evaluation = (nonzeros + one) * unit * (rhs + absolute_norm * iterate);
    let floor = report.threshold + gap + evaluation;
    assert!(reported <= floor, "reported={reported:?}, floor={floor:?}");

    // The reported norm is `‖b - A x‖` of the returned iterate: an
    // independent evaluation agrees within both evaluation errors.
    let zero = <T as NumericElement>::ZERO;
    let padded: Vec<T> = [zero].into_iter().chain(solution).chain([zero]).collect();
    let direct: Vec<T> = padded
        .windows(3)
        .map(|w| load::<T>() - ((one + one) * w[1] - w[0] - w[2]))
        .collect();
    let independent = norm(&direct);
    assert!((reported - independent).abs() <= evaluation + evaluation);
}

fn zero_product_after_a_converged_recurrence_is_a_stall<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let _span = tracing::info_span!("cg_residual_stall_zero_product").entered();
    // Without a scheduled check, CG runs past the recurrence crossing until
    // `<r, r>` underflows to zero, the path that reported `Breakdown`.
    let checked = solve::<T>(BUDGET, 1).0;
    let (report, _, trace) = solve::<T>(BUDGET, BUDGET);
    assert_eq!(report.termination, Termination::Stagnated, "{report:?}");
    assert!(report.iterations > checked.iterations, "{report:?}");
    assert!(report.iterations < BUDGET, "{report:?}");
    let overhead = report.operator_applications - report.iterations;
    assert!((2..=3).contains(&overhead), "{report:?}");
    assert_eq!(trace.len(), 1);
    assert_eq!(trace[0].residual_norm, report.final_residual_norm);
    assert!(report.final_residual_norm > report.threshold);
}

fn a_budget_ending_at_the_stall_check_is_max_iterations<T>(budget: usize)
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    // One check, at iteration `budget`: with a larger budget it classifies
    // the stall, so only the budget differs.
    let (stalled, ..) = solve::<T>(BUDGET, budget);
    assert_eq!(stalled.termination, Termination::Stagnated, "{stalled:?}");
    let (report, ..) = solve::<T>(budget, budget);
    assert_eq!(report.termination, Termination::MaxIterations, "{report:?}");
    assert_eq!(report.operator_applications, budget + 2);
    assert_eq!(report.final_residual_norm, stalled.final_residual_norm);
}

#[test]
fn cg_reports_a_residual_stall_instead_of_breakdown() {
    let _guard =
        tracing::subscriber::set_default(tracing_subscriber::fmt().with_test_writer().finish());
    stall_terminates_at_the_first_measurement::<f32>();
    stall_terminates_at_the_first_measurement::<f64>();
    zero_product_after_a_converged_recurrence_is_a_stall::<f32>();
    zero_product_after_a_converged_recurrence_is_a_stall::<f64>();
    a_budget_ending_at_the_stall_check_is_max_iterations::<f32>(100);
    a_budget_ending_at_the_stall_check_is_max_iterations::<f64>(300);
}
