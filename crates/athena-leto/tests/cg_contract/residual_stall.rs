//! CG stops when `b - A x` stalls at its attainable accuracy.
//!
//! `A = tridiag(-1, 2, -1)` of order 62, load `b_i = 2 / 63²`: the solution
//! is a parabola, `A x` cancels, `‖|A| |x|‖ / ‖b‖ ≈ 1.46e3`, and a relative
//! tolerance of `4 EPSILON` lies below the attainable `‖b - A x‖`.

use core::cell::Cell;

use athena_core::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, IterationObserver, IterationState, KrylovBackend,
    LinearOperator, Preconditioner, SolveReport, Termination,
};
use athena_leto::{CsrOperator, LetoBackend, LetoBackendError};
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

fn operator<T: RealScalar + RealField>() -> CsrOperator<T> {
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
    CsrOperator::new(matrix).expect("invariant: matrix is square")
}

fn solve<T: RealScalar + RealField>(
    budget: usize,
    interval: usize,
) -> (SolveReport<T>, Vec<T>, Vec<IterationState<T>>) {
    solve_with(&operator::<T>(), &Identity, budget, interval)
}

fn solve_with<T, O, P>(
    operator: &O,
    preconditioner: &P,
    budget: usize,
    interval: usize,
) -> (SolveReport<T>, Vec<T>, Vec<IterationState<T>>)
where
    T: RealScalar + RealField,
    O: LinearOperator<LetoBackend<T>>,
    P: Preconditioner<LetoBackend<T>>,
{
    let backend = LetoBackend::<T>::default();
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
        operator,
        preconditioner,
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
    // Without a scheduled check, CG runs past the recurrence crossing to its
    // first zero `<r, z>` or `<p, A p>`; the recurrence does not fix which.
    let checked = solve::<T>(BUDGET, 1).0;
    let matrix = operator::<T>();
    let probe = ProbedOperator::new(&matrix, None);
    let (report, _, trace) = solve_with(&probe, &Identity, BUDGET, BUDGET);
    assert_eq!(report.termination, Termination::Stagnated, "{report:?}");
    assert!(report.iterations > checked.iterations, "{report:?}");
    assert!(report.iterations < BUDGET, "{report:?}");
    // Exactly one measurement of `b - A x`, whichever product reached zero.
    assert_eq!(probe.measurements(), 1, "{report:?}");
    assert_eq!(report.operator_applications, probe.calls.get());
    // One product per completed iteration, plus the abandoned product when
    // the zero is `<p, A p>` rather than `<r, z>`. The recurrence does not
    // fix which dot product reaches zero first, so the two cases are pinned
    // separately in `a_zero_product_ends_the_solve_at_the_stall_exactly`.
    assert!(
        matches!(probe.products() - report.iterations, 0 | 1),
        "{report:?}"
    );
    assert_eq!(trace.len(), 1);
    assert_eq!(trace[0].residual_norm, report.final_residual_norm);
    assert!(report.final_residual_norm > report.threshold);
}

/// Counts applications by input buffer and optionally zeroes the product of
/// the `zeroed`-th application, counted from one.
///
/// The first application evaluates `b - A x_0` on the solution buffer; every
/// later application with that input buffer is a measurement of `b - A x`.
struct ProbedOperator<'a, T> {
    inner: &'a CsrOperator<T>,
    zeroed: Option<usize>,
    calls: Cell<usize>,
    solution: Cell<Option<usize>>,
    on_solution: Cell<usize>,
}

impl<'a, T> ProbedOperator<'a, T> {
    fn new(inner: &'a CsrOperator<T>, zeroed: Option<usize>) -> Self {
        Self {
            inner,
            zeroed,
            calls: Cell::new(0),
            solution: Cell::new(None),
            on_solution: Cell::new(0),
        }
    }

    /// Applications to the solution buffer after the initial residual.
    fn measurements(&self) -> usize {
        self.on_solution.get() - 1
    }

    /// Applications to any other buffer: the products `A p`.
    fn products(&self) -> usize {
        self.calls.get() - self.on_solution.get()
    }
}

impl<T: RealScalar + RealField> LinearOperator<LetoBackend<T>> for ProbedOperator<'_, T> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn apply(
        &self,
        backend: &LetoBackend<T>,
        input: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let address = input
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?
            .as_ptr()
            .addr();
        self.calls.set(self.calls.get() + 1);
        if self.solution.get().is_none() {
            self.solution.set(Some(address));
        }
        if self.solution.get() == Some(address) {
            self.on_solution.set(self.on_solution.get() + 1);
        }
        if self.zeroed != Some(self.calls.get()) {
            return self.inner.apply(backend, input, output);
        }
        output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?
            .fill(<T as NumericElement>::ZERO);
        Ok(())
    }
}

/// Identity that zeroes the output of the `ordinal`-th application.
struct ZeroedPreconditioner {
    ordinal: usize,
    calls: Cell<usize>,
}

impl<T: RealScalar + RealField> Preconditioner<LetoBackend<T>> for ZeroedPreconditioner {
    fn apply(
        &self,
        backend: &LetoBackend<T>,
        residual: <LetoBackend<T> as KrylovBackend>::View<'_>,
        mut output: <LetoBackend<T> as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() != self.ordinal {
            return Identity.apply(backend, residual, output);
        }
        output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?
            .fill(<T as NumericElement>::ZERO);
        Ok(())
    }
}

fn a_zero_product_ends_the_solve_at_the_stall_exactly<T>()
where
    T: RealScalar + RealField + core::fmt::Debug,
{
    let _span = tracing::info_span!("cg_residual_stall_forced_zero").entered();
    // The checked run classifies the stall at iteration `s`; both forced runs
    // reproduce its iterates up to `s` and zero the next product.
    let (checked, ..) = solve::<T>(BUDGET, 1);
    let stall = checked.iterations;
    assert_eq!(checked.termination, Termination::Stagnated, "{checked:?}");

    // `<p, A p> = 0` in iteration `s + 1`: the initial residual is
    // application 1 and the product of iteration `j` is application `j + 1`.
    // The abandoned product is counted, the iteration is not.
    let matrix = operator::<T>();
    let product = ProbedOperator::new(&matrix, Some(stall + 2));
    let (report, _, trace) = solve_with(&product, &Identity, BUDGET, BUDGET);
    assert_eq!(report.termination, Termination::Stagnated, "{report:?}");
    assert_eq!(report.iterations, stall);
    assert_eq!(report.operator_applications, stall + 3);
    assert_eq!(product.measurements(), 1);
    assert_eq!(product.products(), stall + 1);
    assert_eq!(report.final_residual_norm, checked.final_residual_norm);
    assert_eq!(trace.len(), 1);

    // `<r, z> = 0` after iteration `s`: the preconditioner application after
    // iteration `j` is application `j + 1`. No product is abandoned.
    let preconditioner = ZeroedPreconditioner {
        ordinal: stall + 1,
        calls: Cell::new(0),
    };
    let probe = ProbedOperator::new(&matrix, None);
    let (report, _, trace) = solve_with(&probe, &preconditioner, BUDGET, BUDGET);
    assert_eq!(report.termination, Termination::Stagnated, "{report:?}");
    assert_eq!(report.iterations, stall);
    assert_eq!(report.operator_applications, stall + 2);
    assert_eq!(probe.measurements(), 1);
    assert_eq!(probe.products(), stall);
    assert_eq!(report.final_residual_norm, checked.final_residual_norm);
    assert_eq!(trace.len(), 1);
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
    a_zero_product_ends_the_solve_at_the_stall_exactly::<f32>();
    a_zero_product_ends_the_solve_at_the_stall_exactly::<f64>();
    a_budget_ending_at_the_stall_check_is_max_iterations::<f32>(100);
    a_budget_ending_at_the_stall_check_is_max_iterations::<f64>(300);
}
