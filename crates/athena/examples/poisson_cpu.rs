//! Solve a two-cell Poisson system with Leto CPU execution.

use athena::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, Termination,
    cpu::{CsrOperator, LetoBackend},
};
use leto::Array1;
use leto_ops::CsrMatrix;

// dyn exception: top-level binary error aggregation is outside solver paths.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix = CsrMatrix::from_parts(
        vec![2.0_f64, -1.0, -1.0, 2.0],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )?;
    let operator = CsrOperator::new(matrix)?;
    let backend = LetoBackend::<f64>::default();
    let right_hand_side = Array1::from_shape_vec([2], vec![0.0, 3.0])?;
    let mut solution = Array1::zeros([2]);
    let mut workspace = CgWorkspace::new(&backend, 2)?;
    let policy = ConvergencePolicy::new(64.0 * f64::EPSILON, 64.0 * f64::EPSILON, 4)?;

    let report = Cg::<LetoBackend<f64>>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )?;

    assert_eq!(report.termination, Termination::Converged);
    println!("solution = {:?}", solution.as_slice());
    Ok(())
}
