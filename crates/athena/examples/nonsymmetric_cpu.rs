//! Solve a nonsymmetric system with restarted GMRES over Leto.

mod support;

use athena::{
    ConvergencePolicy, Gmres, GmresWorkspace, Identity, Termination,
    cpu::{CsrOperator, LetoBackend},
};
use leto::Array1;

// dyn exception: top-level binary error aggregation is outside solver paths.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = LetoBackend::<f64>::default();
    let operator = CsrOperator::new(support::nonsymmetric_matrix()?)?;
    let right_hand_side = Array1::from_shape_vec([3], vec![2.0, -1.0, 4.0])?;
    let mut solution = Array1::zeros([3]);
    let mut workspace = GmresWorkspace::<_, 2>::new(&backend, 3)?;
    let policy = ConvergencePolicy::new(4096.0 * f64::EPSILON, 4096.0 * f64::EPSILON, 30)?;

    let report = Gmres::<LetoBackend<f64>, 2>::solve_into(
        &backend,
        &operator,
        &Identity,
        &right_hand_side,
        &mut solution,
        &mut workspace,
        policy,
    )?;
    if report.termination != Termination::Converged {
        return Err(format!("solver terminated with {:?}", report.termination).into());
    }

    println!("solution = {:?}", solution.as_slice());
    Ok(())
}
