//! Solve a two-cell Poisson system with Hephaestus WGPU execution.

use athena::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, Termination,
    wgpu::{WgpuBackend, WgpuCsrOperator},
};
use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::WgpuDevice;
use leto_ops::CsrMatrix;

// dyn exception: top-level binary error aggregation is outside solver paths.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::try_default("athena-example")?;
    let backend = WgpuBackend::new(device)?;
    let matrix = CsrMatrix::from_parts(
        vec![2.0_f32, -1.0, -1.0, 2.0],
        vec![0, 1, 0, 1],
        vec![0, 2, 4],
        2,
        2,
    )?;
    let operator = WgpuCsrOperator::from_cpu(&backend, &matrix)?;
    let right_hand_side = backend.device().upload(&[0.0_f32, 3.0])?;
    let mut solution = backend.device().alloc_zeroed(2)?;
    let mut workspace = CgWorkspace::new(&backend, 2)?;
    let policy = ConvergencePolicy::new(64.0 * f32::EPSILON, 64.0 * f32::EPSILON, 4)?;

    let report = Cg::<WgpuBackend>::solve_into(
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

    let mut host_solution = [0.0_f32; 2];
    backend.device().download(&solution, &mut host_solution)?;
    println!("solution = {host_solution:?}");
    Ok(())
}
