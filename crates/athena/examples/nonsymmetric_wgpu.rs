//! Solve a nonsymmetric system with restarted GMRES over Hephaestus WGPU.

mod support;

use athena::{
    ConvergencePolicy, Gmres, GmresWorkspace, Identity, Termination,
    wgpu::{WgpuBackend, WgpuCsrOperator},
};
use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::WgpuDevice;

// dyn exception: top-level binary error aggregation is outside solver paths.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::try_default("athena-gmres-example")?;
    let backend = WgpuBackend::new(device)?;
    let matrix = support::nonsymmetric_matrix::<f32>()?;
    let operator = WgpuCsrOperator::from_cpu(&backend, &matrix)?;
    let right_hand_side = backend.device().upload(&[2.0_f32, -1.0, 4.0])?;
    let mut solution = backend.device().alloc_zeroed(3)?;
    let mut workspace = GmresWorkspace::<_, 3>::new(&backend, 3)?;
    let bound = 8192.0 * f32::EPSILON;
    let policy = ConvergencePolicy::new(bound, bound, 6)?;

    let report = Gmres::<WgpuBackend, 3>::solve_into(
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

    let mut host_solution = [0.0_f32; 3];
    backend.device().download(&solution, &mut host_solution)?;
    println!("solution = {host_solution:?}");
    Ok(())
}
