//! Solve a two-cell Poisson system on an accelerator device.
//!
//! The solver, operator and vectors below name no device API. Only the device
//! handle and its operation bundles do, which is what makes the backend
//! device-neutral: swapping `WgpuDevice` and its op bundles for another
//! Hephaestus device changes the two type aliases and nothing else.

use athena::{
    Cg, CgWorkspace, ConvergencePolicy, Identity, Termination,
    accelerator::{CsrOperator, HephaestusBackend},
};
use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{GpuCsrMatrix, WgpuDevice, WgpuSparseOps, WgpuVectorOps};

type Backend = HephaestusBackend<WgpuDevice, WgpuVectorOps, f32>;
type Operator = CsrOperator<WgpuSparseOps, GpuCsrMatrix<f32>>;

// dyn exception: top-level binary error aggregation is outside solver paths.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::try_default("athena-example")?;
    let operations = WgpuVectorOps::new(&device)?;
    let backend = Backend::new(device, operations);

    // [[2, -1], [-1, 2]] with b = [0, 3] gives x = [1, 2].
    let operator = Operator::from_parts(
        WgpuSparseOps,
        backend.device(),
        &[2.0_f32, -1.0, -1.0, 2.0],
        &[0, 1, 0, 1],
        &[0, 2, 4],
        2,
        2,
    )?;
    let right_hand_side = backend.device().upload(&[0.0_f32, 3.0])?;
    let mut solution = backend.device().alloc_zeroed(2)?;
    let mut workspace = CgWorkspace::new(&backend, 2)?;
    let policy = ConvergencePolicy::new(64.0 * f32::EPSILON, 64.0 * f32::EPSILON, 4)?;

    let report = Cg::<Backend>::solve_into(
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
