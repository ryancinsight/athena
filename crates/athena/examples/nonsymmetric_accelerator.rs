//! Solve a nonsymmetric system with restarted GMRES on an accelerator device.
//!
//! As in the Poisson example, no solver or operator code names a device API.
//! The CSR parts are passed directly because the device-neutral operator
//! uploads from slices rather than from a host matrix.

use athena::{
    ConvergencePolicy, Gmres, GmresWorkspace, Identity, Termination,
    accelerator::{CsrOperator, HephaestusBackend},
};
use hephaestus_core::ComputeDevice;
use hephaestus_wgpu::{GpuCsrMatrix, WgpuDevice, WgpuSparseOps, WgpuVectorOps};

type Backend = HephaestusBackend<WgpuDevice, WgpuVectorOps, f32>;
type Operator = CsrOperator<WgpuSparseOps, GpuCsrMatrix<f32>>;

// dyn exception: top-level binary error aggregation is outside solver paths.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = WgpuDevice::try_default("athena-gmres-example")?;
    let operations = WgpuVectorOps::new(&device)?;
    let backend = Backend::new(device, operations);

    // Nonsymmetric [[4, 1, 0], [2, 3, 1], [0, 1, 2]].
    let operator = Operator::from_parts(
        WgpuSparseOps,
        backend.device(),
        &[4.0_f32, 1.0, 2.0, 3.0, 1.0, 1.0, 2.0],
        &[0, 1, 0, 1, 2, 1, 2],
        &[0, 2, 5, 7],
        3,
        3,
    )?;
    let right_hand_side = backend.device().upload(&[2.0_f32, -1.0, 4.0])?;
    let mut solution = backend.device().alloc_zeroed(3)?;
    let mut workspace = GmresWorkspace::<_, 3>::new(&backend, 3)?;
    let bound = 8192.0 * f32::EPSILON;
    let policy = ConvergencePolicy::new(bound, bound, 6)?;

    let report = Gmres::<Backend, 3>::solve_into(
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
