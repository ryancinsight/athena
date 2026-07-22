//! Prepared reduction identity and value semantics on a real WGPU device.

mod support;

use athena_core::KrylovBackend;
use athena_wgpu::WgpuBackend;
use hephaestus_core::{ComputeDevice, HephaestusError};

use support::device;

#[test]
#[expect(
    clippy::float_cmp,
    reason = "all integer products and partial sums are exactly representable in f32"
)]
fn prepared_reductions_reuse_fixed_allocations() {
    let Some(device) = device("athena-prepared-reduction-contract") else {
        return;
    };
    let backend = WgpuBackend::new(device).expect("WGPU kernels must prepare");
    let left = backend
        .device()
        .upload(&[1.0_f32, 2.0, 3.0])
        .expect("left upload must succeed");
    let right = backend
        .device()
        .upload(&[4.0_f32, 5.0, 6.0])
        .expect("right upload must succeed");
    let replacement = backend
        .device()
        .upload(&[7.0_f32, 8.0, 9.0])
        .expect("replacement upload must succeed");

    let dot = backend
        .prepare_dot(&left, &right)
        .expect("dot preparation must succeed");
    let norm = backend
        .prepare_norm_l2(&left)
        .expect("norm preparation must succeed");

    assert_eq!(
        backend
            .dot_prepared(&dot, &left, &right)
            .expect("prepared dot must execute"),
        32.0
    );
    let norm_value = backend
        .norm_l2_prepared(&norm, &left)
        .expect("prepared norm must execute");
    let norm_bound = 8.0 * f32::EPSILON;
    assert!((norm_value - 14.0_f32.sqrt()).abs() <= norm_bound);

    let mismatch = backend
        .dot_prepared(&dot, &replacement, &right)
        .expect_err("a prepared operation must reject a different allocation");
    assert!(matches!(
        mismatch,
        HephaestusError::DispatchFailed { message }
            if message == "prepared dot left operand received a different device allocation"
    ));
}
