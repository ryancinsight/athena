use std::borrow::Cow;

use hephaestus_core::{BindingDecl, KernelInterface, KernelSource, Wgsl};

use super::VectorParams;

pub(crate) struct ResidualKernel;

impl KernelInterface for ResidualKernel {
    type Params = VectorParams;

    const LABEL: &'static str = "athena-residual";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [256, 1, 1];
}

impl KernelSource<Wgsl> for ResidualKernel {
    const ENTRY: &'static str = "residual";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r"
struct Params {
    scalar: f32,
    len: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> rhs: array<f32>;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read_write> residual_out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        residual_out[index] = rhs[index] - image[index];
    }
}
",
        )
    }
}
