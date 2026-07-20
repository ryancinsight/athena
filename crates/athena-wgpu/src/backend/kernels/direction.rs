use std::borrow::Cow;

use hephaestus_core::{BindingDecl, KernelInterface, KernelSource, Wgsl};

use super::VectorParams;

pub(crate) struct DirectionKernel;

impl KernelInterface for DirectionKernel {
    type Params = VectorParams;

    const LABEL: &'static str = "athena-cg-direction";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [256, 1, 1];
}

impl KernelSource<Wgsl> for DirectionKernel {
    const ENTRY: &'static str = "combine_direction";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r"
struct Params {
    beta: f32,
    len: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> direction: array<f32>;
@group(0) @binding(1) var<storage, read> preconditioned_residual: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn combine_direction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        direction[index] = preconditioned_residual[index] + params.beta * direction[index];
    }
}
",
        )
    }
}
