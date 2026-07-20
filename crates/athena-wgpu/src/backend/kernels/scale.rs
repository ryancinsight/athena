use std::borrow::Cow;

use hephaestus_core::{BindingDecl, KernelInterface, KernelSource, Wgsl};

use super::VectorParams;

pub(crate) struct ScaleKernel;

impl KernelInterface for ScaleKernel {
    type Params = VectorParams;

    const LABEL: &'static str = "athena-scale";
    const BINDINGS: &'static [BindingDecl] = &[BindingDecl::read_write::<f32>()];
    const WORKGROUP: [u32; 3] = [256, 1, 1];
}

impl KernelSource<Wgsl> for ScaleKernel {
    const ENTRY: &'static str = "scale";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r"
struct Params {
    factor: f32,
    len: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> destination: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        destination[index] = params.factor * destination[index];
    }
}
",
        )
    }
}
