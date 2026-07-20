use std::borrow::Cow;

use hephaestus_core::{BindingDecl, KernelInterface, KernelSource, Wgsl};

use super::VectorParams;

pub(crate) struct UpdateKernel;

impl KernelInterface for UpdateKernel {
    type Params = VectorParams;

    const LABEL: &'static str = "athena-cg-update";
    const BINDINGS: &'static [BindingDecl] = &[
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<f32>(),
        BindingDecl::read_write::<f32>(),
        BindingDecl::read_only::<f32>(),
    ];
    const WORKGROUP: [u32; 3] = [256, 1, 1];
}

impl KernelSource<Wgsl> for UpdateKernel {
    const ENTRY: &'static str = "update";

    fn source(&self) -> Cow<'static, str> {
        Cow::Borrowed(
            r"
struct Params {
    alpha: f32,
    len: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> solution: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read_write> residual: array<f32>;
@group(0) @binding(3) var<storage, read> image: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index < params.len) {
        solution[index] = solution[index] + params.alpha * direction[index];
        residual[index] = residual[index] - params.alpha * image[index];
    }
}
",
        )
    }
}
