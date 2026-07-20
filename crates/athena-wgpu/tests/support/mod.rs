use hephaestus_wgpu::WgpuDevice;

pub(crate) fn device(label: &str) -> Option<WgpuDevice> {
    match WgpuDevice::try_default(label) {
        Ok(device) => Some(device),
        Err(error) if std::env::var_os("CI").is_none() => {
            eprintln!("skipping local WGPU contract without an adapter: {error}");
            None
        }
        Err(error) => panic!("CI requires a WGPU adapter: {error}"),
    }
}
