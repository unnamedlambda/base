use pollster::block_on;
use std::sync::Arc;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    InstanceDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, PowerPreference,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use super::{clear_ctx_slot, read_ctx_mut, read_ctx_ref, write_ctx_slot};

// Cached wgpu Device + Queue. Creating many wgpu Devices exhausts OS GPU driver
// handles (~60 limit). One device per process; callers create fresh
// buffers/pipelines per use.
fn cached_gpu_device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
    use std::sync::OnceLock;
    static GPU: OnceLock<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> = OnceLock::new();
    let (d, q) = GPU.get_or_init(|| {
        let instance = wgpu::Instance::new(InstanceDescriptor::default());
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Failed to find GPU adapter");
        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor::default(), None))
            .expect("Failed to create GPU device");
        std::mem::forget(instance);
        std::mem::forget(adapter);
        (Arc::new(device), Arc::new(queue))
    });
    (d.clone(), q.clone())
}

pub(crate) struct CraneliftGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: Vec<wgpu::Buffer>,
    staging_buffers: Vec<wgpu::Buffer>,
    pipelines: Vec<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    pending_encoder: Option<wgpu::CommandEncoder>,
}

pub(crate) unsafe extern "C" fn cl_gpu_init(ctx_slot_ptr: *mut *mut CraneliftGpuContext) {
    let (device, queue) = cached_gpu_device();
    let ctx = Box::new(CraneliftGpuContext {
        device,
        queue,
        buffers: Vec::new(),
        staging_buffers: Vec::new(),
        pipelines: Vec::new(),
        pending_encoder: None,
    });
    let _ = write_ctx_slot(ctx_slot_ptr, Box::into_raw(ctx));
}

pub(crate) unsafe extern "C" fn cl_gpu_create_buffer(
    ctx_ptr: *mut CraneliftGpuContext,
    size: i64,
) -> i32 {
    if size <= 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftGpuContext>(ctx_ptr) else {
            return -1;
        };
        let buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let idx = ctx.buffers.len() as i32;
        ctx.buffers.push(buffer);
        ctx.staging_buffers.push(staging);
        idx
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_gpu_create_pipeline(
    ctx_ptr: *mut CraneliftGpuContext,
    shader_ptr: *const u8,
    bind_ptr: *const u8,
    n_bindings: i32,
) -> i32 {
    if n_bindings < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftGpuContext>(ctx_ptr) else {
            return -1;
        };
        let mut len = 0;
        while *shader_ptr.add(len) != 0 {
            len += 1;
        }
        let shader_src = match std::str::from_utf8(std::slice::from_raw_parts(shader_ptr, len)) {
            Ok(s) => s,
            Err(_) => return -1,
        };
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(shader_src.into()),
        });
        let mut bgl_entries = Vec::new();
        let mut bg_entries = Vec::new();
        let bind_base = bind_ptr;
        let n_bufs = ctx.buffers.len();
        for i in 0..n_bindings as usize {
            let desc_ptr = bind_base.add(i * 8);
            let buf_id = std::ptr::read_unaligned(desc_ptr as *const i32) as usize;
            if buf_id >= n_bufs {
                return -1;
            }
            let read_only = std::ptr::read_unaligned(desc_ptr.add(4) as *const i32) != 0;
            bgl_entries.push(BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
            bg_entries.push((i as u32, buf_id));
        }
        let bgl = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &bgl_entries,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(
                    &ctx.device
                        .create_pipeline_layout(&PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[&bgl],
                            push_constant_ranges: &[],
                        }),
                ),
                module: &shader,
                entry_point: "main",
                compilation_options: PipelineCompilationOptions::default(),
            });
        let entries: Vec<BindGroupEntry> = bg_entries
            .iter()
            .map(|&(binding, buf_id)| BindGroupEntry {
                binding,
                resource: BindingResource::Buffer(ctx.buffers[buf_id].as_entire_buffer_binding()),
            })
            .collect();
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &entries,
        });
        let idx = ctx.pipelines.len() as i32;
        ctx.pipelines.push((pipeline, bind_group));
        idx
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_gpu_upload(
    ctx_ptr: *const CraneliftGpuContext,
    buf_id: i32,
    src_ptr: *const u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftGpuContext>(ctx_ptr) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= ctx.buffers.len() {
            return -1;
        }
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        ctx.queue.write_buffer(&ctx.buffers[bid], 0, data);
        0
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_gpu_upload_ptr(
    ctx_ptr: *const CraneliftGpuContext,
    buf_id: i32,
    src_ptr: *const u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftGpuContext>(ctx_ptr) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= ctx.buffers.len() {
            return -1;
        }
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        ctx.queue.write_buffer(&ctx.buffers[bid], 0, data);
        0
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_gpu_download_ptr(
    ctx_ptr: *mut CraneliftGpuContext,
    buf_id: i32,
    buf_offset: i64,
    dst_ptr: *mut u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || dst_ptr.is_null() || buf_offset < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftGpuContext>(ctx_ptr) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= ctx.buffers.len() {
            return -1;
        }
        let size = size as u64;
        let buf_offset = buf_offset as u64;
        let mut encoder = ctx.pending_encoder.take().unwrap_or_else(|| {
            ctx.device
                .create_command_encoder(&CommandEncoderDescriptor { label: None })
        });
        encoder.copy_buffer_to_buffer(
            &ctx.buffers[bid],
            buf_offset,
            &ctx.staging_buffers[bid],
            0,
            size,
        );
        ctx.queue.submit(Some(encoder.finish()));
        let slice = ctx.staging_buffers[bid].slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::Maintain::Wait);
        let mapped = slice.get_mapped_range();
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        dst.copy_from_slice(&mapped);
        drop(mapped);
        ctx.staging_buffers[bid].unmap();
        0
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_gpu_dispatch(
    ctx_ptr: *mut CraneliftGpuContext,
    pipeline_id: i32,
    wg_x: i32,
    wg_y: i32,
    wg_z: i32,
) -> i32 {
    if pipeline_id < 0 || wg_x <= 0 || wg_y <= 0 || wg_z <= 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftGpuContext>(ctx_ptr) else {
            return -1;
        };
        let pid = pipeline_id as usize;
        if pid >= ctx.pipelines.len() {
            return -1;
        }
        if let Some(enc) = ctx.pending_encoder.take() {
            ctx.queue.submit(Some(enc.finish()));
        }
        let (pipeline, bind_group) = &ctx.pipelines[pid];
        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(wg_x as u32, wg_y as u32, wg_z as u32);
        }
        ctx.pending_encoder = Some(encoder);
        0
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_gpu_download(
    ctx_ptr: *mut CraneliftGpuContext,
    buf_id: i32,
    dst_ptr: *mut u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || dst_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftGpuContext>(ctx_ptr) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= ctx.buffers.len() {
            return -1;
        }
        let size = size as u64;
        let mut encoder = ctx.pending_encoder.take().unwrap_or_else(|| {
            ctx.device
                .create_command_encoder(&CommandEncoderDescriptor { label: None })
        });
        encoder.copy_buffer_to_buffer(&ctx.buffers[bid], 0, &ctx.staging_buffers[bid], 0, size);
        ctx.queue.submit(Some(encoder.finish()));
        let slice = ctx.staging_buffers[bid].slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::Maintain::Wait);
        let mapped = slice.get_mapped_range();
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        dst.copy_from_slice(&mapped);
        drop(mapped);
        ctx.staging_buffers[bid].unmap();
        0
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_gpu_cleanup(ctx_slot_ptr: *mut *mut CraneliftGpuContext) {
    let ctx_ptr = clear_ctx_slot::<CraneliftGpuContext>(ctx_slot_ptr);
    if !ctx_ptr.is_null() {
        drop(Box::from_raw(ctx_ptr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WGSL_VEC_ADD: &str = concat!(
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n",
        "@group(0) @binding(1) var<storage, read> b: array<f32>;\n",
        "@group(0) @binding(2) var<storage, read_write> result: array<f32>;\n",
        "@compute @workgroup_size(64)\n",
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n",
        "    let i = gid.x;\n",
        "    if (i < arrayLength(&a)) { result[i] = a[i] + b[i]; }\n",
        "}\n\0"
    );

    const WGSL_MUL2: &str = concat!(
        "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n",
        "@compute @workgroup_size(64)\n",
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n",
        "    let i = gid.x;\n",
        "    if (i < arrayLength(&data)) { data[i] = data[i] * 2.0; }\n",
        "}\n\0"
    );

    const WGSL_ADD1: &str = concat!(
        "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n",
        "@compute @workgroup_size(64)\n",
        "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n",
        "    let i = gid.x;\n",
        "    if (i < arrayLength(&data)) { data[i] = data[i] + 1.0; }\n",
        "}\n\0"
    );

    fn bind_desc(buf_id: i32, read_only: bool) -> [u8; 8] {
        let mut b = [0u8; 8];
        b[0..4].copy_from_slice(&buf_id.to_le_bytes());
        b[4..8].copy_from_slice(&(read_only as i32).to_le_bytes());
        b
    }

    #[test]
    fn init_then_cleanup_lifecycle() {
        let mut slot: *mut CraneliftGpuContext = std::ptr::null_mut();
        unsafe {
            cl_gpu_init(&mut slot);
            assert!(!slot.is_null());
            cl_gpu_cleanup(&mut slot);
            assert!(slot.is_null());
        }
    }

    #[test]
    fn create_buffer_returns_sequential_ids() {
        let mut slot: *mut CraneliftGpuContext = std::ptr::null_mut();
        unsafe {
            cl_gpu_init(&mut slot);
            assert_eq!(cl_gpu_create_buffer(slot, 256), 0);
            assert_eq!(cl_gpu_create_buffer(slot, 256), 1);
            assert_eq!(cl_gpu_create_buffer(slot, 256), 2);
            cl_gpu_cleanup(&mut slot);
        }
    }

    #[test]
    fn create_buffer_invalid_size_returns_neg1() {
        let mut slot: *mut CraneliftGpuContext = std::ptr::null_mut();
        unsafe {
            cl_gpu_init(&mut slot);
            assert_eq!(cl_gpu_create_buffer(slot, 0), -1);
            assert_eq!(cl_gpu_create_buffer(slot, -1), -1);
            cl_gpu_cleanup(&mut slot);
        }
    }

    #[test]
    fn vec_add_roundtrip() {
        let n: usize = 64;
        let size = (n * 4) as i64;
        let a: Vec<f32> = (1..=n as u32).map(|x| x as f32).collect();
        let b = vec![100.0f32; n];
        let mut result = vec![0.0f32; n];

        let mut bindings = [0u8; 24];
        bindings[0..8].copy_from_slice(&bind_desc(0, true));
        bindings[8..16].copy_from_slice(&bind_desc(1, true));
        bindings[16..24].copy_from_slice(&bind_desc(2, false));

        let mut slot: *mut CraneliftGpuContext = std::ptr::null_mut();
        unsafe {
            cl_gpu_init(&mut slot);

            let buf_a = cl_gpu_create_buffer(slot, size);
            let buf_b = cl_gpu_create_buffer(slot, size);
            let buf_r = cl_gpu_create_buffer(slot, size);

            assert_eq!(cl_gpu_upload(slot, buf_a, a.as_ptr() as *const u8, size), 0);
            assert_eq!(cl_gpu_upload(slot, buf_b, b.as_ptr() as *const u8, size), 0);

            let pip = cl_gpu_create_pipeline(slot, WGSL_VEC_ADD.as_ptr(), bindings.as_ptr(), 3);
            assert!(pip >= 0, "create_pipeline failed");

            assert_eq!(cl_gpu_dispatch(slot, pip, 1, 1, 1), 0);
            assert_eq!(
                cl_gpu_download(slot, buf_r, result.as_mut_ptr() as *mut u8, size),
                0
            );

            cl_gpu_cleanup(&mut slot);
        }

        for i in 0..n {
            let expected = (i + 1) as f32 + 100.0;
            assert!(
                (result[i] - expected).abs() < 0.01,
                "index {i}: got {}, expected {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn multiple_dispatches_before_download() {
        // pending_encoder batching: dispatch ×3 with data[i]*=2 each → data[i]*8
        let n: usize = 64;
        let size = (n * 4) as i64;
        let data: Vec<f32> = (1..=n as u32).map(|x| x as f32).collect();
        let mut result = vec![0.0f32; n];
        let binding = bind_desc(0, false);

        let mut slot: *mut CraneliftGpuContext = std::ptr::null_mut();
        unsafe {
            cl_gpu_init(&mut slot);

            let buf = cl_gpu_create_buffer(slot, size);
            assert_eq!(cl_gpu_upload(slot, buf, data.as_ptr() as *const u8, size), 0);

            let pip = cl_gpu_create_pipeline(slot, WGSL_MUL2.as_ptr(), binding.as_ptr(), 1);
            assert!(pip >= 0);

            assert_eq!(cl_gpu_dispatch(slot, pip, 1, 1, 1), 0);
            assert_eq!(cl_gpu_dispatch(slot, pip, 1, 1, 1), 0);
            assert_eq!(cl_gpu_dispatch(slot, pip, 1, 1, 1), 0);

            assert_eq!(
                cl_gpu_download(slot, buf, result.as_mut_ptr() as *mut u8, size),
                0
            );
            cl_gpu_cleanup(&mut slot);
        }

        for i in 0..n {
            let expected = (i + 1) as f32 * 8.0;
            assert!(
                (result[i] - expected).abs() < 0.01,
                "index {i}: got {}, expected {expected}",
                result[i]
            );
        }
    }

    #[test]
    fn buffer_reuse() {
        // Upload A → dispatch → download, then upload B → dispatch → download on same buffer.
        let n: usize = 64;
        let size = (n * 4) as i64;
        let data_a = vec![10.0f32; n];
        let data_b = vec![100.0f32; n];
        let mut result_a = vec![0.0f32; n];
        let mut result_b = vec![0.0f32; n];
        let binding = bind_desc(0, false);

        let mut slot: *mut CraneliftGpuContext = std::ptr::null_mut();
        unsafe {
            cl_gpu_init(&mut slot);

            let buf = cl_gpu_create_buffer(slot, size);
            let pip = cl_gpu_create_pipeline(slot, WGSL_ADD1.as_ptr(), binding.as_ptr(), 1);
            assert!(pip >= 0);

            assert_eq!(cl_gpu_upload(slot, buf, data_a.as_ptr() as *const u8, size), 0);
            assert_eq!(cl_gpu_dispatch(slot, pip, 1, 1, 1), 0);
            assert_eq!(
                cl_gpu_download(slot, buf, result_a.as_mut_ptr() as *mut u8, size),
                0
            );

            assert_eq!(cl_gpu_upload(slot, buf, data_b.as_ptr() as *const u8, size), 0);
            assert_eq!(cl_gpu_dispatch(slot, pip, 1, 1, 1), 0);
            assert_eq!(
                cl_gpu_download(slot, buf, result_b.as_mut_ptr() as *mut u8, size),
                0
            );

            cl_gpu_cleanup(&mut slot);
        }

        for i in 0..n {
            assert!(
                (result_a[i] - 11.0).abs() < 0.01,
                "A[{i}]: got {}",
                result_a[i]
            );
            assert!(
                (result_b[i] - 101.0).abs() < 0.01,
                "B[{i}]: got {}",
                result_b[i]
            );
        }
    }

    #[test]
    fn upload_ptr_download_ptr_roundtrip() {
        let n: usize = 64;
        let size = (n * 4) as i64;
        let data: Vec<f32> = (1..=n as u32).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; n];

        let mut slot: *mut CraneliftGpuContext = std::ptr::null_mut();
        unsafe {
            cl_gpu_init(&mut slot);
            let buf = cl_gpu_create_buffer(slot, size);
            assert_eq!(
                cl_gpu_upload_ptr(slot, buf, data.as_ptr() as *const u8, size),
                0
            );
            assert_eq!(
                cl_gpu_download_ptr(slot, buf, 0, out.as_mut_ptr() as *mut u8, size),
                0
            );
            cl_gpu_cleanup(&mut slot);
        }

        for i in 0..n {
            assert!((out[i] - (i + 1) as f32).abs() < 0.01, "index {i}: got {}", out[i]);
        }
    }

    #[test]
    fn download_ptr_with_buf_offset() {
        // Upload [A: n f32][B: n f32], download only the B region via buf_offset.
        let n: usize = 64;
        let half = (n * 4) as i64;
        let full = half * 2;

        let mut payload: Vec<f32> = (1..=n as u32).map(|x| x as f32).collect();
        payload.extend((101..=100 + n as u32).map(|x| x as f32));
        let mut out = vec![0.0f32; n];

        let mut slot: *mut CraneliftGpuContext = std::ptr::null_mut();
        unsafe {
            cl_gpu_init(&mut slot);
            let buf = cl_gpu_create_buffer(slot, full);
            assert_eq!(
                cl_gpu_upload_ptr(slot, buf, payload.as_ptr() as *const u8, full),
                0
            );
            assert_eq!(
                cl_gpu_download_ptr(slot, buf, half, out.as_mut_ptr() as *mut u8, half),
                0
            );
            cl_gpu_cleanup(&mut slot);
        }

        for i in 0..n {
            let expected = (i + 101) as f32;
            assert!(
                (out[i] - expected).abs() < 0.01,
                "index {i}: got {}, expected {expected}",
                out[i]
            );
        }
    }

    #[test]
    fn error_codes_for_invalid_args() {
        let data = [0u8; 64];
        let mut dst = [0u8; 64];
        let bad_binding = bind_desc(99, false);
        const WGSL_TRIVIAL: &str = concat!(
            "@group(0) @binding(0) var<storage, read_write> d: array<f32>;\n",
            "@compute @workgroup_size(1)\n",
            "fn main() { d[0] = 1.0; }\n\0"
        );

        let mut slot: *mut CraneliftGpuContext = std::ptr::null_mut();
        unsafe {
            cl_gpu_init(&mut slot);

            assert_eq!(cl_gpu_create_buffer(slot, 0), -1);
            assert_eq!(cl_gpu_upload(slot, 99, data.as_ptr(), 64), -1);
            assert_eq!(cl_gpu_upload_ptr(slot, 99, data.as_ptr(), 64), -1);
            assert_eq!(cl_gpu_download(slot, 99, dst.as_mut_ptr(), 64), -1);
            assert_eq!(cl_gpu_download_ptr(slot, 99, 0, dst.as_mut_ptr(), 64), -1);
            assert_eq!(cl_gpu_dispatch(slot, 99, 1, 1, 1), -1);

            // Create a valid buffer, then create_pipeline with binding that references buf_id=99.
            cl_gpu_create_buffer(slot, 256);
            assert_eq!(
                cl_gpu_create_pipeline(slot, WGSL_TRIVIAL.as_ptr(), bad_binding.as_ptr(), 1),
                -1
            );

            cl_gpu_cleanup(&mut slot);
        }
    }

    #[test]
    fn null_ctx_returns_neg1() {
        let null = std::ptr::null_mut::<CraneliftGpuContext>();
        let data = [0u8; 64];
        let mut dst = [0u8; 64];
        let bind = [0u8; 8];
        unsafe {
            assert_eq!(cl_gpu_create_buffer(null, 64), -1);
            assert_eq!(cl_gpu_upload(null as *const _, 0, data.as_ptr(), 64), -1);
            assert_eq!(cl_gpu_upload_ptr(null as *const _, 0, data.as_ptr(), 64), -1);
            assert_eq!(cl_gpu_download(null, 0, dst.as_mut_ptr(), 64), -1);
            assert_eq!(cl_gpu_download_ptr(null, 0, 0, dst.as_mut_ptr(), 64), -1);
            assert_eq!(cl_gpu_dispatch(null, 0, 1, 1, 1), -1);
            assert_eq!(cl_gpu_create_pipeline(null, data.as_ptr(), bind.as_ptr(), 0), -1);
        }
    }
}
