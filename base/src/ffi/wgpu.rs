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
