//! wgpu through the `wgpu` crate, one call each. An object is boxed and
//! handed out as an `i64`; a call that consumes one frees it. An error no
//! scope catches is printed and the program goes on.

// The functions are named as the symbols a program's calls name.
#![allow(non_snake_case)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::window;

/// A device and its one queue, which wgpu hands over together.
type Device = (wgpu::Device, wgpu::Queue);

fn boxed<T>(x: T) -> i64 {
    Box::into_raw(Box::new(x)) as i64
}

/// The object `h` holds, borrowed.
unsafe fn get<'a, T>(h: i64) -> &'a mut T {
    &mut *(h as *mut T)
}

/// The object `h` holds, taken back: its box is freed.
unsafe fn take<T>(h: i64) -> T {
    *Box::from_raw(h as *mut T)
}

/// `n` words of 8 bytes at `a`; empty for a count below one.
unsafe fn words<'a>(a: *const u64, n: i64) -> &'a [u64] {
    match usize::try_from(n) {
        Ok(n) if n > 0 => std::slice::from_raw_parts(a, n),
        _ => &[],
    }
}

/// `n` bytes at `a` as UTF-8, or `None`.
unsafe fn text<'a>(a: *const u8, n: i64) -> Option<&'a str> {
    match usize::try_from(n).ok()? {
        0 => Some(""),
        n => std::str::from_utf8(std::slice::from_raw_parts(a, n)).ok(),
    }
}

/// A `WGPUTextureFormat` the model names: RGBA8 and BGRA8, linear or sRGB.
fn format(f: i32) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    match f {
        0x16 => Some(F::Rgba8Unorm),
        0x17 => Some(F::Rgba8UnormSrgb),
        0x1B => Some(F::Bgra8Unorm),
        0x1C => Some(F::Bgra8UnormSrgb),
        _ => None,
    }
}

// --- Setup

unsafe extern "C" fn wgpuCreateInstance() -> i64 {
    boxed(wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()))
}

/// `inst`'s high-performance adapter, waited for; `0` when there is none.
unsafe extern "C" fn wgpuInstanceRequestAdapter(inst: i64) -> i64 {
    let inst = get::<wgpu::Instance>(inst);
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    };
    match pollster::block_on(inst.request_adapter(&options)) {
        Ok(a) => boxed(a),
        Err(_) => 0,
    }
}

/// A device of `adapter`, waited for; `0` when it refuses. An error no scope
/// catches is reported, and the program goes on.
unsafe extern "C" fn wgpuAdapterRequestDevice(adapter: i64) -> i64 {
    let adapter = get::<wgpu::Adapter>(adapter);
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
        Ok((device, queue)) => {
            device.on_uncaptured_error(Arc::new(|e: wgpu::Error| eprintln!("wgpu: {e}")));
            boxed::<Device>((device, queue))
        }
        Err(_) => 0,
    }
}

unsafe extern "C" fn wgpuDeviceGetQueue(dev: i64) -> i64 {
    boxed(get::<Device>(dev).1.clone())
}

// --- Objects

/// A zeroed buffer of `size` bytes with `usage` (`WGPUBufferUsage` bits).
unsafe extern "C" fn wgpuDeviceCreateBuffer(dev: i64, usage: i64, size: i64) -> i64 {
    let dev = &get::<Device>(dev).0;
    let (Some(usage), Ok(size)) = (wgpu::BufferUsages::from_bits(usage as u32), u64::try_from(size)) else {
        return 0;
    };
    boxed(dev.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false }))
}

/// A module of the `len` bytes of WGSL at `src`.
unsafe extern "C" fn wgpuDeviceCreateShaderModule(dev: i64, src: *const u8, len: i64) -> i64 {
    let dev = &get::<Device>(dev).0;
    let Some(src) = text(src, len) else { return 0 };
    boxed(dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    }))
}

/// Storage buffers seen by compute: `n` entries at `ents`, each a
/// `(binding, read-only)` pair of `u32`s.
unsafe extern "C" fn wgpuDeviceCreateBindGroupLayout(dev: i64, ents: *const u32, n: i64) -> i64 {
    let dev = &get::<Device>(dev).0;
    let entries: Vec<_> = words(ents as *const u64, n)
        .iter()
        .map(|&e| wgpu::BindGroupLayoutEntry {
            binding: e as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: (e >> 32) as u32 != 0 },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    boxed(dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries }))
}

/// A layout of the `n` group layouts whose handles are at `layouts`.
unsafe extern "C" fn wgpuDeviceCreatePipelineLayout(dev: i64, layouts: *const u64, n: i64) -> i64 {
    let dev = &get::<Device>(dev).0;
    let ls: Vec<_> = words(layouts, n).iter().map(|&h| Some(&*get::<wgpu::BindGroupLayout>(h as i64))).collect();
    boxed(dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &ls,
        immediate_size: 0,
    }))
}

/// A compute pipeline of `module`'s entry point named by the `len` bytes at
/// `entry`, with `layout`.
unsafe extern "C" fn wgpuDeviceCreateComputePipeline(
    dev: i64,
    layout: i64,
    module: i64,
    entry: *const u8,
    len: i64,
) -> i64 {
    let dev = &get::<Device>(dev).0;
    let Some(entry) = text(entry, len) else { return 0 };
    boxed(dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(get::<wgpu::PipelineLayout>(layout)),
        module: get::<wgpu::ShaderModule>(module),
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    }))
}

/// A group of `layout`: `n` entries at `ents`, each a `(binding, buffer)`
/// pair of words, each binding the whole buffer.
unsafe extern "C" fn wgpuDeviceCreateBindGroup(dev: i64, layout: i64, ents: *const u64, n: i64) -> i64 {
    let dev = &get::<Device>(dev).0;
    let pairs = words(ents, n.saturating_mul(2));
    let entries: Vec<_> = pairs
        .chunks_exact(2)
        .map(|p| wgpu::BindGroupEntry {
            binding: p[0] as u32,
            resource: get::<wgpu::Buffer>(p[1] as i64).as_entire_binding(),
        })
        .collect();
    boxed(dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: get::<wgpu::BindGroupLayout>(layout),
        entries: &entries,
    }))
}

/// A pipeline drawing triangles from `module`'s vertex and fragment entry
/// points, named by the bytes at `vs` and `fs`, to one target of `format`
/// with no blending; its layout derived from the shader.
unsafe extern "C" fn wgpuDeviceCreateRenderPipeline(
    dev: i64,
    module: i64,
    vs: *const u8,
    vs_len: i64,
    fs: *const u8,
    fs_len: i64,
    fmt: i32,
) -> i64 {
    let dev = &get::<Device>(dev).0;
    let module = get::<wgpu::ShaderModule>(module);
    let (Some(vs), Some(fs), Some(fmt)) = (text(vs, vs_len), text(fs, fs_len), format(fmt)) else { return 0 };
    let targets = [Some(wgpu::ColorTargetState { format: fmt, blend: None, write_mask: wgpu::ColorWrites::ALL })];
    boxed(dev.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    }))
}

unsafe extern "C" fn wgpuRenderPipelineGetBindGroupLayout(pipe: i64, index: i32) -> i64 {
    let Ok(index) = u32::try_from(index) else { return 0 };
    boxed(get::<wgpu::RenderPipeline>(pipe).get_bind_group_layout(index))
}

// --- Errors

/// A scope catching `dev`'s validation errors, as a handle to pop.
unsafe extern "C" fn wgpuDevicePushErrorScope(dev: i64) -> i64 {
    boxed(get::<Device>(dev).0.push_error_scope(wgpu::ErrorFilter::Validation))
}

/// `scope` popped and freed, waited for: `0` when it caught nothing, `-1`
/// when it caught an error.
unsafe extern "C" fn wgpuErrorScopePop(scope: i64) -> i32 {
    let g = take::<wgpu::ErrorScopeGuard>(scope);
    if pollster::block_on(g.pop()).is_none() { 0 } else { -1 }
}

// --- Recording

unsafe extern "C" fn wgpuDeviceCreateCommandEncoder(dev: i64) -> i64 {
    boxed(get::<Device>(dev).0.create_command_encoder(&wgpu::CommandEncoderDescriptor::default()))
}

/// A compute pass on `enc`, which wgpu locks until the pass ends.
unsafe extern "C" fn wgpuCommandEncoderBeginComputePass(enc: i64) -> i64 {
    let d = wgpu::ComputePassDescriptor { label: None, timestamp_writes: None };
    boxed(get::<wgpu::CommandEncoder>(enc).begin_compute_pass(&d).forget_lifetime())
}

unsafe extern "C" fn wgpuComputePassSetPipeline(pass: i64, pipe: i64) {
    get::<wgpu::ComputePass>(pass).set_pipeline(get::<wgpu::ComputePipeline>(pipe))
}

unsafe extern "C" fn wgpuComputePassSetBindGroup(pass: i64, index: i32, group: i64) {
    let Ok(index) = u32::try_from(index) else { return };
    get::<wgpu::ComputePass>(pass).set_bind_group(index, &*get::<wgpu::BindGroup>(group), &[])
}

unsafe extern "C" fn wgpuComputePassDispatchWorkgroups(pass: i64, x: i32, y: i32, z: i32) {
    get::<wgpu::ComputePass>(pass).dispatch_workgroups(x as u32, y as u32, z as u32)
}

/// `pass` ended and freed; its encoder unlocked.
unsafe extern "C" fn wgpuComputePassEnd(pass: i64) {
    drop(take::<wgpu::ComputePass>(pass))
}

unsafe extern "C" fn wgpuCommandEncoderCopyBufferToBuffer(enc: i64, src: i64, so: i64, dst: i64, dof: i64, n: i64) {
    let (Ok(so), Ok(dof), Ok(n)) = (u64::try_from(so), u64::try_from(dof), u64::try_from(n)) else { return };
    let (src, dst) = (get::<wgpu::Buffer>(src), get::<wgpu::Buffer>(dst));
    get::<wgpu::CommandEncoder>(enc).copy_buffer_to_buffer(src, so, dst, dof, n)
}

/// A render pass on `enc` to `view`, cleared to opaque black where `clear` is
/// nonzero and loaded otherwise.
unsafe extern "C" fn wgpuCommandEncoderBeginRenderPass(enc: i64, view: i64, clear: i32) -> i64 {
    let load = if clear != 0 { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load };
    let attachments = [Some(wgpu::RenderPassColorAttachment {
        view: get::<wgpu::TextureView>(view),
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
    })];
    let d = wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    };
    boxed(get::<wgpu::CommandEncoder>(enc).begin_render_pass(&d).forget_lifetime())
}

unsafe extern "C" fn wgpuRenderPassSetPipeline(pass: i64, pipe: i64) {
    get::<wgpu::RenderPass>(pass).set_pipeline(get::<wgpu::RenderPipeline>(pipe))
}

unsafe extern "C" fn wgpuRenderPassSetBindGroup(pass: i64, index: i32, group: i64) {
    let Ok(index) = u32::try_from(index) else { return };
    get::<wgpu::RenderPass>(pass).set_bind_group(index, &*get::<wgpu::BindGroup>(group), &[])
}

unsafe extern "C" fn wgpuRenderPassDraw(pass: i64, vertices: i32, instances: i32, first_vertex: i32, first_instance: i32) {
    let (v, i, fv, fi) = (vertices as u32, instances as u32, first_vertex as u32, first_instance as u32);
    get::<wgpu::RenderPass>(pass).draw(fv..fv.wrapping_add(v), fi..fi.wrapping_add(i))
}

/// `pass` ended and freed; its encoder unlocked.
unsafe extern "C" fn wgpuRenderPassEnd(pass: i64) {
    drop(take::<wgpu::RenderPass>(pass))
}

/// `enc` finished and freed: its commands, as a command buffer.
unsafe extern "C" fn wgpuCommandEncoderFinish(enc: i64) -> i64 {
    boxed(take::<wgpu::CommandEncoder>(enc).finish())
}

// --- The queue

/// `cmds` run on `queue`, and freed.
unsafe extern "C" fn wgpuQueueSubmit(queue: i64, cmds: i64) {
    get::<wgpu::Queue>(queue).submit(std::iter::once(take::<wgpu::CommandBuffer>(cmds)));
}

/// The `n` bytes at `src` written to `buf` from `off`, before the next submit.
unsafe extern "C" fn wgpuQueueWriteBuffer(queue: i64, buf: i64, off: i64, src: *const u8, n: i64) {
    let (Ok(off), Ok(n)) = (u64::try_from(off), usize::try_from(n)) else { return };
    let data = if n == 0 { &[][..] } else { std::slice::from_raw_parts(src, n) };
    get::<wgpu::Queue>(queue).write_buffer(get::<wgpu::Buffer>(buf), off, data)
}

/// `n` bytes of `buf` from `off` copied to `dst` through a read mapping,
/// waiting for the queue: `0`, or `-1` when wgpu refuses the mapping.
unsafe extern "C" fn wgpuBufferRead(dev: i64, buf: i64, off: i64, dst: *mut u8, n: i64) -> i32 {
    let (Ok(off), Ok(n)) = (u64::try_from(off), u64::try_from(n)) else { return -1 };
    if n == 0 {
        return -1;
    }
    let (dev, buf) = (&get::<Device>(dev).0, get::<wgpu::Buffer>(buf));
    let ok = Arc::new(AtomicBool::new(false));
    let done = ok.clone();
    let slice = buf.slice(off..off + n);
    slice.map_async(wgpu::MapMode::Read, move |r| done.store(r.is_ok(), Ordering::SeqCst));
    if dev.poll(wgpu::PollType::wait_indefinitely()).is_err() || !ok.load(Ordering::SeqCst) {
        return -1;
    }
    std::ptr::copy_nonoverlapping(slice.get_mapped_range().as_ptr(), dst, n as usize);
    buf.unmap();
    0
}

// --- Surfaces

/// A surface of the window `win` (the window library's handle), which keeps
/// the window alive; `0` when wgpu cannot make one.
unsafe extern "C" fn wgpuInstanceCreateSurface(inst: i64, win: i64) -> i64 {
    match get::<wgpu::Instance>(inst).create_surface(window::shared(win)) {
        Ok(s) => boxed(s),
        Err(_) => 0,
    }
}

/// `surf` configured for rendering on `dev`, `width` by `height` in `format`.
unsafe extern "C" fn wgpuSurfaceConfigure(surf: i64, dev: i64, fmt: i32, width: i32, height: i32) {
    let (Some(fmt), Ok(width), Ok(height)) = (format(fmt), u32::try_from(width), u32::try_from(height)) else {
        return;
    };
    let c = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: fmt,
        width,
        height,
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
    };
    get::<wgpu::Surface>(surf).configure(&get::<Device>(dev).0, &c)
}

/// `surf`'s next texture, or `0` when it has none ready: timed out,
/// occluded, outdated or lost.
unsafe extern "C" fn wgpuSurfaceGetCurrentTexture(surf: i64) -> i64 {
    match get::<wgpu::Surface>(surf).get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => boxed(t),
        _ => 0,
    }
}

unsafe extern "C" fn wgpuSurfaceTextureCreateView(tex: i64) -> i64 {
    boxed(get::<wgpu::SurfaceTexture>(tex).texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

/// `tex` shown on its surface, and freed.
unsafe extern "C" fn wgpuSurfaceTexturePresent(tex: i64) {
    take::<wgpu::SurfaceTexture>(tex).present()
}

// --- Releases

macro_rules! release {
    ($($name:ident $t:ty;)*) => {
        $(
            unsafe extern "C" fn $name(h: i64) {
                drop(take::<$t>(h))
            }
        )*
    };
}

release!(
    wgpuInstanceRelease wgpu::Instance;
    wgpuAdapterRelease wgpu::Adapter;
    wgpuDeviceRelease Device;
    wgpuQueueRelease wgpu::Queue;
    wgpuBufferRelease wgpu::Buffer;
    wgpuShaderModuleRelease wgpu::ShaderModule;
    wgpuBindGroupLayoutRelease wgpu::BindGroupLayout;
    wgpuPipelineLayoutRelease wgpu::PipelineLayout;
    wgpuComputePipelineRelease wgpu::ComputePipeline;
    wgpuBindGroupRelease wgpu::BindGroup;
    wgpuCommandEncoderRelease wgpu::CommandEncoder;
    wgpuComputePassEncoderRelease wgpu::ComputePass<'static>;
    wgpuCommandBufferRelease wgpu::CommandBuffer;
    wgpuSurfaceRelease wgpu::Surface<'static>;
    wgpuTextureRelease wgpu::SurfaceTexture;
    wgpuTextureViewRelease wgpu::TextureView;
    wgpuRenderPipelineRelease wgpu::RenderPipeline;
    wgpuRenderPassEncoderRelease wgpu::RenderPass<'static>;
    wgpuErrorScopeRelease wgpu::ErrorScopeGuard;
);

crate::builtin::symbols!(
    wgpuCreateInstance,
    wgpuInstanceRequestAdapter,
    wgpuAdapterRequestDevice,
    wgpuDeviceGetQueue,
    wgpuDeviceCreateBuffer,
    wgpuDeviceCreateShaderModule,
    wgpuDeviceCreateBindGroupLayout,
    wgpuDeviceCreatePipelineLayout,
    wgpuDeviceCreateComputePipeline,
    wgpuDeviceCreateBindGroup,
    wgpuDeviceCreateRenderPipeline,
    wgpuRenderPipelineGetBindGroupLayout,
    wgpuDevicePushErrorScope,
    wgpuErrorScopePop,
    wgpuDeviceCreateCommandEncoder,
    wgpuCommandEncoderBeginComputePass,
    wgpuComputePassSetPipeline,
    wgpuComputePassSetBindGroup,
    wgpuComputePassDispatchWorkgroups,
    wgpuComputePassEnd,
    wgpuCommandEncoderCopyBufferToBuffer,
    wgpuCommandEncoderBeginRenderPass,
    wgpuRenderPassSetPipeline,
    wgpuRenderPassSetBindGroup,
    wgpuRenderPassDraw,
    wgpuRenderPassEnd,
    wgpuCommandEncoderFinish,
    wgpuQueueSubmit,
    wgpuQueueWriteBuffer,
    wgpuBufferRead,
    wgpuInstanceCreateSurface,
    wgpuSurfaceConfigure,
    wgpuSurfaceGetCurrentTexture,
    wgpuSurfaceTextureCreateView,
    wgpuSurfaceTexturePresent,
    wgpuInstanceRelease,
    wgpuAdapterRelease,
    wgpuDeviceRelease,
    wgpuQueueRelease,
    wgpuBufferRelease,
    wgpuShaderModuleRelease,
    wgpuBindGroupLayoutRelease,
    wgpuPipelineLayoutRelease,
    wgpuComputePipelineRelease,
    wgpuBindGroupRelease,
    wgpuCommandEncoderRelease,
    wgpuComputePassEncoderRelease,
    wgpuCommandBufferRelease,
    wgpuSurfaceRelease,
    wgpuTextureRelease,
    wgpuTextureViewRelease,
    wgpuRenderPipelineRelease,
    wgpuRenderPassEncoderRelease,
    wgpuErrorScopeRelease,
);

#[cfg(test)]
mod tests {
    use super::*;

    /// An object is the program's from the call that makes it to its release.
    #[test]
    fn an_instance_is_made_and_released() {
        unsafe {
            let inst = wgpuCreateInstance();
            assert_ne!(inst, 0);
            wgpuInstanceRelease(inst);
        }
    }
}
