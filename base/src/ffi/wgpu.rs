//! wgpu, through the `wgpu` crate: the C functions a program's `Ext.wgpu`
//! calls reach, by symbol.
//!
//! Each is one call into wgpu, its descriptor spelled as arguments; what a
//! call reads from program memory is an array of words or a run of bytes, at
//! an address and a length it is handed. `Host/Wgpu.lean` states them.
//!
//! **Objects.** Every object is the program's: a call that makes one answers
//! a box, as an `i64`, and every other call takes it back. The box says what
//! kind of object it holds, so a handle of the wrong kind is refused — the
//! call answers `0` or `-1`, or does nothing — rather than read as another.
//! A call that consumes its object (ending a pass, finishing an encoder,
//! submitting, presenting, popping an error scope) frees the box; so does
//! each kind's release. Using a box after that is the program's to rule out.
//! Nothing is kept here between calls.
//!
//! **Errors.** An error no scope catches is reported on standard error and
//! the program goes on; a panic inside wgpu answers as a refusal, never
//! crossing into the program.

// The functions are named as the symbols a program's calls name.
#![allow(non_snake_case)]

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::window;

enum Obj {
    Instance(wgpu::Instance),
    Adapter(wgpu::Adapter),
    /// A device and its one queue, which wgpu hands over together.
    Device((wgpu::Device, wgpu::Queue)),
    Queue(wgpu::Queue),
    Buffer(wgpu::Buffer),
    Shader(wgpu::ShaderModule),
    Bgl(wgpu::BindGroupLayout),
    Playout(wgpu::PipelineLayout),
    Cpipe(wgpu::ComputePipeline),
    BindGroup(wgpu::BindGroup),
    Encoder(wgpu::CommandEncoder),
    Cpass(wgpu::ComputePass<'static>),
    CmdBuf(wgpu::CommandBuffer),
    Surface(wgpu::Surface<'static>),
    Texture(wgpu::SurfaceTexture),
    View(wgpu::TextureView),
    Rpipe(wgpu::RenderPipeline),
    Rpass(wgpu::RenderPass<'static>),
    Scope(wgpu::ErrorScopeGuard),
}

fn boxed(o: Obj) -> i64 {
    Box::into_raw(Box::new(o)) as i64
}

/// The object `h` holds, borrowed; `None` for no handle.
unsafe fn obj<'a>(h: i64) -> Option<&'a mut Obj> {
    (h as *mut Obj).as_mut()
}

/// The object `h` holds, taken back: the box is freed when it drops.
unsafe fn take(h: i64) -> Option<Box<Obj>> {
    if h == 0 { None } else { Some(Box::from_raw(h as *mut Obj)) }
}

/// A box taken back and found of another kind, given back to the program.
fn give_back(b: Box<Obj>) {
    let _ = Box::into_raw(b);
}

macro_rules! get {
    ($h:expr, $v:ident) => {
        match obj($h) {
            Some(Obj::$v(x)) => x,
            _ => return Default::default(),
        }
    };
}

/// `f`, with a panic answered as `fail`.
fn guard<T>(fail: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(fail)
}

/// `n` words of 8 bytes at `a`; empty for a negative count or no address.
unsafe fn words<'a>(a: *const u64, n: i64) -> &'a [u64] {
    match usize::try_from(n) {
        Ok(n) if n > 0 && !a.is_null() => std::slice::from_raw_parts(a, n),
        _ => &[],
    }
}

/// `n` bytes at `a` as UTF-8, or `None`.
unsafe fn text<'a>(a: *const u8, n: i64) -> Option<&'a str> {
    let n = usize::try_from(n).ok()?;
    if a.is_null() {
        return (n == 0).then_some("");
    }
    std::str::from_utf8(std::slice::from_raw_parts(a, n)).ok()
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

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

unsafe extern "C" fn wgpuCreateInstance() -> i64 {
    guard(0, || boxed(Obj::Instance(wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()))))
}

/// `inst`'s high-performance adapter, waited for; `0` when there is none.
unsafe extern "C" fn wgpuInstanceRequestAdapter(inst: i64) -> i64 {
    let inst = get!(inst, Instance);
    guard(0, || {
        let options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        };
        match pollster::block_on(inst.request_adapter(&options)) {
            Ok(a) => boxed(Obj::Adapter(a)),
            Err(_) => 0,
        }
    })
}

/// A device of `adapter`, waited for; `0` when it refuses. An error no scope
/// catches is reported, and the program goes on.
unsafe extern "C" fn wgpuAdapterRequestDevice(adapter: i64) -> i64 {
    let adapter = get!(adapter, Adapter);
    guard(0, || match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) {
        Ok((device, queue)) => {
            device.on_uncaptured_error(Arc::new(|e: wgpu::Error| eprintln!("wgpu: {e}")));
            boxed(Obj::Device((device, queue)))
        }
        Err(_) => 0,
    })
}

unsafe extern "C" fn wgpuDeviceGetQueue(dev: i64) -> i64 {
    let dev = get!(dev, Device);
    guard(0, || boxed(Obj::Queue(dev.1.clone())))
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

/// A zeroed buffer of `size` bytes with `usage` (`WGPUBufferUsage` bits).
unsafe extern "C" fn wgpuDeviceCreateBuffer(dev: i64, usage: i64, size: i64) -> i64 {
    let dev = &get!(dev, Device).0;
    let (Some(usage), Ok(size)) = (wgpu::BufferUsages::from_bits(usage as u32), u64::try_from(size)) else {
        return 0;
    };
    guard(0, || {
        let d = wgpu::BufferDescriptor { label: None, size, usage, mapped_at_creation: false };
        boxed(Obj::Buffer(dev.create_buffer(&d)))
    })
}

/// A module of the `len` bytes of WGSL at `src`.
unsafe extern "C" fn wgpuDeviceCreateShaderModule(dev: i64, src: *const u8, len: i64) -> i64 {
    let dev = &get!(dev, Device).0;
    let Some(src) = text(src, len) else { return 0 };
    guard(0, || {
        let d = wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(src.into()) };
        boxed(Obj::Shader(dev.create_shader_module(d)))
    })
}

/// Storage buffers seen by compute: `n` entries at `ents`, each a
/// `(binding, read-only)` pair of `u32`s.
unsafe extern "C" fn wgpuDeviceCreateBindGroupLayout(dev: i64, ents: *const u32, n: i64) -> i64 {
    let dev = &get!(dev, Device).0;
    let es = words(ents as *const u64, n);
    guard(0, || {
        let entries: Vec<_> = es
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
        let d = wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries };
        boxed(Obj::Bgl(dev.create_bind_group_layout(&d)))
    })
}

/// A layout of the `n` group layouts whose handles are at `layouts`.
unsafe extern "C" fn wgpuDeviceCreatePipelineLayout(dev: i64, layouts: *const u64, n: i64) -> i64 {
    let dev = &get!(dev, Device).0;
    let mut ls = Vec::new();
    for &h in words(layouts, n) {
        match obj(h as i64) {
            Some(Obj::Bgl(l)) => ls.push(Some(&*l)),
            _ => return 0,
        }
    }
    guard(0, || {
        let d = wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &ls, immediate_size: 0 };
        boxed(Obj::Playout(dev.create_pipeline_layout(&d)))
    })
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
    let dev = &get!(dev, Device).0;
    let layout = get!(layout, Playout);
    let module = get!(module, Shader);
    let Some(entry) = text(entry, len) else { return 0 };
    guard(0, || {
        let d = wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        };
        boxed(Obj::Cpipe(dev.create_compute_pipeline(&d)))
    })
}

/// A group of `layout`: `n` entries at `ents`, each a `(binding, buffer)`
/// pair of words, each binding the whole buffer.
unsafe extern "C" fn wgpuDeviceCreateBindGroup(dev: i64, layout: i64, ents: *const u64, n: i64) -> i64 {
    let dev = &get!(dev, Device).0;
    let layout = get!(layout, Bgl);
    let ws = match usize::try_from(n) {
        Ok(n) if n > 0 && !ents.is_null() => std::slice::from_raw_parts(ents, 2 * n),
        _ => &[][..],
    };
    let mut entries = Vec::new();
    for pair in ws.chunks_exact(2) {
        match obj(pair[1] as i64) {
            Some(Obj::Buffer(b)) => {
                entries.push(wgpu::BindGroupEntry { binding: pair[0] as u32, resource: b.as_entire_binding() })
            }
            _ => return 0,
        }
    }
    guard(0, || {
        let d = wgpu::BindGroupDescriptor { label: None, layout, entries: &entries };
        boxed(Obj::BindGroup(dev.create_bind_group(&d)))
    })
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
    let dev = &get!(dev, Device).0;
    let module = get!(module, Shader);
    let (Some(vs), Some(fs), Some(fmt)) = (text(vs, vs_len), text(fs, fs_len), format(fmt)) else { return 0 };
    guard(0, || {
        let targets = [Some(wgpu::ColorTargetState { format: fmt, blend: None, write_mask: wgpu::ColorWrites::ALL })];
        let d = wgpu::RenderPipelineDescriptor {
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
        };
        boxed(Obj::Rpipe(dev.create_render_pipeline(&d)))
    })
}

unsafe extern "C" fn wgpuRenderPipelineGetBindGroupLayout(pipe: i64, index: i32) -> i64 {
    let pipe = get!(pipe, Rpipe);
    let Ok(index) = u32::try_from(index) else { return 0 };
    guard(0, || boxed(Obj::Bgl(pipe.get_bind_group_layout(index))))
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A scope catching `dev`'s validation errors, as a handle to pop.
unsafe extern "C" fn wgpuDevicePushErrorScope(dev: i64) -> i64 {
    let dev = &get!(dev, Device).0;
    guard(0, || boxed(Obj::Scope(dev.push_error_scope(wgpu::ErrorFilter::Validation))))
}

/// `scope` popped and freed, waited for: `0` when it caught nothing, `-1`
/// when it caught an error.
unsafe extern "C" fn wgpuErrorScopePop(scope: i64) -> i32 {
    let Some(b) = take(scope) else { return -1 };
    let Obj::Scope(g) = *b else {
        give_back(b);
        return -1;
    };
    guard(-1, || if pollster::block_on(g.pop()).is_none() { 0 } else { -1 })
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

unsafe extern "C" fn wgpuDeviceCreateCommandEncoder(dev: i64) -> i64 {
    let dev = &get!(dev, Device).0;
    guard(0, || boxed(Obj::Encoder(dev.create_command_encoder(&wgpu::CommandEncoderDescriptor::default()))))
}

/// A compute pass on `enc`, which wgpu locks until the pass ends.
unsafe extern "C" fn wgpuCommandEncoderBeginComputePass(enc: i64) -> i64 {
    let enc = get!(enc, Encoder);
    guard(0, || {
        let d = wgpu::ComputePassDescriptor { label: None, timestamp_writes: None };
        boxed(Obj::Cpass(enc.begin_compute_pass(&d).forget_lifetime()))
    })
}

unsafe extern "C" fn wgpuComputePassSetPipeline(pass: i64, pipe: i64) {
    let pass = get!(pass, Cpass);
    let pipe = get!(pipe, Cpipe);
    guard((), || pass.set_pipeline(pipe))
}

unsafe extern "C" fn wgpuComputePassSetBindGroup(pass: i64, index: i32, group: i64) {
    let pass = get!(pass, Cpass);
    let group = get!(group, BindGroup);
    let Ok(index) = u32::try_from(index) else { return };
    guard((), || pass.set_bind_group(index, &*group, &[]))
}

unsafe extern "C" fn wgpuComputePassDispatchWorkgroups(pass: i64, x: i32, y: i32, z: i32) {
    let pass = get!(pass, Cpass);
    guard((), || pass.dispatch_workgroups(x as u32, y as u32, z as u32))
}

/// `pass` ended and freed; its encoder unlocked.
unsafe extern "C" fn wgpuComputePassEnd(pass: i64) {
    let Some(b) = take(pass) else { return };
    if !matches!(*b, Obj::Cpass(_)) {
        return give_back(b);
    }
    guard((), || drop(b))
}

unsafe extern "C" fn wgpuCommandEncoderCopyBufferToBuffer(enc: i64, src: i64, so: i64, dst: i64, dof: i64, n: i64) {
    let enc = get!(enc, Encoder);
    let src = get!(src, Buffer);
    let dst = get!(dst, Buffer);
    let (Ok(so), Ok(dof), Ok(n)) = (u64::try_from(so), u64::try_from(dof), u64::try_from(n)) else { return };
    guard((), || enc.copy_buffer_to_buffer(src, so, dst, dof, n))
}

/// A render pass on `enc` to `view`, cleared to opaque black where `clear` is
/// nonzero and loaded otherwise.
unsafe extern "C" fn wgpuCommandEncoderBeginRenderPass(enc: i64, view: i64, clear: i32) -> i64 {
    let enc = get!(enc, Encoder);
    let view = get!(view, View);
    guard(0, || {
        let load = if clear != 0 { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load };
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view,
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
        boxed(Obj::Rpass(enc.begin_render_pass(&d).forget_lifetime()))
    })
}

unsafe extern "C" fn wgpuRenderPassSetPipeline(pass: i64, pipe: i64) {
    let pass = get!(pass, Rpass);
    let pipe = get!(pipe, Rpipe);
    guard((), || pass.set_pipeline(pipe))
}

unsafe extern "C" fn wgpuRenderPassSetBindGroup(pass: i64, index: i32, group: i64) {
    let pass = get!(pass, Rpass);
    let group = get!(group, BindGroup);
    let Ok(index) = u32::try_from(index) else { return };
    guard((), || pass.set_bind_group(index, &*group, &[]))
}

unsafe extern "C" fn wgpuRenderPassDraw(pass: i64, vertices: i32, instances: i32, first_vertex: i32, first_instance: i32) {
    let pass = get!(pass, Rpass);
    let (v, i, fv, fi) = (vertices as u32, instances as u32, first_vertex as u32, first_instance as u32);
    guard((), || pass.draw(fv..fv.wrapping_add(v), fi..fi.wrapping_add(i)))
}

/// `pass` ended and freed; its encoder unlocked.
unsafe extern "C" fn wgpuRenderPassEnd(pass: i64) {
    let Some(b) = take(pass) else { return };
    if !matches!(*b, Obj::Rpass(_)) {
        return give_back(b);
    }
    guard((), || drop(b))
}

/// `enc` finished and freed: its commands, as a command buffer.
unsafe extern "C" fn wgpuCommandEncoderFinish(enc: i64) -> i64 {
    let Some(b) = take(enc) else { return 0 };
    let Obj::Encoder(e) = *b else {
        give_back(b);
        return 0;
    };
    guard(0, || boxed(Obj::CmdBuf(e.finish())))
}

// ---------------------------------------------------------------------------
// The queue
// ---------------------------------------------------------------------------

/// `cmds` run on `queue`, and freed.
unsafe extern "C" fn wgpuQueueSubmit(queue: i64, cmds: i64) {
    let queue = get!(queue, Queue);
    let Some(b) = take(cmds) else { return };
    let Obj::CmdBuf(c) = *b else { return give_back(b) };
    guard((), || {
        queue.submit(std::iter::once(c));
    })
}

/// The `n` bytes at `src` written to `buf` from `off`, before the next submit.
unsafe extern "C" fn wgpuQueueWriteBuffer(queue: i64, buf: i64, off: i64, src: *const u8, n: i64) {
    let queue = get!(queue, Queue);
    let buf = get!(buf, Buffer);
    let (Ok(off), Ok(n)) = (u64::try_from(off), usize::try_from(n)) else { return };
    if src.is_null() && n > 0 {
        return;
    }
    let data = if n == 0 { &[][..] } else { std::slice::from_raw_parts(src, n) };
    guard((), || queue.write_buffer(buf, off, data))
}

/// `n` bytes of `buf` from `off` copied to `dst` through a read mapping,
/// waiting for the queue: `0`, or `-1` when wgpu refuses the mapping.
unsafe extern "C" fn wgpuBufferRead(dev: i64, buf: i64, off: i64, dst: *mut u8, n: i64) -> i32 {
    let dev = match obj(dev) {
        Some(Obj::Device((d, _))) => d,
        _ => return -1,
    };
    let buf = match obj(buf) {
        Some(Obj::Buffer(b)) => b,
        _ => return -1,
    };
    let (Ok(off), Ok(n)) = (u64::try_from(off), u64::try_from(n)) else { return -1 };
    if dst.is_null() || n == 0 {
        return -1;
    }
    guard(-1, || {
        let ok = Arc::new(AtomicBool::new(false));
        let done = ok.clone();
        let slice = buf.slice(off..off + n);
        slice.map_async(wgpu::MapMode::Read, move |r| done.store(r.is_ok(), Ordering::SeqCst));
        if dev.poll(wgpu::PollType::wait_indefinitely()).is_err() || !ok.load(Ordering::SeqCst) {
            return -1;
        }
        {
            let view = slice.get_mapped_range();
            std::ptr::copy_nonoverlapping(view.as_ptr(), dst, n as usize);
        }
        buf.unmap();
        0
    })
}

// ---------------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------------

/// A surface of the window `win` (the window library's handle), which keeps
/// the window alive; `0` when wgpu cannot make one.
unsafe extern "C" fn wgpuInstanceCreateSurface(inst: i64, win: i64) -> i64 {
    let inst = get!(inst, Instance);
    let Some(w) = window::shared(win) else { return 0 };
    guard(0, || match inst.create_surface(w) {
        Ok(s) => boxed(Obj::Surface(s)),
        Err(_) => 0,
    })
}

/// `surf` configured for rendering on `dev`, `width` by `height` in `format`.
unsafe extern "C" fn wgpuSurfaceConfigure(surf: i64, dev: i64, fmt: i32, width: i32, height: i32) {
    let surf = get!(surf, Surface);
    let dev = &get!(dev, Device).0;
    let (Some(fmt), Ok(width), Ok(height)) = (format(fmt), u32::try_from(width), u32::try_from(height)) else {
        return;
    };
    guard((), || {
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
        surf.configure(dev, &c)
    })
}

/// `surf`'s next texture, or `0` when it has none ready: timed out,
/// occluded, outdated or lost.
unsafe extern "C" fn wgpuSurfaceGetCurrentTexture(surf: i64) -> i64 {
    let surf = get!(surf, Surface);
    guard(0, || match surf.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
            boxed(Obj::Texture(t))
        }
        _ => 0,
    })
}

unsafe extern "C" fn wgpuSurfaceTextureCreateView(tex: i64) -> i64 {
    let tex = get!(tex, Texture);
    guard(0, || boxed(Obj::View(tex.texture.create_view(&wgpu::TextureViewDescriptor::default()))))
}

/// `tex` shown on its surface, and freed.
unsafe extern "C" fn wgpuSurfaceTexturePresent(tex: i64) {
    let Some(b) = take(tex) else { return };
    let Obj::Texture(t) = *b else { return give_back(b) };
    guard((), || t.present())
}

// ---------------------------------------------------------------------------
// Releases
// ---------------------------------------------------------------------------

macro_rules! release {
    ($($name:ident $v:ident)*) => {
        $(
            unsafe extern "C" fn $name(h: i64) {
                let Some(b) = take(h) else { return };
                if !matches!(*b, Obj::$v(..)) {
                    return give_back(b);
                }
                guard((), || drop(b))
            }
        )*
    };
}

release!(
    wgpuInstanceRelease Instance wgpuAdapterRelease Adapter wgpuDeviceRelease Device
    wgpuQueueRelease Queue wgpuBufferRelease Buffer wgpuShaderModuleRelease Shader
    wgpuBindGroupLayoutRelease Bgl wgpuPipelineLayoutRelease Playout
    wgpuComputePipelineRelease Cpipe wgpuBindGroupRelease BindGroup
    wgpuCommandEncoderRelease Encoder wgpuComputePassEncoderRelease Cpass
    wgpuCommandBufferRelease CmdBuf wgpuSurfaceRelease Surface wgpuTextureRelease Texture
    wgpuTextureViewRelease View wgpuRenderPipelineRelease Rpipe wgpuRenderPassEncoderRelease Rpass
    wgpuErrorScopeRelease Scope
);

/// The address of `symbol`, for the functions a program calls itself
/// (`Ext.wgpu`).
pub(crate) fn linked(symbol: &str) -> Option<usize> {
    macro_rules! table {
        ($($f:ident)*) => {
            match symbol {
                $(stringify!($f) => Some($f as usize),)*
                _ => None,
            }
        };
    }
    table!(
        wgpuCreateInstance wgpuInstanceRequestAdapter wgpuAdapterRequestDevice wgpuDeviceGetQueue
        wgpuDeviceCreateBuffer wgpuDeviceCreateShaderModule wgpuDeviceCreateBindGroupLayout
        wgpuDeviceCreatePipelineLayout wgpuDeviceCreateComputePipeline wgpuDeviceCreateBindGroup
        wgpuDeviceCreateRenderPipeline wgpuRenderPipelineGetBindGroupLayout
        wgpuDevicePushErrorScope wgpuErrorScopePop
        wgpuDeviceCreateCommandEncoder wgpuCommandEncoderBeginComputePass wgpuComputePassSetPipeline
        wgpuComputePassSetBindGroup wgpuComputePassDispatchWorkgroups wgpuComputePassEnd
        wgpuCommandEncoderCopyBufferToBuffer wgpuCommandEncoderBeginRenderPass wgpuRenderPassSetPipeline
        wgpuRenderPassSetBindGroup wgpuRenderPassDraw wgpuRenderPassEnd wgpuCommandEncoderFinish
        wgpuQueueSubmit wgpuQueueWriteBuffer wgpuBufferRead
        wgpuInstanceCreateSurface wgpuSurfaceConfigure wgpuSurfaceGetCurrentTexture
        wgpuSurfaceTextureCreateView wgpuSurfaceTexturePresent
        wgpuInstanceRelease wgpuAdapterRelease wgpuDeviceRelease wgpuQueueRelease wgpuBufferRelease
        wgpuShaderModuleRelease wgpuBindGroupLayoutRelease wgpuPipelineLayoutRelease
        wgpuComputePipelineRelease wgpuBindGroupRelease wgpuCommandEncoderRelease
        wgpuComputePassEncoderRelease wgpuCommandBufferRelease wgpuSurfaceRelease wgpuTextureRelease
        wgpuTextureViewRelease wgpuRenderPipelineRelease wgpuRenderPassEncoderRelease wgpuErrorScopeRelease
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handle of another kind is refused, and the object it holds stays the
    /// program's: releasing it by the wrong kind frees nothing.
    #[test]
    fn a_handle_of_another_kind_is_refused() {
        unsafe {
            let inst = wgpuCreateInstance();
            assert_ne!(inst, 0);
            assert_eq!(wgpuDeviceCreateBuffer(inst, 140, 64), 0);
            assert_eq!(wgpuDeviceCreateCommandEncoder(inst), 0);
            assert_eq!(wgpuCommandEncoderFinish(inst), 0);
            assert_eq!(wgpuErrorScopePop(inst), -1);
            wgpuBufferRelease(inst);
            wgpuQueueSubmit(inst, inst);
            assert!(matches!(obj(inst), Some(Obj::Instance(_))));
            wgpuInstanceRelease(inst);
            assert_eq!(wgpuInstanceRequestAdapter(0), 0);
        }
    }
}
