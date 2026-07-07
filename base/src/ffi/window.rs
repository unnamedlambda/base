use std::collections::VecDeque;
use std::time::Duration;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, FragmentState, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, ShaderModuleDescriptor, ShaderSource, StoreOp,
    Surface, SurfaceConfiguration, SurfaceError, TextureFormat, TextureUsages,
    TextureViewDescriptor, VertexState,
};
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::platform::pump_events::EventLoopExtPumpEvents;
use winit::window::Window;

use super::wgpu::{cached_gpu_handles, CraneliftGpuContext, GpuHandles};
use super::{clear_ctx_slot, read_ctx_mut, write_ctx_slot};

// Event record wire format: four i64s, mirrored on the CLIF side.
const EVENT_BYTES: usize = 32;
const EVENT_CLOSE: i64 = 1;
const EVENT_RESIZE: i64 = 2;
const EVENT_KEY_DOWN: i64 = 3;
const EVENT_KEY_UP: i64 = 4;
const EVENT_MOUSE_MOVE: i64 = 5;
const EVENT_MOUSE_DOWN: i64 = 6;
const EVENT_MOUSE_UP: i64 = 7;

fn map_mouse_button(button: winit::event::MouseButton) -> i64 {
    use winit::event::MouseButton as B;
    match button {
        B::Left => 1,
        B::Right => 2,
        B::Middle => 3,
        _ => 0,
    }
}

/// Portable, keyboard-layout-independent ids for physical keys — the game owns
/// what they mean. Unmapped keys return 0 (dropped). Extend as games need more;
/// only the numbering must stay in sync with the CLIF-side key constants.
fn map_key_code(code: winit::keyboard::KeyCode) -> i64 {
    use winit::keyboard::KeyCode as K;
    match code {
        K::Escape => 1,
        K::Space => 2,
        K::ArrowLeft => 3,
        K::ArrowRight => 4,
        K::ArrowUp => 5,
        K::ArrowDown => 6,
        K::KeyW => 10,
        K::KeyA => 11,
        K::KeyS => 12,
        K::KeyD => 13,
        _ => 0,
    }
}

#[derive(Clone, Copy)]
struct EventRecord {
    kind: i64,
    a: i64,
    b: i64,
    c: i64,
}

struct WindowState {
    // `surface` borrows `window`'s raw handles via the 'static transmute in
    // recreate_surface. Rust drops fields in declaration order, so `surface`
    // MUST be declared before `window` to be dropped first — otherwise the
    // window would be freed while the surface still references it.
    surface: Surface<'static>,
    // Never read after construction, but keeps the OS window alive for `surface`.
    #[allow(dead_code)]
    window: Window,
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    // Derived from the shader via auto-layout, not declared here (no sync).
    bind_group_layout: wgpu::BindGroupLayout,
}

pub(crate) struct CraneliftWindowContext {
    event_loop: EventLoop<()>,
    gpu: GpuHandles,
    window: Option<WindowState>,
    pending: VecDeque<EventRecord>,
}

pub(crate) unsafe extern "C" fn cl_window_init(ctx_slot_ptr: *mut *mut CraneliftWindowContext) {
    let raw = std::panic::catch_unwind(|| {
        let event_loop = EventLoop::new().expect("failed to create winit event loop");
        let ctx = Box::new(CraneliftWindowContext {
            event_loop,
            gpu: cached_gpu_handles(),
            window: None,
            pending: VecDeque::new(),
        });
        Box::into_raw(ctx)
    })
    .unwrap_or(std::ptr::null_mut());
    let _ = write_ctx_slot(ctx_slot_ptr, raw);
}

/// Open the single window and build the blit pipeline from the Lean-supplied
/// WGSL at `blit_ptr` (see the module docs for the shader contract).
pub(crate) unsafe extern "C" fn cl_window_open(
    ctx_ptr: *mut CraneliftWindowContext,
    width: i64,
    height: i64,
    title_ptr: *const u8,
    title_len: i64,
    blit_ptr: *const u8,
    blit_len: i64,
) -> i32 {
    if width <= 0
        || height <= 0
        || title_len < 0
        || title_ptr.is_null()
        || blit_len <= 0
        || blit_ptr.is_null()
    {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftWindowContext>(ctx_ptr) else {
            return -1;
        };
        if ctx.window.is_some() {
            return -1; // one window per context
        }
        let title = match std::str::from_utf8(std::slice::from_raw_parts(
            title_ptr,
            title_len as usize,
        )) {
            Ok(s) => s,
            Err(_) => return -1,
        };
        let blit_src =
            match std::str::from_utf8(std::slice::from_raw_parts(blit_ptr, blit_len as usize)) {
                Ok(s) => s,
                Err(_) => return -1,
            };
        let attrs = Window::default_attributes()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width as f64, height as f64));
        #[allow(deprecated)]
        let window = match ctx.event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(_) => return -1,
        };

        // Surface lives as long as the window (kept together in WindowState,
        // dropped before it). See recreate_surface for the 'static rationale.
        let Some(surface) = recreate_surface(&ctx.gpu, &window) else {
            return -1;
        };

        let size = window.inner_size();
        let caps = surface.get_capabilities(&ctx.gpu.adapter);
        // Prefer a non-sRGB format so the fragment writes the game's 0-255 bytes
        // straight through without a gamma re-encode.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or_else(|| caps.formats[0]);
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo, // vsync: paces the loop, no tearing
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&ctx.gpu.device, &config);

        let (pipeline, bind_group_layout) =
            match create_blit_pipeline(&ctx.gpu.device, format, blit_src) {
                Some(p) => p,
                None => return -1,
            };

        ctx.window = Some(WindowState {
            surface,
            window,
            config,
            pipeline,
            bind_group_layout,
        });
        0
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_window_poll(
    ctx_ptr: *mut CraneliftWindowContext,
    events_ptr: *mut u8,
    max_events: i32,
) -> i32 {
    if max_events < 0 || events_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftWindowContext>(ctx_ptr) else {
            return -1;
        };
        pump_events(ctx);
        drain_events(&mut ctx.pending, events_ptr, max_events as usize) as i32
    }))
    .unwrap_or(-1)
}

/// Blit a game framebuffer (a wgpu storage buffer of packed RGBA u32, produced
/// by the compute FFI on the same device) to the window, zero-copy. The
/// framebuffer size is baked into the blit shader, so no dimensions are passed.
///
/// `gpu_ctx_ptr` is the compute context that owns `buf_id`; the same physical
/// device backs both contexts, so the buffer is bound directly into the blit
/// pipeline with no host transfer.
pub(crate) unsafe extern "C" fn cl_window_present_gpu_buffer(
    ctx_ptr: *mut CraneliftWindowContext,
    gpu_ctx_ptr: *mut CraneliftGpuContext,
    buf_id: i32,
) -> i32 {
    if buf_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftWindowContext>(ctx_ptr) else {
            return -1;
        };
        pump_events(ctx);
        let Some(gpu) = read_ctx_mut::<CraneliftGpuContext>(gpu_ctx_ptr) else {
            return -1;
        };
        // Ensure the render dispatch that filled the buffer is submitted before
        // we sample it (same queue => ordered).
        gpu.flush_pending();
        let Some(pixel_buf) = gpu.buffer(buf_id as usize) else {
            return -1;
        };
        let Some(state) = ctx.window.as_mut() else {
            return -1;
        };

        let bind_group = ctx.gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("base_window_blit_bg"),
            layout: &state.bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(pixel_buf.as_entire_buffer_binding()),
            }],
        });

        let frame = match state.surface.get_current_texture() {
            Ok(f) => f,
            Err(SurfaceError::Lost | SurfaceError::Outdated) => {
                state.surface.configure(&ctx.gpu.device, &state.config);
                match state.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => return -1,
                }
            }
            Err(SurfaceError::Timeout) => return 0,
            Err(SurfaceError::OutOfMemory) => return -1,
        };
        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        let mut encoder = ctx
            .gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("base_window_present"),
            });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("base_window_present_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&state.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        ctx.gpu.queue.submit(Some(encoder.finish()));
        frame.present();
        0
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_window_cleanup(ctx_slot_ptr: *mut *mut CraneliftWindowContext) {
    let ctx_ptr = clear_ctx_slot::<CraneliftWindowContext>(ctx_slot_ptr);
    if !ctx_ptr.is_null() {
        drop(Box::from_raw(ctx_ptr));
    }
}

/// Pop up to `max_events` records off `pending` into the caller's 32-byte-strided
/// event buffer, returning the number written. Display-independent (unit-tested).
fn drain_events(
    pending: &mut VecDeque<EventRecord>,
    events_ptr: *mut u8,
    max_events: usize,
) -> usize {
    let count = pending.len().min(max_events);
    for i in 0..count {
        let ev = pending.pop_front().unwrap();
        unsafe {
            let base = events_ptr.add(i * EVENT_BYTES);
            std::ptr::write_unaligned(base as *mut i64, ev.kind);
            std::ptr::write_unaligned(base.add(8) as *mut i64, ev.a);
            std::ptr::write_unaligned(base.add(16) as *mut i64, ev.b);
            std::ptr::write_unaligned(base.add(24) as *mut i64, ev.c);
        }
    }
    count
}

fn recreate_surface(gpu: &GpuHandles, window: &Window) -> Option<Surface<'static>> {
    // SAFETY: the surface is stored alongside the window in WindowState and
    // dropped before it, so the raw handles it borrows stay valid.
    let surface = gpu.instance.create_surface(window).ok()?;
    Some(unsafe { std::mem::transmute::<Surface<'_>, Surface<'static>>(surface) })
}

/// Build the fullscreen-triangle blit pipeline from the Lean-supplied WGSL.
/// Uses wgpu auto-layout (`layout: None`) so the bind-group layout is reflected
/// from the shader rather than declared here; returns that derived layout.
fn create_blit_pipeline(
    device: &wgpu::Device,
    format: TextureFormat,
    shader_src: &str,
) -> Option<(RenderPipeline, wgpu::BindGroupLayout)> {
    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("base_window_blit_shader"),
        source: ShaderSource::Wgsl(shader_src.into()),
    });
    let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("base_window_blit_pipeline"),
        layout: None, // auto: derive the bind-group layout from the shader
        vertex: VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: PipelineCompilationOptions::default(),
        },
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(ColorTargetState {
                format,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
            compilation_options: PipelineCompilationOptions::default(),
        }),
        multiview: None,
    });
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    Some((pipeline, bind_group_layout))
}

fn pump_events(ctx: &mut CraneliftWindowContext) {
    let pending = &mut ctx.pending;
    let window_state = &mut ctx.window;
    let device = &ctx.gpu.device;
    #[allow(deprecated)]
    let _ = ctx
        .event_loop
        .pump_events(Some(Duration::ZERO), |event, _target| {
            let Event::WindowEvent { event, .. } = event else {
                return;
            };
            match event {
                WindowEvent::CloseRequested => {
                    pending.push_back(EventRecord {
                        kind: EVENT_CLOSE,
                        a: 0,
                        b: 0,
                        c: 0,
                    });
                }
                WindowEvent::Resized(size) => {
                    if let Some(state) = window_state.as_mut() {
                        if size.width > 0 && size.height > 0 {
                            state.config.width = size.width;
                            state.config.height = size.height;
                            state.surface.configure(device, &state.config);
                        }
                    }
                    pending.push_back(EventRecord {
                        kind: EVENT_RESIZE,
                        a: i64::from(size.width),
                        b: i64::from(size.height),
                        c: 0,
                    });
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    let winit::keyboard::PhysicalKey::Code(code) = event.physical_key else {
                        return;
                    };
                    let key = map_key_code(code);
                    if key == 0 {
                        return;
                    }
                    pending.push_back(EventRecord {
                        kind: if event.state == ElementState::Pressed {
                            EVENT_KEY_DOWN
                        } else {
                            EVENT_KEY_UP
                        },
                        a: key,
                        b: 0,
                        c: 0,
                    });
                }
                WindowEvent::CursorMoved { position, .. } => {
                    pending.push_back(EventRecord {
                        kind: EVENT_MOUSE_MOVE,
                        a: position.x.round() as i64,
                        b: position.y.round() as i64,
                        c: 0,
                    });
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    pending.push_back(EventRecord {
                        kind: if state == ElementState::Pressed {
                            EVENT_MOUSE_DOWN
                        } else {
                            EVENT_MOUSE_UP
                        },
                        a: map_mouse_button(button),
                        b: 0,
                        c: 0,
                    });
                }
                _ => {}
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_events_layout_and_clamp() {
        let mut pending: VecDeque<EventRecord> = VecDeque::new();
        pending.push_back(EventRecord { kind: EVENT_KEY_DOWN, a: 4, b: 0, c: 0 });
        pending.push_back(EventRecord { kind: EVENT_CLOSE, a: 0, b: 0, c: 0 });
        pending.push_back(EventRecord { kind: EVENT_RESIZE, a: 800, b: 600, c: 0 });

        // Buffer for 2 events; max_events clamps to 2, leaving 1 pending.
        let mut buf = vec![0u8; 2 * EVENT_BYTES];
        let n = drain_events(&mut pending, buf.as_mut_ptr(), 2);
        assert_eq!(n, 2);
        assert_eq!(pending.len(), 1, "unwritten events stay queued");

        let read = |off: usize| i64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        // Record 0: kind at +0, a at +8.
        assert_eq!(read(0), EVENT_KEY_DOWN);
        assert_eq!(read(8), 4);
        // Record 1 begins at +EVENT_BYTES.
        assert_eq!(read(EVENT_BYTES), EVENT_CLOSE);
        // The leftover is the resize, still at the front.
        assert_eq!(pending.front().unwrap().kind, EVENT_RESIZE);
    }

    // The arg-validation guards run before any window/display work, so these
    // are safe headless. A null ctx is fine: invalid args short-circuit before
    // the ctx is read, and valid args then hit the null-ctx check.
    #[test]
    fn null_ctx_returns_neg1() {
        let title = b"x";
        let blit = b"shader";
        let mut buf = [0u8; EVENT_BYTES];
        unsafe {
            assert_eq!(
                cl_window_open(
                    std::ptr::null_mut(), 640, 360,
                    title.as_ptr(), 1, blit.as_ptr(), 6
                ),
                -1
            );
            assert_eq!(cl_window_poll(std::ptr::null_mut(), buf.as_mut_ptr(), 1), -1);
            assert_eq!(
                cl_window_present_gpu_buffer(std::ptr::null_mut(), std::ptr::null_mut(), 0),
                -1
            );
        }
    }

    #[test]
    fn invalid_args_return_neg1() {
        let title = b"x";
        let blit = b"shader";
        let mut buf = [0u8; EVENT_BYTES];
        let n = std::ptr::null_mut();
        unsafe {
            assert_eq!(cl_window_open(n, 0, 360, title.as_ptr(), 1, blit.as_ptr(), 6), -1); // width<=0
            assert_eq!(cl_window_open(n, 640, 0, title.as_ptr(), 1, blit.as_ptr(), 6), -1); // height<=0
            assert_eq!(cl_window_open(n, 640, 360, title.as_ptr(), -1, blit.as_ptr(), 6), -1); // title_len<0
            assert_eq!(cl_window_open(n, 640, 360, title.as_ptr(), 1, blit.as_ptr(), 0), -1); // blit_len<=0
            assert_eq!(cl_window_open(n, 640, 360, std::ptr::null(), 1, blit.as_ptr(), 6), -1); // null title
            assert_eq!(cl_window_open(n, 640, 360, title.as_ptr(), 1, std::ptr::null(), 6), -1); // null blit
            assert_eq!(cl_window_poll(n, buf.as_mut_ptr(), -1), -1); // negative max
            assert_eq!(cl_window_poll(n, std::ptr::null_mut(), 1), -1); // null events buf
            assert_eq!(cl_window_present_gpu_buffer(n, std::ptr::null_mut(), -1), -1); // buf_id<0
        }
    }
}
