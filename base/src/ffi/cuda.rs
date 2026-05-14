use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use super::{clear_ctx_slot, read_cstr_ptr, read_ctx_mut, read_ctx_ref, write_ctx_slot};

pub(crate) struct CraneliftCudaContext {
    device: std::sync::Arc<cudarc::driver::CudaDevice>,
    state: Mutex<CraneliftCudaState>,
}

struct CraneliftCudaState {
    buffers: Vec<Option<cudarc::driver::CudaSlice<u8>>>,
    default_blas: Option<cudarc::cublas::CudaBlas>,
    stream_blas: HashMap<i32, cudarc::cublas::CudaBlas>,
    streams: Vec<Option<CudaOwnedStream>>,
    events: Vec<Option<CudaOwnedEvent>>,
    graphs: Vec<Option<CudaOwnedGraphExec>>,
    pinned_buffers: Vec<Option<CudaPinnedHostBuffer>>,
    main_kernel_cache: std::collections::HashMap<*const u8, RawCudaKernelCacheEntry>,
    named_kernel_cache: std::collections::HashMap<(*const u8, *const u8), RawCudaKernelCacheEntry>,
}

struct RawCudaKernelCacheEntry {
    module: cudarc::driver::sys::CUmodule,
    function: cudarc::driver::sys::CUfunction,
}

struct CudaOwnedStream {
    raw: cudarc::driver::sys::CUstream,
}

struct CudaOwnedEvent {
    raw: cudarc::driver::sys::CUevent,
}

struct CudaOwnedGraphExec {
    raw: cudarc::driver::sys::CUgraphExec,
}

struct CudaPinnedHostBuffer {
    ptr: *mut std::ffi::c_void,
    _size: usize,
}

unsafe impl Send for CudaPinnedHostBuffer {}
unsafe impl Sync for CudaPinnedHostBuffer {}

impl Drop for CudaOwnedStream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cudarc::driver::result::stream::destroy(self.raw) };
        }
    }
}

impl Drop for CudaOwnedEvent {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cudarc::driver::result::event::destroy(self.raw) };
        }
    }
}

impl Drop for CudaOwnedGraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = unsafe { cudarc::driver::sys::lib().cuGraphExecDestroy(self.raw) }.result();
        }
    }
}

impl Drop for CudaPinnedHostBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let _ = unsafe { cudarc::driver::sys::lib().cuMemFreeHost(self.ptr) }.result();
            self.ptr = std::ptr::null_mut();
        }
    }
}

thread_local! {
    static CUDA_BOUND_CTX: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn bind_cuda_ctx_if_needed(ctx: &CraneliftCudaContext) -> bool {
    let key = ctx as *const CraneliftCudaContext as usize;
    CUDA_BOUND_CTX.with(|bound| {
        if bound.get() == key {
            return true;
        }
        if ctx.device.bind_to_thread().is_err() {
            return false;
        }
        bound.set(key);
        true
    })
}

impl Drop for CraneliftCudaContext {
    fn drop(&mut self) {
        let _ = self.device.bind_to_thread();
        // Invalidate per-thread cache so the freed address isn't mistaken for a live context.
        let key = self as *const CraneliftCudaContext as usize;
        CUDA_BOUND_CTX.with(|bound| {
            if bound.get() == key {
                bound.set(0);
            }
        });
        let Ok(state) = self.state.get_mut() else {
            return;
        };
        for (_, entry) in std::mem::take(&mut state.main_kernel_cache) {
            let _ = unsafe { cudarc::driver::result::module::unload(entry.module) };
        }
        for (_, entry) in std::mem::take(&mut state.named_kernel_cache) {
            let _ = unsafe { cudarc::driver::result::module::unload(entry.module) };
        }
        state.stream_blas.clear();
        state.streams.clear();
        state.events.clear();
        state.graphs.clear();
        state.pinned_buffers.clear();
    }
}

fn lock_cuda_state(ctx: &CraneliftCudaContext) -> Result<MutexGuard<'_, CraneliftCudaState>, ()> {
    ctx.state.lock().map_err(|_| ())
}

fn resolve_cuda_stream(
    device: &cudarc::driver::CudaDevice,
    state: &CraneliftCudaState,
    stream_id: i32,
) -> Option<cudarc::driver::sys::CUstream> {
    if stream_id < 0 {
        return Some(*device.cu_stream());
    }
    let sid = stream_id as usize;
    let stream = state.streams.get(sid)?.as_ref()?;
    Some(stream.raw)
}

fn resolve_cuda_event(
    state: &CraneliftCudaState,
    event_id: i32,
) -> Option<cudarc::driver::sys::CUevent> {
    if event_id < 0 {
        return None;
    }
    let eid = event_id as usize;
    let event = state.events.get(eid)?.as_ref()?;
    Some(event.raw)
}

fn resolve_cuda_graph_exec(
    state: &CraneliftCudaState,
    graph_id: i32,
) -> Option<cudarc::driver::sys::CUgraphExec> {
    if graph_id < 0 {
        return None;
    }
    let gid = graph_id as usize;
    let graph = state.graphs.get(gid)?.as_ref()?;
    Some(graph.raw)
}

unsafe fn cuda_buffer_device_ptr(
    state: &CraneliftCudaState,
    buf_id: i32,
) -> Option<cudarc::driver::sys::CUdeviceptr> {
    use cudarc::driver::DevicePtr;

    if buf_id < 0 {
        return None;
    }
    let bid = buf_id as usize;
    let buf = state.buffers.get(bid)?.as_ref()?;
    Some(*buf.device_ptr())
}

fn load_raw_cuda_main_kernel(
    ctx: &CraneliftCudaContext,
    state: &mut CraneliftCudaState,
    kernel_ptr: *const u8,
) -> Result<cudarc::driver::sys::CUfunction, ()> {
    if let Some(entry) = state.main_kernel_cache.get(&kernel_ptr) {
        return Ok(entry.function);
    }
    if !bind_cuda_ctx_if_needed(ctx) {
        return Err(());
    }
    let ptx_src = unsafe { read_cstr_ptr(kernel_ptr) };
    let ptx_cstr = std::ffi::CString::new(ptx_src).map_err(|_| ())?;
    let module = unsafe {
        cudarc::driver::result::module::load_data(ptx_cstr.as_ptr() as *const std::ffi::c_void)
    }
    .map_err(|_| ())?;
    let function = unsafe {
        cudarc::driver::result::module::get_function(
            module,
            std::ffi::CString::new("main").expect("main CString"),
        )
    }
    .map_err(|_| ())?;
    state
        .main_kernel_cache
        .insert(kernel_ptr, RawCudaKernelCacheEntry { module, function });
    Ok(function)
}

fn load_raw_cuda_named_kernel(
    ctx: &CraneliftCudaContext,
    state: &mut CraneliftCudaState,
    kernel_ptr: *const u8,
    name_ptr: *const u8,
) -> Result<cudarc::driver::sys::CUfunction, ()> {
    let key = (kernel_ptr, name_ptr);
    if let Some(entry) = state.named_kernel_cache.get(&key) {
        return Ok(entry.function);
    }
    if !bind_cuda_ctx_if_needed(ctx) {
        return Err(());
    }
    let ptx_src = unsafe { read_cstr_ptr(kernel_ptr) };
    let func_name = unsafe { read_cstr_ptr(name_ptr) };
    let ptx_cstr = std::ffi::CString::new(ptx_src).map_err(|_| ())?;
    let func_cstr = std::ffi::CString::new(func_name).map_err(|_| ())?;
    let module = unsafe {
        cudarc::driver::result::module::load_data(ptx_cstr.as_ptr() as *const std::ffi::c_void)
    }
    .map_err(|_| ())?;
    let function = unsafe { cudarc::driver::result::module::get_function(module, func_cstr) }
        .map_err(|_| ())?;
    state
        .named_kernel_cache
        .insert(key, RawCudaKernelCacheEntry { module, function });
    Ok(function)
}

unsafe fn launch_raw_cuda_kernel(
    state: &CraneliftCudaState,
    function: cudarc::driver::sys::CUfunction,
    n_bufs: i32,
    bind_ptr: *const u8,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
    stream: cudarc::driver::sys::CUstream,
) -> i32 {
    let bind_base = bind_ptr;
    let mut dev_ptrs: Vec<cudarc::driver::sys::CUdeviceptr> = Vec::with_capacity(n_bufs as usize);
    for i in 0..n_bufs as usize {
        let buf_id = std::ptr::read_unaligned(bind_base.add(i * 4) as *const i32);
        let Some(dev_ptr) = cuda_buffer_device_ptr(state, buf_id) else {
            return -1;
        };
        dev_ptrs.push(dev_ptr);
    }
    let mut arg_ptrs: Vec<*mut std::ffi::c_void> = dev_ptrs
        .iter_mut()
        .map(|p| p as *mut cudarc::driver::sys::CUdeviceptr as *mut std::ffi::c_void)
        .collect();
    if let Err(e) = unsafe {
        cudarc::driver::result::launch_kernel(
            function,
            (grid_x as u32, grid_y as u32, grid_z as u32),
            (block_x as u32, block_y as u32, block_z as u32),
            0,
            stream,
            &mut arg_ptrs,
        )
    } {
        eprintln!("cl_cuda_launch: kernel launch failed: {:?}", e);
        return -1;
    }
    0
}

fn cached_cuda_device() -> std::sync::Arc<cudarc::driver::CudaDevice> {
    use std::sync::OnceLock;
    static CUDA: OnceLock<std::sync::Arc<cudarc::driver::CudaDevice>> = OnceLock::new();
    CUDA.get_or_init(|| cudarc::driver::CudaDevice::new(0).expect("Failed to create CUDA device"))
        .clone()
}

pub(crate) unsafe extern "C" fn cl_cuda_init(ctx_slot_ptr: *mut *mut CraneliftCudaContext) {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = cached_cuda_device();
        let cuda_ctx = Box::new(CraneliftCudaContext {
            device,
            state: Mutex::new(CraneliftCudaState {
                buffers: Vec::new(),
                default_blas: None,
                stream_blas: HashMap::new(),
                streams: Vec::new(),
                events: Vec::new(),
                graphs: Vec::new(),
                pinned_buffers: Vec::new(),
                main_kernel_cache: std::collections::HashMap::new(),
                named_kernel_cache: std::collections::HashMap::new(),
            }),
        });
        let _ = write_ctx_slot(ctx_slot_ptr, Box::into_raw(cuda_ctx));
    }))
    .expect("cl_cuda_init panicked");
}

pub(crate) unsafe extern "C" fn cl_cuda_create_buffer(ctx_ptr: *mut CraneliftCudaContext, size: i64) -> i32 {
    if size <= 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        match ctx.device.alloc_zeros::<u8>(size as usize) {
            Ok(buf) => {
                let idx = state.buffers.len() as i32;
                state.buffers.push(Some(buf));
                idx
            }
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_upload(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    src_ptr: *const u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        let Some(buf) = state.buffers[bid].as_mut() else {
            return -1;
        };
        match ctx.device.htod_sync_copy_into(data, buf) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_upload_ptr(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    src_ptr: *const u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        let Some(buf) = state.buffers[bid].as_mut() else {
            return -1;
        };
        match ctx.device.htod_sync_copy_into(data, buf) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_upload_ptr_offset(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    buf_offset: i64,
    src_ptr: *const u8,
    size: i64,
) -> i32 {
    use cudarc::driver::DevicePtr;

    if buf_id < 0 || buf_offset < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        let Some(buf) = state.buffers[bid].as_mut() else {
            return -1;
        };
        let start = buf_offset as usize;
        let byte_len = size as usize;
        // Cuda buffers are allocated as CudaSlice<u8>, so len is in bytes.
        let total_len = {
            use cudarc::driver::DeviceSlice;
            buf.len()
        };
        if start.saturating_add(byte_len) > total_len {
            return -1;
        }
        let dst = (*buf.device_ptr()).saturating_add(buf_offset as u64);
        match unsafe { cudarc::driver::result::memcpy_htod_sync(dst, data) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_stream_create(ctx_ptr: *mut CraneliftCudaContext) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        match cudarc::driver::result::stream::create(
            cudarc::driver::result::stream::StreamKind::NonBlocking,
        ) {
            Ok(raw) => {
                let fence_event = match cudarc::driver::result::event::create(
                    cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
                ) {
                    Ok(event) => event,
                    Err(_) => {
                        let _ = unsafe { cudarc::driver::result::stream::destroy(raw) };
                        return -1;
                    }
                };
                let default_stream = *ctx.device.cu_stream();
                let wait_ok = unsafe {
                    cudarc::driver::result::event::record(fence_event, default_stream).and_then(
                        |_| {
                            cudarc::driver::result::stream::wait_event(
                                raw,
                                fence_event,
                                cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                            )
                        },
                    )
                }
                .is_ok();
                let _ = unsafe { cudarc::driver::result::event::destroy(fence_event) };
                if !wait_ok {
                    let _ = unsafe { cudarc::driver::result::stream::destroy(raw) };
                    return -1;
                }
                let sid = state.streams.len() as i32;
                state.streams.push(Some(CudaOwnedStream { raw }));
                sid
            }
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_stream_sync(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        match unsafe { cudarc::driver::result::stream::synchronize(stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_stream_destroy(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
) -> i32 {
    if stream_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let sid = stream_id as usize;
        if sid >= state.streams.len() {
            return -1;
        }
        match state.streams[sid].take() {
            Some(stream) => {
                state.stream_blas.remove(&stream_id);
                let fence_event = match cudarc::driver::result::event::create(
                    cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING,
                ) {
                    Ok(event) => event,
                    Err(_) => return -1,
                };
                let default_stream = *ctx.device.cu_stream();
                let wait_ok = unsafe {
                    cudarc::driver::result::event::record(fence_event, stream.raw).and_then(|_| {
                        cudarc::driver::result::stream::wait_event(
                            default_stream,
                            fence_event,
                            cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
                        )
                    })
                }
                .is_ok();
                let _ = unsafe { cudarc::driver::result::event::destroy(fence_event) };
                if wait_ok {
                    drop(stream);
                    0
                } else {
                    drop(stream);
                    -1
                }
            }
            None => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_event_create(ctx_ptr: *mut CraneliftCudaContext) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        match cudarc::driver::result::event::create(
            cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
        ) {
            Ok(raw) => {
                let eid = state.events.len() as i32;
                state.events.push(Some(CudaOwnedEvent { raw }));
                eid
            }
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_event_record(
    ctx_ptr: *mut CraneliftCudaContext,
    event_id: i32,
    stream_id: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(event) = resolve_cuda_event(&state, event_id) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        match unsafe { cudarc::driver::result::event::record(event, stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_stream_wait_event(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
    event_id: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        let Some(event) = resolve_cuda_event(&state, event_id) else {
            return -1;
        };
        match unsafe {
            cudarc::driver::result::stream::wait_event(
                stream,
                event,
                cudarc::driver::sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
            )
        } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_event_elapsed_ms_bits(
    ctx_ptr: *mut CraneliftCudaContext,
    start_event_id: i32,
    end_event_id: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(start) = resolve_cuda_event(&state, start_event_id) else {
            return -1;
        };
        let Some(end) = resolve_cuda_event(&state, end_event_id) else {
            return -1;
        };
        match unsafe { cudarc::driver::result::event::elapsed(start, end) } {
            Ok(ms) => i32::from_ne_bytes(ms.to_bits().to_ne_bytes()),
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_event_destroy(
    ctx_ptr: *mut CraneliftCudaContext,
    event_id: i32,
) -> i32 {
    if event_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let eid = event_id as usize;
        if eid >= state.events.len() {
            return -1;
        }
        match state.events[eid].take() {
            Some(_) => 0,
            None => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_begin_capture(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        match unsafe {
            cudarc::driver::sys::lib().cuStreamBeginCapture_v2(
                stream,
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        }
        .result()
        {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_end_capture(
    ctx_ptr: *mut CraneliftCudaContext,
    stream_id: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        let mut graph: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
        if unsafe { cudarc::driver::sys::lib().cuStreamEndCapture(stream, &mut graph) }
            .result()
            .is_err()
            || graph.is_null()
        {
            return -1;
        }
        let mut exec: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
        let instantiate_ok =
            unsafe { cudarc::driver::sys::lib().cuGraphInstantiateWithFlags(&mut exec, graph, 0) }
                .result()
                .is_ok()
                && !exec.is_null();
        let _ = unsafe { cudarc::driver::sys::lib().cuGraphDestroy(graph) }.result();
        if !instantiate_ok {
            return -1;
        }
        let gid = state.graphs.len() as i32;
        state.graphs.push(Some(CudaOwnedGraphExec { raw: exec }));
        gid
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_upload(
    ctx_ptr: *mut CraneliftCudaContext,
    graph_id: i32,
    stream_id: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(graph) = resolve_cuda_graph_exec(&state, graph_id) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        match unsafe { cudarc::driver::sys::lib().cuGraphUpload(graph, stream) }.result() {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_launch(
    ctx_ptr: *mut CraneliftCudaContext,
    graph_id: i32,
    stream_id: i32,
) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(graph) = resolve_cuda_graph_exec(&state, graph_id) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        match unsafe { cudarc::driver::sys::lib().cuGraphLaunch(graph, stream) }.result() {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_graph_destroy(
    ctx_ptr: *mut CraneliftCudaContext,
    graph_id: i32,
) -> i32 {
    if graph_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let gid = graph_id as usize;
        if gid >= state.graphs.len() {
            return -1;
        }
        match state.graphs[gid].take() {
            Some(_) => 0,
            None => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_pinned_alloc(ctx_ptr: *mut CraneliftCudaContext, size: i64) -> i32 {
    if size <= 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let mut ptr = std::ptr::null_mut();
        let result =
            unsafe { cudarc::driver::sys::lib().cuMemAllocHost_v2(&mut ptr, size as usize) };
        if result.result().is_err() || ptr.is_null() {
            return -1;
        }
        let Ok(mut state) = lock_cuda_state(ctx) else {
            let _ = unsafe { cudarc::driver::sys::lib().cuMemFreeHost(ptr) }.result();
            return -1;
        };
        let pid = state.pinned_buffers.len() as i32;
        state.pinned_buffers.push(Some(CudaPinnedHostBuffer {
            ptr,
            _size: size as usize,
        }));
        pid
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_pinned_ptr(ctx_ptr: *mut CraneliftCudaContext, pinned_id: i32) -> i64 {
    if pinned_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let pid = pinned_id as usize;
        let Some(buf) = state.pinned_buffers.get(pid).and_then(|b| b.as_ref()) else {
            return -1;
        };
        buf.ptr as i64
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_pinned_free(
    ctx_ptr: *mut CraneliftCudaContext,
    pinned_id: i32,
) -> i32 {
    if pinned_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let pid = pinned_id as usize;
        if pid >= state.pinned_buffers.len() {
            return -1;
        }
        match state.pinned_buffers[pid].take() {
            Some(_) => 0,
            None => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_upload_ptr_async(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    src_ptr: *const u8,
    size: i64,
    stream_id: i32,
) -> i32 {
    if buf_id < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        let Some(dst) = (unsafe { cuda_buffer_device_ptr(&state, buf_id) }) else {
            return -1;
        };
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        match unsafe { cudarc::driver::result::memcpy_htod_async(dst, data, stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_upload_ptr_offset_async(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    buf_offset: i64,
    src_ptr: *const u8,
    size: i64,
    stream_id: i32,
) -> i32 {
    if buf_id < 0 || buf_offset < 0 || size <= 0 || src_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        let Some(buf) = state.buffers.get(buf_id as usize).and_then(|b| b.as_ref()) else {
            return -1;
        };
        let total_len = {
            use cudarc::driver::DeviceSlice;
            buf.len()
        };
        let start = buf_offset as usize;
        let byte_len = size as usize;
        if start.saturating_add(byte_len) > total_len {
            return -1;
        }
        let Some(base_dst) = (unsafe { cuda_buffer_device_ptr(&state, buf_id) }) else {
            return -1;
        };
        let dst = base_dst.saturating_add(buf_offset as u64);
        let data = std::slice::from_raw_parts(src_ptr, size as usize);
        match unsafe { cudarc::driver::result::memcpy_htod_async(dst, data, stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_download_ptr_async(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    dst_ptr: *mut u8,
    size: i64,
    stream_id: i32,
) -> i32 {
    if buf_id < 0 || size <= 0 || dst_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        if !bind_cuda_ctx_if_needed(ctx) {
            return -1;
        }
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        let Some(src) = (unsafe { cuda_buffer_device_ptr(&state, buf_id) }) else {
            return -1;
        };
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        match unsafe { cudarc::driver::result::memcpy_dtoh_async(dst, src, stream) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_download_ptr(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    dst_ptr: *mut u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || dst_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        let Some(buf) = state.buffers[bid].as_ref() else {
            return -1;
        };
        match ctx.device.dtoh_sync_copy_into(buf, dst) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

/// Download `size` bytes from a GPU buffer at `buf_offset` into a host pointer.
pub(crate) unsafe extern "C" fn cl_cuda_download_ptr_offset(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    buf_offset: i64,
    dst_ptr: *mut u8,
    size: i64,
) -> i32 {
    use cudarc::driver::DevicePtr;
    if buf_id < 0 || buf_offset < 0 || size <= 0 || dst_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        let Some(buf) = state.buffers.get(bid).and_then(|b| b.as_ref()) else {
            return -1;
        };
        let base = *buf.device_ptr();
        let dev = base + buf_offset as u64;
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        match unsafe { cudarc::driver::result::memcpy_dtoh_sync(dst, dev) } {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_download(
    ctx_ptr: *mut CraneliftCudaContext,
    buf_id: i32,
    dst_ptr: *mut u8,
    size: i64,
) -> i32 {
    if buf_id < 0 || size <= 0 || dst_ptr.is_null() {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        let dst = std::slice::from_raw_parts_mut(dst_ptr, size as usize);
        let Some(buf) = state.buffers[bid].as_ref() else {
            return -1;
        };
        match ctx.device.dtoh_sync_copy_into(buf, dst) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_free_buffer(ctx_ptr: *mut CraneliftCudaContext, buf_id: i32) -> i32 {
    if buf_id < 0 {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let bid = buf_id as usize;
        if bid >= state.buffers.len() {
            return -1;
        }
        match state.buffers[bid].take() {
            Some(_) => 0,
            None => -1,
        }
    }))
    .unwrap_or(-1)
}

/// Launch a PTX kernel.
///
/// - `kernel_off`: offset of null-terminated PTX source in shared memory
/// - `n_bufs`: number of buffer arguments
/// - `bind_off`: offset of packed i32 buffer IDs (4 bytes each)
/// - `grid_x/y/z`: grid dimensions
/// - `block_x/y/z`: block dimensions
///
/// The PTX must define `.entry main(...)` with `n_bufs` pointer parameters.
pub(crate) unsafe extern "C" fn cl_cuda_launch(
    ctx_ptr: *mut CraneliftCudaContext,
    kernel_ptr: *const u8,
    n_bufs: i32,
    bind_ptr: *const u8,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
) -> i32 {
    if n_bufs < 0
        || grid_x <= 0
        || grid_y <= 0
        || grid_z <= 0
        || block_x <= 0
        || block_y <= 0
        || block_z <= 0
    {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Ok(function) = load_raw_cuda_main_kernel(ctx, &mut state, kernel_ptr) else {
            eprintln!("cl_cuda_launch: load kernel failed");
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, -1) else {
            return -1;
        };
        unsafe {
            launch_raw_cuda_kernel(
                &state, function, n_bufs, bind_ptr, grid_x, grid_y, grid_z, block_x, block_y,
                block_z, stream,
            )
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_sync(ctx_ptr: *const CraneliftCudaContext) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        match ctx.device.synchronize() {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_cleanup(ctx_slot_ptr: *mut *mut CraneliftCudaContext) {
    let ctx_ptr = clear_ctx_slot::<CraneliftCudaContext>(ctx_slot_ptr);
    if !ctx_ptr.is_null() {
        drop(Box::from_raw(ctx_ptr));
    }
}

/// Launch a PTX kernel by name.
///
/// Same as `cl_cuda_launch` but reads the entry-point name from shared memory
/// at `name_off` instead of hardcoding `"main"`.
pub(crate) unsafe extern "C" fn cl_cuda_launch_named(
    ctx_ptr: *mut CraneliftCudaContext,
    kernel_ptr: *const u8,
    name_ptr: *const u8,
    n_bufs: i32,
    bind_ptr: *const u8,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
) -> i32 {
    if n_bufs < 0
        || grid_x <= 0
        || grid_y <= 0
        || grid_z <= 0
        || block_x <= 0
        || block_y <= 0
        || block_z <= 0
    {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Ok(function) = load_raw_cuda_named_kernel(ctx, &mut state, kernel_ptr, name_ptr) else {
            eprintln!("cl_cuda_launch_named: load kernel failed");
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, -1) else {
            return -1;
        };
        unsafe {
            launch_raw_cuda_kernel(
                &state, function, n_bufs, bind_ptr, grid_x, grid_y, grid_z, block_x, block_y,
                block_z, stream,
            )
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_launch_on_stream(
    ctx_ptr: *mut CraneliftCudaContext,
    kernel_ptr: *const u8,
    n_bufs: i32,
    bind_ptr: *const u8,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
    stream_id: i32,
) -> i32 {
    if n_bufs < 0
        || grid_x <= 0
        || grid_y <= 0
        || grid_z <= 0
        || block_x <= 0
        || block_y <= 0
        || block_z <= 0
    {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Ok(function) = load_raw_cuda_main_kernel(ctx, &mut state, kernel_ptr) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        unsafe {
            launch_raw_cuda_kernel(
                &state, function, n_bufs, bind_ptr, grid_x, grid_y, grid_z, block_x, block_y,
                block_z, stream,
            )
        }
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cuda_launch_named_on_stream(
    ctx_ptr: *mut CraneliftCudaContext,
    kernel_ptr: *const u8,
    name_ptr: *const u8,
    n_bufs: i32,
    bind_ptr: *const u8,
    grid_x: i32,
    grid_y: i32,
    grid_z: i32,
    block_x: i32,
    block_y: i32,
    block_z: i32,
    stream_id: i32,
) -> i32 {
    if n_bufs < 0
        || grid_x <= 0
        || grid_y <= 0
        || grid_z <= 0
        || block_x <= 0
        || block_y <= 0
        || block_z <= 0
    {
        return -1;
    }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Ok(function) = load_raw_cuda_named_kernel(ctx, &mut state, kernel_ptr, name_ptr) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };
        unsafe {
            launch_raw_cuda_kernel(
                &state, function, n_bufs, bind_ptr, grid_x, grid_y, grid_z, block_x, block_y,
                block_z, stream,
            )
        }
    }))
    .unwrap_or(-1)
}

fn ensure_default_cuda_blas<'a>(
    ctx: &CraneliftCudaContext,
    state: &'a mut CraneliftCudaState,
) -> Result<&'a cudarc::cublas::CudaBlas, i32> {
    if state.default_blas.is_none() {
        match cudarc::cublas::CudaBlas::new(ctx.device.clone()) {
            Ok(b) => state.default_blas = Some(b),
            Err(e) => {
                eprintln!("CUDA BLAS init failed: {:?}", e);
                return Err(-1);
            }
        }
    }
    Ok(state.default_blas.as_ref().unwrap())
}

fn set_cuda_blas_stream(
    blas: &cudarc::cublas::CudaBlas,
    stream: cudarc::driver::sys::CUstream,
) -> Result<(), i32> {
    match unsafe { cudarc::cublas::result::set_stream(*blas.handle(), stream as *mut _) } {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("cuBLAS set_stream failed: {:?}", e);
            Err(-1)
        }
    }
}

fn ensure_stream_cuda_blas<'a>(
    ctx: &CraneliftCudaContext,
    state: &'a mut CraneliftCudaState,
    stream_id: i32,
    stream: cudarc::driver::sys::CUstream,
) -> Result<&'a cudarc::cublas::CudaBlas, i32> {
    if let std::collections::hash_map::Entry::Vacant(entry) = state.stream_blas.entry(stream_id) {
        let blas = match cudarc::cublas::CudaBlas::new(ctx.device.clone()) {
            Ok(blas) => blas,
            Err(e) => {
                eprintln!("CUDA BLAS init failed: {:?}", e);
                return Err(-1);
            }
        };
        if let Err(rc) = set_cuda_blas_stream(&blas, stream) {
            return Err(rc);
        }
        entry.insert(blas);
    }
    Ok(state.stream_blas.get(&stream_id).unwrap())
}

/// cuBLAS SGEMM: C = alpha * op(A) * op(B) + beta * C
///
/// - `transa`: 0 = NoTrans, 1 = Trans (for A)
/// - `transb`: 0 = NoTrans, 1 = Trans (for B)
/// - `m`, `n`, `k`: dimensions of op(A) = m×k, op(B) = k×n, C = m×n
/// - `alpha_bits`, `beta_bits`: f32 scalars reinterpreted as i32 bits
/// - `a_buf`, `b_buf`, `c_buf`: buffer IDs (from cl_cuda_create_buffer)
///
/// Leading dimensions follow standard cuBLAS column-major rules:
///   lda = k if transa else m,  ldb = n if transb else k,  ldc = m
pub(crate) unsafe extern "C" fn cl_cublas_sgemm(
    ctx_ptr: *mut CraneliftCudaContext,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha_bits: i32,
    a_buf: i32,
    b_buf: i32,
    beta_bits: i32,
    c_buf: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op_a = if transa != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = if transa != 0 { k } else { m };
        let ldb = if transb != 0 { n } else { k };
        let ldc = m;

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let b_dev = match unsafe { cuda_buffer_device_ptr(&state, b_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let c_dev = match unsafe { cuda_buffer_device_ptr(&state, c_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let blas = match ensure_default_cuda_blas(ctx, &mut state) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemm(
                *blas.handle(),
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha,
                a_dev as *const f32,
                lda,
                b_dev as *const f32,
                ldb,
                &beta,
                c_dev as *mut f32,
                ldc,
            )
        } {
            eprintln!("cl_cublas_sgemm: sgemm failed: {:?}", e);
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}

/// cuBLAS SGEMV: y = alpha * op(A) * x + beta * y
///
/// - `trans`: 0 = NoTrans, 1 = Trans
/// - `m`, `n`: dimensions of A in column-major form
/// - `alpha_bits`, `beta_bits`: f32 scalars reinterpreted as i32 bits
/// - `a_buf`, `x_buf`, `y_buf`: buffer IDs
///
/// For row-major weights shaped [rows, cols], call with `trans=1`, `m=cols`, `n=rows`.
pub(crate) unsafe extern "C" fn cl_cublas_sgemv(
    ctx_ptr: *mut CraneliftCudaContext,
    trans: i32,
    m: i32,
    n: i32,
    alpha_bits: i32,
    a_buf: i32,
    x_buf: i32,
    beta_bits: i32,
    y_buf: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op = if trans != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = m;

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let x_dev = match unsafe { cuda_buffer_device_ptr(&state, x_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let y_dev = match unsafe { cuda_buffer_device_ptr(&state, y_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let blas = match ensure_default_cuda_blas(ctx, &mut state) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemv(
                *blas.handle(),
                op,
                m,
                n,
                &alpha,
                a_dev as *const f32,
                lda,
                x_dev as *const f32,
                1,
                &beta,
                y_dev as *mut f32,
                1,
            )
        } {
            eprintln!("cl_cublas_sgemv: sgemv failed: {:?}", e);
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}

/// cuBLAS SGEMM strided batched: C_i = alpha * op(A_i) * op(B_i) + beta * C_i
///
/// - `transa`: 0 = NoTrans, 1 = Trans
/// - `transb`: 0 = NoTrans, 1 = Trans
/// - `m`, `n`, `k`: dimensions of op(A) = m×k, op(B) = k×n, C = m×n
/// - `stride_a`, `stride_b`, `stride_c`: batch strides in number of f32 elements
/// - `batch_count`: number of matrices/vectors in the batch
/// - `alpha_bits`, `beta_bits`: f32 scalars reinterpreted as i32 bits
///
/// For row-major matrices stored as contiguous `[rows, cols]`, use the same
/// transpose conventions as `cl_cublas_sgemv` / `cl_cublas_sgemm`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn cl_cublas_sgemm_strided_batched(
    ctx_ptr: *mut CraneliftCudaContext,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha_bits: i32,
    a_buf: i32,
    stride_a: i64,
    b_buf: i32,
    stride_b: i64,
    beta_bits: i32,
    c_buf: i32,
    stride_c: i64,
    batch_count: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    if m <= 0
        || n <= 0
        || k <= 0
        || stride_a < 0
        || stride_b < 0
        || stride_c < 0
        || batch_count <= 0
    {
        return -1;
    }

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op_a = if transa != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = if transa != 0 { k } else { m };
        let ldb = if transb != 0 { n } else { k };
        let ldc = m;

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let b_dev = match unsafe { cuda_buffer_device_ptr(&state, b_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let c_dev = match unsafe { cuda_buffer_device_ptr(&state, c_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let blas = match ensure_default_cuda_blas(ctx, &mut state) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemm_strided_batched(
                *blas.handle(),
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha,
                a_dev as *const f32,
                lda,
                stride_a,
                b_dev as *const f32,
                ldb,
                stride_b,
                &beta,
                c_dev as *mut f32,
                ldc,
                stride_c,
                batch_count,
            )
        } {
            eprintln!("cl_cublas_sgemm_strided_batched: sgemm failed: {:?}", e);
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}

pub(crate) unsafe extern "C" fn cl_cublas_sgemv_on_stream(
    ctx_ptr: *mut CraneliftCudaContext,
    trans: i32,
    m: i32,
    n: i32,
    alpha_bits: i32,
    a_buf: i32,
    x_buf: i32,
    beta_bits: i32,
    y_buf: i32,
    stream_id: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op = if trans != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = m;

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let x_dev = match unsafe { cuda_buffer_device_ptr(&state, x_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let y_dev = match unsafe { cuda_buffer_device_ptr(&state, y_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let blas = match ensure_stream_cuda_blas(ctx, &mut state, stream_id, stream) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemv(
                *blas.handle(),
                op,
                m,
                n,
                &alpha,
                a_dev as *const f32,
                lda,
                x_dev as *const f32,
                1,
                &beta,
                y_dev as *mut f32,
                1,
            )
        } {
            eprintln!("cl_cublas_sgemv_on_stream: sgemv failed: {:?}", e);
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn cl_cublas_sgemm_strided_batched_on_stream(
    ctx_ptr: *mut CraneliftCudaContext,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha_bits: i32,
    a_buf: i32,
    stride_a: i64,
    b_buf: i32,
    stride_b: i64,
    beta_bits: i32,
    c_buf: i32,
    stride_c: i64,
    batch_count: i32,
    stream_id: i32,
) -> i32 {
    use cudarc::cublas::sys::cublasOperation_t;

    if m <= 0
        || n <= 0
        || k <= 0
        || stride_a < 0
        || stride_b < 0
        || stride_c < 0
        || batch_count <= 0
    {
        return -1;
    }

    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_mut::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        let Ok(mut state) = lock_cuda_state(ctx) else {
            return -1;
        };
        let Some(stream) = resolve_cuda_stream(&ctx.device, &state, stream_id) else {
            return -1;
        };

        let alpha = f32::from_bits(alpha_bits as u32);
        let beta = f32::from_bits(beta_bits as u32);
        let op_a = if transa != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let op_b = if transb != 0 {
            cublasOperation_t::CUBLAS_OP_T
        } else {
            cublasOperation_t::CUBLAS_OP_N
        };
        let lda = if transa != 0 { k } else { m };
        let ldb = if transb != 0 { n } else { k };
        let ldc = m;

        let a_dev = match unsafe { cuda_buffer_device_ptr(&state, a_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let b_dev = match unsafe { cuda_buffer_device_ptr(&state, b_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let c_dev = match unsafe { cuda_buffer_device_ptr(&state, c_buf) } {
            Some(p) => p,
            None => return -1,
        };
        let blas = match ensure_stream_cuda_blas(ctx, &mut state, stream_id, stream) {
            Ok(blas) => blas,
            Err(rc) => return rc,
        };

        if let Err(e) = unsafe {
            cudarc::cublas::result::sgemm_strided_batched(
                *blas.handle(),
                op_a,
                op_b,
                m,
                n,
                k,
                &alpha,
                a_dev as *const f32,
                lda,
                stride_a,
                b_dev as *const f32,
                ldb,
                stride_b,
                &beta,
                c_dev as *mut f32,
                ldc,
                stride_c,
                batch_count,
            )
        } {
            eprintln!(
                "cl_cublas_sgemm_strided_batched_on_stream: sgemm failed: {:?}",
                e
            );
            return -1;
        }

        0
    }))
    .unwrap_or(-1)
}
