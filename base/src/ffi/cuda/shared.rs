//! Context, owned handles, and the kernel cache every entry point goes through.
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use crate::ffi::{clear_ctx_slot, read_cstr_ptr, read_ctx_ref, write_ctx_slot};

pub(crate) struct CraneliftCudaContext {
    pub(super) device: std::sync::Arc<cudarc::driver::CudaDevice>,
    pub(super) state: Mutex<CraneliftCudaState>,
}

pub(super) struct CraneliftCudaState {
    pub(super) buffers: Vec<Option<cudarc::driver::CudaSlice<u8>>>,
    pub(super) default_blas: Option<cudarc::cublas::CudaBlas>,
    pub(super) stream_blas: HashMap<i32, cudarc::cublas::CudaBlas>,
    pub(super) streams: Vec<Option<CudaOwnedStream>>,
    pub(super) events: Vec<Option<CudaOwnedEvent>>,
    pub(super) graphs: Vec<Option<CudaOwnedGraphExec>>,
    pub(super) pinned_buffers: Vec<Option<CudaPinnedHostBuffer>>,
    pub(super) main_kernel_cache: std::collections::HashMap<*const u8, RawCudaKernelCacheEntry>,
    pub(super) named_kernel_cache: std::collections::HashMap<(*const u8, *const u8), RawCudaKernelCacheEntry>,
}

pub(super) struct RawCudaKernelCacheEntry {
    pub(super) module: cudarc::driver::sys::CUmodule,
    pub(super) function: cudarc::driver::sys::CUfunction,
}

pub(super) struct CudaOwnedStream {
    pub(super) raw: cudarc::driver::sys::CUstream,
}

pub(super) struct CudaOwnedEvent {
    pub(super) raw: cudarc::driver::sys::CUevent,
}

pub(super) struct CudaOwnedGraphExec {
    pub(super) raw: cudarc::driver::sys::CUgraphExec,
}

pub(super) struct CudaPinnedHostBuffer {
    pub(super) ptr: *mut std::ffi::c_void,
    pub(super) _size: usize,
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

pub(super) fn bind_cuda_ctx_if_needed(ctx: &CraneliftCudaContext) -> bool {
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

pub(super) fn lock_cuda_state(ctx: &CraneliftCudaContext) -> Result<MutexGuard<'_, CraneliftCudaState>, ()> {
    ctx.state.lock().map_err(|_| ())
}

pub(super) fn resolve_cuda_stream(
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

pub(super) fn resolve_cuda_event(
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

pub(super) fn resolve_cuda_graph_exec(
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

pub(super) unsafe fn cuda_buffer_device_ptr(
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

pub(super) fn load_raw_cuda_main_kernel(
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
    let ptx_len = ptx_src.len();
    let ptx_cstr = std::ffi::CString::new(ptx_src).map_err(|_| ())?;
    let load = || unsafe {
        cudarc::driver::result::module::load_data(ptx_cstr.as_ptr() as *const std::ffi::c_void)
    };
    let module = match load() {
        Ok(m) => m,
        Err(_) => {
            // A load can fail because the context is out of room for modules.
            // Drop what is resident and try once more: the cache is a
            // launch-cost optimisation, never a correctness condition, so
            // evicting it can only cost time.  A tape of 90 distinct kernels
            // never reaches this path, so the number of modules a context
            // holds is not a bound any artifact here has met.
            //
            // Launches are enqueued asynchronously, so a module may still be
            // running when its entry is dropped.  Waiting for the device first
            // is what makes the eviction safe rather than merely plausible.
            // It also turns a *sticky* error — an illegal access from an
            // earlier launch poisons the context, and every later load returns
            // that error — into something the message below names, rather than
            // into an apparent module-count limit.
            eprintln!("cl_cuda_launch: module load failed with {} resident; evicting",
                      state.main_kernel_cache.len());
            let _ = ctx.device.synchronize();
            for (_, entry) in std::mem::take(&mut state.main_kernel_cache) {
                let _ = unsafe { cudarc::driver::result::module::unload(entry.module) };
            }
            match load() {
                Ok(m) => m,
                Err(e) => {
                    // Say which error: collapsing every driver failure into
                    // `Err(())` makes a module-count limit indistinguishable
                    // from malformed PTX.
                    eprintln!(
                        "cl_cuda_launch: module load failed after evicting {} bytes of PTX: {:?}",
                        ptx_len,
                        e
                    );
                    return Err(());
                }
            }
        }
    };
    let function = match unsafe {
        cudarc::driver::result::module::get_function(
            module,
            std::ffi::CString::new("main").expect("main CString"),
        )
    } {
        Ok(f) => f,
        Err(e) => {
            // A module that loads but has no `main` is a different failure from
            // a module that will not load at all; keep them apart.
            eprintln!(
                "cl_cuda_launch: module loaded ({} bytes of PTX) but `main` is absent: {:?}",
                ptx_len, e
            );
            return Err(());
        }
    };
    state
        .main_kernel_cache
        .insert(kernel_ptr, RawCudaKernelCacheEntry { module, function });
    Ok(function)
}

pub(super) fn load_raw_cuda_named_kernel(
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

pub(super) unsafe fn launch_raw_cuda_kernel(
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

pub(super) fn cached_cuda_device() -> std::sync::Arc<cudarc::driver::CudaDevice> {
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


pub(crate) unsafe extern "C" fn cl_cuda_sync(ctx_ptr: *const CraneliftCudaContext) -> i32 {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let Some(ctx) = read_ctx_ref::<CraneliftCudaContext>(ctx_ptr) else {
            return -1;
        };
        match ctx.device.synchronize() {
            Ok(_) => 0,
            Err(e) => {
                // A launch reports out-of-range addressing only when something
                // waits for it, and the CLIF caller does not read this result.
                eprintln!("cl_cuda_sync: {:?}", e);
                -1
            }
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
