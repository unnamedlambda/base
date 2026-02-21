use base_types::Action;
use pollster::block_on;
use portable_atomic::{AtomicU64, Ordering};
use std::collections::HashMap;
use std::fs;
use std::io::{Read as IoRead, Seek, Write as IoWrite};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{fence, AtomicU32};
use std::sync::Arc;
use tracing::{debug, info, info_span};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    InstanceDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, PowerPreference,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};
use lmdb_zero as lmdb;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::JITBuilder;
use cranelift_module::Module;

/// Cached wgpu Device + Queue (Arc-wrapped for sharing).
/// Creating many wgpu Devices exhausts OS GPU driver handles (~60 limit).
/// One device per process; callers create fresh buffers/pipelines per use.
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
        let (device, queue) = block_on(adapter.request_device(
            &DeviceDescriptor::default(), None,
        ))
        .expect("Failed to create GPU device");
        std::mem::forget(instance);
        std::mem::forget(adapter);
        (Arc::new(device), Arc::new(queue))
    });
    (d.clone(), q.clone())
}

pub const MAILBOX_CLOSED: u64 = u64::MAX;

pub(crate) struct Mailbox(AtomicU64);

pub(crate) enum MailboxPoll {
    Empty,
    Work { start: u32, end: u32, flag: u32 },
    Closed,
}

impl Mailbox {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    pub fn post(&self, start: u32, end: u32, flag: u32) {
        debug!(start, end, flag, "mailbox_post");
        let packed = ((start as u64) << 43) | ((end as u64) << 22) | (flag as u64);
        let mut spin_count = 0u32;
        loop {
            match self
                .0
                .compare_exchange(0, packed, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => break,
                Err(_) => spin_backoff(&mut spin_count),
            }
        }
    }

    pub fn shutdown(&self) {
        debug!("mailbox_shutdown");
        self.0.store(MAILBOX_CLOSED, Ordering::Release);
    }

    pub fn poll(&self) -> MailboxPoll {
        let packed = self.0.swap(0, Ordering::AcqRel);
        if packed == 0 {
            return MailboxPoll::Empty;
        }
        if packed == MAILBOX_CLOSED {
            return MailboxPoll::Closed;
        }
        let start = (packed >> 43) as u32;
        let end = ((packed >> 22) & 0x1F_FFFF) as u32;
        let flag = (packed & 0x3F_FFFF) as u32;
        MailboxPoll::Work { start, end, flag }
    }
}

fn spin_backoff(spin_count: &mut u32) {
    *spin_count += 1;
    if *spin_count < 100 {
        std::hint::spin_loop();
    } else if *spin_count < 1000 {
        std::thread::yield_now();
    } else {
        std::thread::sleep(std::time::Duration::from_micros(1));
    }
}


pub(crate) struct SharedMemory {
    ptr: *mut u8,
}

unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

impl SharedMemory {
    pub fn new(ptr: *mut u8) -> Self {
        Self { ptr }
    }

    pub unsafe fn read(&self, offset: usize, size: usize) -> &[u8] {
        std::slice::from_raw_parts(self.ptr.add(offset), size)
    }

    pub unsafe fn write(&self, offset: usize, data: &[u8]) {
        self.ptr
            .add(offset)
            .copy_from_nonoverlapping(data.as_ptr(), data.len());
    }

    // Use a true atomic op when the pointer is naturally aligned; fall back to
    // an unaligned read + fence otherwise.
    pub unsafe fn load_u64(&self, offset: usize, order: Ordering) -> u64 {
        let ptr = self.ptr.add(offset);
        if (ptr as usize) & 0x7 == 0 {
            return (*(ptr as *const AtomicU64)).load(order);
        }
        let value = std::ptr::read_unaligned(ptr as *const u64);
        if matches!(order, Ordering::Acquire | Ordering::AcqRel | Ordering::SeqCst) {
            fence(Ordering::Acquire);
        }
        value
    }

    pub unsafe fn store_u64(&self, offset: usize, value: u64, order: Ordering) {
        let ptr = self.ptr.add(offset);
        if (ptr as usize) & 0x7 == 0 {
            (*(ptr as *const AtomicU64)).store(value, order);
            return;
        }
        if matches!(order, Ordering::Release | Ordering::AcqRel | Ordering::SeqCst) {
            fence(Ordering::Release);
        }
        std::ptr::write_unaligned(ptr as *mut u64, value);
    }

    pub unsafe fn load_u32(&self, offset: usize, order: Ordering) -> u32 {
        let ptr = self.ptr.add(offset);
        if (ptr as usize) & 0x3 == 0 {
            return (*(ptr as *const AtomicU32)).load(order);
        }
        let value = std::ptr::read_unaligned(ptr as *const u32);
        if matches!(order, Ordering::Acquire | Ordering::AcqRel | Ordering::SeqCst) {
            fence(Ordering::Acquire);
        }
        value
    }

    pub unsafe fn store_u32(&self, offset: usize, value: u32, order: Ordering) {
        let ptr = self.ptr.add(offset);
        if (ptr as usize) & 0x3 == 0 {
            (*(ptr as *const AtomicU32)).store(value, order);
            return;
        }
        if matches!(order, Ordering::Release | Ordering::AcqRel | Ordering::SeqCst) {
            fence(Ordering::Release);
        }
        std::ptr::write_unaligned(ptr as *mut u32, value);
    }

    pub unsafe fn cas_u32(&self, offset: usize, current: u32, new: u32, success: Ordering, failure: Ordering) -> Result<u32, u32> {
        let ptr = self.ptr.add(offset);
        debug_assert!((ptr as usize) & 0x3 == 0, "cas_u32: pointer not 4-byte aligned (offset {offset})");
        (*(ptr as *const AtomicU32)).compare_exchange(current, new, success, failure)
    }

    // CAS requires natural alignment — there is no unaligned fallback.
    pub unsafe fn cas64(&self, offset: usize, expected: u64, new: u64) -> u64 {
        let ptr = self.ptr.add(offset);
        debug_assert!((ptr as usize) & 0x7 == 0, "cas64: pointer not 8-byte aligned (offset {offset})");
        (*(ptr as *const AtomicU64))
            .compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or_else(|x| x)
    }

}

pub(crate) fn order_from_u32(raw: u32) -> Ordering {
    match raw {
        1 => Ordering::Acquire,
        2 => Ordering::Release,
        3 => Ordering::AcqRel,
        4 => Ordering::SeqCst,
        _ => Ordering::Relaxed,
    }
}

pub(crate) unsafe fn load_sized(shared: &SharedMemory, offset: usize, size: u32, order: Ordering) -> u64 {
    match size {
        1 => shared.read(offset, 1)[0] as u64,
        2 => u16::from_le_bytes(shared.read(offset, 2)[0..2].try_into().unwrap()) as u64,
        4 => u32::from_le_bytes(shared.read(offset, 4)[0..4].try_into().unwrap()) as u64,
        8 => shared.load_u64(offset, order),
        _ => 0,
    }
}

pub(crate) fn read_null_terminated_string_from_slice(data: &[u8], offset: usize, max_len: usize) -> String {
    let end = (offset + max_len).min(data.len());
    let bytes = &data[offset..end];
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..len]).into_owned()
}

pub(crate) struct CraneliftHashTableContext {
    tables: HashMap<u32, HashMap<Vec<u8>, Vec<u8>>>,
    next_handle: u32,
}

impl CraneliftHashTableContext {
    fn new() -> Self {
        Self { tables: HashMap::new(), next_handle: 0 }
    }
}

/// Offset in shared memory where the HT context pointer is stored (first 8 bytes of payload).
const CL_HT_CTX_OFFSET: usize = 0;

unsafe extern "C" fn cl_ht_create(ctx: *mut u8) -> u32 {
    let ctx = &mut *(ctx as *mut CraneliftHashTableContext);
    let handle = ctx.next_handle;
    ctx.next_handle += 1;
    ctx.tables.insert(handle, HashMap::new());
    handle
}

unsafe extern "C" fn cl_ht_lookup(
    ctx: *mut u8,
    key: *const u8,
    key_len: u32,
    result: *mut u8,
) -> u32 {
    let ctx = &*(ctx as *const CraneliftHashTableContext);
    let key = std::slice::from_raw_parts(key, key_len as usize);
    if let Some(table) = ctx.tables.get(&0) {
        if let Some(val) = table.get(key) {
            std::ptr::copy_nonoverlapping(val.as_ptr(), result, val.len());
            return val.len() as u32;
        }
    }
    0xFFFF_FFFF
}

unsafe extern "C" fn cl_ht_insert(
    ctx: *mut u8,
    key: *const u8,
    key_len: u32,
    val: *const u8,
    val_len: u32,
) {
    let ctx = &mut *(ctx as *mut CraneliftHashTableContext);
    let key_slice = std::slice::from_raw_parts(key, key_len as usize);
    let val_slice = std::slice::from_raw_parts(val, val_len as usize);
    if let Some(table) = ctx.tables.get_mut(&0) {
        if let Some(existing) = table.get_mut(key_slice) {
            if existing.len() == val_len as usize {
                existing.copy_from_slice(val_slice);
            } else {
                *existing = val_slice.to_vec();
            }
        } else {
            table.insert(key_slice.to_vec(), val_slice.to_vec());
        }
    }
}

unsafe extern "C" fn cl_ht_count(ctx: *mut u8) -> u32 {
    let ctx = &*(ctx as *const CraneliftHashTableContext);
    ctx.tables.get(&0).map(|t| t.len() as u32).unwrap_or(0)
}

unsafe extern "C" fn cl_ht_get_entry(
    ctx: *mut u8,
    index: u32,
    key_out: *mut u8,
    val_out: *mut u8,
) -> i32 {
    let ctx = &*(ctx as *const CraneliftHashTableContext);
    if let Some(table) = ctx.tables.get(&0) {
        if let Some((key, val)) = table.iter().nth(index as usize) {
            std::ptr::copy_nonoverlapping(key.as_ptr(), key_out, key.len());
            std::ptr::copy_nonoverlapping(val.as_ptr(), val_out, val.len());
            return key.len() as i32;
        }
    }
    -1
}

unsafe extern "C" fn cl_ht_increment(
    ctx: *mut u8,
    key: *const u8,
    key_len: u32,
    addend: i64,
) -> i64 {
    let ctx = &mut *(ctx as *mut CraneliftHashTableContext);
    let key_slice = std::slice::from_raw_parts(key, key_len as usize);
    if let Some(table) = ctx.tables.get_mut(&0) {
        if let Some(existing) = table.get_mut(key_slice) {
            let current = i64::from_le_bytes(existing[..8].try_into().unwrap_or([0; 8]));
            let new_val = current + addend;
            existing[..8].copy_from_slice(&new_val.to_le_bytes());
            return new_val;
        }
        table.insert(key_slice.to_vec(), addend.to_le_bytes().to_vec());
    }
    addend
}

struct CraneliftGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffers: Vec<wgpu::Buffer>,
    staging_buffers: Vec<wgpu::Buffer>,
    pipelines: Vec<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    pending_encoder: Option<wgpu::CommandEncoder>,
}

unsafe extern "C" fn cl_gpu_init(ptr: *mut u8) {
    let (device, queue) = cached_gpu_device();
    let ctx = Box::new(CraneliftGpuContext {
        device, queue,
        buffers: Vec::new(),
        staging_buffers: Vec::new(),
        pipelines: Vec::new(),
        pending_encoder: None,
    });
    std::ptr::write_unaligned(ptr as *mut *mut CraneliftGpuContext, Box::into_raw(ctx));
}

unsafe extern "C" fn cl_gpu_create_buffer(ptr: *mut u8, size: i64) -> i32 {
    if size <= 0 { return -1; }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftGpuContext);
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
    })).unwrap_or(-1)
}

unsafe extern "C" fn cl_gpu_create_pipeline(
    ptr: *mut u8,
    shader_off: i64,
    bind_off: i64,
    n_bindings: i32,
) -> i32 {
    if n_bindings < 0 { return -1; }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftGpuContext);
        let shader_ptr = ptr.add(shader_off as usize);
        let mut len = 0;
        while *shader_ptr.add(len) != 0 { len += 1; }
        let shader_src = match std::str::from_utf8(std::slice::from_raw_parts(shader_ptr, len)) {
            Ok(s) => s,
            Err(_) => return -1,
        };
        let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None, source: ShaderSource::Wgsl(shader_src.into()),
        });
        let mut bgl_entries = Vec::new();
        let mut bg_entries = Vec::new();
        let bind_base = ptr.add(bind_off as usize);
        let n_bufs = ctx.buffers.len();
        for i in 0..n_bindings as usize {
            let desc_ptr = bind_base.add(i * 8);
            let buf_id = std::ptr::read_unaligned(desc_ptr as *const i32) as usize;
            if buf_id >= n_bufs { return -1; }
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
        let bgl = ctx.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None, entries: &bgl_entries,
        });
        let pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&ctx.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
            })),
            module: &shader,
            entry_point: "main",
            compilation_options: PipelineCompilationOptions::default(),
        });
        let entries: Vec<BindGroupEntry> = bg_entries.iter().map(|&(binding, buf_id)| {
            BindGroupEntry {
                binding,
                resource: BindingResource::Buffer(ctx.buffers[buf_id].as_entire_buffer_binding()),
            }
        }).collect();
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None, layout: &bgl, entries: &entries,
        });
        let idx = ctx.pipelines.len() as i32;
        ctx.pipelines.push((pipeline, bind_group));
        idx
    })).unwrap_or(-1)
}

unsafe extern "C" fn cl_gpu_upload(ptr: *mut u8, buf_id: i32, src_off: i64, size: i64) -> i32 {
    if buf_id < 0 || size <= 0 { return -1; }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx = &*std::ptr::read_unaligned(ptr as *const *mut CraneliftGpuContext);
        let bid = buf_id as usize;
        if bid >= ctx.buffers.len() { return -1; }
        let data = std::slice::from_raw_parts(ptr.add(src_off as usize), size as usize);
        ctx.queue.write_buffer(&ctx.buffers[bid], 0, data);
        0
    })).unwrap_or(-1)
}

unsafe extern "C" fn cl_gpu_dispatch(ptr: *mut u8, pipeline_id: i32, workgroups: i32) -> i32 {
    if pipeline_id < 0 || workgroups <= 0 { return -1; }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftGpuContext);
        let pid = pipeline_id as usize;
        if pid >= ctx.pipelines.len() { return -1; }
        if let Some(enc) = ctx.pending_encoder.take() {
            ctx.queue.submit(Some(enc.finish()));
        }
        let (pipeline, bind_group) = &ctx.pipelines[pid];
        let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None, timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups as u32, 1, 1);
        }
        ctx.pending_encoder = Some(encoder);
        0
    })).unwrap_or(-1)
}

unsafe extern "C" fn cl_gpu_download(ptr: *mut u8, buf_id: i32, dst_off: i64, size: i64) -> i32 {
    if buf_id < 0 || size <= 0 { return -1; }
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftGpuContext);
        let bid = buf_id as usize;
        if bid >= ctx.buffers.len() { return -1; }
        let size = size as u64;
        let mut encoder = ctx.pending_encoder.take().unwrap_or_else(||
            ctx.device.create_command_encoder(&CommandEncoderDescriptor { label: None })
        );
        encoder.copy_buffer_to_buffer(&ctx.buffers[bid], 0, &ctx.staging_buffers[bid], 0, size);
        ctx.queue.submit(Some(encoder.finish()));
        let slice = ctx.staging_buffers[bid].slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::Maintain::Wait);
        let mapped = slice.get_mapped_range();
        let dst = std::slice::from_raw_parts_mut(ptr.add(dst_off as usize), size as usize);
        dst.copy_from_slice(&mapped);
        drop(mapped);
        ctx.staging_buffers[bid].unmap();
        0
    })).unwrap_or(-1)
}

unsafe extern "C" fn cl_gpu_cleanup(ptr: *mut u8) {
    let ctx_ptr = std::ptr::read_unaligned(ptr as *const *mut CraneliftGpuContext);
    drop(Box::from_raw(ctx_ptr));
}

unsafe fn read_cstr(ptr: *mut u8, off: usize) -> String {
    let start = ptr.add(off);
    let mut len = 0;
    while *start.add(len) != 0 { len += 1; }
    String::from_utf8_lossy(std::slice::from_raw_parts(start, len)).into_owned()
}

unsafe extern "C" fn cl_file_read(
    ptr: *mut u8,
    path_off: i64,
    dst_off: i64,
    file_offset: i64,
    size: i64,
) -> i64 {
    let filename = read_cstr(ptr, path_off as usize);
    let mut file = match fs::File::open(&filename) {
        Ok(f) => f,
        Err(_) => return -1,
    };
    if file_offset > 0 {
        let _ = file.seek(std::io::SeekFrom::Start(file_offset as u64));
    }
    if size == 0 {
        let file_len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
        if file_len == 0 { return 0; }
        let dst = std::slice::from_raw_parts_mut(ptr.add(dst_off as usize), file_len);
        let mut total = 0;
        while total < file_len {
            match file.read(&mut dst[total..]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(_) => break,
            }
        }
        total as i64
    } else {
        let dst = std::slice::from_raw_parts_mut(ptr.add(dst_off as usize), size as usize);
        match file.read(dst) {
            Ok(n) => n as i64,
            Err(_) => -1,
        }
    }
}

unsafe extern "C" fn cl_file_write(
    ptr: *mut u8,
    path_off: i64,
    src_off: i64,
    file_offset: i64,
    size: i64,
) -> i64 {
    let filename = read_cstr(ptr, path_off as usize);
    let mut file = if file_offset == 0 {
        match fs::File::create(&filename) {
            Ok(f) => f,
            Err(_) => return -1,
        }
    } else {
        match fs::OpenOptions::new().write(true).create(true).open(&filename) {
            Ok(mut f) => {
                let _ = f.seek(std::io::SeekFrom::Start(file_offset as u64));
                f
            }
            Err(_) => return -1,
        }
    };
    let written = if size == 0 {
        let base = ptr.add(src_off as usize);
        let mut len = 0;
        while *base.add(len) != 0 { len += 1; }
        if len > 0 {
            let data = std::slice::from_raw_parts(base, len);
            match file.write_all(data) {
                Ok(_) => len as i64,
                Err(_) => -1,
            }
        } else { 0 }
    } else {
        let data = std::slice::from_raw_parts(ptr.add(src_off as usize), size as usize);
        match file.write_all(data) {
            Ok(_) => size,
            Err(_) => -1,
        }
    };
    if written >= 0 { let _ = file.sync_all(); }
    written
}

struct CraneliftNetContext {
    connections: HashMap<u32, TcpStream>,
    listeners: HashMap<u32, TcpListener>,
    next_handle: u32,
}

unsafe extern "C" fn cl_net_init(ptr: *mut u8) {
    let ctx = Box::new(CraneliftNetContext {
        connections: HashMap::new(),
        listeners: HashMap::new(),
        next_handle: 1,
    });
    std::ptr::write_unaligned(ptr as *mut *mut CraneliftNetContext, Box::into_raw(ctx));
}

unsafe extern "C" fn cl_net_listen(ptr: *mut u8, addr_off: i64) -> i64 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftNetContext);
    let addr = read_cstr(ptr, addr_off as usize);
    match TcpListener::bind(&addr) {
        Ok(listener) => {
            let handle = ctx.next_handle;
            ctx.next_handle += 1;
            ctx.listeners.insert(handle, listener);
            handle as i64
        }
        Err(_) => 0,
    }
}

unsafe extern "C" fn cl_net_connect(ptr: *mut u8, addr_off: i64) -> i64 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftNetContext);
    let addr = read_cstr(ptr, addr_off as usize);
    match TcpStream::connect(&addr) {
        Ok(stream) => {
            let handle = ctx.next_handle;
            ctx.next_handle += 1;
            ctx.connections.insert(handle, stream);
            handle as i64
        }
        Err(_) => 0,
    }
}

unsafe extern "C" fn cl_net_accept(ptr: *mut u8, listener: i64) -> i64 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftNetContext);
    if let Some(l) = ctx.listeners.get(&(listener as u32)) {
        if let Ok((stream, _)) = l.accept() {
            let handle = ctx.next_handle;
            ctx.next_handle += 1;
            ctx.connections.insert(handle, stream);
            return handle as i64;
        }
    }
    0
}

unsafe extern "C" fn cl_net_send(ptr: *mut u8, conn: i64, src_off: i64, size: i64) -> i64 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftNetContext);
    if let Some(stream) = ctx.connections.get_mut(&(conn as u32)) {
        let data = std::slice::from_raw_parts(ptr.add(src_off as usize), size as usize);
        match IoWrite::write_all(stream, data) {
            Ok(_) => return 0,
            Err(_) => return -1,
        }
    }
    -1
}

unsafe extern "C" fn cl_net_recv(ptr: *mut u8, conn: i64, dst_off: i64, size: i64) -> i64 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftNetContext);
    if let Some(stream) = ctx.connections.get_mut(&(conn as u32)) {
        let buf = std::slice::from_raw_parts_mut(ptr.add(dst_off as usize), size as usize);
        let mut total = 0;
        while total < size as usize {
            match IoRead::read(stream, &mut buf[total..]) {
                Ok(0) => break,
                Ok(n) => total += n,
                Err(_) => return -1,
            }
        }
        return total as i64;
    }
    -1
}

unsafe extern "C" fn cl_net_cleanup(ptr: *mut u8) {
    let ctx_ptr = std::ptr::read_unaligned(ptr as *const *mut CraneliftNetContext);
    drop(Box::from_raw(ctx_ptr));
}

struct CraneliftLmdbContext {
    envs: HashMap<u32, (lmdb::Environment, liblmdb_sys::MDB_dbi)>,
    active_write_txns: HashMap<u32, *mut liblmdb_sys::MDB_txn>,
    next_handle: u32,
}

impl Drop for CraneliftLmdbContext {
    fn drop(&mut self) {
        for (_handle, txn) in self.active_write_txns.drain() {
            unsafe { liblmdb_sys::mdb_txn_abort(txn); }
        }
    }
}

fn lmdb_raw_begin_txn(env: &lmdb::Environment, readonly: bool) -> *mut liblmdb_sys::MDB_txn {
    let mut txn = std::ptr::null_mut();
    let flags = if readonly { liblmdb_sys::MDB_RDONLY } else { 0 };
    unsafe {
        if liblmdb_sys::mdb_txn_begin(env.as_raw(), std::ptr::null_mut(), flags, &mut txn) == 0 {
            txn
        } else {
            std::ptr::null_mut()
        }
    }
}

fn lmdb_raw_put(txn: *mut liblmdb_sys::MDB_txn, dbi: liblmdb_sys::MDB_dbi, key: &[u8], val: &[u8]) -> bool {
    let mut k = liblmdb_sys::MDB_val { mv_size: key.len(), mv_data: key.as_ptr() as *const _ };
    let mut v = liblmdb_sys::MDB_val { mv_size: val.len(), mv_data: val.as_ptr() as *const _ };
    unsafe { liblmdb_sys::mdb_put(txn, dbi, &mut k, &mut v, 0) == 0 }
}

fn lmdb_raw_get(txn: *mut liblmdb_sys::MDB_txn, dbi: liblmdb_sys::MDB_dbi, key: &[u8]) -> Option<Vec<u8>> {
    let mut k = liblmdb_sys::MDB_val { mv_size: key.len(), mv_data: key.as_ptr() as *const _ };
    let mut v = liblmdb_sys::MDB_val { mv_size: 0, mv_data: std::ptr::null() };
    unsafe {
        if liblmdb_sys::mdb_get(txn, dbi, &mut k, &mut v) == 0 {
            Some(std::slice::from_raw_parts(v.mv_data as *const u8, v.mv_size).to_vec())
        } else {
            None
        }
    }
}

fn lmdb_raw_del(txn: *mut liblmdb_sys::MDB_txn, dbi: liblmdb_sys::MDB_dbi, key: &[u8]) -> bool {
    let mut k = liblmdb_sys::MDB_val { mv_size: key.len(), mv_data: key.as_ptr() as *const _ };
    unsafe { liblmdb_sys::mdb_del(txn, dbi, &mut k, std::ptr::null_mut()) == 0 }
}

fn lmdb_raw_cursor_scan(
    txn: *mut liblmdb_sys::MDB_txn,
    dbi: liblmdb_sys::MDB_dbi,
    start_key: Option<&[u8]>,
    max_entries: usize,
) -> Vec<u8> {
    let mut result = Vec::new();
    result.extend_from_slice(&0u32.to_le_bytes());
    let mut cursor: *mut liblmdb_sys::MDB_cursor = std::ptr::null_mut();
    unsafe {
        if liblmdb_sys::mdb_cursor_open(txn, dbi, &mut cursor) != 0 {
            return result;
        }
        let mut k = liblmdb_sys::MDB_val { mv_size: 0, mv_data: std::ptr::null() };
        let mut v = liblmdb_sys::MDB_val { mv_size: 0, mv_data: std::ptr::null() };
        let first_rc = if let Some(sk) = start_key {
            k.mv_size = sk.len();
            k.mv_data = sk.as_ptr() as *const _;
            liblmdb_sys::mdb_cursor_get(cursor, &mut k, &mut v, liblmdb_sys::MDB_cursor_op::MDB_SET_RANGE)
        } else {
            liblmdb_sys::mdb_cursor_get(cursor, &mut k, &mut v, liblmdb_sys::MDB_cursor_op::MDB_FIRST)
        };
        let mut count = 0u32;
        if first_rc == 0 {
            loop {
                if count >= max_entries as u32 { break; }
                if k.mv_size > u16::MAX as usize || v.mv_size > u16::MAX as usize { break; }
                result.extend_from_slice(&(k.mv_size as u16).to_le_bytes());
                result.extend_from_slice(&(v.mv_size as u16).to_le_bytes());
                result.extend_from_slice(std::slice::from_raw_parts(k.mv_data as *const u8, k.mv_size));
                result.extend_from_slice(std::slice::from_raw_parts(v.mv_data as *const u8, v.mv_size));
                count += 1;
                if liblmdb_sys::mdb_cursor_get(cursor, &mut k, &mut v, liblmdb_sys::MDB_cursor_op::MDB_NEXT) != 0 {
                    break;
                }
            }
        }
        liblmdb_sys::mdb_cursor_close(cursor);
        result[0..4].copy_from_slice(&count.to_le_bytes());
    }
    result
}

unsafe extern "C" fn cl_lmdb_init(ptr: *mut u8) {
    let ctx = Box::new(CraneliftLmdbContext {
        envs: HashMap::new(),
        active_write_txns: HashMap::new(),
        next_handle: 0,
    });
    std::ptr::write_unaligned(ptr as *mut *mut CraneliftLmdbContext, Box::into_raw(ctx));
}

unsafe extern "C" fn cl_lmdb_open(ptr: *mut u8, path_off: i64, map_size_mb: i32) -> i32 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftLmdbContext);
    let path_str = read_cstr(ptr, path_off as usize);
    let map_size = if map_size_mb <= 0 { 1024 * 1024 * 1024 } else { (map_size_mb as usize) * 1024 * 1024 };

    if std::fs::create_dir_all(&path_str).is_err() { return -1; }

    let env = match lmdb::EnvBuilder::new() {
        Ok(mut builder) => {
            builder.set_mapsize(map_size).ok();
            builder.set_maxdbs(1).ok();
            let flags = lmdb::open::WRITEMAP | lmdb::open::NOSYNC;
            match builder.open(&path_str, flags, 0o600) {
                Ok(env) => env,
                Err(_) => return -1,
            }
        }
        Err(_) => return -1,
    };

    let dbi = match lmdb::Database::open(&env, None, &lmdb::DatabaseOptions::defaults()) {
        Ok(db) => db.into_raw(),
        Err(_) => return -1,
    };

    let handle = ctx.next_handle;
    ctx.next_handle += 1;
    ctx.envs.insert(handle, (env, dbi));
    handle as i32
}

unsafe extern "C" fn cl_lmdb_put(
    ptr: *mut u8, handle: u32,
    key_off: i64, key_len: i32,
    val_off: i64, val_len: i32,
) -> i32 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftLmdbContext);
    if let Some((env, dbi)) = ctx.envs.get(&handle) {
        let key = std::slice::from_raw_parts(ptr.add(key_off as usize), key_len as usize);
        let val = std::slice::from_raw_parts(ptr.add(val_off as usize), val_len as usize);
        let dbi = *dbi;

        if let Some(&txn) = ctx.active_write_txns.get(&handle) {
            return if lmdb_raw_put(txn, dbi, key, val) { 0 } else { -1 };
        }
        let txn = lmdb_raw_begin_txn(env, false);
        if !txn.is_null() {
            let ok = lmdb_raw_put(txn, dbi, key, val);
            if !ok {
                liblmdb_sys::mdb_txn_abort(txn);
                return -1;
            }
            if liblmdb_sys::mdb_txn_commit(txn) != 0 {
                return -1;
            }
            return 0;
        }
    }
    -1
}

unsafe extern "C" fn cl_lmdb_get(
    ptr: *mut u8, handle: u32,
    key_off: i64, key_len: i32,
    result_off: i64,
) -> i32 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftLmdbContext);
    if let Some((env, dbi)) = ctx.envs.get(&handle) {
        let key = std::slice::from_raw_parts(ptr.add(key_off as usize), key_len as usize);
        let dbi = *dbi;

        let (txn, owned) = match ctx.active_write_txns.get(&handle) {
            Some(&txn) => (txn, false),
            None => (lmdb_raw_begin_txn(env, true), true),
        };
        if !txn.is_null() {
            if let Some(val) = lmdb_raw_get(txn, dbi, key) {
                let len = val.len() as u32;
                let dst = ptr.add(result_off as usize);
                std::ptr::copy_nonoverlapping(len.to_le_bytes().as_ptr(), dst, 4);
                std::ptr::copy_nonoverlapping(val.as_ptr(), dst.add(4), val.len());
                if owned { liblmdb_sys::mdb_txn_abort(txn); }
                return len as i32;
            }
            if owned { liblmdb_sys::mdb_txn_abort(txn); }
        }
    }
    let dst = ptr.add(result_off as usize);
    std::ptr::copy_nonoverlapping(0xFFFF_FFFFu32.to_le_bytes().as_ptr(), dst, 4);
    -1
}

unsafe extern "C" fn cl_lmdb_delete(
    ptr: *mut u8, handle: u32,
    key_off: i64, key_len: i32,
) -> i32 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftLmdbContext);
    if let Some((env, dbi)) = ctx.envs.get(&handle) {
        let key = std::slice::from_raw_parts(ptr.add(key_off as usize), key_len as usize);
        let dbi = *dbi;

        if let Some(&txn) = ctx.active_write_txns.get(&handle) {
            return if lmdb_raw_del(txn, dbi, key) { 0 } else { -1 };
        }
        let txn = lmdb_raw_begin_txn(env, false);
        if !txn.is_null() {
            let ok = lmdb_raw_del(txn, dbi, key);
            if !ok {
                liblmdb_sys::mdb_txn_abort(txn);
                return -1;
            }
            if liblmdb_sys::mdb_txn_commit(txn) != 0 {
                return -1;
            }
            return 0;
        }
    }
    -1
}

unsafe extern "C" fn cl_lmdb_begin_write_txn(ptr: *mut u8, handle: u32) -> i32 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftLmdbContext);
    if let Some(old_txn) = ctx.active_write_txns.remove(&handle) {
        liblmdb_sys::mdb_txn_abort(old_txn);
    }
    if let Some((env, _)) = ctx.envs.get(&handle) {
        let txn = lmdb_raw_begin_txn(env, false);
        if !txn.is_null() {
            ctx.active_write_txns.insert(handle, txn);
            return 0;
        }
    }
    -1
}

unsafe extern "C" fn cl_lmdb_commit_write_txn(ptr: *mut u8, handle: u32) -> i32 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftLmdbContext);
    if let Some(txn) = ctx.active_write_txns.remove(&handle) {
        return if liblmdb_sys::mdb_txn_commit(txn) == 0 { 0 } else { -1 };
    }
    -1
}

unsafe extern "C" fn cl_lmdb_cursor_scan(
    ptr: *mut u8, handle: u32,
    key_off: i64, key_len: i32,
    max_entries: i32,
    result_off: i64,
) -> i32 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftLmdbContext);
    if let Some((env, dbi)) = ctx.envs.get(&handle) {
        let start_key = if key_len > 0 {
            Some(std::slice::from_raw_parts(ptr.add(key_off as usize), key_len as usize))
        } else {
            None
        };
        let dbi = *dbi;

        let (txn, owned) = match ctx.active_write_txns.get(&handle) {
            Some(&txn) => (txn, false),
            None => (lmdb_raw_begin_txn(env, true), true),
        };
        if !txn.is_null() {
            let result = lmdb_raw_cursor_scan(txn, dbi, start_key, max_entries as usize);
            std::ptr::copy_nonoverlapping(result.as_ptr(), ptr.add(result_off as usize), result.len());
            let count = u32::from_le_bytes(result[0..4].try_into().unwrap());
            if owned { liblmdb_sys::mdb_txn_abort(txn); }
            return count as i32;
        }
    }
    let dst = ptr.add(result_off as usize);
    std::ptr::copy_nonoverlapping(0u32.to_le_bytes().as_ptr(), dst, 4);
    0
}

unsafe extern "C" fn cl_lmdb_sync(ptr: *mut u8, handle: u32) -> i32 {
    let ctx = &*std::ptr::read_unaligned(ptr as *const *mut CraneliftLmdbContext);
    if let Some((env, _)) = ctx.envs.get(&handle) {
        match env.sync(true) {
            Ok(_) => return 0,
            Err(_) => return -1,
        }
    }
    -1
}

unsafe extern "C" fn cl_lmdb_cleanup(ptr: *mut u8) {
    let ctx_ptr = std::ptr::read_unaligned(ptr as *const *mut CraneliftLmdbContext);
    drop(Box::from_raw(ctx_ptr));
}

thread_local! {
    static THREAD_COMPILED_FNS: std::cell::RefCell<Option<Arc<Vec<unsafe extern "C" fn(*mut u8)>>>> = const { std::cell::RefCell::new(None) };
}

struct CraneliftThreadContext {
    threads: HashMap<u32, std::thread::JoinHandle<()>>,
    next_handle: u32,
    compiled_fns: Arc<Vec<unsafe extern "C" fn(*mut u8)>>,
}

unsafe extern "C" fn cl_thread_init(ptr: *mut u8) {
    let compiled_fns = THREAD_COMPILED_FNS.with(|cell| {
        cell.borrow().clone().expect("cl_thread_init: no compiled functions available")
    });
    let ctx = Box::new(CraneliftThreadContext {
        threads: HashMap::new(),
        next_handle: 1,
        compiled_fns,
    });
    std::ptr::write_unaligned(ptr as *mut *mut CraneliftThreadContext, Box::into_raw(ctx));
}

unsafe extern "C" fn cl_thread_spawn(ptr: *mut u8, fn_index: i64, thread_ptr: i64) -> i64 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftThreadContext);
    let idx = fn_index as usize;
    if idx >= ctx.compiled_fns.len() { return -1; }
    let func = ctx.compiled_fns[idx];
    let arg = thread_ptr as usize;
    let handle_id = ctx.next_handle;
    ctx.next_handle += 1;

    let compiled_fns_clone = ctx.compiled_fns.clone();
    let join = std::thread::spawn(move || {
        THREAD_COMPILED_FNS.with(|cell| {
            *cell.borrow_mut() = Some(compiled_fns_clone);
        });
        func(arg as *mut u8);
    });

    ctx.threads.insert(handle_id, join);
    handle_id as i64
}

unsafe extern "C" fn cl_thread_join(ptr: *mut u8, handle: i64) -> i64 {
    let ctx = &mut *std::ptr::read_unaligned(ptr as *const *mut CraneliftThreadContext);
    if let Some(join) = ctx.threads.remove(&(handle as u32)) {
        match join.join() {
            Ok(_) => 0,
            Err(_) => -1,
        }
    } else {
        -1
    }
}

unsafe extern "C" fn cl_thread_cleanup(ptr: *mut u8) {
    let ctx_ptr = std::ptr::read_unaligned(ptr as *const *mut CraneliftThreadContext);
    let mut ctx = Box::from_raw(ctx_ptr);
    for (_, join) in ctx.threads.drain() {
        let _ = join.join();
    }
}

pub(crate) struct CraneliftUnit {
    shared: Arc<SharedMemory>,
    compiled_fns: Arc<Vec<unsafe extern "C" fn(*mut u8)>>,
}

impl CraneliftUnit {
    pub fn new(shared: Arc<SharedMemory>, compiled_fns: Arc<Vec<unsafe extern "C" fn(*mut u8)>>) -> Self {
        Self { shared, compiled_fns }
    }

    pub fn compile(clif_source: &str) -> Arc<Vec<unsafe extern "C" fn(*mut u8)>> {
        info!(ir_len = clif_source.len(), "compiling Cranelift IR");

        let mut functions = cranelift_reader::parse_functions(clif_source)
            .expect("Failed to parse CLIF IR");
        assert!(!functions.is_empty(), "No functions in CLIF IR");

        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        let isa_builder = cranelift_native::builder().expect("Host ISA not supported");
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        builder.symbol("ht_create", cl_ht_create as *const u8);
        builder.symbol("ht_lookup", cl_ht_lookup as *const u8);
        builder.symbol("ht_insert", cl_ht_insert as *const u8);
        builder.symbol("ht_count", cl_ht_count as *const u8);
        builder.symbol("ht_get_entry", cl_ht_get_entry as *const u8);
        builder.symbol("ht_increment", cl_ht_increment as *const u8);

        builder.symbol("cl_gpu_init", cl_gpu_init as *const u8);
        builder.symbol("cl_gpu_create_buffer", cl_gpu_create_buffer as *const u8);
        builder.symbol("cl_gpu_create_pipeline", cl_gpu_create_pipeline as *const u8);
        builder.symbol("cl_gpu_upload", cl_gpu_upload as *const u8);
        builder.symbol("cl_gpu_dispatch", cl_gpu_dispatch as *const u8);
        builder.symbol("cl_gpu_download", cl_gpu_download as *const u8);
        builder.symbol("cl_gpu_cleanup", cl_gpu_cleanup as *const u8);

        builder.symbol("cl_file_read", cl_file_read as *const u8);
        builder.symbol("cl_file_write", cl_file_write as *const u8);

        builder.symbol("cl_net_init", cl_net_init as *const u8);
        builder.symbol("cl_net_listen", cl_net_listen as *const u8);
        builder.symbol("cl_net_connect", cl_net_connect as *const u8);
        builder.symbol("cl_net_accept", cl_net_accept as *const u8);
        builder.symbol("cl_net_send", cl_net_send as *const u8);
        builder.symbol("cl_net_recv", cl_net_recv as *const u8);
        builder.symbol("cl_net_cleanup", cl_net_cleanup as *const u8);

        builder.symbol("cl_lmdb_init", cl_lmdb_init as *const u8);
        builder.symbol("cl_lmdb_open", cl_lmdb_open as *const u8);
        builder.symbol("cl_lmdb_put", cl_lmdb_put as *const u8);
        builder.symbol("cl_lmdb_get", cl_lmdb_get as *const u8);
        builder.symbol("cl_lmdb_delete", cl_lmdb_delete as *const u8);
        builder.symbol("cl_lmdb_begin_write_txn", cl_lmdb_begin_write_txn as *const u8);
        builder.symbol("cl_lmdb_commit_write_txn", cl_lmdb_commit_write_txn as *const u8);
        builder.symbol("cl_lmdb_cursor_scan", cl_lmdb_cursor_scan as *const u8);
        builder.symbol("cl_lmdb_sync", cl_lmdb_sync as *const u8);
        builder.symbol("cl_lmdb_cleanup", cl_lmdb_cleanup as *const u8);

        builder.symbol("cl_thread_init", cl_thread_init as *const u8);
        builder.symbol("cl_thread_spawn", cl_thread_spawn as *const u8);
        builder.symbol("cl_thread_join", cl_thread_join as *const u8);
        builder.symbol("cl_thread_cleanup", cl_thread_cleanup as *const u8);

        let mut module = cranelift_jit::JITModule::new(builder);

        // cranelift_reader parses `%name` as ExternalName::TestCase; fix up to ExternalName::User
        for func in functions.iter_mut() {
            let mut fixups = Vec::new();
            for (fref, data) in func.dfg.ext_funcs.iter() {
                if let cranelift_codegen::ir::ExternalName::TestCase(testcase) = &data.name {
                    let name = testcase.to_string();
                    let name = name.strip_prefix('%').unwrap_or(&name).to_string();
                    let sig = func.dfg.signatures[data.signature].clone();
                    fixups.push((fref, name, sig));
                }
            }
            for (fref, name, sig) in fixups {
                let fid = module
                    .declare_function(&name, cranelift_module::Linkage::Import, &sig)
                    .expect("Failed to declare imported function");
                let user_ref = func.declare_imported_user_function(
                    cranelift_codegen::ir::UserExternalName { namespace: 0, index: fid.as_u32() },
                );
                func.dfg.ext_funcs[fref].name = cranelift_codegen::ir::ExternalName::user(user_ref);
                func.dfg.ext_funcs[fref].colocated = false;
            }
        }

        let mut func_ids = Vec::with_capacity(functions.len());
        for (i, func) in functions.into_iter().enumerate() {
            let name = format!("fn_{}", i);
            let func_id = module
                .declare_function(&name, cranelift_module::Linkage::Local, &func.signature)
                .expect("Failed to declare function");
            let mut ctx = cranelift_codegen::Context::for_function(func);
            module.define_function(func_id, &mut ctx).expect("Failed to compile function");
            func_ids.push(func_id);
        }
        module.finalize_definitions().unwrap();

        let compiled_fns: Vec<unsafe extern "C" fn(*mut u8)> = func_ids
            .iter()
            .map(|&id| {
                let code_ptr = module.get_finalized_function(id);
                unsafe { std::mem::transmute(code_ptr) }
            })
            .collect();

        info!(count = compiled_fns.len(), "Cranelift IR compiled successfully");
        Box::leak(Box::new(module));
        Arc::new(compiled_fns)
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        let fn_idx = (action.src as usize) % self.compiled_fns.len();
        let ptr = self.shared.ptr.add(action.dst as usize);
        (self.compiled_fns[fn_idx])(ptr);
    }
}

pub(crate) fn cranelift_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    compiled_fns: Arc<Vec<unsafe extern "C" fn(*mut u8)>>,
) {
    let _span = info_span!("cranelift_unit").entered();
    info!("Cranelift unit started");

    THREAD_COMPILED_FNS.with(|cell| {
        *cell.borrow_mut() = Some(compiled_fns.clone());
    });

    // Create hash table context for CLIF-callable HT functions and store its pointer
    // at offset 0 in shared memory (HT_CTX_PTR slot, written before JIT code runs).
    let ctx = Box::new(CraneliftHashTableContext::new());
    let ctx_ptr = Box::into_raw(ctx);
    unsafe {
        shared.store_u64(CL_HT_CTX_OFFSET, ctx_ptr as u64, Ordering::Release);
    }

    let mut unit = CraneliftUnit::new(shared.clone(), compiled_fns);
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "cranelift_work_received");
                for idx in start..end {
                    unsafe {
                        unit.execute(&actions[idx as usize]);
                    }
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "cranelift_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("Cranelift unit shutting down");
                unsafe { drop(Box::from_raw(ctx_ptr)); }
                return;
            }
            MailboxPoll::Empty => spin_backoff(&mut spin_count),
        }
    }
}
