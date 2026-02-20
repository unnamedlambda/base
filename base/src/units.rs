use base_types::{Action, Kind};
use pollster::block_on;
use portable_atomic::{AtomicU128, AtomicU64, Ordering};
use quanta::Clock;
use std::collections::HashMap;
use std::fs;
use std::io::{Read as IoRead, Seek, Write as IoWrite};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{fence, AtomicBool, AtomicU32};
use std::sync::Arc;
use tracing::{debug, info, info_span, trace, warn};
use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    InstanceDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, PowerPreference,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};
use wide::{f32x4, i32x4};
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

pub(crate) struct Broadcast {
    epoch: AtomicU64,
    start: AtomicU32,
    end: AtomicU32,
    flag: AtomicU32,
    done: AtomicU32,
    num_workers: u32,
    shutdown: AtomicBool,
}

impl Broadcast {
    pub fn new(num_workers: u32) -> Self {
        Self {
            epoch: AtomicU64::new(0),
            start: AtomicU32::new(0),
            end: AtomicU32::new(0),
            flag: AtomicU32::new(0),
            done: AtomicU32::new(0),
            num_workers,
            shutdown: AtomicBool::new(false),
        }
    }

    pub fn dispatch(&self, start: u32, end: u32, flag: u32) {
        debug!(start, end, flag, workers = self.num_workers, "broadcast_dispatch");
        self.start.store(start, Ordering::Relaxed);
        self.end.store(end, Ordering::Relaxed);
        self.flag.store(flag, Ordering::Relaxed);
        self.done.store(self.num_workers, Ordering::Relaxed);
        self.epoch.fetch_add(1, Ordering::Release);
    }

    pub fn shutdown(&self) {
        debug!("broadcast_shutdown");
        self.shutdown.store(true, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::Release);
    }
}

fn broadcast_step(
    worker_id: u32,
    last_epoch: &mut u64,
    broadcast: &Broadcast,
    actions: &Arc<Vec<Action>>,
    shared: &Arc<SharedMemory>,
    mut exec: impl FnMut(&Action),
) -> bool {
    if broadcast.shutdown.load(Ordering::Acquire) {
        return false;
    }

    let epoch = broadcast.epoch.load(Ordering::Acquire);
    if epoch == *last_epoch {
        return true;
    }

    if broadcast.shutdown.load(Ordering::Acquire) {
        return false;
    }

    let start = broadcast.start.load(Ordering::Relaxed);
    let end = broadcast.end.load(Ordering::Relaxed);
    let flag = broadcast.flag.load(Ordering::Relaxed);
    let num_workers = broadcast.num_workers.max(1);

    let total = end.saturating_sub(start);
    if total == 0 {
        *last_epoch = epoch;
        return true;
    }

    let chunk = (total + num_workers - 1) / num_workers;
    let my_start = start.saturating_add(worker_id.saturating_mul(chunk));
    let my_end = (my_start + chunk).min(end);

    trace!(worker_id, my_start, my_end, "broadcast_work_received");

    for idx in my_start..my_end {
        exec(&actions[idx as usize]);
    }

    if broadcast.done.fetch_sub(1, Ordering::AcqRel) == 1 {
        unsafe {
            shared.store_u64(flag as usize, 1, Ordering::Release);
        }
    }

    *last_epoch = epoch;
    true
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

    pub unsafe fn cas128(&self, offset: usize, expected: u128, new: u128) -> u128 {
        let ptr = self.ptr.add(offset);
        debug_assert!((ptr as usize) & 0xF == 0, "cas128: pointer not 16-byte aligned (offset {offset})");
        (*(ptr as *const AtomicU128))
            .compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or_else(|x| x)
    }
}

fn read_as_u64(shared: &SharedMemory, offset: usize) -> u64 {
    u64::from_le_bytes(unsafe { shared.read(offset, 8)[0..8].try_into().unwrap() })
}

fn read_as_u128(shared: &SharedMemory, offset: usize) -> u128 {
    u128::from_le_bytes(unsafe { shared.read(offset, 16)[0..16].try_into().unwrap() })
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

unsafe fn store_sized(shared: &SharedMemory, offset: usize, size: u32, value: u64, order: Ordering) {
    match size {
        1 => shared.write(offset, &[(value & 0xFF) as u8]),
        2 => shared.write(offset, &((value as u16).to_le_bytes())),
        4 => shared.write(offset, &((value as u32).to_le_bytes())),
        8 => shared.store_u64(offset, value, order),
        _ => {}
    }
}

pub(crate) const QUEUE_HEAD_OFF: usize = 0;
pub(crate) const QUEUE_TAIL_OFF: usize = 4;
pub(crate) const QUEUE_MASK_OFF: usize = 8;
pub(crate) const QUEUE_BASE_OFF: usize = 12;
pub(crate) const QUEUE_RESERVE_OFF: usize = 20;

pub(crate) unsafe fn queue_try_push_packet_mp(
    shared: &SharedMemory,
    queue_desc: usize,
    data_src: usize,
) -> bool {
    let mask = shared.load_u32(queue_desc + QUEUE_MASK_OFF, Ordering::Relaxed);
    let base = shared.load_u32(queue_desc + QUEUE_BASE_OFF, Ordering::Relaxed);
    let cap = mask.wrapping_add(1);

    let reserved = loop {
        let head = shared.load_u32(queue_desc + QUEUE_HEAD_OFF, Ordering::Acquire);
        let reserve = shared.load_u32(queue_desc + QUEUE_RESERVE_OFF, Ordering::Relaxed);
        if reserve.wrapping_sub(head) >= cap {
            return false;
        }
        match shared.cas_u32(
            queue_desc + QUEUE_RESERVE_OFF,
            reserve,
            reserve.wrapping_add(1),
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => break reserve,
            Err(_) => std::hint::spin_loop(),
        }
    };

    let slot = (reserved & mask) as usize;
    let slot_off = base as usize + slot * 8;
    let data = shared.read(data_src, 8);
    shared.write(slot_off, data);

    loop {
        let tail = shared.load_u32(queue_desc + QUEUE_TAIL_OFF, Ordering::Acquire);
        if tail == reserved {
            match shared.cas_u32(
                queue_desc + QUEUE_TAIL_OFF,
                tail,
                tail.wrapping_add(1),
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => std::hint::spin_loop(),
            }
        } else {
            std::hint::spin_loop();
        }
    }

    true
}

fn read_null_terminated_string(shared: &SharedMemory, offset: usize, max_len: usize) -> String {
    unsafe {
        // Read up to max_len bytes
        let bytes = shared.read(offset, max_len);

        // Find null terminator
        let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());

        // Convert to string
        String::from_utf8_lossy(&bytes[..len]).into_owned()
    }
}

pub(crate) fn read_null_terminated_string_from_slice(data: &[u8], offset: usize, max_len: usize) -> String {
    let end = (offset + max_len).min(data.len());
    let bytes = &data[offset..end];
    let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..len]).into_owned()
}

pub(crate) struct MemoryUnit {
    shared: Arc<SharedMemory>,
}

impl MemoryUnit {
    pub fn new(shared: Arc<SharedMemory>) -> Self {
        Self { shared }
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::ConditionalWrite => {
                debug!(dst = action.dst, src = action.src, size = action.size, offset = action.offset, "mem_conditional_write");
                // Read first 8 bytes at offset as condition
                let cond_bytes = self.shared.read(action.offset as usize, 8);
                let condition = u64::from_le_bytes(cond_bytes[0..8].try_into().unwrap());

                if condition != 0 {
                    let src_ptr = self.shared.ptr.add(action.src as usize);
                    let dst_ptr = self.shared.ptr.add(action.dst as usize);
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, action.size as usize);
                }
            }
            Kind::MemCopy => {
                debug!(dst = action.dst, src = action.src, size = action.size, "mem_copy");
                let src_ptr = self.shared.ptr.add(action.src as usize);
                let dst_ptr = self.shared.ptr.add(action.dst as usize);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, action.size as usize);
            }
            Kind::MemWrite => {
                debug!(dst = action.dst, src = action.src, size = action.size, "mem_write");
                // Write immediate value to memory
                // dst = destination address, src = immediate value, size = bytes
                let dst_ptr = self.shared.ptr.add(action.dst as usize);
                match action.size {
                    1 => *dst_ptr = action.src as u8,
                    2 => std::ptr::write_unaligned(dst_ptr as *mut u16, action.src as u16),
                    4 => std::ptr::write_unaligned(dst_ptr as *mut u32, action.src),
                    8 => std::ptr::write_unaligned(dst_ptr as *mut u64, action.src as u64),
                    n => {
                        std::ptr::write_bytes(dst_ptr, action.src as u8, n as usize);
                    }
                }
            }
            Kind::MemCopyIndirect => {
                debug!(dst = action.dst, src = action.src, size = action.size, offset = action.offset, "mem_copy_indirect");
                // src = address containing source pointer (u32), dst = destination, offset added to indirect addr
                let indirect_addr_bytes = self.shared.read(action.src as usize, 4);
                let indirect_addr = u32::from_le_bytes([
                    indirect_addr_bytes[0], indirect_addr_bytes[1],
                    indirect_addr_bytes[2], indirect_addr_bytes[3],
                ]) as usize + action.offset as usize;
                let src_ptr = self.shared.ptr.add(indirect_addr);
                let dst_ptr = self.shared.ptr.add(action.dst as usize);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, action.size as usize);
            }
            Kind::MemStoreIndirect => {
                debug!(dst = action.dst, src = action.src, size = action.size, offset = action.offset, "mem_store_indirect");
                // src = source data, dst = address containing destination pointer (u32), offset added to indirect addr
                let indirect_addr_bytes = self.shared.read(action.dst as usize, 4);
                let indirect_addr = u32::from_le_bytes([
                    indirect_addr_bytes[0], indirect_addr_bytes[1],
                    indirect_addr_bytes[2], indirect_addr_bytes[3],
                ]) as usize + action.offset as usize;
                let src_ptr = self.shared.ptr.add(action.src as usize);
                let dst_ptr = self.shared.ptr.add(indirect_addr);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, action.size as usize);
            }
            Kind::AtomicLoad => {
                let order = order_from_u32(action.offset);
                match action.size {
                    1 => {
                        let value = self.shared.read(action.src as usize, 1)[0];
                        self.shared.write(action.dst as usize, &[value]);
                    }
                    2 => {
                        let value = self.shared.read(action.src as usize, 2);
                        self.shared.write(action.dst as usize, &value[0..2]);
                    }
                    4 => {
                        let value = self.shared.load_u32(action.src as usize, order);
                        self.shared.write(action.dst as usize, &value.to_le_bytes());
                    }
                    8 => {
                        let value = self.shared.load_u64(action.src as usize, order);
                        self.shared.store_u64(action.dst as usize, value, order);
                    }
                    _ => {}
                }
            }
            Kind::AtomicStore => {
                let order = order_from_u32(action.offset);
                match action.size {
                    1 => {
                        let value = self.shared.read(action.src as usize, 1)[0];
                        self.shared.write(action.dst as usize, &[value]);
                    }
                    2 => {
                        let value = self.shared.read(action.src as usize, 2);
                        self.shared.write(action.dst as usize, &value[0..2]);
                    }
                    4 => {
                        let bytes = self.shared.read(action.src as usize, 4);
                        let value = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
                        self.shared.store_u32(action.dst as usize, value, order);
                    }
                    8 => {
                        let bytes = self.shared.read(action.src as usize, 8);
                        let value = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                        self.shared.store_u64(action.dst as usize, value, order);
                    }
                    _ => {}
                }
            }
            Kind::AtomicFetchAdd => {
                let order = order_from_u32(action.size >> 29);
                let op_size = action.size & 0x1FFF_FFFF;
                let addend = load_sized(&self.shared, action.offset as usize, op_size, Ordering::Relaxed);
                let prev = match op_size {
                    4 => {
                        let ptr = self.shared.ptr.add(action.dst as usize) as *const AtomicU32;
                        (*ptr).fetch_add(addend as u32, order) as u64
                    }
                    8 => {
                        let ptr = self.shared.ptr.add(action.dst as usize) as *const AtomicU64;
                        (*ptr).fetch_add(addend as u64, order)
                    }
                    _ => 0,
                };
                store_sized(&self.shared, action.src as usize, op_size, prev, Ordering::Relaxed);
            }
            Kind::AtomicFetchSub => {
                let order = order_from_u32(action.size >> 29);
                let op_size = action.size & 0x1FFF_FFFF;
                let addend = load_sized(&self.shared, action.offset as usize, op_size, Ordering::Relaxed);
                let prev = match op_size {
                    4 => {
                        let ptr = self.shared.ptr.add(action.dst as usize) as *const AtomicU32;
                        (*ptr).fetch_sub(addend as u32, order) as u64
                    }
                    8 => {
                        let ptr = self.shared.ptr.add(action.dst as usize) as *const AtomicU64;
                        (*ptr).fetch_sub(addend as u64, order)
                    }
                    _ => 0,
                };
                store_sized(&self.shared, action.src as usize, op_size, prev, Ordering::Relaxed);
            }
            Kind::QueuePushPacketMP => {
                let ok =
                    queue_try_push_packet_mp(&self.shared, action.dst as usize, action.src as usize);
                self.shared.store_u64(action.offset as usize, if ok { 1 } else { 0 }, Ordering::Release);
            }
            Kind::MemScan => {
                debug!(src = action.src, dst = action.dst, size = action.size, offset = action.offset, "mem_scan");
                // action.src = pattern start offset
                // action.dst = search region start offset
                // action.size = search region size
                // action.offset = pattern size (lower 16 bits) | result offset (upper 16 bits)

                let pattern_size = (action.offset & 0xFFFF) as usize;
                let result_offset = (action.offset >> 16) as usize;

                if pattern_size == 0 || pattern_size > action.size as usize {
                    // Invalid pattern size - write -1 (not found)
                    self.shared.write(result_offset, &(-1i64).to_le_bytes());
                    return;
                }

                let pattern = self.shared.read(action.src as usize, pattern_size);
                let search_region = self.shared.read(action.dst as usize, action.size as usize);

                // Search for pattern in region
                let mut found_offset = -1i64;

                if pattern_size == 1 {
                    // Optimize single byte search
                    if let Some(pos) = search_region.iter().position(|&b| b == pattern[0]) {
                        found_offset = (action.dst as i64) + (pos as i64);
                    }
                } else {
                    // Multi-byte pattern search
                    for i in 0..=(search_region.len() - pattern_size) {
                        if &search_region[i..i + pattern_size] == pattern {
                            found_offset = (action.dst as i64) + (i as i64);
                            break;
                        }
                    }
                }

                // Write result offset (or -1 if not found)
                self.shared
                    .write(result_offset, &found_offset.to_le_bytes());
            }
            Kind::AtomicCAS => {
                debug!(dst = action.dst, src = action.src, size = action.size, "atomic_cas");
                if action.size == 16 {
                    let expected = read_as_u128(&self.shared, action.src as usize);
                    let new = read_as_u128(&self.shared, action.offset as usize);
                    let actual = self.shared.cas128(action.dst as usize, expected, new);
                    self.shared
                        .write(action.src as usize, &actual.to_le_bytes());
                } else if action.size == 4 {
                    let expected_bytes = self.shared.read(action.src as usize, 4);
                    let expected =
                        u32::from_le_bytes(expected_bytes[0..4].try_into().unwrap());
                    let new_bytes = self.shared.read(action.offset as usize, 4);
                    let new =
                        u32::from_le_bytes(new_bytes[0..4].try_into().unwrap());
                    let observed = match self.shared.cas_u32(
                        action.dst as usize,
                        expected,
                        new,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(prev) => prev,
                        Err(actual) => actual,
                    };
                    self.shared
                        .write(action.src as usize, &observed.to_le_bytes());
                } else {
                    let expected = read_as_u64(&self.shared, action.src as usize);
                    let new = read_as_u64(&self.shared, action.offset as usize);
                    let actual = self.shared.cas64(action.dst as usize, expected, new);
                    self.shared
                        .write(action.src as usize, &actual.to_le_bytes()[0..8]);
                }
            }
            Kind::Fence => {
                trace!("fence");
                fence(Ordering::SeqCst);
            }
            Kind::Compare => {
                debug!(src = action.src, dst = action.dst, offset = action.offset, size = action.size, "compare");
                let a_bytes = self.shared.read(action.src as usize, 4);
                let a = i32::from_le_bytes(a_bytes[0..4].try_into().unwrap());
                let b_bytes = self.shared.read(action.offset as usize, 4);
                let b = i32::from_le_bytes(b_bytes[0..4].try_into().unwrap());
                let result: i32 = if action.size == 5 {
                    if a >= b { 1 } else { 0 }
                } else {
                    if a > b { 1 } else { 0 }
                };
                self.shared.write(action.dst as usize, &result.to_le_bytes());
            }
            _ => {}
        }
    }
}

pub(crate) struct FFIUnit {
    shared: Arc<SharedMemory>,
}

impl FFIUnit {
    pub fn new(shared: Arc<SharedMemory>) -> Self {
        Self { shared }
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::FFICall => {
                debug!(src = action.src, dst = action.dst, offset = action.offset, "ffi_call");
                // Read function pointer from memory
                let fn_bytes = self.shared.read(action.src as usize, 8);
                let fn_ptr = usize::from_le_bytes(fn_bytes[0..8].try_into().unwrap());

                if fn_ptr == 0 {
                    warn!("ffi_call skipped: null function pointer");
                    return; // Skip null pointer
                }

                // Get arg pointer - this should be within our memory
                let arg_ptr = self.shared.ptr.add(action.dst as usize);

                // Cast and call
                let func: unsafe extern "C" fn(*mut u8) -> i64 = std::mem::transmute(fn_ptr);
                let result = func(arg_ptr);

                // Store result
                self.shared
                    .write(action.offset as usize, &result.to_le_bytes());
            }
            _ => {}
        }
    }
}

pub(crate) struct HashTableUnit {
    shared: Arc<SharedMemory>,
    tables: HashMap<u32, HashMap<Vec<u8>, Vec<u8>>>,
    next_handle: u32,
}

impl HashTableUnit {
    pub fn new(shared: Arc<SharedMemory>) -> Self {
        Self {
            shared,
            tables: HashMap::new(),
            next_handle: 0,
        }
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::HashTableCreate => {
                debug!(dst = action.dst, "hash_table_create");
                let handle = self.next_handle;
                self.next_handle += 1;
                self.tables.insert(handle, HashMap::new());
                self.shared
                    .write(action.dst as usize, &handle.to_le_bytes());
            }
            Kind::HashTableInsert => {
                debug!(handle = action.offset, key_dst = action.dst, val_src = action.src, "hash_table_insert");
                let handle = action.offset;
                let key_size = (action.size >> 16) as usize;
                let val_size = (action.size & 0xFFFF) as usize;
                let key = self.shared.read(action.dst as usize, key_size).to_vec();
                let val = self.shared.read(action.src as usize, val_size).to_vec();
                if let Some(table) = self.tables.get_mut(&handle) {
                    table.insert(key, val);
                }
            }
            Kind::HashTableLookup => {
                debug!(handle = action.offset, key_dst = action.dst, result_src = action.src, "hash_table_lookup");
                let handle = action.offset;
                let key_size = (action.size >> 16) as usize;
                let key = self.shared.read(action.dst as usize, key_size).to_vec();
                if let Some(table) = self.tables.get(&handle) {
                    if let Some(val) = table.get(&key) {
                        let len = val.len() as u32;
                        self.shared
                            .write(action.src as usize, &len.to_le_bytes());
                        self.shared
                            .write(action.src as usize + 4, val);
                        return;
                    }
                }
                // Not found - write sentinel
                self.shared
                    .write(action.src as usize, &0xFFFF_FFFFu32.to_le_bytes());
            }
            Kind::HashTableDelete => {
                debug!(handle = action.offset, key_dst = action.dst, "hash_table_delete");
                let handle = action.offset;
                let key_size = action.size as usize;
                let key = self.shared.read(action.dst as usize, key_size).to_vec();
                if let Some(table) = self.tables.get_mut(&handle) {
                    table.remove(&key);
                }
            }
            _ => {}
        }
    }
}

pub(crate) fn hash_table_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
) {
    let _span = info_span!("hash_table_unit").entered();
    info!("hash table unit started");
    let mut unit = HashTableUnit::new(shared.clone());
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "hash_table_work_received");
                for idx in start..end {
                    unsafe {
                        unit.execute(&actions[idx as usize]);
                    }
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "hash_table_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("hash table unit shutting down");
                return;
            }
            MailboxPoll::Empty => spin_backoff(&mut spin_count),
        }
    }
}

pub(crate) struct LmdbUnit {
    shared: Arc<SharedMemory>,
    envs: HashMap<u32, (lmdb::Environment, liblmdb_sys::MDB_dbi)>,
    active_write_txns: HashMap<u32, *mut liblmdb_sys::MDB_txn>,
    next_handle: u32,
}

impl Drop for LmdbUnit {
    fn drop(&mut self) {
        for (_handle, txn) in self.active_write_txns.drain() {
            unsafe { liblmdb_sys::mdb_txn_abort(txn); }
        }
    }
}

impl LmdbUnit {
    pub fn new(shared: Arc<SharedMemory>) -> Self {
        Self {
            shared,
            envs: HashMap::new(),
            active_write_txns: HashMap::new(),
            next_handle: 0,
        }
    }

    fn raw_begin_txn(env: &lmdb::Environment, readonly: bool) -> *mut liblmdb_sys::MDB_txn {
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

    fn raw_put(txn: *mut liblmdb_sys::MDB_txn, dbi: liblmdb_sys::MDB_dbi, key: &[u8], val: &[u8]) -> bool {
        let mut k = liblmdb_sys::MDB_val { mv_size: key.len(), mv_data: key.as_ptr() as *const _ };
        let mut v = liblmdb_sys::MDB_val { mv_size: val.len(), mv_data: val.as_ptr() as *const _ };
        unsafe { liblmdb_sys::mdb_put(txn, dbi, &mut k, &mut v, 0) == 0 }
    }

    fn raw_get(txn: *mut liblmdb_sys::MDB_txn, dbi: liblmdb_sys::MDB_dbi, key: &[u8]) -> Option<Vec<u8>> {
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

    fn raw_del(txn: *mut liblmdb_sys::MDB_txn, dbi: liblmdb_sys::MDB_dbi, key: &[u8]) -> bool {
        let mut k = liblmdb_sys::MDB_val { mv_size: key.len(), mv_data: key.as_ptr() as *const _ };
        unsafe { liblmdb_sys::mdb_del(txn, dbi, &mut k, std::ptr::null_mut()) == 0 }
    }

    fn raw_cursor_scan(
        txn: *mut liblmdb_sys::MDB_txn,
        dbi: liblmdb_sys::MDB_dbi,
        start_key: Option<&[u8]>,
        max_entries: usize,
    ) -> Vec<u8> {
        let mut result = Vec::new();
        result.extend_from_slice(&0u32.to_le_bytes()); // count placeholder

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

    pub unsafe fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::LmdbOpen => {
                debug!(dst = action.dst, src = action.src, "lmdb_open");
                let path_str = read_null_terminated_string(
                    &self.shared,
                    action.src as usize,
                    action.offset as usize,
                );

                let map_size = if action.size == 0 {
                    1024 * 1024 * 1024
                } else {
                    (action.size as usize) * 1024 * 1024
                };

                if let Err(_) = std::fs::create_dir_all(&path_str) {
                    self.shared
                        .write(action.dst as usize, &0xFFFF_FFFFu32.to_le_bytes());
                    return;
                }

                let env = match lmdb::EnvBuilder::new() {
                    Ok(mut builder) => {
                        builder.set_mapsize(map_size).ok();
                        builder.set_maxdbs(1).ok();
                        let flags = lmdb::open::WRITEMAP | lmdb::open::NOSYNC;
                        match builder.open(&path_str, flags, 0o600) {
                            Ok(env) => env,
                            Err(_) => {
                                self.shared
                                    .write(action.dst as usize, &0xFFFF_FFFFu32.to_le_bytes());
                                return;
                            }
                        }
                    }
                    Err(_) => {
                        self.shared
                            .write(action.dst as usize, &0xFFFF_FFFFu32.to_le_bytes());
                        return;
                    }
                };

                // Open database once and cache the raw DBI handle
                let dbi = match lmdb::Database::open(&env, None, &lmdb::DatabaseOptions::defaults()) {
                    Ok(db) => db.into_raw(),
                    Err(_) => {
                        self.shared
                            .write(action.dst as usize, &0xFFFF_FFFFu32.to_le_bytes());
                        return;
                    }
                };

                let handle = self.next_handle;
                self.next_handle += 1;
                self.envs.insert(handle, (env, dbi));
                self.shared
                    .write(action.dst as usize, &handle.to_le_bytes());
            }

            Kind::LmdbBeginWriteTxn => {
                debug!(handle = action.offset, "lmdb_begin_write_txn");
                let handle = action.offset;
                // Abort existing txn to avoid deadlock (LMDB allows one write txn per env)
                if let Some(old_txn) = self.active_write_txns.remove(&handle) {
                    liblmdb_sys::mdb_txn_abort(old_txn);
                }
                if let Some((env, _dbi)) = self.envs.get(&handle) {
                    let txn = Self::raw_begin_txn(env, false);
                    if !txn.is_null() {
                        self.active_write_txns.insert(handle, txn);
                    }
                }
            }

            Kind::LmdbCommitWriteTxn => {
                debug!(handle = action.offset, "lmdb_commit_write_txn");
                let handle = action.offset;
                if let Some(txn) = self.active_write_txns.remove(&handle) {
                    liblmdb_sys::mdb_txn_commit(txn);
                }
            }

            Kind::LmdbPut => {
                debug!(handle = action.offset, key_dst = action.dst, val_src = action.src, "lmdb_put");
                let handle = action.offset;
                let key_size = (action.size >> 16) as usize;
                let val_size = (action.size & 0xFFFF) as usize;

                if let Some((env, dbi)) = self.envs.get(&handle) {
                    let key = self.shared.read(action.dst as usize, key_size);
                    let val = self.shared.read(action.src as usize, val_size);

                    let txn = self.active_write_txns.get(&handle).copied();
                    if let Some(txn) = txn {
                        Self::raw_put(txn, *dbi, key, val);
                    } else {
                        let txn = Self::raw_begin_txn(env, false);
                        if !txn.is_null() {
                            Self::raw_put(txn, *dbi, key, val);
                            liblmdb_sys::mdb_txn_commit(txn);
                        }
                    }
                }
            }

            Kind::LmdbGet => {
                debug!(handle = action.offset, key_dst = action.dst, result_src = action.src, "lmdb_get");
                let handle = action.offset;
                let key_size = (action.size >> 16) as usize;

                if let Some((env, dbi)) = self.envs.get(&handle) {
                    let key = self.shared.read(action.dst as usize, key_size);

                    let (txn, owned) = match self.active_write_txns.get(&handle) {
                        Some(&txn) => (txn, false),
                        None => (Self::raw_begin_txn(env, true), true),
                    };
                    if !txn.is_null() {
                        if let Some(val) = Self::raw_get(txn, *dbi, key) {
                            let len = val.len() as u32;
                            self.shared.write(action.src as usize, &len.to_le_bytes());
                            self.shared.write(action.src as usize + 4, &val);
                            if owned { liblmdb_sys::mdb_txn_abort(txn); }
                            return;
                        }
                        if owned { liblmdb_sys::mdb_txn_abort(txn); }
                    }
                }
                self.shared
                    .write(action.src as usize, &0xFFFF_FFFFu32.to_le_bytes());
            }

            Kind::LmdbDelete => {
                debug!(handle = action.offset, key_dst = action.dst, "lmdb_delete");
                let handle = action.offset;
                let key_size = (action.size >> 16) as usize;

                if let Some((env, dbi)) = self.envs.get(&handle) {
                    let key = self.shared.read(action.dst as usize, key_size);

                    let txn = self.active_write_txns.get(&handle).copied();
                    if let Some(txn) = txn {
                        Self::raw_del(txn, *dbi, key);
                    } else {
                        let txn = Self::raw_begin_txn(env, false);
                        if !txn.is_null() {
                            Self::raw_del(txn, *dbi, key);
                            liblmdb_sys::mdb_txn_commit(txn);
                        }
                    }
                }
            }

            Kind::LmdbCursorScan => {
                let handle = action.offset;
                let key_len = (action.size >> 16) as usize;
                let max_entries = (action.size & 0xFFFF) as usize;
                debug!(
                    handle = handle,
                    result_dst = action.dst,
                    key_len = key_len,
                    max_entries = max_entries,
                    "lmdb_cursor_scan"
                );

                if let Some((env, dbi)) = self.envs.get(&handle) {
                    let start_key = if key_len > 0 {
                        Some(self.shared.read(action.src as usize, key_len).to_vec())
                    } else {
                        None
                    };

                    let (txn, owned) = match self.active_write_txns.get(&handle) {
                        Some(&txn) => (txn, false),
                        None => (Self::raw_begin_txn(env, true), true),
                    };
                    if !txn.is_null() {
                        let result = Self::raw_cursor_scan(txn, *dbi, start_key.as_deref(), max_entries);
                        self.shared.write(action.dst as usize, &result);
                        if owned { liblmdb_sys::mdb_txn_abort(txn); }
                    } else {
                        self.shared.write(action.dst as usize, &0u32.to_le_bytes());
                    }
                } else {
                    self.shared
                        .write(action.dst as usize, &0u32.to_le_bytes());
                }
            }

            Kind::LmdbSync => {
                debug!(handle = action.offset, "lmdb_sync");
                let handle = action.offset;
                if let Some((env, _dbi)) = self.envs.get(&handle) {
                    let _ = env.sync(true);
                }
            }

            _ => {}
        }
    }
}

pub(crate) fn lmdb_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
) {
    let _span = info_span!("lmdb_unit").entered();
    info!("LMDB unit started");
    let mut unit = LmdbUnit::new(shared.clone());
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "lmdb_work_received");
                for idx in start..end {
                    unsafe {
                        unit.execute(&actions[idx as usize]);
                    }
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "lmdb_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("LMDB unit shutting down");
                return;
            }
            MailboxPoll::Empty => spin_backoff(&mut spin_count),
        }
    }
}

pub(crate) struct NetworkUnit {
    id: u8,
    shared: Arc<SharedMemory>,
    connections: HashMap<u32, TcpStream>,
    listeners: HashMap<u32, TcpListener>,
    next_handle: u32,
}

impl NetworkUnit {
    pub fn new(id: u8, shared: Arc<SharedMemory>) -> Self {
        Self {
            id,
            shared,
            connections: HashMap::new(),
            listeners: HashMap::new(),
            next_handle: 1,
        }
    }

    pub fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::NetConnect => {
                let addr = read_null_terminated_string(
                    &self.shared,
                    action.src as usize,
                    action.offset as usize,
                );
                debug!(addr = %addr, dst = action.dst, "net_connect");

                if addr.starts_with(':') || addr.contains("0.0.0.0:") {
                    if let Ok(listener) = TcpListener::bind(&addr) {
                        let handle = self.next_handle;
                        self.next_handle += 1;
                        self.listeners.insert(handle, listener);
                        unsafe {
                            self.shared
                                .write(action.dst as usize, &handle.to_le_bytes());
                        }
                    }
                } else if let Ok(stream) = TcpStream::connect(&addr) {
                    let handle = self.next_handle;
                    self.next_handle += 1;
                    self.connections.insert(handle, stream);
                    unsafe {
                        self.shared
                            .write(action.dst as usize, &handle.to_le_bytes());
                    }
                }
            }

            Kind::NetAccept => {
                debug!(src = action.src, dst = action.dst, "net_accept");
                let handle = unsafe {
                    u32::from_le_bytes(
                        self.shared.read(action.src as usize, 4)[0..4]
                            .try_into()
                            .unwrap(),
                    )
                };

                if let Some(listener) = self.listeners.get_mut(&handle) {
                    if let Ok((stream, _addr)) = listener.accept() {
                        let conn_handle = self.next_handle;
                        self.next_handle += 1;
                        self.connections.insert(conn_handle, stream);
                        unsafe {
                            self.shared
                                .write(action.dst as usize, &conn_handle.to_le_bytes());
                        }
                    }
                }
            }

            Kind::NetSend => {
                debug!(dst = action.dst, src = action.src, size = action.size, "net_send");
                let handle = unsafe {
                    u32::from_le_bytes(
                        self.shared.read(action.dst as usize, 4)[0..4]
                            .try_into()
                            .unwrap(),
                    )
                };

                if let Some(stream) = self.connections.get_mut(&handle) {
                    let data = unsafe { self.shared.read(action.src as usize, action.size as usize) };
                    let _ = IoWrite::write_all(stream, data);
                }
            }

            Kind::NetRecv => {
                debug!(src = action.src, dst = action.dst, size = action.size, "net_recv");
                let handle = unsafe {
                    u32::from_le_bytes(
                        self.shared.read(action.src as usize, 4)[0..4]
                            .try_into()
                            .unwrap(),
                    )
                };

                if let Some(stream) = self.connections.get_mut(&handle) {
                    let mut buffer = vec![0u8; action.size as usize];
                    if let Ok(_) = IoRead::read_exact(stream, &mut buffer) {
                        unsafe {
                            self.shared.write(action.dst as usize, &buffer);
                        }
                    }
                }
            }

            _ => {}
        }
    }
}

pub(crate) struct FileUnit {
    id: u8,
    shared: Arc<SharedMemory>,
    buffer: Vec<u8>,
}

impl FileUnit {
    pub fn new(id: u8, shared: Arc<SharedMemory>, buffer_size: usize) -> Self {
        Self {
            id,
            shared,
            buffer: vec![0u8; buffer_size],
        }
    }

    pub fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::FileRead => {
                let filename = read_null_terminated_string(
                    &self.shared,
                    action.src as usize,
                    4096,
                );
                debug!(filename = %filename, dst = action.dst, offset = action.offset, size = action.size, "file_read");

                if let Ok(mut file) = fs::File::open(&filename) {
                    if action.offset > 0 {
                        let _ = file.seek(std::io::SeekFrom::Start(action.offset as u64));
                    }
                    if action.size == 0 {
                        // Read entire file in chunks
                        let mut total_read = 0;
                        let dst_base = action.dst as usize;

                        loop {
                            match file.read(&mut self.buffer) {
                                Ok(0) => break, // EOF
                                Ok(n) => {
                                    unsafe {
                                        self.shared
                                            .write(dst_base + total_read, &self.buffer[..n]);
                                    }
                                    total_read += n;
                                }
                                Err(_) => break,
                            }
                        }
                    } else {
                        // Read specific amount directly into shared memory
                        let read_size = action.size as usize;
                        let dst_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                self.shared.ptr.add(action.dst as usize),
                                read_size,
                            )
                        };
                        let _ = IoRead::read_exact(&mut file, dst_slice);
                    }
                }
            }
            Kind::FileWrite => {
                let filename = read_null_terminated_string(
                    &self.shared,
                    action.dst as usize,
                    4096,
                );
                debug!(filename = %filename, src = action.src, offset = action.offset, size = action.size, "file_write");

                let file_result = if action.offset == 0 {
                    fs::File::create(&filename)
                } else {
                    fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .open(&filename)
                        
                };

                if let Ok(mut file) = file_result {
                    if action.offset > 0 {
                        let _ = file.seek(std::io::SeekFrom::Start(action.offset as u64));
                    }
                    let src_base = action.src as usize;

                    if action.size == 0 {
                        // Null-terminated mode: find the null byte and write up to it
                        let mut len = 0;
                        while len < self.buffer.len() {
                            let byte = unsafe { *self.shared.ptr.add(src_base + len) };
                            if byte == 0 {
                                break;
                            }
                            len += 1;
                        }

                        if len > 0 {
                            let data = unsafe { self.shared.read(src_base, len) };
                            let _ = file.write_all(data);
                        }
                    } else {
                        let mut written = 0;
                        let total_size = action.size as usize;

                        // Write in chunks
                        while written < total_size {
                            let chunk_size = (total_size - written).min(self.buffer.len());
                            let data = unsafe { self.shared.read(src_base + written, chunk_size) };

                            match file.write_all(data) {
                                Ok(_) => written += chunk_size,
                                Err(_) => break,
                            }
                        }
                    }

                    let _ = file.sync_all(); // Ensure data hits disk
                }
            }
            _ => {}
        }
    }
}

pub(crate) struct ComputationalUnit {
    regs_f64: Vec<f64>,
    regs_u64: Vec<u64>,
    shared: Arc<SharedMemory>,
    clock: Clock,
}

impl ComputationalUnit {
    pub fn new(regs: usize, shared: Arc<SharedMemory>) -> Self {
        Self {
            regs_f64: vec![0.0; regs],
            regs_u64: vec![0; regs],
            shared,
            clock: Clock::new(),
        }
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::Approximate => {
                debug!(dst = action.dst, src = action.src, iterations = action.offset, "approximate");
                let base = self.regs_f64[action.src as usize];
                let iterations = action.offset as usize;
                let mut x = base;
                for _ in 0..iterations {
                    x = 0.5 * (x + base / x)
                }
                self.regs_f64[action.dst as usize] = x;
            }
            Kind::Choose => {
                debug!(dst = action.dst, src = action.src, "choose");
                let n = self.regs_u64[action.src as usize];
                if n > 0 {
                    let choice = rand::random::<u64>() % n;
                    self.regs_u64[action.dst as usize] = choice;
                }
            }
            Kind::Timestamp => {
                debug!(dst = action.dst, "timestamp");
                // Store current timestamp in register
                self.regs_u64[action.dst as usize] = self.clock.raw();
            }
            Kind::ComputationalLoadF64 => {
                debug!(dst = action.dst, src = action.src, "comp_load_f64");
                let bytes = self.shared.read(action.src as usize, 8);
                let value = f64::from_le_bytes(bytes[0..8].try_into().unwrap());
                self.regs_f64[action.dst as usize] = value;
            }
            Kind::ComputationalStoreF64 => {
                debug!(src = action.src, offset = action.offset, "comp_store_f64");
                let value = self.regs_f64[action.src as usize];
                self.shared.write(action.offset as usize, &value.to_le_bytes());
            }
            Kind::ComputationalLoadU64 => {
                debug!(dst = action.dst, src = action.src, "comp_load_u64");
                let bytes = self.shared.read(action.src as usize, 8);
                let value = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                self.regs_u64[action.dst as usize] = value;
            }
            Kind::ComputationalStoreU64 => {
                debug!(src = action.src, offset = action.offset, "comp_store_u64");
                let value = self.regs_u64[action.src as usize];
                self.shared.write(action.offset as usize, &value.to_le_bytes());
            }
            _ => {}
        }
    }
}

pub(crate) struct SimdUnit {
    id: u8,
    regs_f32: Vec<f32x4>,
    regs_i32: Vec<i32x4>,
    shared: Arc<SharedMemory>,
}

impl SimdUnit {
    pub fn new(
        id: u8,
        regs: usize,
        shared: Arc<SharedMemory>,
    ) -> Self {
        Self {
            id,
            regs_f32: vec![f32x4::splat(0.0); regs],
            regs_i32: vec![i32x4::splat(0); regs],
            shared,
        }
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::SimdLoad => {
                trace!(dst = action.dst, src = action.src, "simd_load_f32");
                let vals = self.shared.read(action.src as usize, 16);
                let f0 = f32::from_le_bytes([vals[0], vals[1], vals[2], vals[3]]);
                let f1 = f32::from_le_bytes([vals[4], vals[5], vals[6], vals[7]]);
                let f2 = f32::from_le_bytes([vals[8], vals[9], vals[10], vals[11]]);
                let f3 = f32::from_le_bytes([vals[12], vals[13], vals[14], vals[15]]);

                self.regs_f32[action.dst as usize] = f32x4::from([f0, f1, f2, f3]);
            }
            Kind::SimdAdd => {
                trace!(dst = action.dst, src = action.src, offset = action.offset, "simd_add_f32");
                let a = self.regs_f32[action.src as usize];
                let b = self.regs_f32[action.offset as usize];
                self.regs_f32[action.dst as usize] = a + b;
            }
            Kind::SimdMul => {
                trace!(dst = action.dst, src = action.src, offset = action.offset, "simd_mul_f32");
                let a = self.regs_f32[action.src as usize];
                let b = self.regs_f32[action.offset as usize];
                self.regs_f32[action.dst as usize] = a * b;
            }
            Kind::SimdStore => {
                trace!(src = action.src, offset = action.offset, "simd_store_f32");
                let reg_data = self.regs_f32[action.src as usize].to_array();
                let write_offset = action.offset as usize;

                let mut bytes = [0u8; 16];
                for (i, &val) in reg_data.iter().enumerate() {
                    bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
                }
                self.shared.write(write_offset, &bytes);
            }
            Kind::SimdLoadI32 => {
                trace!(dst = action.dst, src = action.src, "simd_load_i32");
                let vals = self.shared.read(action.src as usize, 16);
                let i0 = i32::from_le_bytes([vals[0], vals[1], vals[2], vals[3]]);
                let i1 = i32::from_le_bytes([vals[4], vals[5], vals[6], vals[7]]);
                let i2 = i32::from_le_bytes([vals[8], vals[9], vals[10], vals[11]]);
                let i3 = i32::from_le_bytes([vals[12], vals[13], vals[14], vals[15]]);

                self.regs_i32[action.dst as usize] = i32x4::from([i0, i1, i2, i3]);
            }
            Kind::SimdAddI32 => {
                trace!(dst = action.dst, src = action.src, offset = action.offset, "simd_add_i32");
                let a = self.regs_i32[action.src as usize];
                let b = self.regs_i32[action.offset as usize];
                self.regs_i32[action.dst as usize] = a + b;
            }
            Kind::SimdMulI32 => {
                trace!(dst = action.dst, src = action.src, offset = action.offset, "simd_mul_i32");
                let a = self.regs_i32[action.src as usize];
                let b = self.regs_i32[action.offset as usize];
                self.regs_i32[action.dst as usize] = a * b;
            }
            Kind::SimdDivI32 => {
                trace!(dst = action.dst, src = action.src, offset = action.offset, "simd_div_i32");
                let a = self.regs_i32[action.src as usize].to_array();
                let b = self.regs_i32[action.offset as usize].to_array();
                let result = i32x4::from([
                    if b[0] != 0 { a[0] / b[0] } else { 0 },
                    if b[1] != 0 { a[1] / b[1] } else { 0 },
                    if b[2] != 0 { a[2] / b[2] } else { 0 },
                    if b[3] != 0 { a[3] / b[3] } else { 0 },
                ]);
                self.regs_i32[action.dst as usize] = result;
            }
            Kind::SimdSubI32 => {
                trace!(dst = action.dst, src = action.src, offset = action.offset, "simd_sub_i32");
                let a = self.regs_i32[action.src as usize];
                let b = self.regs_i32[action.offset as usize];
                self.regs_i32[action.dst as usize] = a - b;
            }
            Kind::SimdStoreI32 => {
                trace!(src = action.src, offset = action.offset, "simd_store_i32");
                let reg_data = self.regs_i32[action.src as usize].to_array();
                let write_offset = action.offset as usize;

                let mut bytes = [0u8; 16];
                for (i, &val) in reg_data.iter().enumerate() {
                    bytes[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
                }
                self.shared.write(write_offset, &bytes);
            }
            _ => {}
        }
    }
}

pub(crate) struct GpuUnit {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    compute_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    shared: Arc<SharedMemory>,
}

impl GpuUnit {
    pub fn new(shared: Arc<SharedMemory>, shader_source: &str, gpu_size: usize, _backends: Backends) -> Self {
        let _span = info_span!("gpu_init", gpu_size).entered();
        info!("initializing GPU unit");

        let (device, queue) = cached_gpu_device();
        info!("GPU device acquired (cached)");

        let compute_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Compute"),
            size: gpu_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Staging"),
            size: gpu_size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Compute"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        info!(shader_len = shader_source.len(), "GPU shader compiled");

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: PipelineCompilationOptions::default(),
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(compute_buffer.as_entire_buffer_binding()),
            }],
        });
        info!("GPU pipeline created");

        Self {
            device,
            queue,
            compute_buffer,
            staging_buffer,
            pipeline,
            bind_group,
            shared,
        }
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::Dispatch => {
                debug!(
                    src = action.src,
                    dst = action.dst,
                    size = action.size,
                    workgroups = (action.size + 63) / 64,
                    "gpu_dispatch"
                );
                // Read input from action.src, size bytes
                let data = self.shared.read(action.src as usize, action.size as usize);

                // Write to GPU buffer, run shader, read result
                self.queue.write_buffer(&self.compute_buffer, 0, data);

                let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Compute Encoder"),
                });

                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("Compute Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups((action.size + 63) / 64, 1, 1);
                }

                encoder.copy_buffer_to_buffer(&self.compute_buffer, 0, &self.staging_buffer, 0, action.size as u64);
                self.queue.submit(Some(encoder.finish()));

                // Read back and write to action.dst
                let buffer_slice = self.staging_buffer.slice(..);
                buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
                self.device.poll(wgpu::Maintain::Wait);

                let result = buffer_slice.get_mapped_range();
                self.shared.write(action.dst as usize, &result[..action.size as usize]);
                drop(result);
                self.staging_buffer.unmap();
                debug!("gpu_dispatch_readback_complete");
            }
            _ => {
                // Other GPU actions (CreateBuffer, etc.) not yet implemented
            }
        }
    }
}

pub(crate) fn memory_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    broadcast: Arc<Broadcast>,
    worker_id: u32,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
) {
    let _span = info_span!("memory_unit", worker_id).entered();
    info!("memory unit started");
    let mut unit = MemoryUnit::new(shared.clone());
    let mut last_epoch = 0u64;
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "memory_work_received");
                for idx in start..end {
                    unsafe {
                        unit.execute(&actions[idx as usize]);
                    }
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "memory_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("memory unit shutting down");
                return;
            }
            MailboxPoll::Empty => {
                let prev_epoch = last_epoch;
                if !broadcast_step(
                    worker_id,
                    &mut last_epoch,
                    &broadcast,
                    &actions,
                    &shared,
                    |action| unsafe { unit.execute(action); },
                ) {
                    return;
                }
                if last_epoch != prev_epoch {
                    spin_count = 0;
                } else {
                    spin_backoff(&mut spin_count);
                }
            }
        }
    }
}

pub(crate) fn ffi_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
) {
    let _span = info_span!("ffi_unit").entered();
    info!("FFI unit started");
    let mut unit = FFIUnit::new(shared.clone());
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "ffi_work_received");
                for idx in start..end {
                    unsafe {
                        unit.execute(&actions[idx as usize]);
                    }
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "ffi_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("FFI unit shutting down");
                return;
            }
            MailboxPoll::Empty => spin_backoff(&mut spin_count),
        }
    }
}

pub(crate) fn network_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
) {
    let _span = info_span!("network_unit").entered();
    info!("network unit started");
    let mut unit = NetworkUnit::new(0, shared.clone());
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "network_work_received");
                for idx in start..end {
                    unit.execute(&actions[idx as usize]);
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "network_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("network unit shutting down");
                return;
            }
            MailboxPoll::Empty => spin_backoff(&mut spin_count),
        }
    }
}

pub(crate) fn file_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    buffer_size: usize,
) {
    let _span = info_span!("file_unit").entered();
    info!("file unit started");
    let mut unit = FileUnit::new(0, shared.clone(), buffer_size);
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "file_work_received");
                for idx in start..end {
                    unit.execute(&actions[idx as usize]);
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "file_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("file unit shutting down");
                return;
            }
            MailboxPoll::Empty => spin_backoff(&mut spin_count),
        }
    }
}

pub(crate) fn computational_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    broadcast: Arc<Broadcast>,
    worker_id: u32,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    regs: usize,
) {
    let _span = info_span!("computational_unit", worker_id).entered();
    info!("computational unit started");
    let mut unit = ComputationalUnit::new(regs, shared.clone());
    let mut last_epoch = 0u64;
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "computational_work_received");
                for idx in start..end {
                    unsafe {
                        unit.execute(&actions[idx as usize]);
                    }
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "computational_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("computational unit shutting down");
                return;
            }
            MailboxPoll::Empty => {
                let prev_epoch = last_epoch;
                if !broadcast_step(
                    worker_id,
                    &mut last_epoch,
                    &broadcast,
                    &actions,
                    &shared,
                    |action| unsafe { unit.execute(action); },
                ) {
                    return;
                }
                if last_epoch != prev_epoch {
                    spin_count = 0;
                } else {
                    spin_backoff(&mut spin_count);
                }
            }
        }
    }
}

pub(crate) fn simd_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    broadcast: Arc<Broadcast>,
    worker_id: u32,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    regs: usize,
) {
    let _span = info_span!("simd_unit", worker_id).entered();
    info!("SIMD unit started");
    let mut unit = SimdUnit::new(0, regs, shared.clone());
    let mut last_epoch = 0u64;
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "simd_work_received");
                for idx in start..end {
                    unsafe {
                        unit.execute(&actions[idx as usize]);
                    }
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "simd_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("SIMD unit shutting down");
                return;
            }
            MailboxPoll::Empty => {
                let prev_epoch = last_epoch;
                if !broadcast_step(
                    worker_id,
                    &mut last_epoch,
                    &broadcast,
                    &actions,
                    &shared,
                    |action| unsafe { unit.execute(action); },
                ) {
                    return;
                }
                if last_epoch != prev_epoch {
                    spin_count = 0;
                } else {
                    spin_backoff(&mut spin_count);
                }
            }
        }
    }
}

pub(crate) fn gpu_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    shader_source: String,
    gpu_size: usize,
    backends: Backends,
) {
    let _span = info_span!("gpu_unit").entered();
    info!("GPU unit started");
    let mut gpu = GpuUnit::new(shared.clone(), &shader_source, gpu_size, backends);
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "gpu_work_received");
                for idx in start..end {
                    unsafe {
                        gpu.execute(&actions[idx as usize]);
                    }
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "gpu_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("GPU unit shutting down");
                return;
            }
            MailboxPoll::Empty => spin_backoff(&mut spin_count),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memscan_single_byte() {
        let mut memory = vec![0u8; 1024];

        // Pattern to search for (single byte 0x42)
        memory[100] = 0x42;

        // Data to search in
        memory[200..210]
            .copy_from_slice(&[0x00, 0x11, 0x22, 0x42, 0x33, 0x44, 0x42, 0x55, 0x66, 0x77]);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::MemScan,
            src: 100,                // Pattern at offset 100
            dst: 200,                // Search from offset 200
            size: 10,                // Search 10 bytes
            offset: 1 | (300 << 16), // Pattern size 1, result at offset 300
        };

        unsafe {
            unit.execute(&action);
            let result = i64::from_le_bytes(shared.read(300, 8)[0..8].try_into().unwrap());
            assert_eq!(result, 203); // Found at offset 203
        }
    }

    #[test]
    fn test_memscan_multi_byte() {
        let mut memory = vec![0u8; 1024];

        // Pattern to search for (3 bytes: 0xAA, 0xBB, 0xCC)
        memory[100..103].copy_from_slice(&[0xAA, 0xBB, 0xCC]);

        // Data to search in
        memory[200..215].copy_from_slice(&[
            0x00, 0x11, 0x22, 0xAA, 0xBB, 0xCC, 0x33, 0x44, 0xAA, 0xBB, 0xCC, 0x55, 0x66, 0x77,
            0x88,
        ]);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::MemScan,
            src: 100,                // Pattern at offset 100
            dst: 200,                // Search from offset 200
            size: 15,                // Search 15 bytes
            offset: 3 | (300 << 16), // Pattern size 3, result at offset 300
        };

        unsafe {
            unit.execute(&action);
            let result = i64::from_le_bytes(shared.read(300, 8)[0..8].try_into().unwrap());
            assert_eq!(result, 203); // Found at offset 203 (first occurrence)
        }
    }

    #[test]
    fn test_memscan_not_found() {
        let mut memory = vec![0u8; 1024];

        // Pattern to search for
        memory[100] = 0xFF;

        // Data to search in (doesn't contain 0xFF)
        memory[200..210]
            .copy_from_slice(&[0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99]);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::MemScan,
            src: 100,
            dst: 200,
            size: 10,
            offset: 1 | (300 << 16),
        };

        unsafe {
            unit.execute(&action);
            let result = i64::from_le_bytes(shared.read(300, 8)[0..8].try_into().unwrap());
            assert_eq!(result, -1); // Not found
        }
    }

    #[test]
    fn test_atomic_load_store() {
        let mut memory = vec![0u8; 256];
        memory[64..72].copy_from_slice(&0xDEADBEEFu64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let store = Action {
            kind: Kind::AtomicStore,
            dst: 32,
            src: 64,
            offset: 0,
            size: 8,
        };
        let load = Action {
            kind: Kind::AtomicLoad,
            dst: 40,
            src: 32,
            offset: 0,
            size: 8,
        };

        unsafe {
            unit.execute(&store);
            unit.execute(&load);
            let stored = u64::from_le_bytes(shared.read(32, 8)[0..8].try_into().unwrap());
            let loaded = u64::from_le_bytes(shared.read(40, 8)[0..8].try_into().unwrap());
            assert_eq!(stored, 0xDEADBEEF);
            assert_eq!(loaded, 0xDEADBEEF);
        }
    }

    #[test]
    fn test_atomic_fetch_add_sub() {
        let mut memory = vec![0u8; 256];
        memory[32..40].copy_from_slice(&10u64.to_le_bytes());
        memory[64..72].copy_from_slice(&5u64.to_le_bytes());
        memory[72..80].copy_from_slice(&3u64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let add = Action {
            kind: Kind::AtomicFetchAdd,
            dst: 32,
            src: 40,
            offset: 64,
            size: 8,
        };
        let sub = Action {
            kind: Kind::AtomicFetchSub,
            dst: 32,
            src: 48,
            offset: 72,
            size: 8,
        };

        unsafe {
            unit.execute(&add);
            unit.execute(&sub);
            let final_val = u64::from_le_bytes(shared.read(32, 8)[0..8].try_into().unwrap());
            let prev_add = u64::from_le_bytes(shared.read(40, 8)[0..8].try_into().unwrap());
            let prev_sub = u64::from_le_bytes(shared.read(48, 8)[0..8].try_into().unwrap());
            assert_eq!(prev_add, 10);
            assert_eq!(prev_sub, 15);
            assert_eq!(final_val, 12);
        }
    }

    #[test]
    fn test_simd_unit_creation() {
        let memory = vec![0u8; 1024];
        let shared = Arc::new(SharedMemory::new(memory.as_ptr() as *mut u8));

        let unit = SimdUnit::new(0, 16, shared);
        assert_eq!(unit.id, 0);
        assert_eq!(unit.regs_f32.len(), 16);
        assert_eq!(unit.regs_i32.len(), 16);
    }

    #[test]
    fn test_simd_i32x4_operations() {
        let mut memory = vec![0u8; 1024];
        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = SimdUnit::new(0, 4, shared.clone());

        // Setup test data: [1, 2, 3, 4] and [10, 20, 30, 40]
        unsafe {
            memory[0..4].copy_from_slice(&1i32.to_le_bytes());
            memory[4..8].copy_from_slice(&2i32.to_le_bytes());
            memory[8..12].copy_from_slice(&3i32.to_le_bytes());
            memory[12..16].copy_from_slice(&4i32.to_le_bytes());

            memory[16..20].copy_from_slice(&10i32.to_le_bytes());
            memory[20..24].copy_from_slice(&20i32.to_le_bytes());
            memory[24..28].copy_from_slice(&30i32.to_le_bytes());
            memory[28..32].copy_from_slice(&40i32.to_le_bytes());
        }

        // Test Load, Add, Mul, Store
        let load_a = Action { kind: Kind::SimdLoadI32, dst: 0, src: 0, offset: 0, size: 16 };
        let load_b = Action { kind: Kind::SimdLoadI32, dst: 1, src: 16, offset: 0, size: 16 };
        let add = Action { kind: Kind::SimdAddI32, dst: 2, src: 0, offset: 1, size: 0 };
        let mul = Action { kind: Kind::SimdMulI32, dst: 3, src: 0, offset: 1, size: 0 };
        let store_add = Action { kind: Kind::SimdStoreI32, dst: 0, src: 2, offset: 100, size: 16 };
        let store_mul = Action { kind: Kind::SimdStoreI32, dst: 0, src: 3, offset: 116, size: 16 };

        unsafe {
            unit.execute(&load_a);
            unit.execute(&load_b);
            unit.execute(&add);
            unit.execute(&mul);
            unit.execute(&store_add);
            unit.execute(&store_mul);

            // Verify addition: [11, 22, 33, 44]
            let add_result = shared.read(100, 16);
            assert_eq!(i32::from_le_bytes(add_result[0..4].try_into().unwrap()), 11);
            assert_eq!(i32::from_le_bytes(add_result[4..8].try_into().unwrap()), 22);
            assert_eq!(i32::from_le_bytes(add_result[8..12].try_into().unwrap()), 33);
            assert_eq!(i32::from_le_bytes(add_result[12..16].try_into().unwrap()), 44);

            // Verify multiplication: [10, 40, 90, 160]
            let mul_result = shared.read(116, 16);
            assert_eq!(i32::from_le_bytes(mul_result[0..4].try_into().unwrap()), 10);
            assert_eq!(i32::from_le_bytes(mul_result[4..8].try_into().unwrap()), 40);
            assert_eq!(i32::from_le_bytes(mul_result[8..12].try_into().unwrap()), 90);
            assert_eq!(i32::from_le_bytes(mul_result[12..16].try_into().unwrap()), 160);
        }
    }

    #[test]
    fn test_computational_unit_creation() {
        let mut memory = vec![0u8; 1024];
        let shared = unsafe { SharedMemory::new(memory.as_mut_ptr()) };
        let unit = ComputationalUnit::new(32, Arc::new(shared));
        assert_eq!(unit.regs_f64.len(), 32);
        assert_eq!(unit.regs_u64.len(), 32);
    }

    #[test]
    fn test_approximate_action() {
        let mut memory = vec![0u8; 1024];
        let shared = unsafe { SharedMemory::new(memory.as_mut_ptr()) };
        let mut unit = ComputationalUnit::new(8, Arc::new(shared));

        unit.regs_f64[1] = 16.0;

        let action = Action {
            kind: Kind::Approximate,
            dst: 2,
            src: 1,
            offset: 10,
            size: 0,
        };

        unsafe {
            unit.execute(&action);
        }

        // sqrt(16) = 4
        assert_eq!(unit.regs_f64[2], 4.0);
    }

    #[test]
    fn test_choose_action() {
        let mut memory = vec![0u8; 1024];
        let shared = unsafe { SharedMemory::new(memory.as_mut_ptr()) };
        let mut unit = ComputationalUnit::new(8, Arc::new(shared));

        unit.regs_u64[1] = 100;

        // Choose from [0, 100)
        let action = Action {
            kind: Kind::Choose,
            dst: 2,
            src: 1,
            offset: 0,
            size: 0,
        };

        unsafe {
            unit.execute(&action);
        }

        // Result should be in range [0, 100)
        assert!(unit.regs_u64[2] < 100);
    }

    #[test]
    fn test_choose_ranges() {
        let mut memory = vec![0u8; 1024];
        let shared = unsafe { SharedMemory::new(memory.as_mut_ptr()) };
        let mut unit = ComputationalUnit::new(8, Arc::new(shared));

        for n in [1u64, 10, 50, 1000] {
            unit.regs_u64[0] = n;

            let action = Action {
                kind: Kind::Choose,
                dst: 1,
                src: 0,
                offset: 0,
                size: 0,
            };

            for _ in 0..10 {
                unsafe {
                    unit.execute(&action);
                }
                assert!(unit.regs_u64[1] < n);
            }
        }
    }

    #[test]
    fn test_memcopy_basic() {
        let mut memory = vec![0u8; 1024];
        memory[100..104].copy_from_slice(&[1, 2, 3, 4]);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::MemCopy,
            dst: 200,
            src: 100,
            offset: 0,
            size: 4,
        };

        unsafe {
            unit.execute(&action);
            let copied = shared.read(200, 4);
            assert_eq!(copied, &[1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_conditional_write_true() {
        let mut memory = vec![0u8; 1024];

        // Set condition to 1 (true)
        memory[0..8].copy_from_slice(&1u64.to_le_bytes());
        // Set source data
        memory[100..104].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::ConditionalWrite,
            dst: 200,
            src: 100,
            offset: 0, // condition at offset 0
            size: 4,
        };

        unsafe {
            unit.execute(&action);
            let result = shared.read(200, 4);
            assert_eq!(result, &[0xAA, 0xBB, 0xCC, 0xDD]);
        }
    }

    #[test]
    fn test_conditional_write_false() {
        let mut memory = vec![0u8; 1024];

        // Set condition to 0 (false)
        memory[0..8].copy_from_slice(&0u64.to_le_bytes());
        // Set source data
        memory[100..104].copy_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        // Pre-fill destination with different data
        memory[200..204].copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::ConditionalWrite,
            dst: 200,
            src: 100,
            offset: 0,
            size: 4,
        };

        unsafe {
            unit.execute(&action);
            // Destination should be unchanged
            let result = shared.read(200, 4);
            assert_eq!(result, &[0x11, 0x22, 0x33, 0x44]);
        }
    }

    #[test]
    fn test_memory_compare_action() {
        let mut memory = vec![0u8; 256];
        memory[0..4].copy_from_slice(&5i32.to_le_bytes());
        memory[16..20].copy_from_slice(&3i32.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::Compare,
            dst: 32,
            src: 0,
            offset: 16,
            size: 0,
        };

        // 5 > 3 → 1
        unsafe {
            unit.execute(&action);
        }
        let result = i32::from_le_bytes(unsafe { shared.read(32, 4) }[0..4].try_into().unwrap());
        assert_eq!(result, 1);

        // 3 > 5 → 0
        let action2 = Action { src: 16, offset: 0, ..action };
        unsafe {
            unit.execute(&action2);
        }
        let result = i32::from_le_bytes(unsafe { shared.read(32, 4) }[0..4].try_into().unwrap());
        assert_eq!(result, 0);

        // 5 > 5 → 0
        let action3 = Action { src: 0, offset: 0, ..action };
        unsafe {
            unit.execute(&action3);
        }
        let result = i32::from_le_bytes(unsafe { shared.read(32, 4) }[0..4].try_into().unwrap());
        assert_eq!(result, 0);
    }

    #[test]
    fn test_large_memcopy() {
        let mut memory = vec![0u8; 65536];

        // Fill source with pattern
        for i in 0..1000 {
            memory[1000 + i] = (i % 256) as u8;
        }

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::MemCopy,
            dst: 5000,
            src: 1000,
            offset: 0,
            size: 1000,
        };

        unsafe {
            unit.execute(&action);
            let result = shared.read(5000, 1000);
            for i in 0..1000 {
                assert_eq!(result[i], (i % 256) as u8);
            }
        }
    }

    #[test]
    fn test_overlapping_memcopy() {
        let mut memory = vec![0u8; 1024];
        memory[100..110].copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        // Copy with overlap (src=100, dst=105, overlaps at 105-109)
        let action = Action {
            kind: Kind::MemCopy,
            dst: 105,
            src: 100,
            offset: 0,
            size: 5,
        };

        unsafe {
            unit.execute(&action);
            let result = shared.read(105, 5);
            assert_eq!(result, &[0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_conditional_with_integer_values() {
        let mut memory = vec![0u8; 1024];

        // Test with different condition values
        for (cond_val, should_copy) in [
            (0u64, false),
            (1u64, true),
            (u64::MAX, true),
            (42u64, true),
        ] {
            memory[0..8].copy_from_slice(&cond_val.to_le_bytes());
            memory[100..108].copy_from_slice(&42u64.to_le_bytes());
            memory[200..208].copy_from_slice(&0u64.to_le_bytes());

            let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
            let mut unit = MemoryUnit::new(shared.clone());

            let action = Action {
                kind: Kind::ConditionalWrite,
                dst: 200,
                src: 100,
                offset: 0,
                size: 8,
            };

            unsafe {
                unit.execute(&action);
                let result_bytes = shared.read(200, 8);
                let result = u64::from_le_bytes(result_bytes.try_into().unwrap());

                if should_copy {
                    assert_eq!(result, 42, "Condition {} should copy", cond_val);
                } else {
                    assert_eq!(result, 0, "Condition {} should not copy", cond_val);
                }
            }
        }
    }
}

#[cfg(test)]
mod ffi_tests {
    use super::*;

    #[no_mangle]
    pub extern "C" fn test_add_numbers(args: *mut u8) -> i64 {
        unsafe {
            // Use byte copying to handle any alignment
            let mut a_bytes = [0u8; 8];
            let mut b_bytes = [0u8; 8];
            std::ptr::copy_nonoverlapping(args, a_bytes.as_mut_ptr(), 8);
            std::ptr::copy_nonoverlapping(args.add(8), b_bytes.as_mut_ptr(), 8);

            let a = i64::from_le_bytes(a_bytes);
            let b = i64::from_le_bytes(b_bytes);
            a + b
        }
    }

    #[no_mangle]
    pub extern "C" fn test_return_42(_args: *mut u8) -> i64 {
        42
    }

    #[test]
    fn test_ffi_basic_call() {
        let mut memory = vec![0u8; 1024];

        // Store function pointer at offset 0
        let fn_ptr = test_return_42 as usize as u64;
        memory[0..8].copy_from_slice(&fn_ptr.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FFIUnit::new(shared.clone());

        let action = Action {
            kind: Kind::FFICall,
            src: 0,      // function pointer location
            dst: 100,    // args location (unused for this function)
            offset: 200, // result location
            size: 0,     // unused
        };

        unsafe {
            unit.execute(&action);

            let result = i64::from_le_bytes(shared.read(200, 8)[0..8].try_into().unwrap());
            assert_eq!(result, 42);
        }
    }

    #[test]
    fn test_ffi_with_args() {
        let mut memory = vec![0u8; 1024];

        // Get function pointer as usize first
        let fn_ptr = test_add_numbers as *const () as usize;

        // Store as little-endian bytes
        memory[0..8].copy_from_slice(&fn_ptr.to_le_bytes());

        // Store arguments at offset 100
        memory[100..108].copy_from_slice(&10i64.to_le_bytes());
        memory[108..116].copy_from_slice(&32i64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FFIUnit::new(shared.clone());

        let action = Action {
            kind: Kind::FFICall,
            src: 0,      // function pointer location
            dst: 100,    // args location
            offset: 200, // result location
            size: 0,
        };

        unsafe {
            unit.execute(&action);

            let result = i64::from_le_bytes(shared.read(200, 8)[0..8].try_into().unwrap());
            assert_eq!(result, 42); // 10 + 32 = 42
        }
    }

    #[test]
    fn test_ffi_null_pointer_safety() {
        let mut memory = vec![0u8; 1024];

        // Store null pointer
        memory[0..8].copy_from_slice(&0u64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FFIUnit::new(shared.clone());

        let action = Action {
            kind: Kind::FFICall,
            src: 0,
            dst: 100,
            offset: 200,
            size: 0,
        };

        unsafe {
            // Should not crash, just skip
            unit.execute(&action);

            // Result area should be untouched (still zeros)
            let result = i64::from_le_bytes(shared.read(200, 8)[0..8].try_into().unwrap());
            assert_eq!(result, 0);
        }
    }
}

#[cfg(test)]
mod network_tests {
    use super::*;
    use std::time::Duration;
    
    #[test]
    fn test_network_connect_and_send() {
        // Start a test server
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn server task
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 5];
            stream.read_exact(&mut buf).unwrap();
            assert_eq!(&buf, b"hello");
        });

        // Setup memory
        let mut memory = vec![0u8; 1024];
        let addr_str = addr.to_string();
        memory[0..addr_str.len()].copy_from_slice(addr_str.as_bytes());
        memory[addr_str.len()] = 0;

        // Store data to send
        memory[100..105].copy_from_slice(b"hello");

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = NetworkUnit::new(0, shared.clone());

        // Connect
        let connect_action = Action {
            kind: Kind::NetConnect,
            src: 0,      // address at offset 0
            dst: 200,    // store handle at 200
            offset: 100, // max address length
            size: 0,
        };

        unit.execute(&connect_action);
        let handle = unsafe {
            u32::from_le_bytes(
                shared.read(200, 4)[0..4].try_into().unwrap(),
            )
        };
        assert!(handle != 0);

        // Send data
        let send_action = Action {
            kind: Kind::NetSend,
            src: 100, // data at offset 100
            dst: 200, // handle at offset 200
            offset: 0,
            size: 5, // send 5 bytes
        };

        unit.execute(&send_action);
    }

    #[test]
    fn test_network_listen_accept_recv() {
        let mut memory = vec![0u8; 1024];

        // Setup listener address
        let addr = "0.0.0.0:0"; // OS assigns port
        memory[0..addr.len()].copy_from_slice(addr.as_bytes());
        memory[addr.len()] = 0;

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = NetworkUnit::new(0, shared.clone());

        // Create listener
        let listen_action = Action {
            kind: Kind::NetConnect, // NetConnect with 0.0.0.0 = listen
            src: 0,                 // address
            dst: 200,               // store listener handle
            offset: 50,
            size: 0,
        };

        unit.execute(&listen_action);

        // Get actual listening port for client
        let handle = unsafe { u32::from_le_bytes(shared.read(200, 4)[0..4].try_into().unwrap()) };
        let actual_addr = unit.listeners.get(&handle).unwrap().local_addr().unwrap();

        // Spawn client
        let client_task = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            let mut stream = TcpStream::connect(actual_addr).unwrap();
            stream.write_all(b"test").unwrap();
        });

        // Accept connection
        let accept_action = Action {
            kind: Kind::NetAccept,
            src: 200, // listener handle
            dst: 300, // store connection handle
            offset: 0,
            size: 0,
        };

        unit.execute(&accept_action);

        // Receive data
        let recv_action = Action {
            kind: Kind::NetRecv,
            src: 300, // connection handle
            dst: 400, // store received data
            offset: 0,
            size: 4, // exact bytes to receive
        };

        unit.execute(&recv_action);

        // Verify received data
        unsafe {
            let data = shared.read(400, 4);
            assert_eq!(data, b"test");
        }

        client_task.join().unwrap();
    }

    #[test]
    fn test_network_echo_server() {
        let mut memory = vec![0u8; 1024];
        memory[0..10].copy_from_slice(b"0.0.0.0:0\0");

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = NetworkUnit::new(0, shared.clone());

        // Start listener
        let listen_action = Action {
            kind: Kind::NetConnect,
            src: 0,
            dst: 200,
            offset: 50,
            size: 0,
        };
        unit.execute(&listen_action);

        let handle = unsafe { u32::from_le_bytes(shared.read(200, 4)[0..4].try_into().unwrap()) };
        let addr = unit.listeners.get(&handle).unwrap().local_addr().unwrap();

        // Client task
        let client = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            stream.write_all(b"echo").unwrap();

            let mut buf = [0u8; 4];
            stream.read_exact(&mut buf).unwrap();
            assert_eq!(&buf, b"echo");
        });

        // Accept
        let accept = Action {
            kind: Kind::NetAccept,
            src: 200,
            dst: 300,
            offset: 0,
            size: 0,
        };
        unit.execute(&accept);

        // Receive
        let recv = Action {
            kind: Kind::NetRecv,
            src: 300,
            dst: 400,
            offset: 0,
            size: 4, // exact bytes to receive ("echo")
        };
        unit.execute(&recv);

        // Echo back
        let send = Action {
            kind: Kind::NetSend,
            src: 400,
            dst: 300,
            offset: 0,
            size: 4,
        };
        unit.execute(&send);

        client.join().unwrap();
    }

    #[test]
    fn test_multiple_connections() {
        let mut memory = vec![0u8; 2048];
        memory[0..10].copy_from_slice(b"0.0.0.0:0\0");

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = NetworkUnit::new(0, shared.clone());

        // Create listener
        unit.execute(&Action {
            kind: Kind::NetConnect,
            src: 0,
            dst: 200,
            offset: 50,
            size: 0,
        })
        ;

        let handle = unsafe { u32::from_le_bytes(shared.read(200, 4)[0..4].try_into().unwrap()) };
        let addr = unit.listeners.get(&handle).unwrap().local_addr().unwrap();

        // Spawn multiple clients
        let mut clients = vec![];
        for i in 0..3 {
            let addr = addr.clone();
            clients.push(std::thread::spawn(move || {
                let mut stream = TcpStream::connect(addr).unwrap();
                stream.write_all(&[i as u8]).unwrap();
            }));
        }

        // Accept all connections
        let mut handles = vec![];
        for i in 0..3 {
            let accept = Action {
                kind: Kind::NetAccept,
                src: 200,
                dst: 300 + i * 4,
                offset: 0,
                size: 0,
            };
            unit.execute(&accept);
            handles.push(300 + i * 4);
        }

        // Receive from all
        for (i, &handle_offset) in handles.iter().enumerate() {
            let recv = Action {
                kind: Kind::NetRecv,
                src: handle_offset as u32,
                dst: 400 + i as u32 * 100,
                offset: 0,
                size: 1,
            };
            unit.execute(&recv);
        }

        for client in clients {
            client.join().unwrap();
        }
    }

    #[test]
    fn test_network_handle_persistence() {
        let mut memory = vec![0u8; 1024];
        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = NetworkUnit::new(0, shared.clone());

        // Verify handles increment
        memory[0..10].copy_from_slice(b"0.0.0.0:0\0");

        for i in 0..3 {
            let action = Action {
                kind: Kind::NetConnect,
                src: 0,
                dst: 200 + i * 4,
                offset: 50,
                size: 0,
            };
            unit.execute(&action);

            let handle = unsafe {
                u32::from_le_bytes(
                    shared.read((200 + i * 4) as usize, 4)[0..4]
                        .try_into()
                        .unwrap(),
                )
            };
            assert_eq!(handle, i + 1);
        }
    }

    #[test]
    fn test_invalid_connection() {
        let mut memory = vec![0u8; 1024];

        // Invalid address
        let addr = "invalid.address:99999";
        memory[0..addr.len()].copy_from_slice(addr.as_bytes());
        memory[addr.len()] = 0;

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = NetworkUnit::new(0, shared.clone());

        let action = Action {
            kind: Kind::NetConnect,
            src: 0,
            dst: 200,
            offset: 50,
            size: 0,
        };

        unit.execute(&action);
        let handle = unsafe {
            u32::from_le_bytes(shared.read(200, 4)[0..4].try_into().unwrap())
        };
        assert_eq!(handle, 0); // Should fail gracefully
    }
}

#[cfg(test)]
mod file_tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    #[test]
    fn test_file_write_and_read() {
        let mut memory = vec![0u8; 1024];
        let test_file = "test_output.txt";
        let test_data = b"Hello, BASE!";

        // Setup: Store filename at offset 0
        let filename_bytes = test_file.as_bytes();
        memory[0..filename_bytes.len()].copy_from_slice(filename_bytes);
        memory[filename_bytes.len()] = 0; // null terminator

        // Store data at offset 100
        memory[100..100 + test_data.len()].copy_from_slice(test_data);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FileUnit::new(0, shared.clone(), 1024);

        // Test FileWrite
        let write_action = Action {
            kind: Kind::FileWrite,
            src: 100,                     // data at offset 100
            dst: 0,                       // filename at offset 0
            offset: 0,                    // file byte offset
            size: test_data.len() as u32, // write 12 bytes
        };

        unit.execute(&write_action);

        // Verify file was created
        assert!(Path::new(test_file).exists());
        let file_contents = fs::read(test_file).unwrap();
        assert_eq!(file_contents, test_data);

        // Test FileRead
        // Clear the data area first
        for i in 200..212 {
            memory[i] = 0;
        }

        let read_action = Action {
            kind: Kind::FileRead,
            src: 0,     // filename at offset 0
            dst: 200,   // write data to offset 200
            offset: 0,  // file byte offset
            size: 100,  // max bytes to read
        };

        unit.execute(&read_action);

        unsafe {
            let read_data = shared.read(200, test_data.len());
            assert_eq!(read_data, test_data);
        }

        // Cleanup
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_file_read_nonexistent() {
        let mut memory = vec![0u8; 1024];

        // Store filename for nonexistent file
        let filename = "nonexistent_file.txt";
        memory[0..filename.len()].copy_from_slice(filename.as_bytes());
        memory[filename.len()] = 0;

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FileUnit::new(0, shared.clone(), 1024);

        let action = Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 100,
            offset: 0,
            size: 100,
        };

        unit.execute(&action);

        // Memory should remain unchanged (all zeros)
        unsafe {
            let data = shared.read(100, 10);
            assert_eq!(data.iter().filter(|&&b| b != 0).count(), 0);
        }
    }

    #[test]
    fn test_file_size_limits() {
        let mut memory = vec![0u8; 1024];
        let test_file = "test_size_limit.txt";
        let test_data = b"This is a longer test string for size limiting";

        // Setup filename
        let filename_bytes = test_file.as_bytes();
        memory[0..filename_bytes.len()].copy_from_slice(filename_bytes);
        memory[filename_bytes.len()] = 0;

        // Write test file with full data
        fs::write(test_file, test_data).unwrap();

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FileUnit::new(0, shared.clone(), 1024);

        // Read only first 10 bytes
        let action = Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 100,
            offset: 0,
            size: 10, // Limit to 10 bytes
        };

        unit.execute(&action);

        unsafe {
            let read_data = shared.read(100, 10);
            assert_eq!(read_data, &test_data[..10]);

            // Verify we didn't write beyond the limit
            let byte_11 = shared.read(110, 1)[0];
            assert_eq!(byte_11, 0);
        }

        // Cleanup
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_filename_with_path() {
        let mut memory = vec![0u8; 1024];
        let test_dir = "test_dir";
        let test_file = "test_dir/test_file.txt";
        let test_data = b"Path test";

        fs::create_dir_all(test_dir).ok();

        // Store filename with path
        memory[0..test_file.len()].copy_from_slice(test_file.as_bytes());
        memory[test_file.len()] = 0;

        // Store data
        memory[100..100 + test_data.len()].copy_from_slice(test_data);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FileUnit::new(0, shared.clone(), 1024);

        let action = Action {
            kind: Kind::FileWrite,
            src: 100,
            dst: 0,
            offset: 0,
            size: test_data.len() as u32,
        };

        unit.execute(&action);

        assert!(Path::new(test_file).exists());
        let contents = fs::read(test_file).unwrap();
        assert_eq!(contents, test_data);

        // Cleanup
        fs::remove_file(test_file).ok();
        fs::remove_dir(test_dir).ok();
    }

    #[test]
    fn test_binary_data() {
        let mut memory = vec![0u8; 1024];
        let test_file = "test_binary.bin";

        // Binary data including zeros
        let binary_data = vec![0xFF, 0x00, 0x42, 0x00, 0xDE, 0xAD, 0xBE, 0xEF];

        // Setup filename
        memory[0..test_file.len()].copy_from_slice(test_file.as_bytes());
        memory[test_file.len()] = 0;

        // Store binary data
        memory[100..100 + binary_data.len()].copy_from_slice(&binary_data);

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FileUnit::new(0, shared.clone(), 1024);

        // Write binary file
        let write_action = Action {
            kind: Kind::FileWrite,
            src: 100,
            dst: 0,
            offset: 0,
            size: binary_data.len() as u32,
        };

        unit.execute(&write_action);

        // Read it back
        let read_action = Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 200,
            offset: 0,
            size: 0, // Read entire file
        };

        unit.execute(&read_action);

        unsafe {
            let read_data = shared.read(200, binary_data.len());
            assert_eq!(read_data, &binary_data);
        }

        // Cleanup
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_file_read_with_offset() {
        let mut memory = vec![0u8; 1024];
        let test_file = "test_read_offset.txt";
        let test_data = b"Hello, World!";

        fs::write(test_file, test_data).unwrap();

        let filename_bytes = test_file.as_bytes();
        memory[0..filename_bytes.len()].copy_from_slice(filename_bytes);
        memory[filename_bytes.len()] = 0;

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FileUnit::new(0, shared.clone(), 1024);

        // Read from byte offset 7 ("World!")
        let action = Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 100,
            offset: 7,
            size: 6,
        };

        unit.execute(&action);

        unsafe {
            let read_data = shared.read(100, 6);
            assert_eq!(read_data, b"World!");
        }

        fs::remove_file(test_file).ok();
    }

}

#[cfg(test)]
mod atomic_tests {
    use super::*;

    #[test]
    fn test_cas64_success() {
        let mut memory = vec![0u8; 1024];

        // Initialize value to 42
        memory[104..112].copy_from_slice(&42u64.to_le_bytes());

        // Expected: 42, New: 100
        memory[208..216].copy_from_slice(&42u64.to_le_bytes());
        memory[304..312].copy_from_slice(&100u64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::AtomicCAS,
            dst: 104,    // target location
            src: 208,    // expected value location
            offset: 304, // new value location
            size: 8,     // 64-bit
        };

        unsafe {
            unit.execute(&action);

            // Should have swapped to 100
            let result = u64::from_le_bytes(shared.read(104, 8)[0..8].try_into().unwrap());
            assert_eq!(result, 100);

            // Old value (42) should be written back to src
            let old = u64::from_le_bytes(shared.read(208, 8)[0..8].try_into().unwrap());
            assert_eq!(old, 42);
        }
    }

    #[test]
    fn test_cas64_failure() {
        let mut memory = vec![0u8; 1024];

        // Initialize value to 42
        memory[104..112].copy_from_slice(&42u64.to_le_bytes());

        memory[208..216].copy_from_slice(&50u64.to_le_bytes());
        memory[304..312].copy_from_slice(&100u64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::AtomicCAS,
            dst: 104,
            src: 208,
            offset: 304,
            size: 8,
        };

        unsafe {
            unit.execute(&action);

            // Should still be 42 (CAS failed)
            let result = u64::from_le_bytes(shared.read(104, 8)[0..8].try_into().unwrap());
            assert_eq!(result, 42);

            // Actual value (42) should be written back to src
            let actual = u64::from_le_bytes(shared.read(208, 8)[0..8].try_into().unwrap());
            assert_eq!(actual, 42);
        }
    }

    #[test]
    fn test_file_write_with_offset() {
        let test_file = "test_write_offset.txt";
        let initial_data = b"AAAAAABBBBBB"; // 12 bytes

        // First write: create file with initial content
        fs::write(test_file, initial_data).unwrap();

        let mut memory = vec![0u8; 1024];

        // Setup filename at offset 0
        let filename_bytes = test_file.as_bytes();
        memory[0..filename_bytes.len()].copy_from_slice(filename_bytes);
        memory[filename_bytes.len()] = 0; // null terminator

        // Setup data to write at offset 100: "XX"
        memory[100..102].copy_from_slice(b"XX");

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = FileUnit::new(0, shared.clone(), 4096);

        // Write "XX" at byte offset 4 in the file
        let action = Action {
            kind: Kind::FileWrite,
            src: 100,        // data at offset 100
            dst: 0,          // filename at offset 0
            offset: 4,       // write to file byte position 4
            size: 2,         // write 2 bytes
        };

        unit.execute(&action);

        // Verify file now contains "AAAAXXBBBBBB"
        let file_contents = fs::read(test_file).unwrap();
        assert_eq!(file_contents, b"AAAAXXBBBBBB");

        // Cleanup
        fs::remove_file(test_file).ok();
    }

    #[test]
    fn test_cas_loop_increment() {
        let mut memory = vec![0u8; 1024];

        // Initialize counter to 0
        memory[104..112].copy_from_slice(&0u64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        // Simulate increment using CAS loop
        for expected_val in 0u64..10 {
            // Set expected value
            memory[208..216].copy_from_slice(&expected_val.to_le_bytes());

            // Set new value (expected + 1)
            memory[304..312].copy_from_slice(&(expected_val + 1).to_le_bytes());

            let action = Action {
                kind: Kind::AtomicCAS,
                dst: 104,
                src: 208,
                offset: 304,
                size: 8,
            };

            unsafe {
                unit.execute(&action);
            }
        }

        // Counter should be 10
        unsafe {
            let final_val = u64::from_le_bytes(shared.read(104, 8)[0..8].try_into().unwrap());
            assert_eq!(final_val, 10);
        }
    }

    #[test]
    fn test_cas128_success() {
        let mut memory = vec![0u8; 1024];

        // Use offset 112 instead of 100 (112 = 7 * 16, so it's 16-byte aligned)
        // Initialize 128-bit value (pointer: 0x1000, generation: 1)
        let initial = ((0x1000u128) << 64) | 1;
        memory[112..128].copy_from_slice(&initial.to_le_bytes());

        // Expected: same as initial
        memory[208..224].copy_from_slice(&initial.to_le_bytes());

        // New value (pointer: 0x2000, generation: 2)
        let new_val = ((0x2000u128) << 64) | 2;
        memory[304..320].copy_from_slice(&new_val.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::AtomicCAS,
            dst: 112,
            src: 208,
            offset: 304,
            size: 16,
        };

        unsafe {
            unit.execute(&action);

            // Should have swapped to new value
            let result = u128::from_le_bytes(shared.read(112, 16)[0..16].try_into().unwrap());
            assert_eq!(result, new_val);
        }
    }

    #[test]
    fn test_cas_aba_protection() {
        let mut memory = vec![0u8; 1024];

        // 128-bit value with pointer and generation
        // Use offset 112 for 16-byte alignment
        let ptr_a = 0x1000u64;
        let gen_1 = 1u64;
        let value_1 = ((ptr_a as u128) << 64) | (gen_1 as u128);

        memory[112..128].copy_from_slice(&value_1.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        // Try to CAS with same pointer but old generation (should fail)
        let old_gen_value = ((ptr_a as u128) << 64) | 0u128;
        memory[208..224].copy_from_slice(&old_gen_value.to_le_bytes());

        let new_value = ((0x2000u128) << 64) | 2u128;
        memory[304..320].copy_from_slice(&new_value.to_le_bytes());

        let action = Action {
            kind: Kind::AtomicCAS,
            dst: 112,
            src: 208,
            offset: 304,
            size: 16,
        };

        unsafe {
            unit.execute(&action);

            // Should have failed - generation mismatch
            let result = u128::from_le_bytes(shared.read(112, 16)[0..16].try_into().unwrap());
            assert_eq!(result, value_1);

            // Actual value written back shows the real generation
            let actual = u128::from_le_bytes(shared.read(208, 16)[0..16].try_into().unwrap());
            assert_eq!(actual, value_1);
        }
    }

    #[test]
    fn test_fence() {
        let mut memory = vec![0u8; 1024];
        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared);

        let action = Action {
            kind: Kind::Fence,
            dst: 0,
            src: 0,
            offset: 0,
            size: 0, // All fields ignored for fence
        };

        unsafe {
            // Fence doesn't crash and provides ordering guarantee
            unit.execute(&action);
            // Can't really test the effect without multiple threads
            // but at least verify it executes
        }
    }

    #[test]
    fn test_fence_ordering() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        use std::thread;

        let mut memory = vec![0u8; 1024];
        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));

        let data_ready = Arc::new(AtomicBool::new(false));
        let data_ready_clone = data_ready.clone();

        let shared_clone = shared.clone();
        let handle = thread::spawn(move || {
            let mut unit = MemoryUnit::new(shared_clone);

            // Write data
            let data = 42u64.to_le_bytes();
            unsafe {
                unit.shared.write(0, &data);
            }

            // Fence to ensure write completes before flag
            let fence_action = Action {
                kind: Kind::Fence,
                dst: 0,
                src: 0,
                offset: 0,
                size: 0,
            };
            unsafe {
                unit.execute(&fence_action);
            }

            // Signal ready
            data_ready_clone.store(true, Ordering::Release);
        });

        // Wait for data
        while !data_ready.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }

        // Read should see 42
        unsafe {
            let value = u64::from_le_bytes(shared.read(0, 8)[0..8].try_into().unwrap());
            assert_eq!(value, 42);
        }

        handle.join().unwrap();
    }
}

#[cfg(test)]
mod concurrent_tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_concurrent_simd_units() {
        // Test multiple SIMD units writing to different memory regions concurrently
        let mut memory = vec![0u8; 65536];
        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let counter = Arc::new(AtomicU32::new(0));
        let mut handles = vec![];

        // Spawn 4 SIMD units working in parallel
        for unit_id in 0..4u8 {
            let shared_clone = shared.clone();
            let counter_clone = counter.clone();

            handles.push(std::thread::spawn(move || {
                let mut unit = SimdUnit::new(
                    unit_id,
                    16,
                    shared_clone,
                );

                // Each unit does some SIMD operations
                let action = Action {
                    kind: Kind::SimdStore,
                    src: 0,
                    offset: 0,
                    size: 16,
                    dst: 0,
                };

                unsafe {
                    unit.execute(&action);
                }
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify we got results from all units
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_concurrent_file_operations() {
        // Test multiple file operations happening concurrently
        let mut memory = vec![0u8; 4096];
        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));

        // Setup multiple files
        let files = ["test1.txt", "test2.txt", "test3.txt"];
        let mut handles = vec![];

        for (i, filename) in files.iter().enumerate() {
            let offset = i * 100;
            memory[offset..offset + filename.len()].copy_from_slice(filename.as_bytes());
            memory[offset + filename.len()] = 0;

            // Write test data
            let data = format!("Data {}", i);
            memory[1000 + offset..1000 + offset + data.len()].copy_from_slice(data.as_bytes());

            let shared_clone = shared.clone();
            handles.push(std::thread::spawn(move || {
                let mut unit = FileUnit::new(i as u8, shared_clone, 1024);

                let action = Action {
                    kind: Kind::FileWrite,
                    src: (1000 + offset) as u32,
                    dst: offset as u32,
                    offset: 0,
                    size: data.len() as u32,
                };

                unit.execute(&action)
            }));
        }

        // Wait for all writes
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all files exist
        for filename in &files {
            assert!(std::path::Path::new(filename).exists());
            std::fs::remove_file(filename).ok();
        }
    }

    #[test]
    fn test_atomic_cas_contention() {
        // Test CAS under contention from multiple threads
        let mut memory = vec![0u8; 1024];
        memory[104..112].copy_from_slice(&0u64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let counter = Arc::new(AtomicU32::new(0));

        let mut handles = vec![];
        for _ in 0..10 {
            let shared_clone = shared.clone();
            let counter_clone = counter.clone();

            handles.push(std::thread::spawn(move || {
                // Try to increment the value using CAS
                loop {
                    let current = unsafe {
                        u64::from_le_bytes(shared_clone.read(104, 8)[0..8].try_into().unwrap())
                    };

                    let result = unsafe { shared_clone.cas64(104, current, current + 1) };

                    if result == current {
                        counter_clone.fetch_add(1, Ordering::SeqCst);
                        break;
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All 10 threads should have succeeded
        assert_eq!(counter.load(Ordering::SeqCst), 10);

        // Final value should be 10
        let final_val =
            unsafe { u64::from_le_bytes(shared.read(104, 8)[0..8].try_into().unwrap()) };
        assert_eq!(final_val, 10);
    }
}
