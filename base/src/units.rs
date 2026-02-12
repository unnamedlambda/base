use base_types::{Action, Kind};
use pollster::block_on;
use portable_atomic::{AtomicU128, AtomicU64, Ordering};
use quanta::Clock;
use std::collections::HashMap;
use std::sync::atomic::{fence, AtomicBool, AtomicU32};
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, info, info_span, trace, warn, Instrument};
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

async fn spin_backoff_async(spin_count: &mut u32) {
    *spin_count += 1;
    if *spin_count < 50 {
        std::hint::spin_loop();
        tokio::task::yield_now().await;
    } else if *spin_count < 200 {
        tokio::task::yield_now().await;
    } else {
        tokio::time::sleep(std::time::Duration::from_micros(50)).await;
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

    pub async fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::NetConnect => {
                // Read address from memory
                let addr = read_null_terminated_string(
                    &self.shared,
                    action.src as usize,
                    action.offset as usize,
                );
                debug!(addr = %addr, dst = action.dst, "net_connect");

                // Connect to remote OR bind+listen based on format
                if addr.starts_with(':') || addr.contains("0.0.0.0:") {
                    // It's a listener
                    if let Ok(listener) = TcpListener::bind(addr).await {
                        let handle = self.next_handle;
                        self.next_handle += 1;
                        self.listeners.insert(handle, listener);

                        // Write handle to dst
                        unsafe {
                            self.shared
                                .write(action.dst as usize, &handle.to_le_bytes());
                        }
                    }
                } else if let Ok(stream) = TcpStream::connect(addr).await {
                    // It's a connection
                    let handle = self.next_handle;
                    self.next_handle += 1;
                    self.connections.insert(handle, stream);

                    // Write handle to dst
                    unsafe {
                        self.shared
                            .write(action.dst as usize, &handle.to_le_bytes());
                    }
                }
            }

            Kind::NetAccept => {
                debug!(src = action.src, dst = action.dst, "net_accept");
                // Read listener handle from src
                let handle = unsafe {
                    u32::from_le_bytes(
                        self.shared.read(action.src as usize, 4)[0..4]
                            .try_into()
                            .unwrap(),
                    )
                };

                if let Some(listener) = self.listeners.get_mut(&handle) {
                    if let Ok((stream, _addr)) = listener.accept().await {
                        let conn_handle = self.next_handle;
                        self.next_handle += 1;
                        self.connections.insert(conn_handle, stream);

                        // Write new connection handle to dst
                        unsafe {
                            self.shared
                                .write(action.dst as usize, &conn_handle.to_le_bytes());
                        }
                    }
                }
            }

            Kind::NetSend => {
                debug!(dst = action.dst, src = action.src, size = action.size, "net_send");
                // Read connection handle from dst
                let handle = unsafe {
                    u32::from_le_bytes(
                        self.shared.read(action.dst as usize, 4)[0..4]
                            .try_into()
                            .unwrap(),
                    )
                };

                if let Some(stream) = self.connections.get_mut(&handle) {
                    let data = unsafe { self.shared.read(action.src as usize, action.size as usize) };
                    let _ = stream.write(data).await;
                }
            }

            Kind::NetRecv => {
                debug!(src = action.src, dst = action.dst, size = action.size, "net_recv");
                // Read connection handle from src
                let handle = unsafe {
                    u32::from_le_bytes(
                        self.shared.read(action.src as usize, 4)[0..4]
                            .try_into()
                            .unwrap(),
                    )
                };

                if let Some(stream) = self.connections.get_mut(&handle) {
                    let mut buffer = vec![0u8; action.size as usize];
                    if let Ok(n) = stream.read(&mut buffer).await {
                        unsafe {
                            self.shared.write(action.dst as usize, &buffer[..n]);
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

    pub async fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::FileRead => {
                let filename = read_null_terminated_string(
                    &self.shared,
                    action.src as usize,
                    4096,
                );
                debug!(filename = %filename, dst = action.dst, offset = action.offset, size = action.size, "file_read");

                if let Ok(mut file) = fs::File::open(&filename).await {
                    if action.offset > 0 {
                        let _ = file.seek(std::io::SeekFrom::Start(action.offset as u64)).await;
                    }
                    if action.size == 0 {
                        // Read entire file in chunks
                        let mut total_read = 0;
                        let dst_base = action.dst as usize;

                        loop {
                            match file.read(&mut self.buffer).await {
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
                        // Read specific amount
                        let read_size = (action.size as usize).min(self.buffer.len());
                        if let Ok(n) = file.read(&mut self.buffer[..read_size]).await {
                            unsafe {
                                self.shared.write(action.dst as usize, &self.buffer[..n]);
                            }
                        }
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
                    fs::File::create(&filename).await
                } else {
                    fs::OpenOptions::new()
                        .write(true)
                        .create(true)
                        .open(&filename)
                        .await
                };

                if let Ok(mut file) = file_result {
                    if action.offset > 0 {
                        let _ = file.seek(std::io::SeekFrom::Start(action.offset as u64)).await;
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
                            let _ = file.write_all(data).await;
                        }
                    } else {
                        let mut written = 0;
                        let total_size = action.size as usize;

                        // Write in chunks
                        while written < total_size {
                            let chunk_size = (total_size - written).min(self.buffer.len());
                            let data = unsafe { self.shared.read(src_base + written, chunk_size) };

                            match file.write_all(data).await {
                                Ok(_) => written += chunk_size,
                                Err(_) => break,
                            }
                        }
                    }

                    let _ = file.sync_all().await; // Ensure data hits disk
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
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    shared: Arc<SharedMemory>,
}

impl GpuUnit {
    pub fn new(shared: Arc<SharedMemory>, shader_source: &str, gpu_size: usize, backends: Backends) -> Self {
        let _span = info_span!("gpu_init", gpu_size, ?backends).entered();
        info!("initializing GPU unit");

        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Failed to find adapter");
        info!("GPU adapter acquired");

        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor::default(), None))
            .expect("Failed to create device");
        info!("GPU device created");

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

pub(crate) async fn network_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
) {
    async move {
        info!("network unit started");
    let mut unit = NetworkUnit::new(0, shared.clone());
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "network_work_received");
                for idx in start..end {
                    unit.execute(&actions[idx as usize]).await;
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
            MailboxPoll::Empty => spin_backoff_async(&mut spin_count).await,
        }
    }
    }
    .instrument(info_span!("network_unit"))
    .await
}

pub(crate) async fn file_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    buffer_size: usize,
) {
    async move {
        info!("file unit started");
    let mut unit = FileUnit::new(0, shared.clone(), buffer_size);
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { start, end, flag } => {
                debug!(start, end, flag, "file_work_received");
                for idx in start..end {
                    unit.execute(&actions[idx as usize]).await;
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
            MailboxPoll::Empty => spin_backoff_async(&mut spin_count).await,
        }
    }
    }
    .instrument(info_span!("file_unit"))
    .await
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

pub(crate) struct CraneliftUnit {
    shared: Arc<SharedMemory>,
    compiled_fn: unsafe extern "C" fn(*mut u8),
    _module: cranelift_jit::JITModule,
}

impl CraneliftUnit {
    pub fn new(shared: Arc<SharedMemory>, clif_source: &str) -> Self {
        info!(ir_len = clif_source.len(), "compiling Cranelift IR");

        // Parse CLIF text
        let functions = cranelift_reader::parse_functions(clif_source)
            .expect("Failed to parse CLIF IR");
        assert!(!functions.is_empty(), "No functions in CLIF IR");
        let func = functions.into_iter().next().unwrap();

        // Create JIT module with host ISA
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        let isa_builder = cranelift_native::builder().expect("Host ISA not supported");
        let isa = isa_builder.finish(settings::Flags::new(flag_builder)).unwrap();
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = cranelift_jit::JITModule::new(builder);

        // Declare function with the parsed signature
        let func_id = module
            .declare_function("main", cranelift_module::Linkage::Local, &func.signature)
            .expect("Failed to declare function");

        // Define function from parsed IR
        let mut ctx = cranelift_codegen::Context::for_function(func);
        module
            .define_function(func_id, &mut ctx)
            .expect("Failed to compile function");
        module.finalize_definitions().unwrap();

        // Get function pointer
        let code_ptr = module.get_finalized_function(func_id);
        let compiled_fn: unsafe extern "C" fn(*mut u8) =
            unsafe { std::mem::transmute(code_ptr) };

        info!("Cranelift IR compiled successfully");

        CraneliftUnit {
            shared,
            compiled_fn,
            _module: module,
        }
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        let ptr = self.shared.ptr.add(action.dst as usize);
        (self.compiled_fn)(ptr);
    }
}

pub(crate) fn cranelift_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    clif_source: String,
) {
    let _span = info_span!("cranelift_unit").entered();
    info!("Cranelift unit started");
    let mut unit = CraneliftUnit::new(shared.clone(), &clif_source);
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
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn test_network_connect_and_send() {
        // Start a test server
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn server task
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 5];
            stream.read_exact(&mut buf).await.unwrap();
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

        unit.execute(&connect_action).await;
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

        unit.execute(&send_action).await;
    }

    #[tokio::test]
    async fn test_network_listen_accept_recv() {
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

        unit.execute(&listen_action).await;

        // Get actual listening port for client
        let handle = unsafe { u32::from_le_bytes(shared.read(200, 4)[0..4].try_into().unwrap()) };
        let actual_addr = unit.listeners.get(&handle).unwrap().local_addr().unwrap();

        // Spawn client
        let client_task = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let mut stream = TcpStream::connect(actual_addr).await.unwrap();
            stream.write_all(b"test").await.unwrap();
        });

        // Accept connection
        let accept_action = Action {
            kind: Kind::NetAccept,
            src: 200, // listener handle
            dst: 300, // store connection handle
            offset: 0,
            size: 0,
        };

        unit.execute(&accept_action).await;

        // Receive data
        let recv_action = Action {
            kind: Kind::NetRecv,
            src: 300, // connection handle
            dst: 400, // store received data
            offset: 0,
            size: 100, // max bytes to receive
        };

        unit.execute(&recv_action).await;

        // Verify received data
        unsafe {
            let data = shared.read(400, 4);
            assert_eq!(data, b"test");
        }

        client_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_network_echo_server() {
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
        unit.execute(&listen_action).await;

        let handle = unsafe { u32::from_le_bytes(shared.read(200, 4)[0..4].try_into().unwrap()) };
        let addr = unit.listeners.get(&handle).unwrap().local_addr().unwrap();

        // Client task
        let client = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            stream.write_all(b"echo").await.unwrap();

            let mut buf = [0u8; 4];
            stream.read_exact(&mut buf).await.unwrap();
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
        unit.execute(&accept).await;

        // Receive
        let recv = Action {
            kind: Kind::NetRecv,
            src: 300,
            dst: 400,
            offset: 0,
            size: 100,
        };
        unit.execute(&recv).await;

        // Echo back - use the same size as recv action
        let send = Action {
            kind: Kind::NetSend,
            src: 400,
            dst: 300,
            offset: 0,
            size: recv.size,
        };
        unit.execute(&send).await;

        client.await.unwrap();
    }

    #[tokio::test]
    async fn test_multiple_connections() {
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
        .await;

        let handle = unsafe { u32::from_le_bytes(shared.read(200, 4)[0..4].try_into().unwrap()) };
        let addr = unit.listeners.get(&handle).unwrap().local_addr().unwrap();

        // Spawn multiple clients
        let mut clients = vec![];
        for i in 0..3 {
            let addr = addr.clone();
            clients.push(tokio::spawn(async move {
                let mut stream = TcpStream::connect(addr).await.unwrap();
                stream.write_all(&[i as u8]).await.unwrap();
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
            unit.execute(&accept).await;
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
            unit.execute(&recv).await;
        }

        for client in clients {
            client.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_network_handle_persistence() {
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
            unit.execute(&action).await;

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

    #[tokio::test]
    async fn test_invalid_connection() {
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

        unit.execute(&action).await;
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

    #[tokio::test]
    async fn test_file_write_and_read() {
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

        unit.execute(&write_action).await;

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

        unit.execute(&read_action).await;

        unsafe {
            let read_data = shared.read(200, test_data.len());
            assert_eq!(read_data, test_data);
        }

        // Cleanup
        fs::remove_file(test_file).ok();
    }

    #[tokio::test]
    async fn test_file_read_nonexistent() {
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

        unit.execute(&action).await;

        // Memory should remain unchanged (all zeros)
        unsafe {
            let data = shared.read(100, 10);
            assert_eq!(data.iter().filter(|&&b| b != 0).count(), 0);
        }
    }

    #[tokio::test]
    async fn test_file_size_limits() {
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

        unit.execute(&action).await;

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

    #[tokio::test]
    async fn test_filename_with_path() {
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

        unit.execute(&action).await;

        assert!(Path::new(test_file).exists());
        let contents = fs::read(test_file).unwrap();
        assert_eq!(contents, test_data);

        // Cleanup
        fs::remove_file(test_file).ok();
        fs::remove_dir(test_dir).ok();
    }

    #[tokio::test]
    async fn test_binary_data() {
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

        unit.execute(&write_action).await;

        // Read it back
        let read_action = Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 200,
            offset: 0,
            size: 0, // Read entire file
        };

        unit.execute(&read_action).await;

        unsafe {
            let read_data = shared.read(200, binary_data.len());
            assert_eq!(read_data, &binary_data);
        }

        // Cleanup
        fs::remove_file(test_file).ok();
    }

    #[tokio::test]
    async fn test_file_read_with_offset() {
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

        unit.execute(&action).await;

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

    #[tokio::test]
    async fn test_file_write_with_offset() {
        let test_file = "test_write_offset.txt";
        let initial_data = b"AAAAAABBBBBB"; // 12 bytes

        // First write: create file with initial content
        fs::write(test_file, initial_data).await.unwrap();

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

        unit.execute(&action).await;

        // Verify file now contains "AAAAXXBBBBBB"
        let file_contents = fs::read(test_file).await.unwrap();
        assert_eq!(file_contents, b"AAAAXXBBBBBB");

        // Cleanup
        fs::remove_file(test_file).await.ok();
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_simd_units() {
        // Test multiple SIMD units writing to different memory regions concurrently
        let mut memory = vec![0u8; 65536];
        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let counter = Arc::new(AtomicU32::new(0));
        let mut handles = vec![];

        // Spawn 4 SIMD units working in parallel
        for unit_id in 0..4u8 {
            let shared_clone = shared.clone();
            let counter_clone = counter.clone();

            handles.push(tokio::spawn(async move {
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
            handle.await.unwrap();
        }

        // Verify we got results from all units
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn test_concurrent_file_operations() {
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
            handles.push(tokio::spawn(async move {
                let mut unit = FileUnit::new(i as u8, shared_clone, 1024);

                let action = Action {
                    kind: Kind::FileWrite,
                    src: (1000 + offset) as u32,
                    dst: offset as u32,
                    offset: 0,
                    size: data.len() as u32,
                };

                unit.execute(&action).await
            }));
        }

        // Wait for all writes
        for handle in handles {
            handle.await.unwrap();
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
