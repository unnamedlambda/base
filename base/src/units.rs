use base_types::{Action, Kind};
use pollster::block_on;
use portable_atomic::{AtomicU128, AtomicU64, Ordering};
use quanta::Clock;
use std::collections::HashMap;
use std::sync::atomic::fence;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    InstanceDescriptor, PipelineCompilationOptions, PipelineLayoutDescriptor, PowerPreference,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};
use wide::f32x4;

#[derive(Clone, Copy, Debug)]
pub(crate) struct QueueItem {
    pub action_index: u32,
    pub offset: u32,
    pub size: u16,
    pub unit_id: u8,
    pub _pad: u8,
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

    pub unsafe fn cas64(&self, offset: usize, expected: u64, new: u64) -> u64 {
        let ptr = self.ptr.add(offset) as *mut AtomicU64;
        (*ptr)
            .compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst)
            .unwrap_or_else(|x| x)
    }

    pub unsafe fn cas128(&self, offset: usize, expected: u128, new: u128) -> u128 {
        let ptr = self.ptr.add(offset) as *mut AtomicU128;
        (*ptr)
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
                // Read first 8 bytes at offset as condition
                let cond_bytes = self.shared.read(action.offset as usize, 8);
                let condition = f64::from_le_bytes(cond_bytes[0..8].try_into().unwrap());

                if condition != 0.0 {
                    let src_ptr = self.shared.ptr.add(action.src as usize);
                    let dst_ptr = self.shared.ptr.add(action.dst as usize);
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, action.size as usize);
                }
            }
            Kind::MemCopy => {
                let src_ptr = self.shared.ptr.add(action.src as usize);
                let dst_ptr = self.shared.ptr.add(action.dst as usize);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, action.size as usize);
            }
            Kind::MemScan => {
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
                if action.size == 16 {
                    let expected = read_as_u128(&self.shared, action.src as usize);
                    let new = read_as_u128(&self.shared, action.offset as usize);
                    let actual = self.shared.cas128(action.dst as usize, expected, new);
                    self.shared
                        .write(action.src as usize, &actual.to_le_bytes());
                } else {
                    let expected = read_as_u64(&self.shared, action.src as usize);
                    let new = read_as_u64(&self.shared, action.offset as usize);
                    let actual = self.shared.cas64(action.dst as usize, expected, new);
                    self.shared
                        .write(action.src as usize, &actual.to_le_bytes()[0..8]);
                }
            }
            Kind::Fence => {
                fence(Ordering::SeqCst);
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
                // Read function pointer from memory
                let fn_bytes = self.shared.read(action.src as usize, 8);
                let fn_ptr = usize::from_le_bytes(fn_bytes[0..8].try_into().unwrap());

                if fn_ptr == 0 {
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

    pub async fn execute(&mut self, action: &Action) -> Option<QueueItem> {
        match action.kind {
            Kind::NetConnect => {
                // Read address from memory
                let addr = read_null_terminated_string(
                    &self.shared,
                    action.src as usize,
                    action.offset as usize,
                );

                // Connect to remote OR bind+listen based on format
                if addr.starts_with(':') || addr.contains("0.0.0.0:") {
                    // It's a listener
                    let listener = TcpListener::bind(addr).await.ok()?;
                    let handle = self.next_handle;
                    self.next_handle += 1;
                    self.listeners.insert(handle, listener);

                    // Write handle to dst
                    unsafe {
                        self.shared
                            .write(action.dst as usize, &handle.to_le_bytes());
                    }

                    Some(QueueItem {
                        action_index: 0,
                        offset: action.dst,
                        size: 4,
                        unit_id: self.id,
                        _pad: 0,
                    })
                } else {
                    // It's a connection
                    let stream = TcpStream::connect(addr).await.ok()?;
                    let handle = self.next_handle;
                    self.next_handle += 1;
                    self.connections.insert(handle, stream);

                    // Write handle to dst
                    unsafe {
                        self.shared
                            .write(action.dst as usize, &handle.to_le_bytes());
                    }

                    Some(QueueItem {
                        action_index: 0,
                        offset: action.dst,
                        size: 4,
                        unit_id: self.id,
                        _pad: 0,
                    })
                }
            }

            Kind::NetAccept => {
                // Read listener handle from src
                let handle = unsafe {
                    u32::from_le_bytes(
                        self.shared.read(action.src as usize, 4)[0..4]
                            .try_into()
                            .unwrap(),
                    )
                };

                let listener = self.listeners.get_mut(&handle)?;
                let (stream, _addr) = listener.accept().await.ok()?;

                let conn_handle = self.next_handle;
                self.next_handle += 1;
                self.connections.insert(conn_handle, stream);

                // Write new connection handle to dst
                unsafe {
                    self.shared
                        .write(action.dst as usize, &conn_handle.to_le_bytes());
                }

                Some(QueueItem {
                    action_index: 0,
                    offset: action.dst,
                    size: 4,
                    unit_id: self.id,
                    _pad: 0,
                })
            }

            Kind::NetSend => {
                // Read connection handle from dst
                let handle = unsafe {
                    u32::from_le_bytes(
                        self.shared.read(action.dst as usize, 4)[0..4]
                            .try_into()
                            .unwrap(),
                    )
                };

                let stream = self.connections.get_mut(&handle)?;
                let data = unsafe { self.shared.read(action.src as usize, action.size as usize) };

                let n = stream.write(data).await.ok()?;

                Some(QueueItem {
                    action_index: 0,
                    offset: action.src,
                    size: n as u16,
                    unit_id: self.id,
                    _pad: 0,
                })
            }

            Kind::NetRecv => {
                // Read connection handle from src
                let handle = unsafe {
                    u32::from_le_bytes(
                        self.shared.read(action.src as usize, 4)[0..4]
                            .try_into()
                            .unwrap(),
                    )
                };

                let stream = self.connections.get_mut(&handle)?;
                let mut buffer = vec![0u8; action.size as usize];
                let n = stream.read(&mut buffer).await.ok()?;

                unsafe {
                    self.shared.write(action.dst as usize, &buffer[..n]);
                }

                Some(QueueItem {
                    action_index: 0,
                    offset: action.dst,
                    size: n as u16,
                    unit_id: self.id,
                    _pad: 0,
                })
            }

            _ => None,
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

    pub async fn execute(&mut self, action: &Action) -> Option<QueueItem> {
        match action.kind {
            Kind::FileRead => {
                let filename = read_null_terminated_string(
                    &self.shared,
                    action.src as usize,
                    action.offset as usize,
                );

                match fs::File::open(&filename).await {
                    Ok(mut file) => {
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

                            Some(QueueItem {
                                action_index: 0,
                                offset: action.dst,
                                size: (total_read.min(u16::MAX as usize)) as u16,
                                unit_id: self.id,
                                _pad: 0,
                            })
                        } else {
                            // Read specific amount
                            let read_size = (action.size as usize).min(self.buffer.len());
                            match file.read(&mut self.buffer[..read_size]).await {
                                Ok(n) => {
                                    unsafe {
                                        self.shared.write(action.dst as usize, &self.buffer[..n]);
                                    }
                                    Some(QueueItem {
                                        action_index: 0,
                                        offset: action.dst,
                                        size: n as u16,
                                        unit_id: self.id,
                                        _pad: 0,
                                    })
                                }
                                Err(_) => None,
                            }
                        }
                    }
                    Err(_) => None,
                }
            }
            Kind::FileWrite => {
                let filename = read_null_terminated_string(
                    &self.shared,
                    action.dst as usize,
                    action.offset as usize,
                );

                match fs::File::create(&filename).await {
                    Ok(mut file) => {
                        let mut written = 0;
                        let total_size = action.size as usize;
                        let src_base = action.src as usize;

                        // Write in chunks
                        while written < total_size {
                            let chunk_size = (total_size - written).min(self.buffer.len());
                            let data = unsafe { self.shared.read(src_base + written, chunk_size) };

                            match file.write_all(data).await {
                                Ok(_) => written += chunk_size,
                                Err(_) => break,
                            }
                        }

                        let _ = file.sync_all().await; // Ensure data hits disk

                        None
                    }
                    Err(_) => None,
                }
            }
            _ => None,
        }
    }
}

pub(crate) struct ComputationalUnit {
    regs: Vec<f64>,
    clock: Clock,
}

impl ComputationalUnit {
    pub fn new(regs: usize) -> Self {
        Self {
            regs: vec![0.0; regs],
            clock: Clock::new(),
        }
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::Approximate => {
                let base = self.regs[action.src as usize];
                let iterations = action.offset as usize;
                let mut x = base;
                for _ in 0..iterations {
                    x = 0.5 * (x + base / x)
                }
                self.regs[action.dst as usize] = x;
            }
            Kind::Choose => {
                let n = self.regs[action.src as usize] as u64;
                if n > 0 {
                    let choice = rand::random::<u64>() % n;
                    self.regs[action.dst as usize] = choice as f64;
                }
            }
            Kind::Compare => {
                let a = self.regs[action.src as usize];
                let b = self.regs[action.offset as usize];
                self.regs[action.dst as usize] = if a > b { 1.0 } else { 0.0 };
            }
            Kind::Timestamp => {
                // Store current timestamp in register
                self.regs[action.dst as usize] = self.clock.raw() as f64;
            }
            _ => {}
        }
    }
}

pub(crate) struct SimdUnit {
    id: u8,
    regs: Vec<f32x4>,
    scratch: Arc<Vec<u8>>,
    scratch_offset: usize,
    scratch_size: usize,
    shared: Arc<SharedMemory>,
    shared_offset: usize,
}

impl SimdUnit {
    pub fn new(
        id: u8,
        regs: usize,
        scratch: Arc<Vec<u8>>,
        scratch_offset: usize,
        scratch_size: usize,
        shared: Arc<SharedMemory>,
        shared_offset: usize,
    ) -> Self {
        Self {
            id,
            regs: vec![f32x4::splat(0.0); regs],
            scratch,
            scratch_offset,
            scratch_size,
            shared,
            shared_offset,
        }
    }

    pub unsafe fn execute(&mut self, action: &Action) -> Option<QueueItem> {
        match action.kind {
            Kind::SimdLoad => {
                let offset = self.scratch_offset + action.offset as usize;
                if offset + 16 <= self.scratch_offset + self.scratch_size {
                    let vals = &self.scratch[offset..offset + 16];
                    let floats: Vec<f32> = vals
                        .chunks(4)
                        .map(|chunk| {
                            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                            f32::from_le_bytes(bytes)
                        })
                        .collect();
                    if floats.len() >= 4 {
                        self.regs[action.dst as usize] =
                            f32x4::from([floats[0], floats[1], floats[2], floats[3]]);
                    }
                }
                None
            }
            Kind::SimdAdd => {
                let a = self.regs[action.src as usize];
                let b = self.regs[action.offset as usize];
                self.regs[action.dst as usize] = a + b;
                None
            }
            Kind::SimdMul => {
                let a = self.regs[action.src as usize];
                let b = self.regs[action.offset as usize];
                self.regs[action.dst as usize] = a * b;
                None
            }
            Kind::SimdStore => {
                let reg_data = self.regs[action.src as usize].to_array();
                let write_offset = self.shared_offset + (action.offset as usize);

                let bytes: Vec<u8> = reg_data.iter().flat_map(|f| f.to_le_bytes()).collect();
                self.shared.write(write_offset, &bytes);

                Some(QueueItem {
                    action_index: 0,
                    offset: write_offset as u32,
                    size: 16,
                    unit_id: self.id,
                    _pad: 0,
                })
            }
            _ => None,
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
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Failed to find adapter");

        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor::default(), None))
            .expect("Failed to create device");

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
            }
            _ => {
                // Other GPU actions (CreateBuffer, etc.) not yet implemented
            }
        }
    }
}

pub(crate) async fn memory_unit_task(
    mut rx: mpsc::Receiver<QueueItem>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
) {
    let mut unit = MemoryUnit::new(shared.clone());

    while let Some(item) = rx.recv().await {
        let action = &actions[item.action_index as usize];
        unsafe {
            unit.execute(action);
        }

        // Write completion flag
        unsafe {
            let flag_offset = action.offset as usize;
            shared.write(flag_offset, &1u64.to_le_bytes());
        }
    }
}

pub(crate) async fn ffi_unit_task(
    mut rx: mpsc::Receiver<QueueItem>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
) {
    let mut unit = FFIUnit::new(shared.clone());

    while let Some(item) = rx.recv().await {
        let action = &actions[item.action_index as usize];
        unsafe {
            unit.execute(action);
        }

        // Write completion flag
        unsafe {
            let flag_offset = action.offset as usize;
            shared.write(flag_offset, &1u64.to_le_bytes());
        }
    }
}

pub(crate) async fn network_unit_task(
    mut rx: mpsc::Receiver<QueueItem>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
) {
    let mut unit = NetworkUnit::new(0, shared.clone());

    while let Some(item) = rx.recv().await {
        let action = &actions[item.action_index as usize];

        unit.execute(action).await;

        // Write completion flag
        unsafe {
            let flag_offset = action.offset as usize;
            shared.write(flag_offset, &1u64.to_le_bytes());
        }
    }
}

pub(crate) async fn file_unit_task(
    mut rx: mpsc::Receiver<QueueItem>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    buffer_size: usize,
) {
    let mut unit = FileUnit::new(0, shared.clone(), buffer_size);

    while let Some(item) = rx.recv().await {
        let action = &actions[item.action_index as usize];

        unit.execute(action).await;

        // Write completion flag
        unsafe {
            let flag_offset = action.offset as usize;
            shared.write(flag_offset, &1u64.to_le_bytes());
        }
    }
}

pub(crate) async fn computational_unit_task(
    mut rx: mpsc::Receiver<QueueItem>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    regs: usize,
) {
    let mut unit = ComputationalUnit::new(regs);

    while let Some(item) = rx.recv().await {
        let action = &actions[item.action_index as usize];
        unsafe {
            unit.execute(action);
        }

        // Write completion flag
        unsafe {
            let flag_offset = action.offset as usize;
            shared.write(flag_offset, &1u64.to_le_bytes());
        }
    }
}

pub(crate) async fn simd_unit_task(
    mut rx: mpsc::Receiver<QueueItem>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    regs: usize,
    scratch: Arc<Vec<u8>>,
    scratch_offset: usize,
    scratch_size: usize,
    shared_offset: usize,
) {
    let mut unit = SimdUnit::new(
        0,
        regs,
        scratch,
        scratch_offset,
        scratch_size,
        shared.clone(),
        shared_offset,
    );

    while let Some(item) = rx.recv().await {
        let action = &actions[item.action_index as usize];
        unsafe {
            unit.execute(action);
        }

        // Write completion flag
        unsafe {
            let flag_offset = action.offset as usize;
            shared.write(flag_offset, &1u64.to_le_bytes());
        }
    }
}

pub(crate) async fn gpu_unit_task(
    mut rx: mpsc::Receiver<QueueItem>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    shader_source: String,
    gpu_size: usize,
    backends: Backends,
) {
    let mut gpu = GpuUnit::new(shared.clone(), &shader_source, gpu_size, backends);

    while let Some(item) = rx.recv().await {
        let action = &actions[item.action_index as usize];
        unsafe { gpu.execute(action); }

        // Write completion flag
        unsafe {
            shared.write(action.offset as usize, &1u64.to_le_bytes());
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
    fn test_simd_unit_creation() {
        let memory = vec![0u8; 1024];
        let shared = Arc::new(SharedMemory::new(memory.as_ptr() as *mut u8));
        let scratch = Arc::new(vec![0u8; 256]);

        let unit = SimdUnit::new(0, 16, scratch, 0, 256, shared, 0);
        assert_eq!(unit.id, 0);
        assert_eq!(unit.regs.len(), 16);
    }

    #[test]
    fn test_queue_item_size() {
        // Ensure QueueItem is small for efficient channel passing
        assert_eq!(std::mem::size_of::<QueueItem>(), 8);
    }

    #[test]
    fn test_computational_unit_creation() {
        let unit = ComputationalUnit::new(32);
        assert_eq!(unit.regs.len(), 32);
    }

    #[test]
    fn test_approximate_action() {
        let mut unit = ComputationalUnit::new(8);

        unit.regs[1] = 16.0;

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
        assert_eq!(unit.regs[2], 4.0);
    }

    #[test]
    fn test_choose_action() {
        let mut unit = ComputationalUnit::new(8);

        unit.regs[1] = 100.0;

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
        assert!(unit.regs[2] >= 0.0);
        assert!(unit.regs[2] < 100.0);
    }

    #[test]
    fn test_choose_ranges() {
        let mut unit = ComputationalUnit::new(8);

        for n in [1.0, 10.0, 50.0, 1000.0] {
            unit.regs[0] = n;

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
                assert!(unit.regs[1] >= 0.0);
                assert!(unit.regs[1] < n);
            }
        }
    }

    #[test]
    fn test_compare_action() {
        let mut unit = ComputationalUnit::new(8);

        // Test greater than
        unit.regs[0] = 5.0;
        unit.regs[1] = 3.0;

        let action = Action {
            kind: Kind::Compare,
            dst: 2,
            src: 0,
            offset: 1,
            size: 0,
        };

        unsafe {
            unit.execute(&action);
        }
        assert_eq!(unit.regs[2], 1.0); // 5 > 3 is true

        // Test less than
        unit.regs[0] = 2.0;
        unit.regs[1] = 7.0;

        unsafe {
            unit.execute(&action);
        }
        assert_eq!(unit.regs[2], 0.0); // 2 > 7 is false

        // Test equal
        unit.regs[0] = 4.0;
        unit.regs[1] = 4.0;

        unsafe {
            unit.execute(&action);
        }
        assert_eq!(unit.regs[2], 0.0); // 4 > 4 is false (not greater)
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

        // Set condition to 1.0 (true)
        memory[0..8].copy_from_slice(&1.0f64.to_le_bytes());
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

        // Set condition to 0.0 (false)
        memory[0..8].copy_from_slice(&0.0f64.to_le_bytes());
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
    fn test_conditional_with_float_values() {
        let mut memory = vec![0u8; 1024];

        // Test with different condition values
        for (cond_val, should_copy) in [
            (0.0f64, false),
            (1.0f64, true),
            (-1.0f64, true),
            (0.001f64, true),
        ] {
            memory[0..8].copy_from_slice(&cond_val.to_le_bytes());
            memory[100..108].copy_from_slice(&42.0f64.to_le_bytes());
            memory[200..208].copy_from_slice(&0.0f64.to_le_bytes());

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
                let result = f64::from_le_bytes(result_bytes.try_into().unwrap());

                if should_copy {
                    assert_eq!(result, 42.0, "Condition {} should copy", cond_val);
                } else {
                    assert_eq!(result, 0.0, "Condition {} should not copy", cond_val);
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

        let result = unit.execute(&connect_action).await;
        assert!(result.is_some());

        // Send data
        let send_action = Action {
            kind: Kind::NetSend,
            src: 100, // data at offset 100
            dst: 200, // handle at offset 200
            offset: 0,
            size: 5, // send 5 bytes
        };

        let result = unit.execute(&send_action).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().size, 5);
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

        let result = unit.execute(&listen_action).await;
        assert!(result.is_some());

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

        let result = unit.execute(&accept_action).await;
        assert!(result.is_some());

        // Receive data
        let recv_action = Action {
            kind: Kind::NetRecv,
            src: 300, // connection handle
            dst: 400, // store received data
            offset: 0,
            size: 100, // max bytes to receive
        };

        let result = unit.execute(&recv_action).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().size, 4);

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
        unit.execute(&listen_action).await.unwrap();

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
        unit.execute(&accept).await.unwrap();

        // Receive
        let recv = Action {
            kind: Kind::NetRecv,
            src: 300,
            dst: 400,
            offset: 0,
            size: 100,
        };
        let result = unit.execute(&recv).await.unwrap();

        // Echo back
        let send = Action {
            kind: Kind::NetSend,
            src: 400,
            dst: 300,
            offset: 0,
            size: result.size as u32,
        };
        unit.execute(&send).await.unwrap();

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
        .await
        .unwrap();

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
            unit.execute(&accept).await.unwrap();
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
            unit.execute(&recv).await.unwrap();
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
            unit.execute(&action).await.unwrap();

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
        let mut unit = NetworkUnit::new(0, shared);

        let action = Action {
            kind: Kind::NetConnect,
            src: 0,
            dst: 200,
            offset: 50,
            size: 0,
        };

        let result = unit.execute(&action).await;
        assert!(result.is_none()); // Should fail gracefully
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
            offset: 50,                   // max filename length
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
            offset: 50, // max filename length
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
            offset: 50,
            size: 100,
        };

        let result = unit.execute(&action).await;
        // Should return None for failed read
        assert!(result.is_none());

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
            offset: 50,
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
            offset: 100, // Longer for path
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
            offset: 50,
            size: binary_data.len() as u32,
        };

        unit.execute(&write_action).await;

        // Read it back
        let read_action = Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 200,
            offset: 50,
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
}

#[cfg(test)]
mod atomic_tests {
    use super::*;

    #[test]
    fn test_cas64_success() {
        let mut memory = vec![0u8; 1024];

        // Initialize value to 42
        memory[100..108].copy_from_slice(&42u64.to_le_bytes());

        // Expected: 42, New: 100
        memory[200..208].copy_from_slice(&42u64.to_le_bytes());
        memory[300..308].copy_from_slice(&100u64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::AtomicCAS,
            dst: 100,    // target location
            src: 200,    // expected value location
            offset: 300, // new value location
            size: 8,     // 64-bit
        };

        unsafe {
            unit.execute(&action);

            // Should have swapped to 100
            let result = u64::from_le_bytes(shared.read(100, 8)[0..8].try_into().unwrap());
            assert_eq!(result, 100);

            // Old value (42) should be written back to src
            let old = u64::from_le_bytes(shared.read(200, 8)[0..8].try_into().unwrap());
            assert_eq!(old, 42);
        }
    }

    #[test]
    fn test_cas64_failure() {
        let mut memory = vec![0u8; 1024];

        // Initialize value to 42
        memory[100..108].copy_from_slice(&42u64.to_le_bytes());

        memory[200..208].copy_from_slice(&50u64.to_le_bytes());
        memory[300..308].copy_from_slice(&100u64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        let action = Action {
            kind: Kind::AtomicCAS,
            dst: 100,
            src: 200,
            offset: 300,
            size: 8,
        };

        unsafe {
            unit.execute(&action);

            // Should still be 42 (CAS failed)
            let result = u64::from_le_bytes(shared.read(100, 8)[0..8].try_into().unwrap());
            assert_eq!(result, 42);

            // Actual value (42) should be written back to src
            let actual = u64::from_le_bytes(shared.read(200, 8)[0..8].try_into().unwrap());
            assert_eq!(actual, 42);
        }
    }

    #[test]
    fn test_cas_loop_increment() {
        let mut memory = vec![0u8; 1024];

        // Initialize counter to 0
        memory[100..108].copy_from_slice(&0u64.to_le_bytes());

        let shared = Arc::new(SharedMemory::new(memory.as_mut_ptr()));
        let mut unit = MemoryUnit::new(shared.clone());

        // Simulate increment using CAS loop
        for expected_val in 0u64..10 {
            // Set expected value
            memory[200..208].copy_from_slice(&expected_val.to_le_bytes());

            // Set new value (expected + 1)
            memory[300..308].copy_from_slice(&(expected_val + 1).to_le_bytes());

            let action = Action {
                kind: Kind::AtomicCAS,
                dst: 100,
                src: 200,
                offset: 300,
                size: 8,
            };

            unsafe {
                unit.execute(&action);
            }
        }

        // Counter should be 10
        unsafe {
            let final_val = u64::from_le_bytes(shared.read(100, 8)[0..8].try_into().unwrap());
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
        let scratch = Arc::new(vec![0u8; 16384]);

        let (tx, mut rx) = mpsc::channel(100);
        let mut handles = vec![];

        // Spawn 4 SIMD units working in parallel
        for unit_id in 0..4u8 {
            let shared_clone = shared.clone();
            let scratch_clone = scratch.clone();
            let tx_clone = tx.clone();

            handles.push(tokio::spawn(async move {
                let mut unit = SimdUnit::new(
                    unit_id,
                    16,
                    scratch_clone,
                    unit_id as usize * 4096,
                    4096,
                    shared_clone,
                    unit_id as usize * 1024,
                );

                // Each unit does some SIMD operations
                let action = Action {
                    kind: Kind::SimdStore,
                    src: 0,
                    offset: 0,
                    size: 16,
                    dst: 0,
                };

                if let Some(item) = unsafe { unit.execute(&action) } {
                    tx_clone.send(item).await.unwrap();
                }
            }));
        }

        drop(tx);

        // Collect all results
        let mut items = vec![];
        while let Some(item) = rx.recv().await {
            items.push(item);
        }

        // Verify we got results from all units
        assert_eq!(items.len(), 4);

        // Verify each unit wrote to its own region
        let mut unit_ids: Vec<u8> = items.iter().map(|i| i.unit_id).collect();
        unit_ids.sort();
        assert_eq!(unit_ids, vec![0, 1, 2, 3]);
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
                    offset: 50,
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
        memory[100..108].copy_from_slice(&0u64.to_le_bytes());

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
                        u64::from_le_bytes(shared_clone.read(100, 8)[0..8].try_into().unwrap())
                    };

                    let result = unsafe { shared_clone.cas64(100, current, current + 1) };

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
            unsafe { u64::from_le_bytes(shared.read(100, 8)[0..8].try_into().unwrap()) };
        assert_eq!(final_val, 10);
    }
}
