use crate::types::{Action, Kind};
use pollster::block_on;
use quanta::Clock;
use std::sync::Arc;
use tokio::sync::mpsc;
use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor, DeviceDescriptor,
    InstanceDescriptor, PipelineLayoutDescriptor, PowerPreference, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};
use wide::f32x4;

#[derive(Clone, Copy, Debug)]
pub(crate) struct QueueItem {
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

pub(crate) struct WriteUnit {
    shared: Arc<SharedMemory>,
}

impl WriteUnit {
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
            Kind::FileRead => {
                // Read filename from memory (null-terminated string)
                // action.src = memory offset where filename starts
                // action.offset = max filename length to read
                let filename = read_null_terminated_string(
                    &self.shared,
                    action.src as usize,
                    action.offset as usize,
                );

                // Read entire file into memory
                // action.dst = memory offset to write file contents
                // action.size = max bytes to read from file (0 = entire file)
                let data = std::fs::read(&filename).unwrap_or_default();

                if action.size == 0 {
                    // Write entire file
                    self.shared.write(action.dst as usize, &data);
                } else {
                    // Write up to size bytes
                    let len = data.len().min(action.size as usize);
                    self.shared.write(action.dst as usize, &data[..len]);
                }
            }
            Kind::FileWrite => {
                // Read filename from memory (null-terminated string)
                // action.dst = memory offset where filename starts
                // action.offset = max filename length to read
                let filename = read_null_terminated_string(
                    &self.shared,
                    action.dst as usize,
                    action.offset as usize,
                );

                // Write data from memory to file
                // action.src = memory offset of data to write
                // action.size = number of bytes to write
                let data = self.shared.read(action.src as usize, action.size as usize);
                let _ = std::fs::write(&filename, data);
            }
            _ => {}
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
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    shared: Arc<SharedMemory>,
}

impl GpuUnit {
    pub fn new(shared: Arc<SharedMemory>, gpu_size: usize, backends: Backends) -> Self {
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

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Compute"),
            source: ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0)
                var<storage, read_write> data: array<f32>;
                
                @compute @workgroup_size(64)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let i = id.x;
                    if (i < arrayLength(&data)) {
                        data[i] = data[i] * 2.0;
                    }
                }
            "#
                .into(),
            ),
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
            pipeline,
            bind_group,
            shared,
        }
    }

    pub unsafe fn process_batch(&mut self, items: &[QueueItem]) {
        if items.is_empty() {
            return;
        }

        let mut batch_data = Vec::with_capacity(items.len() * 16);

        for item in items {
            let data = self.shared.read(item.offset as usize, item.size as usize);
            batch_data.extend_from_slice(data);
        }

        self.queue
            .write_buffer(&self.compute_buffer, 0, &batch_data);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((items.len() as u32 * 4 + 63) / 64, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
    }
}

pub(crate) async fn write_unit_task(
    actions: Arc<Vec<Action>>,
    indices: Vec<usize>,
    shared: Arc<SharedMemory>,
) {
    let mut unit = WriteUnit::new(shared);

    for idx in indices {
        unsafe {
            unit.execute(&actions[idx]);
        }
    }
}

pub(crate) async fn computational_unit_task(
    actions: Arc<Vec<Action>>,
    indices: Vec<usize>,
    regs: usize,
) {
    let mut unit = ComputationalUnit::new(regs);

    for idx in indices {
        unsafe {
            unit.execute(&actions[idx]);
        }
    }
}

pub(crate) async fn simd_unit_task(
    id: u8,
    actions: Arc<Vec<Action>>,
    indices: Vec<usize>,
    scratch: Arc<Vec<u8>>,
    scratch_offset: usize,
    scratch_size: usize,
    shared: Arc<SharedMemory>,
    shared_offset: usize,
    regs: usize,
    tx: mpsc::Sender<QueueItem>,
) {
    let mut unit = SimdUnit::new(
        id,
        regs,
        scratch,
        scratch_offset,
        scratch_size,
        shared,
        shared_offset,
    );

    for idx in indices {
        if let Some(item) = unsafe { unit.execute(&actions[idx]) } {
            let _ = tx.send(item).await;
        }
    }
}

pub(crate) async fn gpu_unit_task(
    mut rx: mpsc::Receiver<QueueItem>,
    shared: Arc<SharedMemory>,
    gpu_size: usize,
    batch_size: usize,
    backends: Backends,
) {
    let mut gpu = GpuUnit::new(shared, gpu_size, backends);
    let mut batch = Vec::with_capacity(batch_size);

    while let Some(item) = rx.recv().await {
        batch.push(item);

        if batch.len() >= batch_size {
            unsafe {
                gpu.process_batch(&batch);
            }
            batch.clear();
        }
    }

    if !batch.is_empty() {
        unsafe {
            gpu.process_batch(&batch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut unit = WriteUnit::new(shared.clone());

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
        let mut unit = WriteUnit::new(shared.clone());

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
        let mut unit = WriteUnit::new(shared.clone());

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
        let mut unit = WriteUnit::new(shared.clone());

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
        let mut unit = WriteUnit::new(shared.clone());

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
            let mut unit = WriteUnit::new(shared.clone());

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
        let mut unit = WriteUnit::new(shared.clone());
        
        // Test FileWrite
        let write_action = Action {
            kind: Kind::FileWrite,
            src: 100,        // data at offset 100
            dst: 0,          // filename at offset 0
            offset: 50,      // max filename length
            size: test_data.len() as u32,  // write 12 bytes
        };
        
        unsafe {
            unit.execute(&write_action);
        }
        
        // Verify file was created
        assert!(Path::new(test_file).exists());
        let file_contents = fs::read(test_file).unwrap();
        assert_eq!(file_contents, test_data);
        
        // Test FileRead
        // Clear the data area first
        memory[200..212].fill(0);
        
        let read_action = Action {
            kind: Kind::FileRead,
            src: 0,          // filename at offset 0
            dst: 200,        // write data to offset 200
            offset: 50,      // max filename length
            size: 100,       // max bytes to read
        };
        
        unsafe {
            unit.execute(&read_action);
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
        let mut unit = WriteUnit::new(shared.clone());
        
        let action = Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 100,
            offset: 50,
            size: 100,
        };
        
        unsafe {
            unit.execute(&action);
            // Should write empty data (unwrap_or_default)
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
        let mut unit = WriteUnit::new(shared.clone());
        
        // Read only first 10 bytes
        let action = Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 100,
            offset: 50,
            size: 10,  // Limit to 10 bytes
        };
        
        unsafe {
            unit.execute(&action);
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
        let mut unit = WriteUnit::new(shared.clone());
        
        let action = Action {
            kind: Kind::FileWrite,
            src: 100,
            dst: 0,
            offset: 100,  // Longer for path
            size: test_data.len() as u32,
        };
        
        unsafe {
            unit.execute(&action);
        }
        
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
        let mut unit = WriteUnit::new(shared.clone());
        
        // Write binary file
        let write_action = Action {
            kind: Kind::FileWrite,
            src: 100,
            dst: 0,
            offset: 50,
            size: binary_data.len() as u32,
        };
        
        unsafe {
            unit.execute(&write_action);
        }
        
        // Read it back
        let read_action = Action {
            kind: Kind::FileRead,
            src: 0,
            dst: 200,
            offset: 50,
            size: 0,  // Read entire file
        };
        
        unsafe {
            unit.execute(&read_action);
            let read_data = shared.read(200, binary_data.len());
            assert_eq!(read_data, &binary_data);
        }
        
        // Cleanup
        fs::remove_file(test_file).ok();
    }
}
