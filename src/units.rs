use crate::types::{Action, Kind};
use pollster::block_on;
use std::sync::Arc;
use tokio::sync::mpsc;
use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BufferBindingType, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    DeviceDescriptor, InstanceDescriptor, PipelineLayoutDescriptor, PowerPreference,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
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
pub(crate) struct ComputationalUnit {
    regs: Vec<f64>,
}

impl ComputationalUnit {
    pub fn new(regs: usize) -> Self {
        Self {
            regs: vec![0.0; regs],
        }
    }

    pub unsafe fn execute(&mut self, action: &Action) {
        match action.kind {
            Kind::Approximate => {
                let x = self.regs[action.src as usize];
                let _epsilon = if action.offset > 0 {
                    self.regs[action.offset as usize]
                } else {
                    1e-9  // Default precision
                };
                self.regs[action.dst as usize] = x.sqrt();
            }
            Kind::Choose => {
                let n = self.regs[action.src as usize] as u64;
                if n > 0 {
                    let choice = rand::random::<u64>() % n;
                    self.regs[action.dst as usize] = choice as f64;
                }
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
            offset: 0,
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
}