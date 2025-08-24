use serde::{Deserialize, Serialize};
use wgpu::Backends;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    CreateBuffer = 0,
    WriteBuffer = 1,
    CreateShader = 2,
    CreatePipeline = 3,
    Dispatch = 4,
    ReadBuffer = 5,
    SimdLoad = 10,
    SimdAdd = 11,
    SimdMul = 12,
    SimdStore = 13,
    ConditionalWrite = 14,
    MemCopy = 15,
    Approximate = 20,
    Choose = 21,
    Compare = 22,
    Timestamp = 23,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Action {
    pub kind: Kind,
    pub dst: u32,
    pub src: u32,
    pub offset: u32,
    pub size: u32,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct Algorithm {
    pub actions: Vec<Action>,
    pub payloads: Vec<u8>,
    pub state: State,
    pub queues: QueueSpec,
    pub units: UnitSpec,
    pub simd_assignments: Vec<u8>,
    pub computational_assignments: Vec<u8>,
    pub write_assignments: Vec<u8>,
    pub worker_threads: Option<usize>,
    pub blocking_threads: Option<usize>,
    pub stack_size: Option<usize>,
    pub timeout_ms: Option<u64>,
    pub thread_name_prefix: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct State {
    pub regs_per_unit: usize,
    pub unit_scratch_offsets: Vec<usize>,
    pub unit_scratch_size: usize,
    pub shared_data_offset: usize,
    pub shared_data_size: usize,
    pub gpu_offset: usize,
    pub gpu_size: usize,
    pub computational_regs: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueueSpec {
    pub capacity: usize,
    pub batch_size: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UnitSpec {
    pub simd_units: usize,
    pub gpu_enabled: bool,
    pub computational_enabled: bool,
    pub backends_bits: u32,
    pub features_bits: u64,
}

impl Default for Algorithm {
    fn default() -> Self {
        Algorithm {
            actions: Vec::new(),
            payloads: vec![0u8; 65536],
            state: State {
                regs_per_unit: 16,
                unit_scratch_offsets: vec![0, 4096, 8192, 12288],
                unit_scratch_size: 4096,
                shared_data_offset: 16384,
                shared_data_size: 16384,
                gpu_offset: 32768,
                gpu_size: 32768,
                computational_regs: 32,
            },
            queues: QueueSpec {
                capacity: 256,
                batch_size: 16,
            },
            units: UnitSpec {
                simd_units: 4,
                gpu_enabled: true,
                computational_enabled: true,
                backends_bits: Backends::all().bits(),
                features_bits: 0,
            },
            simd_assignments: Vec::new(),
            computational_assignments: Vec::new(),
            write_assignments: Vec::new(),
            worker_threads: None,
            blocking_threads: None,
            stack_size: None,
            timeout_ms: None,
            thread_name_prefix: None,
        }
    }
}
