use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    FileRead = 16,
    FileWrite = 17,
    MemScan = 18,
    Approximate = 20,
    Choose = 21,
    Compare = 22,
    Timestamp = 23,
    AtomicCAS = 24,
    ConditionalJump = 25,
    Fence = 26,
    NetConnect = 27,
    NetAccept = 28,
    NetSend = 29,
    NetRecv = 30,
    FFICall = 31,
    AsyncDispatch = 32,
    Wait = 33,
    MemWrite = 34,
    SimdLoadI32 = 37,
    SimdAddI32 = 38,
    SimdMulI32 = 39,
    SimdStoreI32 = 40,
    SimdDivI32 = 41,
    SimdSubI32 = 43,
    MemCopyIndirect = 44,
    MemStoreIndirect = 45,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Action {
    pub kind: Kind,
    pub dst: u32,
    pub src: u32,
    pub offset: u32,
    pub size: u32,
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
    pub file_buffer_size: usize,
    pub gpu_shader_offsets: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UnitSpec {
    pub simd_units: usize,
    pub gpu_units: usize,
    pub computational_units: usize,
    pub file_units: usize,
    pub network_units: usize,
    pub memory_units: usize,
    pub ffi_units: usize,
    pub backends_bits: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Algorithm {
    pub actions: Vec<Action>,
    pub payloads: Vec<u8>,
    pub state: State,
    pub units: UnitSpec,
    pub simd_assignments: Vec<u8>,
    pub computational_assignments: Vec<u8>,
    pub memory_assignments: Vec<u8>,
    pub file_assignments: Vec<u8>,
    pub network_assignments: Vec<u8>,
    pub ffi_assignments: Vec<u8>,
    pub gpu_assignments: Vec<u8>,
    pub worker_threads: Option<usize>,
    pub blocking_threads: Option<usize>,
    pub stack_size: Option<usize>,
    pub timeout_ms: Option<u64>,
    pub thread_name_prefix: Option<String>,
}
