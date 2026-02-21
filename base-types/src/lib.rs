use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    FileRead = 16,
    FileWrite = 17,
    ConditionalJump = 25,
    AsyncDispatch = 32,
    Wait = 33,
    WaitUntil = 76,
    Park = 77,
    Wake = 78,
    KernelStart = 80,
    KernelSubmit = 81,
    KernelWait = 82,
    KernelStop = 83,
    KernelSubmitIndirect = 84,
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
    pub file_buffer_size: usize,
    pub cranelift_ir_offsets: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UnitSpec {
    pub file_units: usize,
    pub cranelift_units: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Algorithm {
    pub actions: Vec<Action>,
    pub payloads: Vec<u8>,
    pub state: State,
    pub units: UnitSpec,
    pub file_assignments: Vec<u8>,
    pub cranelift_assignments: Vec<u8>,
    pub worker_threads: Option<usize>,
    pub blocking_threads: Option<usize>,
    pub stack_size: Option<usize>,
    pub timeout_ms: Option<u64>,
    pub thread_name_prefix: Option<String>,
}
