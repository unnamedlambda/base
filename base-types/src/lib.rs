use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Describe = 0,
    ClifCall = 24,
    ConditionalJump = 25,
    ClifCallAsync = 32,
    Wait = 33,
    WaitUntil = 76,
    Park = 77,
    Wake = 78,
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
    pub cranelift_ir_offsets: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UnitSpec {
    pub cranelift_units: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OutputType {
    I64,
    F64,
    Utf8,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OutputColumn {
    pub name: String,
    pub dtype: OutputType,
    pub data_offset: usize,
    pub len_offset: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OutputBatchSchema {
    pub columns: Vec<OutputColumn>,
    pub row_count_offset: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Algorithm {
    pub actions: Vec<Action>,
    pub payloads: Vec<u8>,
    pub state: State,
    pub units: UnitSpec,
    pub cranelift_assignments: Vec<u8>,
    pub worker_threads: Option<usize>,
    pub blocking_threads: Option<usize>,
    pub stack_size: Option<usize>,
    pub timeout_ms: Option<u64>,
    pub thread_name_prefix: Option<String>,
    pub additional_shared_memory: usize,
    #[serde(default)]
    pub output: Vec<OutputBatchSchema>,
}
