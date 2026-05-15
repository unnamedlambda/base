use serde::{Deserialize, Serialize};

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

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuntimeHeader {
    pub data_ptr_offset: usize,
    pub data_len_offset: usize,
    pub out_ptr_offset: usize,
    pub out_len_offset: usize,
}

impl Default for RuntimeHeader {
    fn default() -> Self {
        Self {
            data_ptr_offset: 0x18,
            data_len_offset: 0x20,
            out_ptr_offset: 0x28,
            out_len_offset: 0x30,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BaseConfig {
    pub cranelift_ir: String,
    pub memory_size: usize,
    pub runtime_header: RuntimeHeader,
    #[serde(default)]
    pub context_offset: usize,
    #[serde(default)]
    pub initial_memory: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Algorithm {
    pub fn_idx: u32,
    pub output: Vec<OutputBatchSchema>,
}
