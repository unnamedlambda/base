use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
pub struct IoOffsets {
    pub data_ptr: usize,
    pub data_len: usize,
    pub out_ptr: usize,
    pub out_len: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Setup {
    pub cranelift_ir: String,
    pub memory_size: usize,
    pub io_offsets: IoOffsets,
    #[serde(default)]
    pub initial_memory: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Algorithm {
    pub fn_idx: u32,
    pub output: Vec<OutputBatchSchema>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Artifact {
    pub setup: Setup,
    pub main: Algorithm,
    #[serde(default)]
    pub extras: HashMap<String, Algorithm>,
}

impl Artifact {
    pub fn from_bytes(bytes: &[u8]) -> Artifact {
        bincode::deserialize(bytes).expect("failed to deserialize artifact")
    }
}
