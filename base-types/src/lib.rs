use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod clif;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Setup {
    /// The program the runtime compiles.
    pub clif: clif::Program,
    pub memory_size: usize,
    #[serde(default)]
    pub initial_memory: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Algorithm {
    pub fn_idx: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Artifact {
    pub setup: Setup,
    pub main: Algorithm,
    /// Ordered, so that serializing the same artifact twice gives the same
    /// bytes. A `HashMap` here makes the encoding depend on iteration order,
    /// which leaves two builds of one artifact byte-different.
    #[serde(default)]
    pub extras: BTreeMap<String, Algorithm>,
}

impl Artifact {
    pub fn from_bytes(bytes: &[u8]) -> Artifact {
        bincode::deserialize(bytes).expect("failed to deserialize artifact")
    }
}
