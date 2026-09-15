use serde::{Deserialize, Serialize};

pub mod clif;

/// The board's control, as data: the functions a host compiles, the memory
/// they run in, and the bytes that memory starts with.
///
/// This is the whole of the wire format. An entry point is a function index —
/// which function does what is the generator's knowledge, and stays with
/// whoever built the artifact.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Artifact {
    /// Compiled as a unit. A function's `u0:N` index is its position here.
    pub functions: Vec<clif::Function>,
    pub memory_size: usize,
    #[serde(default)]
    pub initial_memory: Vec<u8>,
}

impl Artifact {
    pub fn from_bytes(bytes: &[u8]) -> Artifact {
        bincode::deserialize(bytes).expect("failed to deserialize artifact")
    }
}
