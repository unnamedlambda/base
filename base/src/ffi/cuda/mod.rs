//! The CUDA FFI, split by concern.  Every entry point stays `pub(crate)` and
//! keeps its name: `jit.rs` registers them by symbol, so this is a move.

mod shared;
mod memory;
mod stream;
mod launch;
mod blas;

pub(crate) use blas::*;
pub(crate) use launch::*;
pub(crate) use memory::*;
pub(crate) use shared::*;
pub(crate) use stream::*;

#[cfg(test)]
mod tests;
