//! base's integration tests, as one binary. The corpora that drive a device
//! are binaries of their own: CUDA's and wgpu's state is the process's.

mod common;

mod c_api;
mod call;
mod hprog_corpus;
mod hprog_cpu_corpus;
mod hprog_diff;
mod hprog_lmdb_corpus;
mod hprog_local_corpus;
mod hprog_native_corpus;
mod hprog_serial_corpus;
mod hprog_thread_corpus;
mod hprog_usb_corpus;
mod instructions;
mod load;
mod programs;
mod totality;
