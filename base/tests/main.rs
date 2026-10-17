//! base's integration tests, as one binary.

mod common;

mod c_api;
mod call;
mod hprog_absent_corpus;
mod hprog_corpus;
mod hprog_cpu_corpus;
mod hprog_cuda_corpus;
mod hprog_diff;
mod hprog_driver_corpus;
mod hprog_gpu_corpus;
mod hprog_lmdb_corpus;
mod hprog_local_corpus;
mod hprog_native_corpus;
mod hprog_serial_corpus;
mod hprog_stream_corpus;
mod hprog_thread_corpus;
mod hprog_usb_corpus;
mod hprog_wgpu_corpus;
mod hprog_window_corpus;
mod instructions;
mod load;
mod programs;
mod totality;
