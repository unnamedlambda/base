//! What a program may call outside itself, and at what signature.
//!
//! The table is the whole of it: a program naming anything else is refused
//! when it is built, rather than resolved against whatever the process happens
//! to have loaded. Each signature is read off the Rust function's own type, so
//! it is not a second statement that could disagree with the function — the
//! `as` cast in each entry is checked by the compiler.

use std::sync::OnceLock;

use base_types::clif::ClifTy;

use crate::ffi::{
    cl_cosf, cl_powf, cl_sinf, cuda, file, ht, lmdb, native, net, stdio, thread, wgpu as gpu,
    window,
};

/// A host function a program may import.
pub(crate) struct Import {
    pub(crate) name: &'static str,
    /// The function's address. Code, never written, so sharing it between
    /// threads is sound.
    pub(crate) addr: usize,
    pub(crate) params: Vec<ClifTy>,
    pub(crate) result: Option<ClifTy>,
}

/// How a Rust parameter or result type crosses into CLIF.
///
/// Every pointer is an `i64`: the platform is 64-bit, and a program holds an
/// address as a plain integer.
trait Abi {
    const TY: ClifTy;
}

impl Abi for i32 {
    const TY: ClifTy = ClifTy::I32;
}
impl Abi for u32 {
    const TY: ClifTy = ClifTy::I32;
}
impl Abi for i64 {
    const TY: ClifTy = ClifTy::I64;
}
impl Abi for u64 {
    const TY: ClifTy = ClifTy::I64;
}
impl Abi for usize {
    const TY: ClifTy = ClifTy::I64;
}
impl Abi for f32 {
    const TY: ClifTy = ClifTy::F32;
}
impl Abi for f64 {
    const TY: ClifTy = ClifTy::F64;
}
impl<T> Abi for *mut T {
    const TY: ClifTy = ClifTy::I64;
}
impl<T> Abi for *const T {
    const TY: ClifTy = ClifTy::I64;
}

/// What a function answers: nothing, or one value.
trait Answer {
    const TY: Option<ClifTy>;
}

impl Answer for () {
    const TY: Option<ClifTy> = None;
}
impl<T: Abi> Answer for T {
    const TY: Option<ClifTy> = Some(T::TY);
}

/// A C function pointer, and the signature its type spells.
trait HostFn: Copy {
    fn addr(self) -> usize;
    fn params() -> Vec<ClifTy>;
    fn result() -> Option<ClifTy>;
}

macro_rules! host_fn {
    ($($a:ident)*) => {
        impl<R: Answer, $($a: Abi),*> HostFn for unsafe extern "C" fn($($a),*) -> R {
            fn addr(self) -> usize {
                self as usize
            }
            fn params() -> Vec<ClifTy> {
                vec![$($a::TY),*]
            }
            fn result() -> Option<ClifTy> {
                R::TY
            }
        }
    };
}

host_fn!();
host_fn!(A);
host_fn!(A B);
host_fn!(A B C);
host_fn!(A B C D);
host_fn!(A B C D E);
host_fn!(A B C D E F);
host_fn!(A B C D E F G);
host_fn!(A B C D E F G H);
host_fn!(A B C D E F G H I);
host_fn!(A B C D E F G H I J);
host_fn!(A B C D E F G H I J K);
host_fn!(A B C D E F G H I J K L);
host_fn!(A B C D E F G H I J K L M);
host_fn!(A B C D E F G H I J K L M N);
host_fn!(A B C D E F G H I J K L M N O);
host_fn!(A B C D E F G H I J K L M N O P);
host_fn!(A B C D E F G H I J K L M N O P Q);
host_fn!(A B C D E F G H I J K L M N O P Q S);
host_fn!(A B C D E F G H I J K L M N O P Q S T);
host_fn!(A B C D E F G H I J K L M N O P Q S T U);
host_fn!(A B C D E F G H I J K L M N O P Q S T U V);
host_fn!(A B C D E F G H I J K L M N O P Q S T U V W);
host_fn!(A B C D E F G H I J K L M N O P Q S T U V W X);

fn entry<F: HostFn>(name: &'static str, f: F) -> Import {
    Import { name, addr: f.addr(), params: F::params(), result: F::result() }
}

/// Every function a program may import.
pub(crate) fn imports() -> &'static [Import] {
    static TABLE: OnceLock<Vec<Import>> = OnceLock::new();
    TABLE.get_or_init(|| {
        vec![
        // Hash table
        entry("cl_ht_init", ht::cl_ht_init as unsafe extern "C" fn(*mut *mut _)),
        entry("cl_ht_cleanup", ht::cl_ht_cleanup as unsafe extern "C" fn(*mut *mut _)),
        entry("ht_create", ht::cl_ht_create as unsafe extern "C" fn(*mut _) -> u32),
        entry("ht_lookup", ht::cl_ht_lookup as unsafe extern "C" fn(*const _, *const u8, u32, *mut u8) -> u32),
        entry("ht_insert", ht::cl_ht_insert as unsafe extern "C" fn(*mut _, *const u8, u32, *const u8, u32)),
        entry("ht_count", ht::cl_ht_count as unsafe extern "C" fn(*const _) -> u32),
        entry("ht_get_entry", ht::cl_ht_get_entry as unsafe extern "C" fn(*const _, u32, *mut u8, *mut u8) -> i32),
        entry("ht_increment", ht::cl_ht_increment as unsafe extern "C" fn(*mut _, *const u8, u32, i64) -> i64),

        // wgpu (cross-platform GPU)
        entry("cl_gpu_init", gpu::cl_gpu_init as unsafe extern "C" fn(*mut *mut _)),
        entry("cl_gpu_create_buffer", gpu::cl_gpu_create_buffer as unsafe extern "C" fn(*mut _, i64) -> i32),
        entry("cl_gpu_create_pipeline", gpu::cl_gpu_create_pipeline as unsafe extern "C" fn(*mut _, *const u8, *const u8, i32) -> i32),
        entry("cl_gpu_upload", gpu::cl_gpu_upload as unsafe extern "C" fn(*const _, i32, *const u8, i64) -> i32),
        entry("cl_gpu_upload_ptr", gpu::cl_gpu_upload_ptr as unsafe extern "C" fn(*const _, i32, *const u8, i64) -> i32),
        entry("cl_gpu_dispatch", gpu::cl_gpu_dispatch as unsafe extern "C" fn(*mut _, i32, i32, i32, i32) -> i32),
        entry("cl_gpu_download", gpu::cl_gpu_download as unsafe extern "C" fn(*mut _, i32, *mut u8, i64) -> i32),
        entry("cl_gpu_download_ptr", gpu::cl_gpu_download_ptr as unsafe extern "C" fn(*mut _, i32, i64, *mut u8, i64) -> i32),
        entry("cl_gpu_cleanup", gpu::cl_gpu_cleanup as unsafe extern "C" fn(*mut *mut _)),

        // Window / input / present (shares the wgpu device for zero-copy present)
        entry("cl_window_init", window::cl_window_init as unsafe extern "C" fn(*mut *mut _)),
        entry("cl_window_open", window::cl_window_open as unsafe extern "C" fn(*mut _, i64, i64, *const u8, i64, *const u8, i64) -> i32),
        entry("cl_window_poll", window::cl_window_poll as unsafe extern "C" fn(*mut _, *mut u8, i32) -> i32),
        entry("cl_window_present_gpu_buffer", window::cl_window_present_gpu_buffer as unsafe extern "C" fn(*mut _, *mut _, i32) -> i32),
        entry("cl_window_cleanup", window::cl_window_cleanup as unsafe extern "C" fn(*mut *mut _)),

        // CUDA core
        entry("cl_cuda_init", cuda::cl_cuda_init as unsafe extern "C" fn(*mut *mut _)),
        entry("cl_cuda_create_buffer", cuda::cl_cuda_create_buffer as unsafe extern "C" fn(*mut _, i64) -> i32),
        entry("cl_cuda_upload", cuda::cl_cuda_upload as unsafe extern "C" fn(*mut _, i32, *const u8, i64) -> i32),
        entry("cl_cuda_upload_ptr", cuda::cl_cuda_upload_ptr as unsafe extern "C" fn(*mut _, i32, *const u8, i64) -> i32),
        entry("cl_cuda_upload_ptr_offset", cuda::cl_cuda_upload_ptr_offset as unsafe extern "C" fn(*mut _, i32, i64, *const u8, i64) -> i32),
        entry("cl_cuda_upload_ptr_async", cuda::cl_cuda_upload_ptr_async as unsafe extern "C" fn(*mut _, i32, *const u8, i64, i32) -> i32),
        entry("cl_cuda_upload_ptr_offset_async", cuda::cl_cuda_upload_ptr_offset_async as unsafe extern "C" fn(*mut _, i32, i64, *const u8, i64, i32) -> i32),
        entry("cl_cuda_download", cuda::cl_cuda_download as unsafe extern "C" fn(*mut _, i32, *mut u8, i64) -> i32),
        entry("cl_cuda_download_ptr", cuda::cl_cuda_download_ptr as unsafe extern "C" fn(*mut _, i32, *mut u8, i64) -> i32),
        entry("cl_cuda_download_ptr_offset", cuda::cl_cuda_download_ptr_offset as unsafe extern "C" fn(*mut _, i32, i64, *mut u8, i64) -> i32),
        entry("cl_cuda_download_ptr_async", cuda::cl_cuda_download_ptr_async as unsafe extern "C" fn(*mut _, i32, *mut u8, i64, i32) -> i32),
        entry("cl_cuda_free_buffer", cuda::cl_cuda_free_buffer as unsafe extern "C" fn(*mut _, i32) -> i32),
        entry("cl_cuda_stream_create", cuda::cl_cuda_stream_create as unsafe extern "C" fn(*mut _) -> i32),
        entry("cl_cuda_stream_sync", cuda::cl_cuda_stream_sync as unsafe extern "C" fn(*mut _, i32) -> i32),
        entry("cl_cuda_stream_destroy", cuda::cl_cuda_stream_destroy as unsafe extern "C" fn(*mut _, i32) -> i32),
        entry("cl_cuda_event_create", cuda::cl_cuda_event_create as unsafe extern "C" fn(*mut _) -> i32),
        entry("cl_cuda_event_record", cuda::cl_cuda_event_record as unsafe extern "C" fn(*mut _, i32, i32) -> i32),
        entry("cl_cuda_stream_wait_event", cuda::cl_cuda_stream_wait_event as unsafe extern "C" fn(*mut _, i32, i32) -> i32),
        entry("cl_cuda_event_elapsed_ms_bits", cuda::cl_cuda_event_elapsed_ms_bits as unsafe extern "C" fn(*mut _, i32, i32) -> i32),
        entry("cl_cuda_event_destroy", cuda::cl_cuda_event_destroy as unsafe extern "C" fn(*mut _, i32) -> i32),
        entry("cl_cuda_graph_begin_capture", cuda::cl_cuda_graph_begin_capture as unsafe extern "C" fn(*mut _, i32) -> i32),
        entry("cl_cuda_graph_end_capture", cuda::cl_cuda_graph_end_capture as unsafe extern "C" fn(*mut _, i32) -> i32),
        entry("cl_cuda_graph_upload", cuda::cl_cuda_graph_upload as unsafe extern "C" fn(*mut _, i32, i32) -> i32),
        entry("cl_cuda_graph_launch", cuda::cl_cuda_graph_launch as unsafe extern "C" fn(*mut _, i32, i32) -> i32),
        entry("cl_cuda_graph_destroy", cuda::cl_cuda_graph_destroy as unsafe extern "C" fn(*mut _, i32) -> i32),
        entry("cl_cuda_pinned_alloc", cuda::cl_cuda_pinned_alloc as unsafe extern "C" fn(*mut _, i64) -> i32),
        entry("cl_cuda_pinned_ptr", cuda::cl_cuda_pinned_ptr as unsafe extern "C" fn(*mut _, i32) -> i64),
        entry("cl_cuda_pinned_ptr_at", cuda::cl_cuda_pinned_ptr_at as unsafe extern "C" fn(*mut _, i32, i64, i64) -> i64),
        entry("cl_cuda_pinned_free", cuda::cl_cuda_pinned_free as unsafe extern "C" fn(*mut _, i32) -> i32),
        entry("cl_cuda_mem_info_free", cuda::cl_cuda_mem_info_free as unsafe extern "C" fn(*mut _) -> i64),
        entry("cl_cuda_mem_info_total", cuda::cl_cuda_mem_info_total as unsafe extern "C" fn(*mut _) -> i64),
        entry("cl_cuda_launch", cuda::cl_cuda_launch as unsafe extern "C" fn(*mut _, *const u8, i32, *const u8, i32, i32, i32, i32, i32, i32) -> i32),
        entry("cl_cuda_launch_named", cuda::cl_cuda_launch_named as unsafe extern "C" fn(*mut _, *const u8, *const u8, i32, *const u8, i32, i32, i32, i32, i32, i32) -> i32),
        entry("cl_cuda_launch_on_stream", cuda::cl_cuda_launch_on_stream as unsafe extern "C" fn(*mut _, *const u8, i32, *const u8, i32, i32, i32, i32, i32, i32, i32) -> i32),
        entry("cl_cuda_launch_named_on_stream", cuda::cl_cuda_launch_named_on_stream as unsafe extern "C" fn(*mut _, *const u8, *const u8, i32, *const u8, i32, i32, i32, i32, i32, i32, i32) -> i32),
        entry("cl_cuda_sync", cuda::cl_cuda_sync as unsafe extern "C" fn(*const _) -> i32),
        entry("cl_cuda_cleanup", cuda::cl_cuda_cleanup as unsafe extern "C" fn(*mut *mut _)),

        // cuBLAS
        entry("cl_cublas_sgemm", cuda::cl_cublas_sgemm as unsafe extern "C" fn(*mut _, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32),
        entry("cl_cublas_sgemv", cuda::cl_cublas_sgemv as unsafe extern "C" fn(*mut _, i32, i32, i32, i32, i32, i32, i32, i32) -> i32),
        entry("cl_cublas_sgemv_on_stream", cuda::cl_cublas_sgemv_on_stream as unsafe extern "C" fn(*mut _, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32),
        entry("cl_cublas_sgemm_strided_batched", cuda::cl_cublas_sgemm_strided_batched as unsafe extern "C" fn(*mut _, i32, i32, i32, i32, i32, i32, i32, i64, i32, i64, i32, i32, i64, i32, i64, i64, i64, i32, i32, i32) -> i32),
        entry("cl_cublas_sgemm_strided_batched_on_stream", cuda::cl_cublas_sgemm_strided_batched_on_stream as unsafe extern "C" fn(*mut _, i32, i32, i32, i32, i32, i32, i32, i64, i32, i64, i32, i32, i64, i32, i32, i64, i64, i64, i32, i32, i32) -> i32),
        entry("cl_cublas_ptr_array", cuda::cl_cublas_ptr_array as unsafe extern "C" fn(*mut _, i32, i32, i32, i64) -> i32),
        entry("cl_cublas_sgemm_batched_on_stream", cuda::cl_cublas_sgemm_batched_on_stream as unsafe extern "C" fn(*mut _, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32),
        entry("cl_cublas_gemm_ex_bf16", cuda::cl_cublas_gemm_ex_bf16 as unsafe extern "C" fn(*mut _, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i32, i32, i32) -> i32),
        entry("cl_cublas_gemm_strided_batched_ex_bf16", cuda::cl_cublas_gemm_strided_batched_ex_bf16 as unsafe extern "C" fn(*mut _, i32, i32, i32, i32, i32, i32, i32, i64, i32, i64, i32, i32, i64, i32, i64, i64, i64, i32, i32, i32) -> i32),

        // Native code the program carries: place it, unmap it, and the two
        // questions that decide which code to carry in (see ffi/native.rs).
        // Calling it is not an import: it is `Callee::Native`.
        entry("cl_native_load", native::cl_native_load as unsafe extern "C" fn(*const u8, i64) -> i64),
        entry("cl_native_free", native::cl_native_free as unsafe extern "C" fn(i64) -> i32),
        entry("cl_native_arch", native::cl_native_arch as unsafe extern "C" fn() -> i32),
        entry("cl_cpu_has", native::cl_cpu_has as unsafe extern "C" fn(*const u8) -> i32),
        // File + math + stdio
        entry("cl_file_read", file::cl_file_read as unsafe extern "C" fn(*mut u8, i64, i64, i64, i64) -> i64),
        entry("cl_file_read_to_ptr", file::cl_file_read_to_ptr as unsafe extern "C" fn(*const u8, *mut u8, i64, i64) -> i64),
        entry("cl_file_write", file::cl_file_write as unsafe extern "C" fn(*mut u8, i64, i64, i64, i64) -> i64),
        entry("cl_file_write_from_ptr", file::cl_file_write_from_ptr as unsafe extern "C" fn(*const u8, *const u8, i64, i64) -> i64),
        entry("cl_sinf", cl_sinf as unsafe extern "C" fn(f32) -> f32),
        entry("cl_cosf", cl_cosf as unsafe extern "C" fn(f32) -> f32),
        entry("cl_powf", cl_powf as unsafe extern "C" fn(f32, f32) -> f32),
        entry("cl_stdin_readline", stdio::cl_stdin_readline as unsafe extern "C" fn(*mut u8, i64, i64) -> i64),
        entry("cl_stdout_write", stdio::cl_stdout_write as unsafe extern "C" fn(*mut u8, i64, i64) -> i64),

        // Net
        entry("cl_net_init", net::cl_net_init as unsafe extern "C" fn(*mut *mut _)),
        entry("cl_net_listen", net::cl_net_listen as unsafe extern "C" fn(*mut _, *const u8) -> i64),
        entry("cl_net_listener_port", net::cl_net_listener_port as unsafe extern "C" fn(*const _, i64) -> i64),
        entry("cl_net_connect", net::cl_net_connect as unsafe extern "C" fn(*mut _, *const u8) -> i64),
        entry("cl_net_accept", net::cl_net_accept as unsafe extern "C" fn(*mut _, i64) -> i64),
        entry("cl_net_send", net::cl_net_send as unsafe extern "C" fn(*mut _, i64, *const u8, i64) -> i64),
        entry("cl_net_recv", net::cl_net_recv as unsafe extern "C" fn(*mut _, i64, *mut u8, i64) -> i64),
        entry("cl_net_cleanup", net::cl_net_cleanup as unsafe extern "C" fn(*mut *mut _)),

        // LMDB
        entry("cl_lmdb_init", lmdb::cl_lmdb_init as unsafe extern "C" fn(*mut *mut _)),
        entry("cl_lmdb_open", lmdb::cl_lmdb_open as unsafe extern "C" fn(*mut _, *const u8, i32) -> i32),
        entry("cl_lmdb_put", lmdb::cl_lmdb_put as unsafe extern "C" fn(*mut _, u32, *const u8, i32, *const u8, i32) -> i32),
        entry("cl_lmdb_get", lmdb::cl_lmdb_get as unsafe extern "C" fn(*mut _, u32, *const u8, i32, *mut u8) -> i32),
        entry("cl_lmdb_delete", lmdb::cl_lmdb_delete as unsafe extern "C" fn(*mut _, u32, *const u8, i32) -> i32),
        entry("cl_lmdb_begin_write_txn", lmdb::cl_lmdb_begin_write_txn as unsafe extern "C" fn(*mut _, u32) -> i32),
        entry("cl_lmdb_commit_write_txn", lmdb::cl_lmdb_commit_write_txn as unsafe extern "C" fn(*mut _, u32) -> i32),
        entry("cl_lmdb_cursor_scan", lmdb::cl_lmdb_cursor_scan as unsafe extern "C" fn(*mut _, u32, *const u8, i32, i32, *mut u8) -> i32),
        entry("cl_lmdb_sync", lmdb::cl_lmdb_sync as unsafe extern "C" fn(*const _, u32) -> i32),
        entry("cl_lmdb_cleanup", lmdb::cl_lmdb_cleanup as unsafe extern "C" fn(*mut *mut _)),

        // Threads
        entry("cl_thread_init", thread::cl_thread_init as unsafe extern "C" fn(*mut *mut _)),
        entry("cl_thread_spawn", thread::cl_thread_spawn as unsafe extern "C" fn(*mut _, i64, *mut u8) -> i64),
        entry("cl_thread_join", thread::cl_thread_join as unsafe extern "C" fn(*mut _, i64) -> i64),
        entry("cl_thread_cleanup", thread::cl_thread_cleanup as unsafe extern "C" fn(*mut *mut _)),
        entry("cl_thread_call", thread::cl_thread_call as unsafe extern "C" fn(*const _, i64, *mut u8) -> i64),
        ]
    })
}

/// The import a program names, if base provides one by that name.
pub(crate) fn lookup(name: &str) -> Option<&'static Import> {
    imports().iter().find(|i| i.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base_types::clif::{Block, BlockRef, Callee, Function, Inst, Val};

    #[test]
    fn names_are_unique() {
        let mut names: Vec<_> = imports().iter().map(|i| i.name).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len());
    }

    /// Every import, named by a program and with its address taken, links: the
    /// table names only functions that exist, at the signature the table itself
    /// hands the JIT.
    #[test]
    fn every_import_links_at_its_own_signature() {
        let mut insts = Vec::new();
        for (n, import) in imports().iter().enumerate() {
            insts.push(Inst::FuncAddr(
                Val(100 + n as u32),
                Callee::Import(import.name.to_string()),
            ));
        }
        insts.push(Inst::Ret(None));
        let f = Function {
            entry_name: None,
            blocks: vec![Block { reference: BlockRef(0), params: vec![(Val(0), ClifTy::I64)], insts }],
        };
        if let Err(e) = crate::jit::compile(&[f]) {
            panic!("{e}");
        }
    }
}
