use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::JITBuilder;
use cranelift_module::Module;
use std::sync::Arc;
use tracing::info;

use crate::ffi::{
    cl_cosf, cl_powf, cl_sinf, cuda, file, ht, lmdb, net, stdio, thread, wgpu as gpu, window,
};

thread_local! {
    pub(crate) static THREAD_COMPILED_FNS: std::cell::RefCell<Option<Arc<Vec<unsafe extern "C" fn(*mut u8)>>>> = const { std::cell::RefCell::new(None) };
}

fn register_symbols(builder: &mut JITBuilder) {
    // Hash table
    builder.symbol("cl_ht_init", ht::cl_ht_init as *const u8);
    builder.symbol("cl_ht_cleanup", ht::cl_ht_cleanup as *const u8);
    builder.symbol("ht_create", ht::cl_ht_create as *const u8);
    builder.symbol("ht_lookup", ht::cl_ht_lookup as *const u8);
    builder.symbol("ht_insert", ht::cl_ht_insert as *const u8);
    builder.symbol("ht_count", ht::cl_ht_count as *const u8);
    builder.symbol("ht_get_entry", ht::cl_ht_get_entry as *const u8);
    builder.symbol("ht_increment", ht::cl_ht_increment as *const u8);

    // wgpu (cross-platform GPU)
    builder.symbol("cl_gpu_init", gpu::cl_gpu_init as *const u8);
    builder.symbol("cl_gpu_create_buffer", gpu::cl_gpu_create_buffer as *const u8);
    builder.symbol("cl_gpu_create_pipeline", gpu::cl_gpu_create_pipeline as *const u8);
    builder.symbol("cl_gpu_upload", gpu::cl_gpu_upload as *const u8);
    builder.symbol("cl_gpu_upload_ptr", gpu::cl_gpu_upload_ptr as *const u8);
    builder.symbol("cl_gpu_dispatch", gpu::cl_gpu_dispatch as *const u8);
    builder.symbol("cl_gpu_download", gpu::cl_gpu_download as *const u8);
    builder.symbol("cl_gpu_download_ptr", gpu::cl_gpu_download_ptr as *const u8);
    builder.symbol("cl_gpu_cleanup", gpu::cl_gpu_cleanup as *const u8);

    // Window / input / present (shares the wgpu device for zero-copy present)
    builder.symbol("cl_window_init", window::cl_window_init as *const u8);
    builder.symbol("cl_window_open", window::cl_window_open as *const u8);
    builder.symbol("cl_window_poll", window::cl_window_poll as *const u8);
    builder.symbol(
        "cl_window_present_gpu_buffer",
        window::cl_window_present_gpu_buffer as *const u8,
    );
    builder.symbol("cl_window_cleanup", window::cl_window_cleanup as *const u8);

    // CUDA core
    builder.symbol("cl_cuda_init", cuda::cl_cuda_init as *const u8);
    builder.symbol("cl_cuda_create_buffer", cuda::cl_cuda_create_buffer as *const u8);
    builder.symbol("cl_cuda_upload", cuda::cl_cuda_upload as *const u8);
    builder.symbol("cl_cuda_upload_ptr", cuda::cl_cuda_upload_ptr as *const u8);
    builder.symbol("cl_cuda_upload_ptr_offset", cuda::cl_cuda_upload_ptr_offset as *const u8);
    builder.symbol("cl_cuda_upload_ptr_async", cuda::cl_cuda_upload_ptr_async as *const u8);
    builder.symbol("cl_cuda_upload_ptr_offset_async", cuda::cl_cuda_upload_ptr_offset_async as *const u8);
    builder.symbol("cl_cuda_download", cuda::cl_cuda_download as *const u8);
    builder.symbol("cl_cuda_download_ptr", cuda::cl_cuda_download_ptr as *const u8);
    builder.symbol("cl_cuda_download_ptr_offset", cuda::cl_cuda_download_ptr_offset as *const u8);
    builder.symbol("cl_cuda_download_ptr_async", cuda::cl_cuda_download_ptr_async as *const u8);
    builder.symbol("cl_cuda_free_buffer", cuda::cl_cuda_free_buffer as *const u8);
    builder.symbol("cl_cuda_stream_create", cuda::cl_cuda_stream_create as *const u8);
    builder.symbol("cl_cuda_stream_sync", cuda::cl_cuda_stream_sync as *const u8);
    builder.symbol("cl_cuda_stream_destroy", cuda::cl_cuda_stream_destroy as *const u8);
    builder.symbol("cl_cuda_event_create", cuda::cl_cuda_event_create as *const u8);
    builder.symbol("cl_cuda_event_record", cuda::cl_cuda_event_record as *const u8);
    builder.symbol("cl_cuda_stream_wait_event", cuda::cl_cuda_stream_wait_event as *const u8);
    builder.symbol("cl_cuda_event_elapsed_ms_bits", cuda::cl_cuda_event_elapsed_ms_bits as *const u8);
    builder.symbol("cl_cuda_event_destroy", cuda::cl_cuda_event_destroy as *const u8);
    builder.symbol("cl_cuda_graph_begin_capture", cuda::cl_cuda_graph_begin_capture as *const u8);
    builder.symbol("cl_cuda_graph_end_capture", cuda::cl_cuda_graph_end_capture as *const u8);
    builder.symbol("cl_cuda_graph_upload", cuda::cl_cuda_graph_upload as *const u8);
    builder.symbol("cl_cuda_graph_launch", cuda::cl_cuda_graph_launch as *const u8);
    builder.symbol("cl_cuda_graph_destroy", cuda::cl_cuda_graph_destroy as *const u8);
    builder.symbol("cl_cuda_pinned_alloc", cuda::cl_cuda_pinned_alloc as *const u8);
    builder.symbol("cl_cuda_pinned_ptr", cuda::cl_cuda_pinned_ptr as *const u8);
    builder.symbol("cl_cuda_pinned_ptr_at", cuda::cl_cuda_pinned_ptr_at as *const u8);
    builder.symbol("cl_cuda_pinned_free", cuda::cl_cuda_pinned_free as *const u8);
    builder.symbol("cl_cuda_mem_info_free", cuda::cl_cuda_mem_info_free as *const u8);
    builder.symbol("cl_cuda_mem_info_total", cuda::cl_cuda_mem_info_total as *const u8);
    builder.symbol("cl_cuda_launch", cuda::cl_cuda_launch as *const u8);
    builder.symbol("cl_cuda_launch_named", cuda::cl_cuda_launch_named as *const u8);
    builder.symbol("cl_cuda_launch_on_stream", cuda::cl_cuda_launch_on_stream as *const u8);
    builder.symbol("cl_cuda_launch_named_on_stream", cuda::cl_cuda_launch_named_on_stream as *const u8);
    builder.symbol("cl_cuda_sync", cuda::cl_cuda_sync as *const u8);
    builder.symbol("cl_cuda_cleanup", cuda::cl_cuda_cleanup as *const u8);

    // cuBLAS
    builder.symbol("cl_cublas_sgemm", cuda::cl_cublas_sgemm as *const u8);
    builder.symbol("cl_cublas_sgemv", cuda::cl_cublas_sgemv as *const u8);
    builder.symbol("cl_cublas_sgemv_on_stream", cuda::cl_cublas_sgemv_on_stream as *const u8);
    builder.symbol("cl_cublas_sgemm_strided_batched", cuda::cl_cublas_sgemm_strided_batched as *const u8);
    builder.symbol("cl_cublas_sgemm_strided_batched_on_stream", cuda::cl_cublas_sgemm_strided_batched_on_stream as *const u8);
    builder.symbol("cl_cublas_ptr_array", cuda::cl_cublas_ptr_array as *const u8);
    builder.symbol("cl_cublas_sgemm_batched_on_stream", cuda::cl_cublas_sgemm_batched_on_stream as *const u8);
    builder.symbol("cl_cublas_gemm_ex_bf16", cuda::cl_cublas_gemm_ex_bf16 as *const u8);
    builder.symbol(
        "cl_cublas_gemm_strided_batched_ex_bf16",
        cuda::cl_cublas_gemm_strided_batched_ex_bf16 as *const u8,
    );

    // File + math + stdio
    builder.symbol("cl_file_read", file::cl_file_read as *const u8);
    builder.symbol("cl_file_read_to_ptr", file::cl_file_read_to_ptr as *const u8);
    builder.symbol("cl_file_write", file::cl_file_write as *const u8);
    builder.symbol("cl_file_write_from_ptr", file::cl_file_write_from_ptr as *const u8);
    builder.symbol("cl_sinf", cl_sinf as *const u8);
    builder.symbol("cl_cosf", cl_cosf as *const u8);
    builder.symbol("cl_powf", cl_powf as *const u8);
    builder.symbol("cl_stdin_readline", stdio::cl_stdin_readline as *const u8);
    builder.symbol("cl_stdout_write", stdio::cl_stdout_write as *const u8);

    // Net
    builder.symbol("cl_net_init", net::cl_net_init as *const u8);
    builder.symbol("cl_net_listen", net::cl_net_listen as *const u8);
    builder.symbol(
        "cl_net_listener_port",
        net::cl_net_listener_port as *const u8,
    );
    builder.symbol("cl_net_connect", net::cl_net_connect as *const u8);
    builder.symbol("cl_net_accept", net::cl_net_accept as *const u8);
    builder.symbol("cl_net_send", net::cl_net_send as *const u8);
    builder.symbol("cl_net_recv", net::cl_net_recv as *const u8);
    builder.symbol("cl_net_cleanup", net::cl_net_cleanup as *const u8);

    // LMDB
    builder.symbol("cl_lmdb_init", lmdb::cl_lmdb_init as *const u8);
    builder.symbol("cl_lmdb_open", lmdb::cl_lmdb_open as *const u8);
    builder.symbol("cl_lmdb_put", lmdb::cl_lmdb_put as *const u8);
    builder.symbol("cl_lmdb_get", lmdb::cl_lmdb_get as *const u8);
    builder.symbol("cl_lmdb_delete", lmdb::cl_lmdb_delete as *const u8);
    builder.symbol("cl_lmdb_begin_write_txn", lmdb::cl_lmdb_begin_write_txn as *const u8);
    builder.symbol("cl_lmdb_commit_write_txn", lmdb::cl_lmdb_commit_write_txn as *const u8);
    builder.symbol("cl_lmdb_cursor_scan", lmdb::cl_lmdb_cursor_scan as *const u8);
    builder.symbol("cl_lmdb_sync", lmdb::cl_lmdb_sync as *const u8);
    builder.symbol("cl_lmdb_cleanup", lmdb::cl_lmdb_cleanup as *const u8);

    // Threads
    builder.symbol("cl_thread_init", thread::cl_thread_init as *const u8);
    builder.symbol("cl_thread_spawn", thread::cl_thread_spawn as *const u8);
    builder.symbol("cl_thread_join", thread::cl_thread_join as *const u8);
    builder.symbol("cl_thread_cleanup", thread::cl_thread_cleanup as *const u8);
    builder.symbol("cl_thread_call", thread::cl_thread_call as *const u8);
}

/// The JIT module every compilation starts from: host ISA, speed, all FFI
/// symbols registered.
fn new_module() -> cranelift_jit::JITModule {
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_native::builder().expect("Host ISA not supported");
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    register_symbols(&mut builder);
    cranelift_jit::JITModule::new(builder)
}

/// Finalizes a module whose functions have all been defined, and hands back
/// pointers to them.
fn finalize(
    mut module: cranelift_jit::JITModule,
    func_ids: Vec<cranelift_module::FuncId>,
) -> Result<
    (
        cranelift_jit::JITModule,
        Arc<Vec<unsafe extern "C" fn(*mut u8)>>,
    ),
    String,
> {
    module.finalize_definitions().map_err(|e| format!("{e}"))?;
    let compiled_fns: Vec<unsafe extern "C" fn(*mut u8)> = func_ids
        .iter()
        .map(|&id| {
            let code_ptr = module.get_finalized_function(id);
            unsafe { std::mem::transmute(code_ptr) }
        })
        .collect();
    info!(count = compiled_fns.len(), "CLIF compiled successfully");
    Ok((module, Arc::new(compiled_fns)))
}

/// Compiles the program an artifact carries.
///
/// Unlike the text path, callees are declared while the function is built, so
/// there is no name to rewrite afterward and no dependence on declaration order
/// happening to match the indices a parser recovered.
pub(crate) fn compile_program(
    prog: &base_types::clif::Program,
) -> Result<
    (
        cranelift_jit::JITModule,
        Arc<Vec<unsafe extern "C" fn(*mut u8)>>,
    ),
    String,
> {
    info!(functions = prog.functions.len(), "compiling CLIF program");
    let mut module = new_module();

    // Declared before any body is built, so `u0:N` resolves to FuncId(N).
    let mut func_ids = Vec::with_capacity(prog.functions.len());
    for (i, f) in prog.functions.iter().enumerate() {
        if f.index as usize != i {
            return Err(format!(
                "function at position {i} declares index u0:{} — they must agree",
                f.index
            ));
        }
        let mut sig = cranelift_codegen::ir::Signature::new(
            cranelift_codegen::isa::CallConv::SystemV,
        );
        sig.params.push(cranelift_codegen::ir::AbiParam::new(
            cranelift_codegen::ir::types::I64,
        ));
        func_ids.push(
            module
                .declare_function(&format!("fn_{i}"), cranelift_module::Linkage::Local, &sig)
                .map_err(|e| format!("declaring fn_{i}: {e}"))?,
        );
    }

    let mut decoded = Vec::with_capacity(prog.functions.len());
    for f in &prog.functions {
        let mut declare = |callee: &base_types::clif::Callee,
                           sig: &cranelift_codegen::ir::Signature| {
            match callee {
                base_types::clif::Callee::Import(name) => module
                    .declare_function(name, cranelift_module::Linkage::Import, sig)
                    .map(|id| id.as_u32())
                    .map_err(|e| format!("declaring import {name}: {e}")),
                base_types::clif::Callee::Local(n) => func_ids
                    .get(*n as usize)
                    .map(|id| id.as_u32())
                    .ok_or_else(|| format!("call to u0:{n}, which the program does not define")),
            }
        };
        decoded.push(crate::clif_decode::decode_function(f, &mut declare)?);
    }

    let dump = std::env::var("BASE_DISASM").is_ok();
    // The decoded function in Cranelift's own textual form. The artifact
    // carries instruction records, not text, so this is the only way to read
    // what was handed to the backend rather than what came out of it.
    let dump_clif = std::env::var("BASE_DUMP_CLIF").is_ok();
    for (i, func) in decoded.into_iter().enumerate() {
        if dump_clif {
            eprintln!("=== clif fn {i} ===\n{}", func.display());
        }
        let mut ctx = cranelift_codegen::Context::for_function(func);
        if dump {
            ctx.set_disasm(true);
        }
        module
            .define_function(func_ids[i], &mut ctx)
            // Debug rather than Display: a verifier rejection names the
            // offending instructions only in the former.
            .map_err(|e| format!("compiling u0:{i}: {e:?}"))?;
        if dump {
            if let Some(vc) = ctx.compiled_code().and_then(|c| c.vcode.as_deref()) {
                eprintln!("=== fn {i} ===\n{vc}");
            }
        }
    }

    finalize(module, func_ids)
}

