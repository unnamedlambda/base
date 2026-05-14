use base_types::Action;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::JITBuilder;
use cranelift_module::Module;
use portable_atomic::Ordering;
use std::sync::Arc;
use tracing::{debug, info, info_span};

use crate::coordination::{spin_backoff, Mailbox, MailboxPoll, SharedMemory};
use crate::ffi::{
    cl_cosf, cl_powf, cl_sinf, cuda, file, ht, lmdb, net, stdio, thread, wgpu as gpu,
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
    builder.symbol("cl_cuda_pinned_free", cuda::cl_cuda_pinned_free as *const u8);
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

pub(crate) fn compile_cranelift_ir(
    clif_source: &str,
) -> Result<
    (
        cranelift_jit::JITModule,
        Arc<Vec<unsafe extern "C" fn(*mut u8)>>,
    ),
    String,
> {
    info!(ir_len = clif_source.len(), "compiling Cranelift IR");

    let mut functions =
        cranelift_reader::parse_functions(clif_source).map_err(|e| format!("{e}"))?;
    if functions.is_empty() {
        return Err("No functions in CLIF IR".into());
    }

    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed").unwrap();
    let isa_builder = cranelift_native::builder().expect("Host ISA not supported");
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .unwrap();
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    register_symbols(&mut builder);

    let mut module = cranelift_jit::JITModule::new(builder);

    // cranelift_reader parses `%name` as ExternalName::TestCase; fix up to ExternalName::User
    for func in functions.iter_mut() {
        let mut fixups = Vec::new();
        for (fref, data) in func.dfg.ext_funcs.iter() {
            if let cranelift_codegen::ir::ExternalName::TestCase(testcase) = &data.name {
                let name = testcase.to_string();
                let name = name.strip_prefix('%').unwrap_or(&name).to_string();
                let sig = func.dfg.signatures[data.signature].clone();
                fixups.push((fref, name, sig));
            }
        }
        for (fref, name, sig) in fixups {
            let fid = module
                .declare_function(&name, cranelift_module::Linkage::Import, &sig)
                .expect("Failed to declare imported function");
            let user_ref =
                func.declare_imported_user_function(cranelift_codegen::ir::UserExternalName {
                    namespace: 0,
                    index: fid.as_u32(),
                });
            func.dfg.ext_funcs[fref].name = cranelift_codegen::ir::ExternalName::user(user_ref);
            func.dfg.ext_funcs[fref].colocated = false;
        }
    }

    let mut func_ids = Vec::with_capacity(functions.len());
    for (i, func) in functions.into_iter().enumerate() {
        let name = format!("fn_{}", i);
        let func_id = module
            .declare_function(&name, cranelift_module::Linkage::Local, &func.signature)
            .expect("Failed to declare function");
        let mut ctx = cranelift_codegen::Context::for_function(func);
        module
            .define_function(func_id, &mut ctx)
            .expect("Failed to compile function");
        func_ids.push(func_id);
    }
    module.finalize_definitions().unwrap();

    let compiled_fns: Vec<unsafe extern "C" fn(*mut u8)> = func_ids
        .iter()
        .map(|&id| {
            let code_ptr = module.get_finalized_function(id);
            unsafe { std::mem::transmute(code_ptr) }
        })
        .collect();

    info!(
        count = compiled_fns.len(),
        "Cranelift IR compiled successfully"
    );
    Ok((module, Arc::new(compiled_fns)))
}

pub(crate) fn cranelift_unit_task_mailbox(
    mailbox: Arc<Mailbox>,
    actions: Arc<Vec<Action>>,
    shared: Arc<SharedMemory>,
    compiled_fns: Arc<Vec<unsafe extern "C" fn(*mut u8)>>,
) {
    let _span = info_span!("cranelift_unit").entered();
    info!("Cranelift unit started");

    THREAD_COMPILED_FNS.with(|cell| {
        *cell.borrow_mut() = Some(compiled_fns.clone());
    });

    let compiled = &*compiled_fns;
    let ptr = shared.ptr;
    let mut spin_count = 0u32;

    loop {
        match mailbox.poll() {
            MailboxPoll::Work { src, count, flag } => {
                debug!(src, count, flag, "cranelift_work_received");
                let start = src as usize;
                let end = start + count as usize;
                for idx in start..end {
                    let desc = &actions[idx];
                    let fn_idx = (desc.src as usize) % compiled.len();
                    unsafe { compiled[fn_idx](ptr.add(desc.dst as usize)) };
                }
                unsafe {
                    shared.store_u64(flag as usize, 1, Ordering::Release);
                }
                debug!(flag, "cranelift_work_complete");
                spin_count = 0;
            }
            MailboxPoll::Closed => {
                info!("Cranelift unit shutting down");
                return;
            }
            MailboxPoll::Empty => spin_backoff(&mut spin_count),
        }
    }
}
