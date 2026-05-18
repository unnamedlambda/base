use arrow_array::{Float64Array, Int64Array, StringArray};
use arrow_schema::{DataType, Field, Schema};
use base::{run, Base, RecordBatch};
use base_types::{
    Algorithm, Setup, OutputBatchSchema, OutputColumn, OutputType, RuntimeHeader,
};
use std::fs;
use std::sync::Arc;
use tempfile::TempDir;

fn legacy_runtime_header() -> RuntimeHeader {
    RuntimeHeader {
        data_ptr_offset: 8,
        data_len_offset: 16,
        out_ptr_offset: 24,
        out_len_offset: 32,
    }
}

fn cranelift_config(memory: Vec<u8>, cranelift_ir: String) -> Setup {
    Setup {
        cranelift_ir,
        memory_size: memory.len(),
        runtime_header: legacy_runtime_header(),
        initial_memory: memory,
    }
}

fn cranelift_algorithm(fn_idx: u32) -> Algorithm {
    Algorithm {
        fn_idx,
        output: vec![],
    }
}

fn create_cranelift_algorithm(
    fn_idx: u32,
    memory: Vec<u8>,
    cranelift_ir: String,
) -> (Setup, Algorithm) {
    (cranelift_config(memory, cranelift_ir), cranelift_algorithm(fn_idx))
}

#[test]
fn test_cranelift_basic_compilation() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_basic.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    // Single CLIF function that writes 8 bytes at offset 2000 to the file at offset 3000.
    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2000
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&42u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 42);
}

#[test]
fn test_cranelift_arithmetic_add() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_add.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    // Add operands at 2000/2008, store at 2016, write 2016 to file.
    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = load.i64 v0+2000
    v2 = load.i64 v0+2008
    v3 = iadd v1, v2
    store.i64 v3, v0+2016
    v4 = iconst.i64 3000
    v5 = iconst.i64 2016
    v6 = iconst.i64 0
    v7 = iconst.i64 8
    v8 = call fn0(v0, v4, v5, v6, v7)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&100u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&200u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 300);
}

#[test]
fn test_cranelift_arithmetic_multiply() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_mul.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = load.i64 v0+2000
    v2 = load.i64 v0+2008
    v3 = imul v1, v2
    store.i64 v3, v0+2016
    v4 = iconst.i64 3000
    v5 = iconst.i64 2016
    v6 = iconst.i64 0
    v7 = iconst.i64 8
    v8 = call fn0(v0, v4, v5, v6, v7)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&7u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&9u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 63);
}

#[test]
fn test_cranelift_memory_operations() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_mem.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = load.i32 v0+2000
    v2 = load.i32 v0+2004
    v3 = load.i32 v0+2008
    v4 = iadd v1, v2
    v5 = iadd v4, v3
    store.i32 v5, v0+2012
    v6 = iconst.i64 3000
    v7 = iconst.i64 2012
    v8 = iconst.i64 0
    v9 = iconst.i64 4
    v10 = call fn0(v0, v6, v7, v8, v9)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2004].copy_from_slice(&10u32.to_le_bytes());
    memory[2004..2008].copy_from_slice(&20u32.to_le_bytes());
    memory[2008..2012].copy_from_slice(&30u32.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(result, 60);
}

#[test]
fn test_cranelift_conditional_logic() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_cond.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = load.i64 v0+2000
    v2 = load.i64 v0+2008
    v3 = load.i64 v0+2016
    v4 = icmp_imm eq v1, 0
    brif v4, block2, block1

block1:
    store.i64 v2, v0+2024
    jump block3

block2:
    store.i64 v3, v0+2024
    jump block3

block3:
    v5 = iconst.i64 3000
    v6 = iconst.i64 2024
    v7 = iconst.i64 0
    v8 = iconst.i64 8
    v9 = call fn0(v0, v5, v6, v7, v8)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    // condition=1, value_a=100, value_b=200 → should store value_a
    memory[2000..2008].copy_from_slice(&1u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&100u64.to_le_bytes());
    memory[2016..2024].copy_from_slice(&200u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 100);
}

#[test]
fn test_clif_ffi_all_symbols_linkable() {
    // Authoritative check that every FFI symbol registered in jit.rs is
    // resolvable from CLIF. Generates a function that takes each symbol's
    // address (via func_addr) and stores it; if any symbol were missing,
    // Base::new would fail to link the module.
    //
    // Per-FFI smoke tests below exercise the runtime call path; this one
    // exists so that adding a new FFI symbol without wiring it into jit.rs
    // is caught by a dedicated, fast-failing test.
    let symbols: &[&str] = &[
        "cl_ht_init", "cl_ht_cleanup", "ht_create", "ht_lookup", "ht_insert",
        "ht_count", "ht_get_entry", "ht_increment",
        "cl_gpu_init", "cl_gpu_create_buffer", "cl_gpu_create_pipeline",
        "cl_gpu_upload", "cl_gpu_upload_ptr", "cl_gpu_dispatch", "cl_gpu_download",
        "cl_gpu_download_ptr", "cl_gpu_cleanup",
        "cl_cuda_init", "cl_cuda_create_buffer", "cl_cuda_upload",
        "cl_cuda_upload_ptr", "cl_cuda_upload_ptr_offset", "cl_cuda_upload_ptr_async",
        "cl_cuda_upload_ptr_offset_async", "cl_cuda_download", "cl_cuda_download_ptr",
        "cl_cuda_download_ptr_offset", "cl_cuda_download_ptr_async", "cl_cuda_free_buffer",
        "cl_cuda_stream_create", "cl_cuda_stream_sync", "cl_cuda_stream_destroy",
        "cl_cuda_event_create", "cl_cuda_event_record", "cl_cuda_stream_wait_event",
        "cl_cuda_event_elapsed_ms_bits", "cl_cuda_event_destroy",
        "cl_cuda_graph_begin_capture", "cl_cuda_graph_end_capture",
        "cl_cuda_graph_upload", "cl_cuda_graph_launch", "cl_cuda_graph_destroy",
        "cl_cuda_pinned_alloc", "cl_cuda_pinned_ptr", "cl_cuda_pinned_free",
        "cl_cuda_launch", "cl_cuda_launch_named", "cl_cuda_launch_on_stream",
        "cl_cuda_launch_named_on_stream", "cl_cuda_sync", "cl_cuda_cleanup",
        "cl_cublas_sgemm", "cl_cublas_sgemv", "cl_cublas_sgemv_on_stream",
        "cl_cublas_sgemm_strided_batched", "cl_cublas_sgemm_strided_batched_on_stream",
        "cl_file_read", "cl_file_read_to_ptr", "cl_file_write", "cl_file_write_from_ptr",
        "cl_sinf", "cl_cosf", "cl_powf",
        "cl_stdin_readline", "cl_stdout_write",
        "cl_net_init", "cl_net_listen", "cl_net_listener_port", "cl_net_connect",
        "cl_net_accept", "cl_net_send", "cl_net_recv", "cl_net_cleanup",
        "cl_lmdb_init", "cl_lmdb_open", "cl_lmdb_put", "cl_lmdb_get", "cl_lmdb_delete",
        "cl_lmdb_begin_write_txn", "cl_lmdb_commit_write_txn", "cl_lmdb_cursor_scan",
        "cl_lmdb_sync", "cl_lmdb_cleanup",
        "cl_thread_init", "cl_thread_spawn", "cl_thread_join", "cl_thread_cleanup",
        "cl_thread_call",
    ];

    let mut decls = String::new();
    let mut body = String::new();
    for (i, sym) in symbols.iter().enumerate() {
        decls.push_str(&format!("    fn{i} = %{sym} sig0\n"));
        // func_addr forces the linker to resolve the symbol.
        body.push_str(&format!("    v{} = func_addr.i64 fn{i}\n", i + 100));
        body.push_str(&format!(
            "    store.i64 notrap aligned v{}, v0+0\n",
            i + 100
        ));
    }
    let clif_ir = format!(
        "function u0:0(i64) system_v {{\n\
         \x20   sig0 = (i64) -> i32 system_v\n\
         {decls}\n\
         block0(v0: i64):\n\
         {body}    return\n\
         }}"
    );

    let memory = vec![0u8; 4096];
    let (config, algorithm) =
        create_cranelift_algorithm(0, memory, clif_ir.clone());
    run(config, algorithm).expect("all FFI symbols must be linkable from CLIF");
}

#[test]
fn test_clif_ffi_file_smoke() {
    // Runtime smoke: exercises cl_file_read, cl_file_write, cl_file_read_to_ptr,
    // cl_file_write_from_ptr via a real round-trip.
    let temp_dir = TempDir::new().unwrap();
    let path_a = temp_dir.path().join("smoke_a.bin");
    let path_b = temp_dir.path().join("smoke_b.bin");
    let path_a_str = format!("{}\0", path_a.to_str().unwrap());
    let path_b_str = format!("{}\0", path_b.to_str().unwrap());

    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    sig1 = (i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0
    fn2 = %cl_file_write_from_ptr sig1
    fn3 = %cl_file_read_to_ptr sig1

block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 5
    v5 = call fn0(v0, v1, v2, v3, v4)
    v6 = iconst.i64 3100
    v7 = call fn1(v0, v1, v6, v3, v4)
    v8 = iadd_imm v0, 2256
    v9 = iadd_imm v0, 3000
    v10 = call fn2(v8, v9, v3, v4)
    v11 = iadd_imm v0, 3200
    v12 = call fn3(v8, v11, v3, v4)
    return
}"#;

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + path_a_str.len()].copy_from_slice(path_a_str.as_bytes());
    memory[2256..2256 + path_b_str.len()].copy_from_slice(path_b_str.as_bytes());
    memory[3000..3005].copy_from_slice(b"hello");


    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    assert_eq!(&fs::read(&path_a).unwrap(), b"hello");
    assert_eq!(&fs::read(&path_b).unwrap(), b"hello");
}

#[test]
fn test_clif_ffi_gpu_smoke() {
    // Runtime smoke: exercises the wgpu FFI call path
    // (init → create_buffer → upload → dispatch → download → cleanup).
    // Symbol linkability is verified by test_clif_ffi_all_symbols_linkable.
    let wgsl = "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
                @compute @workgroup_size(64)\n\
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                    let i = gid.x;\n\
                    if (i < arrayLength(&data)) { data[i] = data[i] * 2.0; }\n\
                }\n";

    let shader_off = 2000usize;
    let bind_off = 3000usize;
    let data_off = 4000usize;
    let result_off = 5000usize;
    let n: usize = 64;
    let data_bytes = n * 4;

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v
    sig5 = (i64, i32, i64, i64) -> i32 system_v
    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload sig2
    fn3 = %cl_gpu_create_pipeline sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig5
    fn6 = %cl_gpu_cleanup sig0
block0(v0: i64):
    v90 = iadd_imm v0, 0
    call fn0(v90)
    v91 = load.i64 notrap aligned v0+0
    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v91, v1)
    v3 = iadd_imm v0, {data_off}
    v10 = call fn2(v91, v2, v3, v1)
    v4 = iadd_imm v0, {shader_off}
    v5 = iadd_imm v0, {bind_off}
    v6 = iconst.i32 1
    v7 = call fn3(v91, v4, v5, v6)
    v11 = call fn4(v91, v7, v6, v6, v6)
    v8 = iadd_imm v0, {result_off}
    v12 = call fn5(v91, v2, v8, v1)
    call fn6(v90)
    return
}}"#,
        data_bytes = data_bytes,
        data_off = data_off,
        shader_off = shader_off,
        bind_off = bind_off,
        result_off = result_off,
    );

    let mut memory = vec![0u8; 6144];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    let shader_bytes = wgsl.as_bytes();
    memory[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    memory[shader_off + shader_bytes.len()] = 0;

    // 1 binding: buf0 read_write
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    memory[bind_off + 4..bind_off + 8].copy_from_slice(&0i32.to_le_bytes());

    for i in 0..n {
        memory[data_off + i * 4..data_off + i * 4 + 4]
            .copy_from_slice(&((i + 1) as f32).to_le_bytes());
    }


    let (config, algorithm) =
        create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();
}

#[test]
fn test_clif_ffi_net_smoke() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("net_smoke_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let addr_str = format!("127.0.0.1:{}\0", port);

    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 5];
        stream.read_exact(&mut buf).unwrap();
        stream.write_all(&buf).unwrap();
    });

    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i64 system_v
    sig2 = (i64, i64, i64, i64) -> i64 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_net_init sig0
    fn1 = %cl_net_connect sig1
    fn2 = %cl_net_send sig2
    fn3 = %cl_net_recv sig2
    fn4 = %cl_net_cleanup sig0
    fn5 = %cl_file_write sig3

block0(v0: i64):
    call fn0(v0)
    v1 = load.i64 notrap aligned v0+0
    v2 = iadd_imm v0, 2000
    v3 = call fn1(v1, v2)
    v4 = iadd_imm v0, 3000
    v5 = iconst.i64 5
    v6 = call fn2(v1, v3, v4, v5)
    v7 = iadd_imm v0, 3100
    v8 = call fn3(v1, v3, v7, v5)
    v9 = iconst.i64 2100
    v10 = iconst.i64 3100
    v11 = iconst.i64 0
    v12 = call fn5(v0, v9, v10, v11, v5)
    call fn4(v0)
    return
}"#;

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + addr_str.len()].copy_from_slice(addr_str.as_bytes());
    memory[2100..2100 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[3000..3005].copy_from_slice(b"hello");


    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();
    server.join().unwrap();

    assert_eq!(&fs::read(&verify_file).unwrap()[..5], b"hello");
}

#[test]
fn test_clif_ffi_lmdb_smoke() {
    // Runtime smoke: exercises the lmdb FFI call path
    // (init → open → put → get → cursor_scan → cleanup).
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("lmdb_smoke");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());

    // Memory layout:
    //   0:     reserved (lmdb ctx ptr)
    //   2000:  db path (null-terminated)
    //   3000:  key "hello" (5 bytes)
    //   3100:  value "world" (5 bytes)
    //   3200:  get result buffer (4-byte len + value)
    //   3500:  cursor scan result buffer
    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_cursor_scan sig4
    fn5 = %cl_lmdb_cleanup sig0
block0(v0: i64):
    call fn0(v0)
    v91 = load.i64 notrap aligned v0+0
    v1 = iadd_imm v0, 2000
    v2 = iconst.i32 10
    v3 = call fn1(v91, v1, v2)
    v4 = iadd_imm v0, 3000
    v5 = iconst.i32 5
    v6 = iadd_imm v0, 3100
    v10 = call fn2(v91, v3, v4, v5, v6, v5)
    v7 = iadd_imm v0, 3200
    v11 = call fn3(v91, v3, v4, v5, v7)
    v8 = iadd_imm v0, 3500
    v9 = iconst.i64 0
    v14 = iconst.i32 0
    v12 = iconst.i32 100
    v13 = call fn4(v91, v3, v9, v14, v12, v8)
    call fn5(v0)
    return
}"#;

    let mut memory = vec![0u8; 6144];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[3000..3005].copy_from_slice(b"hello");
    memory[3100..3105].copy_from_slice(b"world");


    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();
}

#[test]
fn test_clif_ffi_thread_smoke() {
    // Runtime smoke: exercises the thread FFI call path
    // (init → spawn → join → call → cleanup).
    // Memory layout:
    //   16-23:   thread context pointer slot
    //   200-207: spawn target writes 42 here
    //   208-215: cl_thread_call writes 99 here
    //   3000+:   verify file path
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("thread_smoke.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    let mut memory = vec![0u8; 8192];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64, i64) -> i64 system_v
    fn2 = %cl_thread_join sig2
    sig3 = (i64) system_v
    fn3 = %cl_thread_cleanup sig3
    sig4 = (i64, i64, i64) -> i64 system_v
    fn4 = %cl_thread_call sig4
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn5 = %cl_file_write sig5
block0(v0: i64):
    v1 = iadd_imm v0, 16
    call fn0(v1)
    v10 = load.i64 notrap aligned v0+16
    v2 = iconst.i64 1
    v3 = iadd_imm v0, 200
    v4 = call fn1(v10, v2, v3)
    v5 = call fn2(v10, v4)
    v6 = iconst.i64 2
    v7 = iadd_imm v0, 208
    v8 = call fn4(v10, v6, v7)
    call fn3(v1)
    v20 = iconst.i64 3000
    v21 = iconst.i64 200
    v22 = iconst.i64 0
    v23 = iconst.i64 16
    v24 = call fn5(v0, v20, v21, v22, v23)
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 42
    store.i64 v1, v0
    return
}

function u0:2(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 99
    store.i64 v1, v0
    return
}"#;

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 16);
    assert_eq!(u64::from_le_bytes(contents[0..8].try_into().unwrap()), 42);
    assert_eq!(u64::from_le_bytes(contents[8..16].try_into().unwrap()), 99);
}

#[test]
fn test_clif_atomic_rmw_add() {
    // Verifies Cranelift's atomic_rmw.i64 IR op compiles and runs through our JIT.
    // Memory: accumulator at offset 64 (init 0), file path at offset 3000.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("atomic_rmw.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());
    memory[64..72].copy_from_slice(&0u64.to_le_bytes());

    // Two atomic adds (10 then 32) onto the accumulator, then write it to a file.
    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iadd_imm v0, 64
    v2 = iconst.i64 10
    v3 = atomic_rmw.i64 little add v1, v2
    v4 = iconst.i64 32
    v5 = atomic_rmw.i64 little add v1, v4
    v6 = iconst.i64 3000
    v7 = iconst.i64 64
    v8 = iconst.i64 0
    v9 = iconst.i64 8
    v10 = call fn0(v0, v6, v7, v8, v9)
    return
}"#;

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let acc = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(acc, 42, "accumulator should be 10 + 32 = 42");
}

#[test]
fn test_clif_call_basic() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("clif_call_basic.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2000
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&42u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());


    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 42);
}

#[test]
fn test_clif_call_multiple_functions() {
    // ClifCall can invoke different functions via src index.
    // fn0 writes value A to file A, fn1 writes value B to file B.
    let temp_dir = TempDir::new().unwrap();
    let test_file_a = temp_dir.path().join("clif_call_fn0.txt");
    let test_file_b = temp_dir.path().join("clif_call_fn1.txt");
    let file_a_str = format!("{}\0", test_file_a.to_str().unwrap());
    let file_b_str = format!("{}\0", test_file_b.to_str().unwrap());

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2256
    v2 = iconst.i64 3008
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + file_a_str.len()].copy_from_slice(file_a_str.as_bytes());
    memory[2256..2256 + file_b_str.len()].copy_from_slice(file_b_str.as_bytes());
    memory[3000..3008].copy_from_slice(&100u64.to_le_bytes());
    memory[3008..3016].copy_from_slice(&200u64.to_le_bytes());

    // Demonstrates JIT-once, run-many: one Base, two execute() calls picking different
    // fn_idx into the same compiled module.
    let mut base = Base::new(cranelift_config(memory, clif_ir.to_string())).unwrap();
    base.execute(&cranelift_algorithm(0), &[]).unwrap();
    base.execute(&cranelift_algorithm(1), &[]).unwrap();

    assert!(test_file_a.exists());
    let contents_a = fs::read(&test_file_a).unwrap();
    assert_eq!(
        u64::from_le_bytes(contents_a[0..8].try_into().unwrap()),
        100
    );

    assert!(test_file_b.exists());
    let contents_b = fs::read(&test_file_b).unwrap();
    assert_eq!(
        u64::from_le_bytes(contents_b[0..8].try_into().unwrap()),
        200
    );
}

#[test]
fn test_clif_call_arithmetic() {
    // ClifCall runs a CLIF function that does arithmetic then writes the result to a file.
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("clif_call_arith.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = load.i64 v0+2000
    v2 = load.i64 v0+2008
    v3 = iadd v1, v2
    store.i64 v3, v0+2016
    v4 = iconst.i64 3000
    v5 = iconst.i64 2016
    v6 = iconst.i64 0
    v7 = iconst.i64 8
    v8 = call fn0(v0, v4, v5, v6, v7)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&30u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&12u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 42, "30 + 12 = 42");
}

#[test]
fn test_clif_call_sequential_mutations() {
    // Multiple ClifCall actions run sequentially, each mutating shared memory.
    // fn0: store 10 at offset 2000
    // Three execute() calls on the same Base, each running a different fn:
    //   fn0 stores 10 at offset 2000
    //   fn1 loads 2000, multiplies by 5, stores at 2008
    //   fn2 writes offset 2008 to file
    // Shared memory persists across execute() calls, demonstrating run-many semantics.
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("clif_call_seq.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    v1 = iconst.i64 10
    store.i64 v1, v0+2000
    return
}}

function u0:1(i64) system_v {{
block0(v0: i64):
    v1 = load.i64 v0+2000
    v2 = iconst.i64 5
    v3 = imul v1, v2
    store.i64 v3, v0+2008
    return
}}

function u0:2(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2008
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let mut base = Base::new(cranelift_config(memory, clif_ir.to_string())).unwrap();
    base.execute(&cranelift_algorithm(0), &[]).unwrap();
    base.execute(&cranelift_algorithm(1), &[]).unwrap();
    base.execute(&cranelift_algorithm(2), &[]).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 50, "10 * 5 = 50");
}

#[test]
fn test_clif_call_no_workers_needed() {
    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    v1 = iconst.i64 77
    store.i64 v1, v0+2000
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);


    // cranelift_units: 0 — no workers
    let (_config, _algorithm) = create_cranelift_algorithm(0, memory, clif_ir.to_string());

    // Rebuild with file write verification
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("clif_call_no_workers.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir2 = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 77
    store.i64 v1, v0+2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 2000
    v4 = iconst.i64 0
    v5 = iconst.i64 8
    v6 = call fn0(v0, v2, v3, v4, v5)
    return
}}"#
    );

    let mut memory2 = vec![0u8; 4096];
    let clif_bytes2 = format!("{}\0", clif_ir2).into_bytes();
    memory2[0..clif_bytes2.len()].copy_from_slice(&clif_bytes2);
    memory2[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());


    let (config2, algorithm2) =
        create_cranelift_algorithm(0, memory2, clif_ir2.to_string());
    run(config2, algorithm2).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 77);
}

#[test]
fn test_clif_call_file_read_write() {
    // ClifCall can do file read followed by file write.
    // fn0: read input file into memory, fn1: write from memory to output file.
    let temp_dir = TempDir::new().unwrap();
    let input_file = temp_dir.path().join("clif_call_input.bin");
    let output_file = temp_dir.path().join("clif_call_output.bin");

    // Create input file with known data
    let input_data: Vec<u8> = (0..256).map(|i| i as u8).collect();
    fs::write(&input_file, &input_data).unwrap();

    let input_str = format!("{}\0", input_file.to_str().unwrap());
    let output_str = format!("{}\0", output_file.to_str().unwrap());

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_read sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 256
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2256
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 256
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#
    );

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + input_str.len()].copy_from_slice(input_str.as_bytes());
    memory[2256..2256 + output_str.len()].copy_from_slice(output_str.as_bytes());

    // Two execute() calls on one Base: fn0 reads input file, fn1 writes output file.
    let mut base = Base::new(cranelift_config(memory, clif_ir.to_string())).unwrap();
    base.execute(&cranelift_algorithm(0), &[]).unwrap();
    base.execute(&cranelift_algorithm(1), &[]).unwrap();

    assert!(output_file.exists());
    let output_data = fs::read(&output_file).unwrap();
    assert_eq!(output_data, input_data, "output should match input");
}

fn create_output_algorithm(
    clif_ir: &str,
    memory: Vec<u8>,
    output: Vec<OutputBatchSchema>,
) -> (Setup, Algorithm) {
    let mut p = memory;
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    if p.len() < clif_bytes.len() {
        p.resize(clif_bytes.len().max(p.len()), 0);
    }
    p[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    let config = Setup {
        cranelift_ir: clif_ir.to_string(),
        memory_size: p.len(),
        runtime_header: legacy_runtime_header(),
        initial_memory: p,
    };
    let algorithm = Algorithm {
        fn_idx: 0,
        output,
    };
    (config, algorithm)
}

#[test]
fn test_output_no_schema_returns_empty() {
    // A simple CLIF that writes a value but has no output schema —
    // execute should return an empty Vec<RecordBatch>.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 42
    v2 = iconst.i64 2000
    v3 = iadd v0, v2
    store.i64 v1, v3
    return
}"#;

    let memory = vec![0u8; 4096];
    let (cfg, alg) = create_output_algorithm(clif_ir, memory, vec![]);
    let batches = run(cfg, alg).unwrap();
    assert!(batches.is_empty());
}

#[test]
fn test_output_single_i64_column() {
    // CLIF writes i64 value 99 at offset 2000 and row_count=1 at offset 2008.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 99
    v2 = iconst.i64 2000
    v3 = iadd v0, v2
    store.i64 v1, v3
    v4 = iconst.i64 1
    v5 = iconst.i64 2008
    v6 = iadd v0, v5
    store.i64 v4, v6
    return
}"#;

    let memory = vec![0u8; 4096];
    let output = vec![OutputBatchSchema {
        row_count_offset: 2008,
        columns: vec![OutputColumn {
            name: "value".to_string(),
            dtype: OutputType::I64,
            data_offset: 2000,
            len_offset: 0,
        }],
    }];

    let (cfg, alg) = create_output_algorithm(clif_ir, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert_eq!(batches.len(), 1);

    let expected = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![99i64]))],
    )
    .unwrap();
    assert_eq!(batches[0], expected);
}

#[test]
fn test_output_i64_and_f64_columns() {
    // CLIF writes an i64 at 2000, an f64 at 2008, and row_count=1 at 2016.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 42
    v2 = iconst.i64 2000
    v3 = iadd v0, v2
    store.i64 v1, v3
    v4 = f64const 0x1.921fb54442d18p1
    v5 = iconst.i64 2008
    v6 = iadd v0, v5
    store.f64 v4, v6
    v7 = iconst.i64 1
    v8 = iconst.i64 2016
    v9 = iadd v0, v8
    store.i64 v7, v9
    return
}"#;

    let memory = vec![0u8; 4096];
    let output = vec![OutputBatchSchema {
        row_count_offset: 2016,
        columns: vec![
            OutputColumn {
                name: "count".to_string(),
                dtype: OutputType::I64,
                data_offset: 2000,
                len_offset: 0,
            },
            OutputColumn {
                name: "pi".to_string(),
                dtype: OutputType::F64,
                data_offset: 2008,
                len_offset: 0,
            },
        ],
    }];

    let (cfg, alg) = create_output_algorithm(clif_ir, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert_eq!(batches.len(), 1);

    let expected = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("count", DataType::Int64, false),
            Field::new("pi", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![42i64])),
            Arc::new(Float64Array::from(vec![std::f64::consts::PI])),
        ],
    )
    .unwrap();
    assert_eq!(batches[0], expected);
}

#[test]
fn test_output_utf8_single_row() {
    // CLIF writes "hello" (5 bytes) at offset 2000, string length 5 at offset 2008,
    // and row_count=1 at offset 2016.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 0x6f6c6c6568
    v2 = iconst.i64 2000
    v3 = iadd v0, v2
    store.i64 v1, v3
    v4 = iconst.i64 5
    v5 = iconst.i64 2008
    v6 = iadd v0, v5
    store.i64 v4, v6
    v7 = iconst.i64 1
    v8 = iconst.i64 2016
    v9 = iadd v0, v8
    store.i64 v7, v9
    return
}"#;

    let memory = vec![0u8; 4096];
    let output = vec![OutputBatchSchema {
        row_count_offset: 2016,
        columns: vec![OutputColumn {
            name: "greeting".to_string(),
            dtype: OutputType::Utf8,
            data_offset: 2000,
            len_offset: 2008,
        }],
    }];

    let (cfg, alg) = create_output_algorithm(clif_ir, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert_eq!(batches.len(), 1);

    let expected = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "greeting",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["hello"]))],
    )
    .unwrap();
    assert_eq!(batches[0], expected);
}

#[test]
fn test_output_multi_row_i64() {
    // CLIF writes 3 i64 values at offsets 2000, 2008, 2016, and row_count=3 at 2024.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 10
    v2 = iconst.i64 2000
    v3 = iadd v0, v2
    store.i64 v1, v3
    v4 = iconst.i64 20
    v5 = iconst.i64 2008
    v6 = iadd v0, v5
    store.i64 v4, v6
    v7 = iconst.i64 30
    v8 = iconst.i64 2016
    v9 = iadd v0, v8
    store.i64 v7, v9
    v10 = iconst.i64 3
    v11 = iconst.i64 2024
    v12 = iadd v0, v11
    store.i64 v10, v12
    return
}"#;

    let memory = vec![0u8; 4096];
    let output = vec![OutputBatchSchema {
        row_count_offset: 2024,
        columns: vec![OutputColumn {
            name: "values".to_string(),
            dtype: OutputType::I64,
            data_offset: 2000,
            len_offset: 0,
        }],
    }];

    let (cfg, alg) = create_output_algorithm(clif_ir, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert_eq!(batches.len(), 1);

    let expected = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "values",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![10i64, 20, 30]))],
    )
    .unwrap();
    assert_eq!(batches[0], expected);
}

#[test]
fn test_output_zero_row_count_skips_batch() {
    // CLIF writes nothing — row_count stays 0 in zeroed memory.
    // The batch should be skipped entirely.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    return
}"#;

    let memory = vec![0u8; 4096];
    let output = vec![OutputBatchSchema {
        row_count_offset: 2000,
        columns: vec![OutputColumn {
            name: "x".to_string(),
            dtype: OutputType::I64,
            data_offset: 2008,
            len_offset: 0,
        }],
    }];

    let (cfg, alg) = create_output_algorithm(clif_ir, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert!(batches.is_empty());
}

#[test]
fn test_output_multiple_batches() {
    // Two output schemas — each becomes a separate RecordBatch.
    // Batch 1: single i64 at 2000, row_count at 2008.
    // Batch 2: single f64 at 2016, row_count at 2024.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 7
    v2 = iconst.i64 2000
    v3 = iadd v0, v2
    store.i64 v1, v3
    v4 = iconst.i64 1
    v5 = iconst.i64 2008
    v6 = iadd v0, v5
    store.i64 v4, v6
    v7 = f64const 0x1.c000000000000p2
    v8 = iconst.i64 2016
    v9 = iadd v0, v8
    store.f64 v7, v9
    v10 = iconst.i64 1
    v11 = iconst.i64 2024
    v12 = iadd v0, v11
    store.i64 v10, v12
    return
}"#;

    let memory = vec![0u8; 4096];
    let output = vec![
        OutputBatchSchema {
            row_count_offset: 2008,
            columns: vec![OutputColumn {
                name: "integer_val".to_string(),
                dtype: OutputType::I64,
                data_offset: 2000,
                len_offset: 0,
            }],
        },
        OutputBatchSchema {
            row_count_offset: 2024,
            columns: vec![OutputColumn {
                name: "float_val".to_string(),
                dtype: OutputType::F64,
                data_offset: 2016,
                len_offset: 0,
            }],
        },
    ];

    let (cfg, alg) = create_output_algorithm(clif_ir, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert_eq!(batches.len(), 2);

    let expected_0 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "integer_val",
            DataType::Int64,
            false,
        )])),
        vec![Arc::new(Int64Array::from(vec![7i64]))],
    )
    .unwrap();
    assert_eq!(batches[0], expected_0);

    let expected_1 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "float_val",
            DataType::Float64,
            false,
        )])),
        vec![Arc::new(Float64Array::from(vec![7.0f64]))],
    )
    .unwrap();
    assert_eq!(batches[1], expected_1);
}

#[test]
fn test_output_utf8_multi_row() {
    // CLIF writes two null-terminated strings at offset 2000: "abc\0def\0"
    // len_offset at 2100 holds total byte length (not used for multi-row; strings are null-terminated).
    // row_count=2 at 2108.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 0x0066656400636261
    v2 = iconst.i64 2000
    v3 = iadd v0, v2
    store.i64 v1, v3
    v4 = iconst.i64 7
    v5 = iconst.i64 2100
    v6 = iadd v0, v5
    store.i64 v4, v6
    v7 = iconst.i64 2
    v8 = iconst.i64 2108
    v9 = iadd v0, v8
    store.i64 v7, v9
    return
}"#;

    let memory = vec![0u8; 4096];
    let output = vec![OutputBatchSchema {
        row_count_offset: 2108,
        columns: vec![OutputColumn {
            name: "words".to_string(),
            dtype: OutputType::Utf8,
            data_offset: 2000,
            len_offset: 2100,
        }],
    }];

    let (cfg, alg) = create_output_algorithm(clif_ir, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert_eq!(batches.len(), 1);

    let expected = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "words",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["abc", "def"]))],
    )
    .unwrap();
    assert_eq!(batches[0], expected);
}

#[test]
fn test_output_multiple_batches_multi_row_mixed() {
    // Batch 0: summary — 1 row with I64 "total" and F64 "average"
    // Batch 1: detail — 3 rows with I64 "id" and Utf8 "name"
    //
    // Layout (all in additional_shared_memory region starting at offset 2000):
    //   2000: batch0 row_count (8 bytes) = 1
    //   2008: batch0 col0 "total" i64 = 300
    //   2016: batch0 col1 "average" f64 = 100.0
    //   2024: batch1 row_count (8 bytes) = 3
    //   2032: batch1 col0 "id" i64[3] = [1, 2, 3] (24 bytes)
    //   2056: batch1 col1 "name" strings = "alice\0bob\0charlie\0" (19 bytes)
    //   2080: batch1 col1 len_offset (8 bytes) = 19
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    ; batch0 row_count = 1
    v1 = iconst.i64 1
    v2 = iconst.i64 2000
    v3 = iadd v0, v2
    store.i64 v1, v3

    ; batch0 total = 300
    v4 = iconst.i64 300
    v5 = iconst.i64 2008
    v6 = iadd v0, v5
    store.i64 v4, v6

    ; batch0 average = 100.0
    v7 = f64const 0x1.9000000000000p6
    v8 = iconst.i64 2016
    v9 = iadd v0, v8
    store.f64 v7, v9

    ; batch1 row_count = 3
    v10 = iconst.i64 3
    v11 = iconst.i64 2024
    v12 = iadd v0, v11
    store.i64 v10, v12

    ; batch1 id[0] = 1
    v13 = iconst.i64 1
    v14 = iconst.i64 2032
    v15 = iadd v0, v14
    store.i64 v13, v15

    ; batch1 id[1] = 2
    v16 = iconst.i64 2
    v17 = iconst.i64 2040
    v18 = iadd v0, v17
    store.i64 v16, v18

    ; batch1 id[2] = 3
    v19 = iconst.i64 3
    v20 = iconst.i64 2048
    v21 = iadd v0, v20
    store.i64 v19, v21

    ; batch1 names: "alice\0bob\0charlie\0" packed at 2056
    ; "alice\0bo" = 0x6f62_0065_6369_6c61
    v22 = iconst.i64 0x6f62006563696c61
    v23 = iconst.i64 2056
    v24 = iadd v0, v23
    store.i64 v22, v24
    ; "b\0charli" = 0x696c_7261_6863_0062
    v25 = iconst.i64 0x696c726168630062
    v26 = iconst.i64 2064
    v27 = iadd v0, v26
    store.i64 v25, v27
    ; "e\0" + padding = 0x0065
    v28 = iconst.i64 0x0065
    v29 = iconst.i64 2072
    v30 = iadd v0, v29
    store.i64 v28, v30

    ; batch1 name len_offset = 19
    v31 = iconst.i64 19
    v32 = iconst.i64 2080
    v33 = iadd v0, v32
    store.i64 v31, v33

    return
}"#;

    let memory = vec![0u8; 4096];
    let output = vec![
        OutputBatchSchema {
            row_count_offset: 2000,
            columns: vec![
                OutputColumn {
                    name: "total".to_string(),
                    dtype: OutputType::I64,
                    data_offset: 2008,
                    len_offset: 0,
                },
                OutputColumn {
                    name: "average".to_string(),
                    dtype: OutputType::F64,
                    data_offset: 2016,
                    len_offset: 0,
                },
            ],
        },
        OutputBatchSchema {
            row_count_offset: 2024,
            columns: vec![
                OutputColumn {
                    name: "id".to_string(),
                    dtype: OutputType::I64,
                    data_offset: 2032,
                    len_offset: 0,
                },
                OutputColumn {
                    name: "name".to_string(),
                    dtype: OutputType::Utf8,
                    data_offset: 2056,
                    len_offset: 2080,
                },
            ],
        },
    ];

    let (cfg, alg) = create_output_algorithm(clif_ir, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert_eq!(batches.len(), 2);

    // Batch 0: summary
    let expected_0 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("total", DataType::Int64, false),
            Field::new("average", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![300i64])),
            Arc::new(Float64Array::from(vec![100.0f64])),
        ],
    )
    .unwrap();
    assert_eq!(batches[0], expected_0);

    // Batch 1: detail
    let expected_1 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 2, 3])),
            Arc::new(StringArray::from(vec!["alice", "bob", "charlie"])),
        ],
    )
    .unwrap();
    assert_eq!(batches[1], expected_1);
}

#[test]
fn test_output_multiple_batches_partial_skip() {
    // Three schemas declared, but only batch 0 and batch 2 have row_count > 0.
    // Batch 1 should be skipped, resulting in 2 returned batches.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    ; batch0: row_count=1, value=42
    v1 = iconst.i64 1
    v2 = iconst.i64 2000
    v3 = iadd v0, v2
    store.i64 v1, v3
    v4 = iconst.i64 42
    v5 = iconst.i64 2008
    v6 = iadd v0, v5
    store.i64 v4, v6

    ; batch1: row_count stays 0 (skipped)

    ; batch2: row_count=2, values=[10, 20]
    v7 = iconst.i64 2
    v8 = iconst.i64 2032
    v9 = iadd v0, v8
    store.i64 v7, v9
    v10 = iconst.i64 10
    v11 = iconst.i64 2040
    v12 = iadd v0, v11
    store.i64 v10, v12
    v13 = iconst.i64 20
    v14 = iconst.i64 2048
    v15 = iadd v0, v14
    store.i64 v13, v15

    return
}"#;

    let memory = vec![0u8; 4096];
    let output = vec![
        OutputBatchSchema {
            row_count_offset: 2000,
            columns: vec![OutputColumn {
                name: "a".to_string(),
                dtype: OutputType::I64,
                data_offset: 2008,
                len_offset: 0,
            }],
        },
        OutputBatchSchema {
            row_count_offset: 2016, // stays 0 — skipped
            columns: vec![OutputColumn {
                name: "b".to_string(),
                dtype: OutputType::F64,
                data_offset: 2024,
                len_offset: 0,
            }],
        },
        OutputBatchSchema {
            row_count_offset: 2032,
            columns: vec![OutputColumn {
                name: "c".to_string(),
                dtype: OutputType::I64,
                data_offset: 2040,
                len_offset: 0,
            }],
        },
    ];

    let (cfg, alg) = create_output_algorithm(clif_ir, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert_eq!(
        batches.len(),
        2,
        "middle batch with row_count=0 should be skipped"
    );

    let expected_0 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![42i64]))],
    )
    .unwrap();
    assert_eq!(batches[0], expected_0);

    let expected_1 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("c", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![10i64, 20]))],
    )
    .unwrap();
    assert_eq!(batches[1], expected_1);
}

#[test]
fn test_base_single_execute_matches_standalone() {
    // Base::new + execute should produce the same result as standalone run.
    // CLIF: load i64 from offset 100, multiply by 7, store at 200, row_count=1 at 208.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+100
    v2 = iconst.i64 7
    v3 = imul v1, v2
    store v3, v0+200
    v4 = iconst.i64 1
    store v4, v0+208
    return
}"#
    .to_string();

    let mut memory = vec![0u8; 4096];
    memory[100..108].copy_from_slice(&6i64.to_le_bytes());

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "result".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];

    // Standalone
    let config1 = Setup {
        cranelift_ir: clif_ir.clone(),
        memory_size: memory.len(),
        runtime_header: legacy_runtime_header(),
        initial_memory: memory.clone(),
    };
    let alg1 = Algorithm {
        fn_idx: 0,
        output: output_schema.clone(),
    };
    let batches1 = run(config1, alg1).unwrap();

    // Base struct
    let config2 = Setup {
        cranelift_ir: clif_ir,
        memory_size: memory.len(),
        runtime_header: legacy_runtime_header(),
        initial_memory: memory,
    };
    let alg2 = Algorithm {
        fn_idx: 0,
        output: output_schema,
    };
    let mut base = Base::new(config2).unwrap();
    let batches2 = base.execute(&alg2, &[]).unwrap();

    // Both should produce 6 * 7 = 42
    assert_eq!(batches1.len(), 1);
    assert_eq!(batches2.len(), 1);
    let col1 = batches1[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let col2 = batches2[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col1.value(0), 42);
    assert_eq!(col2.value(0), 42);
}

#[test]
fn test_base_multi_execute_different_data() {
    // Compile once, execute twice with different input data via pointer.
    // CLIF reads i64 from data pointer, multiplies by 3, stores result at 200, row_count=1 at 208.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+8
    v2 = load.i64 v1
    v3 = iconst.i64 3
    v4 = imul v2, v3
    store v4, v0+200
    v5 = iconst.i64 1
    v6 = iconst.i64 208
    v7 = iadd v0, v6
    store.i64 v5, v7
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "result".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];

    // First execute: input = 10, expect 30
    let data1 = 10i64.to_le_bytes();
    let batches1 = base
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: output_schema.clone(),
            },
            &data1,
        )
        .unwrap();
    assert_eq!(batches1.len(), 1);
    let col1 = batches1[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col1.value(0), 30);

    // Second execute: input = 100, expect 300
    let data2 = 100i64.to_le_bytes();
    let batches2 = base
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: output_schema,
            },
            &data2,
        )
        .unwrap();
    assert_eq!(batches2.len(), 1);
    let col2 = batches2[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col2.value(0), 300);
}

#[test]
fn test_base_multi_execute_different_actions() {
    // Compile once with two CLIF functions, execute with different action sequences.
    // fn0: stores 42 at offset 200, row_count=1 at 208
    // fn1: stores 99 at offset 200, row_count=1 at 208
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 42
    store v1, v0+200
    v2 = iconst.i64 1
    v3 = iconst.i64 208
    v4 = iadd v0, v3
    store.i64 v2, v4
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 99
    store v1, v0+200
    v2 = iconst.i64 1
    v3 = iconst.i64 208
    v4 = iadd v0, v3
    store.i64 v2, v4
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "val".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];

    // First execute: call fn0 only
    let alg1 = Algorithm {
        fn_idx: 0,
        output: output_schema.clone(),
    };
    let batches1 = base.execute(&alg1, &vec![0u8; 4096]).unwrap();
    let col1 = batches1[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col1.value(0), 42);

    // Second execute: call fn1 only
    let alg2 = Algorithm {
        fn_idx: 1,
        output: output_schema,
    };
    let batches2 = base.execute(&alg2, &vec![0u8; 4096]).unwrap();
    let col2 = batches2[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col2.value(0), 99);
}

#[test]
fn test_base_multi_execute_accumulates_in_memory() {
    // Accumulator in shared memory persists across executes.
    // CLIF: load accumulator from v0+200, add input from data pointer, store back.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+200
    v2 = load.i64 v0+8
    v3 = load.i64 v2
    v4 = iadd v1, v3
    store v4, v0+200
    v5 = iconst.i64 1
    v6 = iconst.i64 208
    v7 = iadd v0, v6
    store.i64 v5, v7
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "total".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];

    // Execute 1: add 10 → total = 10
    let d1 = 10i64.to_le_bytes();
    let batches = base
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: output_schema.clone(),
            },
            &d1,
        )
        .unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col.value(0), 10);

    // Execute 2: add 25 → total = 35
    let d2 = 25i64.to_le_bytes();
    let batches = base
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: output_schema.clone(),
            },
            &d2,
        )
        .unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col.value(0), 35);

    // Execute 3: add 5 → total = 40
    let d3 = 5i64.to_le_bytes();
    let batches = base
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: output_schema,
            },
            &d3,
        )
        .unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col.value(0), 40);
}

#[test]
fn test_base_multi_execute_with_file_io() {
    // Compile once, write different files on each execute using initial_memory for layout.
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("out1.bin");
    let file2 = temp_dir.path().join("out2.bin");
    let file1_str = format!("{}\0", file1.to_str().unwrap());
    let file2_str = format!("{}\0", file2.to_str().unwrap());

    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 256
    v2 = iconst.i64 512
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}"#
    .to_string();

    // Execute 1: write value 42 to file1
    let mut mem1 = vec![0u8; 4096];
    mem1[256..256 + file1_str.len()].copy_from_slice(file1_str.as_bytes());
    mem1[512..520].copy_from_slice(&42u64.to_le_bytes());
    let config1 = Setup {
        cranelift_ir: clif_ir.clone(),
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: mem1,
    };
    let mut base = Base::new(config1).unwrap();
    base.execute(
        &Algorithm {
            fn_idx: 0,
            output: vec![],
        },
        &[],
    )
    .unwrap();
    assert!(file1.exists());
    let data1 = fs::read(&file1).unwrap();
    assert_eq!(u64::from_le_bytes(data1[..8].try_into().unwrap()), 42);

    // Execute 2: write value 99 to file2 — new Base with different initial_memory
    let mut mem2 = vec![0u8; 4096];
    mem2[256..256 + file2_str.len()].copy_from_slice(file2_str.as_bytes());
    mem2[512..520].copy_from_slice(&99u64.to_le_bytes());
    let config2 = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: mem2,
    };
    let mut base2 = Base::new(config2).unwrap();
    base2
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: vec![],
            },
            &[],
        )
        .unwrap();
    assert!(file2.exists());
    let data2 = fs::read(&file2).unwrap();
    assert_eq!(u64::from_le_bytes(data2[..8].try_into().unwrap()), 99);
}

#[test]
fn test_base_multi_execute_varying_cranelift_units() {
    // Same config, but different cranelift_units per execute.
    // fn0: stores 1 at offset 200
    // Workers also call fn0, each adding to the same location (but with sync ClifCall
    // only the interpreter calls it, so this just verifies units can vary).
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 1
    store v1, v0+200
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Execute with 0 units
    base.execute(
        &Algorithm {
            fn_idx: 0,
            output: vec![],
        },
        &vec![0u8; 4096],
    )
    .unwrap();

    // Execute with 2 units
    base.execute(
        &Algorithm {
            fn_idx: 0,
            output: vec![],
        },
        &vec![0u8; 4096],
    )
    .unwrap();

    // Execute with 4 units
    base.execute(
        &Algorithm {
            fn_idx: 0,
            output: vec![],
        },
        &vec![0u8; 4096],
    )
    .unwrap();
}

#[test]
fn test_base_initial_memory_and_data_pointer_coexist() {
    // initial_memory provides static config at v0+100, data pointer provides dynamic input.
    // CLIF reads both and adds them.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+100
    v2 = load.i64 v0+8
    v3 = load.i64 v2
    v4 = iadd v1, v3
    store v4, v0+300
    v5 = iconst.i64 1
    store v5, v0+308
    return
}"#
    .to_string();

    let mut mem = vec![0u8; 4096];
    mem[100..108].copy_from_slice(&11i64.to_le_bytes());

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: mem,
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 308,
        columns: vec![OutputColumn {
            name: "sum".to_string(),
            dtype: OutputType::I64,
            data_offset: 300,
            len_offset: 0,
        }],
    }];

    let data = 99i64.to_le_bytes();
    let batches = base
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: output_schema,
            },
            &data,
        )
        .unwrap();

    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col.value(0), 110); // 11 + 99
}

#[test]
fn test_base_persistent_memory_survives_across_executes() {
    // Shared memory persists across executes. fn0 seeds a value, fn1 reads it.
    // CLIF fn0: stores 77 at offset 200
    // CLIF fn1: reads data pointer input + offset 200 → stores at 300
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 77
    store v1, v0+200
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+8
    v2 = load.i64 v1
    v3 = load.i64 v0+200
    v4 = iadd v2, v3
    store v4, v0+300
    v5 = iconst.i64 1
    store v5, v0+308
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Execute 1: seed 77 at offset 200
    base.execute(
        &Algorithm {
            fn_idx: 0,
            output: vec![],
        },
        &[],
    )
    .unwrap();

    // Execute 2: input=5 via pointer, read persistent 77 from offset 200
    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 308,
        columns: vec![OutputColumn {
            name: "sum".to_string(),
            dtype: OutputType::I64,
            data_offset: 300,
            len_offset: 0,
        }],
    }];

    let data = 5i64.to_le_bytes();
    let batches = base
        .execute(
            &Algorithm {
                fn_idx: 1,
                output: output_schema,
            },
            &data,
        )
        .unwrap();

    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    // 5 (data pointer) + 77 (persistent) = 82
    assert_eq!(col.value(0), 82);
}

#[test]
fn test_base_empty_data_leaves_memory_intact() {
    // Empty memory don't touch memory at all — persistent state survives.
    // CLIF: accumulate into offset 200 (read, add 1, store back). row_count at 208.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+200
    v2 = iconst.i64 1
    v3 = iadd v1, v2
    store v3, v0+200
    v4 = iconst.i64 1
    store v4, v0+208
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "counter".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];
    // Three executes with empty memory — counter should increment each time
    for expected in 1..=3 {
        let batches = base
            .execute(
                &Algorithm {
                    fn_idx: 0,
                    output: output_schema.clone(),
                },
                &[],
            )
            .unwrap();
        let col = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), expected);
    }
}

#[test]
fn test_base_data_pointer_updates_each_execute() {
    // Data pointer is updated each execute call with fresh caller buffer.
    // CLIF reads two i64s from data pointer and adds them.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+8
    v2 = load.i64 v1
    v3 = load.i64 v1+8
    v4 = iadd v2, v3
    store v4, v0+200
    v5 = iconst.i64 1
    store v5, v0+208
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "result".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];
    // Execute 1: 10 + 20 = 30
    let mut d1 = vec![0u8; 16];
    d1[0..8].copy_from_slice(&10i64.to_le_bytes());
    d1[8..16].copy_from_slice(&20i64.to_le_bytes());
    let batches = base
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: output_schema.clone(),
            },
            &d1,
        )
        .unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col.value(0), 30);

    // Execute 2: 100 + 200 = 300 — pointer should update to new buffer
    let mut d2 = vec![0u8; 16];
    d2[0..8].copy_from_slice(&100i64.to_le_bytes());
    d2[8..16].copy_from_slice(&200i64.to_le_bytes());
    let batches = base
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: output_schema,
            },
            &d2,
        )
        .unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col.value(0), 300);
}

#[test]
fn test_base_output_in_persistent_region() {
    // Shared memory persists across executes. CLIF appends values from data pointer
    // into a growing buffer at offset 500+.
    // fn0: reads input from data_ptr, reads count from offset 400, stores at 500+8*count, increments count.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+8
    v2 = load.i64 v1
    v3 = load.i64 v0+400
    v4 = iconst.i64 8
    v5 = imul v3, v4
    v6 = iconst.i64 500
    v7 = iadd v5, v6
    v8 = iadd v0, v7
    store v2, v8
    v9 = iconst.i64 1
    v10 = iadd v3, v9
    store v10, v0+400
    store v10, v0+408
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Execute 3 times with values 100, 200, 300
    for &val in &[100i64, 200, 300] {
        let d = val.to_le_bytes();
        base.execute(
            &Algorithm {
                fn_idx: 0,
                output: vec![],
            },
            &d,
        )
        .unwrap();
    }

    // Final read: count at 400 should be 3, values at 500/508/516 should be 100/200/300
    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 408,
        columns: vec![OutputColumn {
            name: "values".to_string(),
            dtype: OutputType::I64,
            data_offset: 500,
            len_offset: 0,
        }],
    }];

    // One more execute to read output — pass a dummy input
    let d = 999i64.to_le_bytes();
    let batches = base
        .execute(
            &Algorithm {
                fn_idx: 0,
                output: output_schema,
            },
            &d,
        )
        .unwrap();

    // count is now 4 (we did 4 executes), values: 100, 200, 300, 999
    assert_eq!(batches.len(), 1);
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col.len(), 4);
    assert_eq!(col.value(0), 100);
    assert_eq!(col.value(1), 200);
    assert_eq!(col.value(2), 300);
    assert_eq!(col.value(3), 999);
}

#[test]
fn clif_parse_error_garbage_ir() {
    let config = Setup {
        cranelift_ir: "this is not valid CLIF".to_string(),
        memory_size: 256,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let Err(err) = Base::new(config) else {
        panic!("expected ClifParse error for garbage IR");
    };
    assert!(matches!(err, base::Error::ClifParse(_)));
}

#[test]
fn clif_parse_error_via_run() {
    let config = Setup {
        cranelift_ir: "not valid clif at all {}[]".to_string(),
        memory_size: 256,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let algorithm = Algorithm {
        fn_idx: 0,
        output: vec![],
    };
    let Err(err) = run(config, algorithm) else {
        panic!("expected ClifParse error for invalid CLIF via run()");
    };
    assert!(matches!(err, base::Error::ClifParse(_)));
}

#[test]
fn clif_parse_error_incomplete_function() {
    let config = Setup {
        cranelift_ir: "function %f0(i64) {\n".to_string(),
        memory_size: 256,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let Err(err) = Base::new(config) else {
        panic!("expected ClifParse error for incomplete function");
    };
    assert!(matches!(err, base::Error::ClifParse(_)));
}

#[test]
fn clif_parse_error_empty_ir_no_error() {
    // Empty string should NOT error — it skips compilation entirely
    let config = Setup {
        cranelift_ir: String::new(),
        memory_size: 256,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let base = Base::new(config);
    assert!(base.is_ok());
}

#[test]
fn test_clif_ffi_cuda_smoke() {
    // Runtime smoke: exercises the cuda FFI call path
    // (init → create_buffer → upload → launch → sync → download → cleanup).
    let ptx = "\
.version 7.0
.target sm_50
.address_size 64

.visible .entry main(
    .param .u64 data_ptr
)
{
    .reg .u32 %r0;
    .reg .u64 %rd, %off;
    .reg .f32 %fv, %fc;

    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %off, %r0;
    shl.b64 %off, %off, 2;

    ld.param.u64 %rd, [data_ptr];
    add.u64 %rd, %rd, %off;

    ld.global.f32 %fv, [%rd];
    mov.f32 %fc, 0f40000000;
    mul.f32 %fv, %fv, %fc;
    st.global.f32 [%rd], %fv;

    ret;
}\0";

    let ptx_off = 2000usize;
    let bind_off = 3000usize;
    let data_off = 4000usize;
    let result_off = 5000usize;
    let n: usize = 4;
    let data_bytes = n * 4;

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v
    sig4 = (i64) -> i32 system_v
    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload sig2
    fn3 = %cl_cuda_launch sig3
    fn4 = %cl_cuda_sync sig4
    fn5 = %cl_cuda_download sig2
    fn6 = %cl_cuda_cleanup sig0
block0(v0: i64):
    v90 = iadd_imm v0, 0
    call fn0(v90)
    v91 = load.i64 notrap aligned v0+0
    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v91, v1)
    v3 = iadd_imm v0, {data_off}
    v10 = call fn2(v91, v2, v3, v1)
    v4 = iadd_imm v0, {ptx_off}
    v5 = iconst.i32 1
    v6 = iadd_imm v0, {bind_off}
    v7 = iconst.i32 4
    v11 = call fn3(v91, v4, v5, v6, v5, v5, v5, v7, v5, v5)
    v12 = call fn4(v91)
    v9 = iadd_imm v0, {result_off}
    v13 = call fn5(v91, v2, v9, v1)
    call fn6(v90)
    return
}}"#,
        data_bytes = data_bytes,
        data_off = data_off,
        ptx_off = ptx_off,
        bind_off = bind_off,
        result_off = result_off,
    );

    let mut memory = vec![0u8; 6144];
    let ptx_bytes = ptx.as_bytes();
    memory[ptx_off..ptx_off + ptx_bytes.len()].copy_from_slice(ptx_bytes);
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    for i in 0..n {
        memory[data_off + i * 4..data_off + i * 4 + 4]
            .copy_from_slice(&((i + 1) as f32).to_le_bytes());
    }


    let (config, algorithm) =
        create_cranelift_algorithm(0, memory, clif_ir.to_string());
    run(config, algorithm).unwrap();
}


#[test]
fn test_cublas_sgemv_on_stream_reuse() {
    let rows: usize = 2;
    let cols: usize = 3;
    let a_elems: usize = rows * cols;
    let x_elems: usize = cols;
    let y_elems: usize = rows;
    let a_bytes: usize = a_elems * 4;
    let x_bytes: usize = x_elems * 4;
    let y_bytes: usize = y_elems * 4;
    let mem_size: usize = 0x0400;

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> i32 system_v
    sig4 = (i64) -> i32 system_v
    sig5 = (i64, i32) -> i32 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload_ptr sig2
    fn3 = %cl_cuda_download_ptr sig2
    fn4 = %cl_cublas_sgemv_on_stream sig3
    fn5 = %cl_cuda_stream_create sig4
    fn6 = %cl_cuda_stream_sync sig5
    fn7 = %cl_cuda_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)
    v91 = load.i64 notrap aligned v0+0

    v10 = iconst.i64 {a_bytes}
    v11 = iconst.i64 {x_bytes}
    v12 = iconst.i64 {y_bytes}
    v13 = call fn1(v91, v10)
    v14 = call fn1(v91, v11)
    v15 = call fn1(v91, v12)
    v16 = call fn2(v91, v13, v1, v10)
    v17 = iadd v1, v10
    v18 = call fn2(v91, v14, v17, v11)
    v19 = call fn5(v91)

    v20 = iconst.i32 1
    v21 = iconst.i32 {cols}
    v22 = iconst.i32 {rows}
    v23 = iconst.i32 0x3f800000
    v24 = iconst.i32 0
    v25 = call fn4(v91, v20, v21, v22, v23, v13, v14, v24, v15, v19)
    v26 = call fn6(v91, v19)
    v27 = call fn3(v91, v15, v2, v12)

    call fn7(v90)
    return
}}"#,
        a_bytes = a_bytes,
        x_bytes = x_bytes,
        y_bytes = y_bytes,
        cols = cols,
        rows = rows,
    );

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![0u8; mem_size],
    };
    let mut base = Base::new(config).unwrap();
    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    let a1: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x1: [f32; 3] = [1.0, 1.0, 1.0];
    let expected1: [f32; 2] = [6.0, 15.0];
    let mut payload1 = Vec::with_capacity(a_bytes + x_bytes);
    for v in a1 {
        payload1.extend_from_slice(&v.to_le_bytes());
    }
    for v in x1 {
        payload1.extend_from_slice(&v.to_le_bytes());
    }
    let mut out1 = vec![0u8; y_bytes];
    base.execute_into(&alg, &payload1, &mut out1).unwrap();
    for (i, expected) in expected1.iter().enumerate() {
        let actual = f32::from_le_bytes(out1[i * 4..i * 4 + 4].try_into().unwrap());
        assert!((actual - expected).abs() < 0.01);
    }

    let a2: [f32; 6] = [-1.0, 0.0, 2.0, 3.0, -2.0, 1.0];
    let x2: [f32; 3] = [2.0, -1.0, 4.0];
    let expected2: [f32; 2] = [6.0, 12.0];
    let mut payload2 = Vec::with_capacity(a_bytes + x_bytes);
    for v in a2 {
        payload2.extend_from_slice(&v.to_le_bytes());
    }
    for v in x2 {
        payload2.extend_from_slice(&v.to_le_bytes());
    }
    let mut out2 = vec![0u8; y_bytes];
    base.execute_into(&alg, &payload2, &mut out2).unwrap();
    for (i, expected) in expected2.iter().enumerate() {
        let actual = f32::from_le_bytes(out2[i * 4..i * 4 + 4].try_into().unwrap());
        assert!((actual - expected).abs() < 0.01);
    }
}

#[test]
fn test_cublas_sgemm_strided_batched_on_stream_reuse() {
    let batch_count: usize = 2;
    let m: usize = 2;
    let k: usize = 3;
    let a_elems: usize = batch_count * m * k;
    let x_elems: usize = batch_count * k;
    let y_elems: usize = batch_count * m;
    let a_bytes: usize = a_elems * 4;
    let x_bytes: usize = x_elems * 4;
    let y_bytes: usize = y_elems * 4;
    let mem_size: usize = 0x0800;

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i32, i32, i32, i32, i32, i32, i32, i64, i32, i64, i32, i32, i64, i32, i32) -> i32 system_v
    sig4 = (i64) -> i32 system_v
    sig5 = (i64, i32) -> i32 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload_ptr sig2
    fn3 = %cl_cuda_download_ptr sig2
    fn4 = %cl_cublas_sgemm_strided_batched_on_stream sig3
    fn5 = %cl_cuda_stream_create sig4
    fn6 = %cl_cuda_stream_sync sig5
    fn7 = %cl_cuda_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)

    v91 = load.i64 notrap aligned v0+0
    v10 = iconst.i64 {a_bytes}
    v11 = iconst.i64 {x_bytes}
    v12 = iconst.i64 {y_bytes}
    v13 = call fn1(v91, v10)
    v14 = call fn1(v91, v11)
    v15 = call fn1(v91, v12)
    v16 = call fn2(v91, v13, v1, v10)
    v17 = iadd v1, v10
    v18 = call fn2(v91, v14, v17, v11)
    v19 = call fn5(v91)

    v20 = iconst.i32 1
    v21 = iconst.i32 0
    v22 = iconst.i32 {m}
    v23 = iconst.i32 1
    v24 = iconst.i32 {k}
    v25 = iconst.i32 0x3f800000
    v26 = iconst.i64 {stride_a}
    v27 = iconst.i64 {stride_b}
    v28 = iconst.i64 {stride_c}
    v29 = iconst.i32 {batch_count}
    v30 = call fn4(v91, v20, v21, v22, v23, v24, v25, v13, v26, v14, v27, v21, v15, v28, v29, v19)

    v31 = call fn6(v91, v19)
    v32 = call fn3(v91, v15, v2, v12)

    call fn7(v90)
    return
}}"#,
        a_bytes = a_bytes,
        x_bytes = x_bytes,
        y_bytes = y_bytes,
        m = m,
        k = k,
        stride_a = m * k,
        stride_b = k,
        stride_c = m,
        batch_count = batch_count,
    );

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![0u8; mem_size],
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    let a1: [f32; 12] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let x1: [f32; 6] = [1.0, 1.0, 1.0, 2.0, 0.0, -1.0];
    let expected1: [f32; 4] = [6.0, 15.0, 5.0, 8.0];

    let mut payload1 = Vec::with_capacity(a_bytes + x_bytes);
    for v in a1 {
        payload1.extend_from_slice(&v.to_le_bytes());
    }
    for v in x1 {
        payload1.extend_from_slice(&v.to_le_bytes());
    }
    let mut out1 = vec![0u8; y_bytes];
    base.execute_into(&alg, &payload1, &mut out1).unwrap();

    for (i, expected) in expected1.iter().enumerate() {
        let actual = f32::from_le_bytes(out1[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - expected).abs() < 0.01,
            "Run 1, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    let a2: [f32; 12] = [
        -1.0, 0.0, 1.0, 2.0, -2.0, 0.5, 3.0, 1.0, -4.0, 0.0, 2.0, 5.0,
    ];
    let x2: [f32; 6] = [3.0, -1.0, 2.0, -2.0, 4.0, 1.0];
    let expected2: [f32; 4] = [-1.0, 9.0, -6.0, 13.0];

    let mut payload2 = Vec::with_capacity(a_bytes + x_bytes);
    for v in a2 {
        payload2.extend_from_slice(&v.to_le_bytes());
    }
    for v in x2 {
        payload2.extend_from_slice(&v.to_le_bytes());
    }
    let mut out2 = vec![0u8; y_bytes];
    base.execute_into(&alg, &payload2, &mut out2).unwrap();

    for (i, expected) in expected2.iter().enumerate() {
        let actual = f32::from_le_bytes(out2[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - expected).abs() < 0.01,
            "Run 2, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_data_ptr_clif_reads_caller_buffer_directly() {
    // CLIF reads data_ptr from offset 8, data_len from offset 16,
    // then loads a value from the caller's buffer via the pointer.
    // This is the zero-copy path — no shared memory copy needed.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    ; load data_ptr from offset 8
    v1 = load.i64 v0+8
    ; load data_len from offset 16
    v2 = load.i64 v0+16
    ; read first i64 from caller's buffer
    v3 = load.i64 v1
    ; read second i64 from caller's buffer (offset 8)
    v4 = load.i64 v1+8
    v5 = iadd v3, v4
    ; store result and row_count
    store v5, v0+200
    store v2, v0+208
    v6 = iconst.i64 1
    store v6, v0+216
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let mut data = vec![0u8; 16];
    data[0..8].copy_from_slice(&100i64.to_le_bytes());
    data[8..16].copy_from_slice(&200i64.to_le_bytes());

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 216,
        columns: vec![
            OutputColumn {
                name: "sum".to_string(),
                dtype: OutputType::I64,
                data_offset: 200,
                len_offset: 0,
            },
            OutputColumn {
                name: "len".to_string(),
                dtype: OutputType::I64,
                data_offset: 208,
                len_offset: 0,
            },
        ],
    }];
    let alg = Algorithm {
        fn_idx: 0,
        output: output_schema,
    };

    let batches = base.execute(&alg, &data).unwrap();
    let sum = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let len = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        sum.value(0),
        300,
        "should read 100+200 from caller buffer via pointer"
    );
    assert_eq!(len.value(0), 16, "data_len should be 16");
}

#[test]
fn test_data_ptr_written_even_when_data_empty() {
    // Offsets 8-16 are always written — even with empty data.
    // Seed those offsets with sentinels to verify they get overwritten.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+16
    store v1, v0+200
    v3 = iconst.i64 1
    store v3, v0+208
    return
}"#
    .to_string();

    let mut initial = vec![0u8; 4096];
    initial[16..24].copy_from_slice(&0xCAFEBABEu64.to_le_bytes());

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: initial,
    };

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "len".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];
    let alg = Algorithm {
        fn_idx: 0,
        output: output_schema,
    };

    let batches = run(config, alg).unwrap();
    let len = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        len.value(0),
        0,
        "data_len should be 0 for empty data, sentinel overwritten"
    );
}

#[test]
fn test_out_ptr_written_even_when_out_empty() {
    // Offsets 24-32 are always written — even with empty out.
    // Seed those offsets with sentinels to verify they get overwritten.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+32
    store v1, v0+200
    v3 = iconst.i64 1
    store v3, v0+208
    return
}"#
    .to_string();

    let mut initial = vec![0u8; 4096];
    initial[32..40].copy_from_slice(&0x22222222u64.to_le_bytes());

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: initial,
    };

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "len".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];
    let alg = Algorithm {
        fn_idx: 0,
        output: output_schema,
    };

    let batches = run(config, alg).unwrap();
    let len = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        len.value(0),
        0,
        "out_len should be 0 for empty out, sentinel overwritten"
    );
}

#[test]
fn test_execute_into_clif_writes_to_caller_out_buffer() {
    // CLIF reads out_ptr from offset 24, writes a computed value into caller's out buffer.
    // This tests the full zero-copy output path.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    ; read data_ptr, load input from caller's data buffer
    v1 = load.i64 v0+8
    v2 = load.i64 v1
    ; compute: input * 7
    v3 = iconst.i64 7
    v4 = imul v2, v3
    ; read out_ptr, write result into caller's out buffer
    v5 = load.i64 v0+24
    store v4, v5
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let mut data = vec![0u8; 8];
    data[0..8].copy_from_slice(&6i64.to_le_bytes());

    let mut out = vec![0u8; 8];

    let alg = Algorithm {
        fn_idx: 0,
        output: vec![],
    };

    base.execute_into(&alg, &data, &mut out).unwrap();
    let result = i64::from_le_bytes(out[0..8].try_into().unwrap());
    assert_eq!(
        result, 42,
        "CLIF should write 6*7=42 into caller's out buffer"
    );
}

#[test]
fn test_execute_into_multiple_calls_different_data() {
    // execute_into called twice with different data and out buffers.
    // Verifies pointers are updated each call.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+8
    v2 = load.i64 v1
    v3 = load.i64 v0+24
    store v2, v3
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        fn_idx: 0,
        output: vec![],
    };

    // Call 1: data=111
    let data1 = 111i64.to_le_bytes().to_vec();
    let mut out1 = vec![0u8; 8];
    base.execute_into(&alg, &data1, &mut out1).unwrap();
    assert_eq!(i64::from_le_bytes(out1[0..8].try_into().unwrap()), 111);

    // Call 2: data=222, different buffers
    let data2 = 222i64.to_le_bytes().to_vec();
    let mut out2 = vec![0u8; 8];
    base.execute_into(&alg, &data2, &mut out2).unwrap();
    assert_eq!(i64::from_le_bytes(out2[0..8].try_into().unwrap()), 222);

    // out1 should be unchanged from call 2
    assert_eq!(i64::from_le_bytes(out1[0..8].try_into().unwrap()), 111);
}

#[test]
fn test_data_ptr_with_large_buffer_no_shared_mem_copy() {
    // Data buffer is larger than memory_size. The data pointer gives CLIF
    // access to the full buffer without copying it into shared memory.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    ; read data_ptr and data_len
    v1 = load.i64 v0+8
    v2 = load.i64 v0+16
    ; read last i64 from caller buffer: data_ptr + data_len - 8
    v3 = iconst.i64 8
    v4 = isub v2, v3
    v5 = iadd v1, v4
    v6 = load.i64 v5
    store v6, v0+200
    store v2, v0+208
    v7 = iconst.i64 1
    store v7, v0+216
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 256,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Data is 1KB — much larger than memory_size (256)
    let mut data = vec![0u8; 1024];
    // Write sentinel at the very end
    data[1016..1024].copy_from_slice(&999i64.to_le_bytes());

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 216,
        columns: vec![
            OutputColumn {
                name: "last_val".to_string(),
                dtype: OutputType::I64,
                data_offset: 200,
                len_offset: 0,
            },
            OutputColumn {
                name: "len".to_string(),
                dtype: OutputType::I64,
                data_offset: 208,
                len_offset: 0,
            },
        ],
    }];
    let alg = Algorithm {
        fn_idx: 0,
        output: output_schema,
    };

    let batches = base.execute(&alg, &data).unwrap();
    let last_val = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let len = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        last_val.value(0),
        999,
        "CLIF should read last value from caller buffer via pointer"
    );
    assert_eq!(len.value(0), 1024, "data_len should be full buffer size");
}

#[test]
fn test_initial_memory_and_data_coexist() {
    // initial_memory sets up static config (e.g., a multiplier at offset 100).
    // data provides dynamic input via pointer.
    // CLIF reads multiplier from shared memory AND input from data pointer.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    ; read static multiplier from shared memory (set by initial_memory)
    v1 = load.i64 v0+100
    ; read dynamic input from data pointer
    v2 = load.i64 v0+8
    v3 = load.i64 v2
    ; multiply
    v4 = imul v1, v3
    store v4, v0+200
    v5 = iconst.i64 1
    store v5, v0+208
    return
}"#
    .to_string();

    let mut initial = vec![0u8; 4096];
    // Static multiplier = 13
    initial[100..108].copy_from_slice(&13i64.to_le_bytes());

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: initial,
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "product".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];
    let alg = Algorithm {
        fn_idx: 0,
        output: output_schema,
    };

    // Dynamic input = 7
    let data = 7i64.to_le_bytes().to_vec();
    let batches = base.execute(&alg, &data).unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        col.value(0),
        91,
        "13 * 7 = 91: static config from initial_memory, dynamic input via pointer"
    );
}

#[test]
fn test_execute_into_out_buffer_larger_than_memory() {
    // Out buffer can be any size — it's caller-owned, not bounded by memory_size.
    // CLIF writes multiple values into a large out buffer.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+24
    v2 = load.i64 v0+32
    ; write values at out[0], out[8], out[16]
    v3 = iconst.i64 100
    store v3, v1
    v4 = iconst.i64 200
    store v4, v1+8
    v5 = iconst.i64 300
    store v5, v1+16
    ; write out_len at the end for verification
    store v2, v1+24
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 64,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        fn_idx: 0,
        output: vec![],
    };

    // Tiny shared memory (64 bytes) but large out buffer
    let data = vec![0u8; 8]; // need non-empty data so pointers at 8-16 get written, but we need out ptrs
    let mut out = vec![0u8; 32];
    base.execute_into(&alg, &data, &mut out).unwrap();

    let v0 = i64::from_le_bytes(out[0..8].try_into().unwrap());
    let v1 = i64::from_le_bytes(out[8..16].try_into().unwrap());
    let v2 = i64::from_le_bytes(out[16..24].try_into().unwrap());
    let v3 = i64::from_le_bytes(out[24..32].try_into().unwrap());
    assert_eq!(v0, 100);
    assert_eq!(v1, 200);
    assert_eq!(v2, 300);
    assert_eq!(v3, 32, "out_len should be 32");
}

#[test]
fn test_run_with_data_argument() {
    // The standalone run() function also accepts data.
    // Verify the pointer path works through the simple API.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+8
    v2 = load.i64 v1
    store v2, v0+200
    v3 = iconst.i64 1
    store v3, v0+208
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "val".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];
    let alg = Algorithm {
        fn_idx: 0,
        output: output_schema,
    };

    let data = 777i64.to_le_bytes().to_vec();
    let mut base = Base::new(config).unwrap();
    let batches = base.execute(&alg, &data).unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        col.value(0),
        777,
        "execute() should pass data pointer through to CLIF"
    );
}

#[test]
fn test_data_single_byte_still_writes_pointer() {
    // Even a 1-byte data buffer should write the pointer.
    // Edge case: smallest possible non-empty data.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+16
    store v1, v0+200
    v2 = iconst.i64 1
    store v2, v0+208
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![OutputColumn {
            name: "len".to_string(),
            dtype: OutputType::I64,
            data_offset: 200,
            len_offset: 0,
        }],
    }];
    let alg = Algorithm {
        fn_idx: 0,
        output: output_schema,
    };

    let data = vec![42u8]; // single byte
    let mut base = Base::new(config).unwrap();
    let batches = base.execute(&alg, &data).unwrap();
    let col = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(col.value(0), 1, "data_len should be 1 for single-byte data");
}

#[test]
fn test_data_ptr_survives_across_multi_execute() {
    // Multiple execute calls with data — each call gets fresh pointers.
    // Verify that stale pointers from previous calls don't leak.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+8
    v2 = load.i64 v1
    v3 = load.i64 v0+16
    store v2, v0+200
    store v3, v0+208
    v4 = iconst.i64 1
    store v4, v0+216
    return
}"#
    .to_string();

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 216,
        columns: vec![
            OutputColumn {
                name: "val".to_string(),
                dtype: OutputType::I64,
                data_offset: 200,
                len_offset: 0,
            },
            OutputColumn {
                name: "len".to_string(),
                dtype: OutputType::I64,
                data_offset: 208,
                len_offset: 0,
            },
        ],
    }];
    let alg = Algorithm {
        fn_idx: 0,
        output: output_schema,
    };

    // Call 1: 8-byte buffer
    let data1 = 11i64.to_le_bytes().to_vec();
    let b1 = base.execute(&alg, &data1).unwrap();
    let v1 = b1[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let l1 = b1[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(v1.value(0), 11);
    assert_eq!(l1.value(0), 8);

    // Call 2: 16-byte buffer (different size!)
    let mut data2 = vec![0u8; 16];
    data2[0..8].copy_from_slice(&22i64.to_le_bytes());
    let b2 = base.execute(&alg, &data2).unwrap();
    let v2 = b2[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let l2 = b2[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(v2.value(0), 22);
    assert_eq!(l2.value(0), 16, "data_len should reflect new buffer size");
}

#[test]
fn test_gpu_upload_ptr_download_ptr_vecadd() {
    // Tests cl_gpu_upload_ptr and cl_gpu_download_ptr via execute_into:
    // uploads A+B from caller's data pointer, computes C[i]=A[i]+B[i] on GPU,
    // downloads C to caller's out pointer. No shared memory data copying.
    let n: usize = 64;

    let wgsl = "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
                @compute @workgroup_size(64)\n\
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                    let n = arrayLength(&data) / 2u;\n\
                    let i = gid.x;\n\
                    if (i >= n) { return; }\n\
                    data[i] = data[i] + data[n + i];\n\
                }\n";

    // Memory layout:
    //   0x0000  reserved (40 bytes)
    //   0x0100  WGSL shader (null-terminated)
    //   0x1100  bind descriptor (8 bytes: [buf_id=0, read_only=0])
    let shader_off: usize = 0x0100;
    let bind_off: usize = 0x1100;
    let mem_size: usize = 0x1200;

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i64, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i64) -> i32 system_v
    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v
    sig5 = (i64, i32, i64, i64, i64) -> i32 system_v

    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_create_pipeline sig2
    fn3 = %cl_gpu_upload_ptr sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download_ptr sig5
    fn6 = %cl_gpu_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x10
    v3 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)

    v91 = load.i64 notrap aligned v0+0
    v4 = call fn1(v91, v2)
    v5 = call fn3(v91, v4, v1, v2)
    v6 = iadd_imm v0, {shader_off}
    v7 = iadd_imm v0, {bind_off}
    v8 = iconst.i32 1
    v9 = call fn2(v91, v6, v7, v8)
    v10 = call fn4(v91, v9, v8, v8, v8)
    v11 = ushr_imm v2, 3
    v12 = ishl_imm v11, 2
    v13 = iconst.i64 0
    v14 = call fn5(v91, v4, v13, v3, v12)
    call fn6(v90)
    return
}}"#,
        shader_off = shader_off,
        bind_off = bind_off,
    );

    let mut memory = vec![0u8; mem_size];
    let shader_bytes = wgsl.as_bytes();
    memory[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    memory[shader_off + shader_bytes.len()] = 0;
    // bind desc: buf_id=0, read_only=0
    memory[bind_off..bind_off + 8].copy_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: memory,
    };
    let mut base = Base::new(config).unwrap();

    // Build payload: [A: 64 f32s][B: 64 f32s]
    let mut payload = vec![0u8; n * 4 * 2];
    for i in 0..n {
        let a_val = (i + 1) as f32;
        let b_val = 100.0f32;
        payload[i * 4..i * 4 + 4].copy_from_slice(&a_val.to_le_bytes());
        payload[n * 4 + i * 4..n * 4 + i * 4 + 4].copy_from_slice(&b_val.to_le_bytes());
    }

    let mut out = vec![0u8; n * 4];
    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    base.execute_into(&alg, &payload, &mut out).unwrap();

    for i in 0..n {
        let actual = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 + 100.0;
        assert!(
            (actual - expected).abs() < 0.01,
            "Element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_gpu_download_ptr_with_offset() {
    // Tests cl_gpu_download_ptr with a non-zero buf_offset.
    // Allocates a buffer with [A: 64 floats][B: 64 floats], uploads both,
    // then downloads only the B portion (offset = 64*4) to the out pointer.
    let n: usize = 64;

    // Shader does nothing — we just want to test upload + offset download
    let wgsl = "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
                @compute @workgroup_size(64)\n\
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                }\n";

    let shader_off: usize = 0x0100;
    let bind_off: usize = 0x1100;
    let mem_size: usize = 0x1200;

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i32, i64, i64, i64) -> i32 system_v

    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload_ptr sig2
    fn3 = %cl_gpu_download_ptr sig3
    fn4 = %cl_gpu_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x10
    v3 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)

    v91 = load.i64 notrap aligned v0+0
    ; create buffer for full data (2*64*4 = 512 bytes)
    v4 = call fn1(v91, v2)
    ; upload all data from payload
    v5 = call fn2(v91, v4, v1, v2)
    ; download only second half: buf_offset = 256, size = 256, to out_ptr
    v6 = iconst.i64 {half}
    v7 = call fn3(v91, v4, v6, v3, v6)
    call fn4(v90)
    return
}}"#,
        half = n * 4,
    );

    let mut memory = vec![0u8; mem_size];
    let shader_bytes = wgsl.as_bytes();
    memory[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    memory[shader_off + shader_bytes.len()] = 0;
    memory[bind_off..bind_off + 8].copy_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: memory,
    };
    let mut base = Base::new(config).unwrap();

    // Payload: [A: 1.0..64.0][B: 101.0..164.0]
    let mut payload = vec![0u8; n * 4 * 2];
    for i in 0..n {
        let a_val = (i + 1) as f32;
        let b_val = (i + 101) as f32;
        payload[i * 4..i * 4 + 4].copy_from_slice(&a_val.to_le_bytes());
        payload[n * 4 + i * 4..n * 4 + i * 4 + 4].copy_from_slice(&b_val.to_le_bytes());
    }

    let mut out = vec![0u8; n * 4];
    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    base.execute_into(&alg, &payload, &mut out).unwrap();

    // out should contain the B values (101.0..164.0), not A values
    for i in 0..n {
        let actual = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 101) as f32;
        assert!(
            (actual - expected).abs() < 0.01,
            "Element {}: expected {} (B region), got {} — buf_offset download may be broken",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_cuda_upload_ptr_download_ptr_vecadd() {
    // Tests cl_cuda_upload_ptr and cl_cuda_download_ptr with execute_into.
    // Uploads A+B from caller's data pointer via PTX kernel C[i]=A[i]+B[i],
    // downloads C to caller's out pointer. No shared memory data copying.
    let n: usize = 64;
    let data_bytes: usize = n * 4;

    let ptx = ".version 7.0\n\
               .target sm_50\n\
               .address_size 64\n\
               \n\
               .visible .entry main(\n\
                   .param .u64 a_ptr,\n\
                   .param .u64 b_ptr,\n\
                   .param .u64 c_ptr\n\
               )\n\
               {\n\
                   .reg .u32 %r0;\n\
                   .reg .u64 %ra, %rb, %rc, %off;\n\
                   .reg .f32 %fa, %fb, %fr;\n\
               \n\
                   mov.u32 %r0, %tid.x;\n\
                   cvt.u64.u32 %off, %r0;\n\
                   shl.b64 %off, %off, 2;\n\
               \n\
                   ld.param.u64 %ra, [a_ptr];\n\
                   ld.param.u64 %rb, [b_ptr];\n\
                   ld.param.u64 %rc, [c_ptr];\n\
               \n\
                   add.u64 %ra, %ra, %off;\n\
                   add.u64 %rb, %rb, %off;\n\
                   add.u64 %rc, %rc, %off;\n\
               \n\
                   ld.global.f32 %fa, [%ra];\n\
                   ld.global.f32 %fb, [%rb];\n\
                   add.f32 %fr, %fa, %fb;\n\
                   st.global.f32 [%rc], %fr;\n\
               \n\
                   ret;\n\
               }\n\0";

    // Memory layout:
    //   0x0000  reserved (40 bytes)
    //   0x0100  PTX source (null-terminated)
    //   0x1100  bind descriptor (12 bytes: 3 × i32 buf_ids = [0, 1, 2])
    let ptx_off: usize = 0x0100;
    let bind_off: usize = 0x1100;
    let mem_size: usize = 0x1200;

    // CLIF: uses cl_cuda_upload_ptr / cl_cuda_download_ptr with payload pointers
    // sig for upload_ptr/download_ptr: (ptr: i64, buf_id: i32, abs_ptr: i64, size: i64) -> i32
    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload_ptr sig2
    fn3 = %cl_cuda_download_ptr sig2
    fn4 = %cl_cuda_launch sig3
    fn5 = %cl_cuda_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x10
    v3 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)

    v91 = load.i64 notrap aligned v0+0

    ; create 3 buffers of {data_bytes} bytes each
    v4 = iconst.i64 {data_bytes}
    v5 = call fn1(v91, v4)
    v6 = call fn1(v91, v4)
    v7 = call fn1(v91, v4)

    ; upload A from data_ptr
    v8 = call fn2(v91, v5, v1, v4)

    ; upload B from data_ptr + data_bytes
    v9 = iadd v1, v4
    v10 = call fn2(v91, v6, v9, v4)

    ; launch PTX kernel: grid(1,1,1) block(64,1,1)
    v11 = iadd_imm v0, {ptx_off}
    v12 = iconst.i32 3
    v13 = iadd_imm v0, {bind_off}
    v14 = iconst.i32 1
    v15 = iconst.i32 64
    v16 = call fn4(v91, v11, v12, v13, v14, v14, v14, v15, v14, v14)

    ; download result from buf 2 to out_ptr
    v17 = call fn3(v91, v7, v3, v4)

    call fn5(v90)
    return
}}"#,
        data_bytes = data_bytes,
        ptx_off = ptx_off,
        bind_off = bind_off,
    );

    let mut memory = vec![0u8; mem_size];
    let ptx_bytes = ptx.as_bytes();
    memory[ptx_off..ptx_off + ptx_bytes.len()].copy_from_slice(ptx_bytes);
    // bind desc: buf_id=0 (A), buf_id=1 (B), buf_id=2 (C)
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    memory[bind_off + 4..bind_off + 8].copy_from_slice(&1i32.to_le_bytes());
    memory[bind_off + 8..bind_off + 12].copy_from_slice(&2i32.to_le_bytes());

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: memory,
    };
    let mut base = Base::new(config).unwrap();

    // Build payload: [A: 64 f32s][B: 64 f32s]
    let mut payload = vec![0u8; n * 4 * 2];
    for i in 0..n {
        let a_val = (i + 1) as f32;
        let b_val = 100.0f32;
        payload[i * 4..i * 4 + 4].copy_from_slice(&a_val.to_le_bytes());
        payload[n * 4 + i * 4..n * 4 + i * 4 + 4].copy_from_slice(&b_val.to_le_bytes());
    }

    let mut out = vec![0u8; n * 4];
    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    base.execute_into(&alg, &payload, &mut out).unwrap();

    for i in 0..n {
        let actual = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 + 100.0;
        assert!(
            (actual - expected).abs() < 0.01,
            "Element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_cuda_download_ptr_different_data() {
    // Tests cl_cuda_upload_ptr and cl_cuda_download_ptr with two different payloads.
    // First execute: uploads A=[1..64] + B=[100..100], expects C=[101..164].
    // Second execute: uploads A=[200..200] + B=[1..64], expects C=[201..264].
    // Verifies the _ptr functions work correctly across multiple execute_into calls.
    let n: usize = 64;
    let data_bytes: usize = n * 4;

    // Simple PTX: C[i] = A[i] + B[i], 2 buffers in-place on buf 0, result in buf 1
    let ptx = ".version 7.0\n\
               .target sm_50\n\
               .address_size 64\n\
               \n\
               .visible .entry main(\n\
                   .param .u64 a_ptr,\n\
                   .param .u64 b_ptr\n\
               )\n\
               {\n\
                   .reg .u32 %r0;\n\
                   .reg .u64 %ra, %rb, %off;\n\
                   .reg .f32 %fa, %fb, %fr;\n\
               \n\
                   mov.u32 %r0, %tid.x;\n\
                   cvt.u64.u32 %off, %r0;\n\
                   shl.b64 %off, %off, 2;\n\
               \n\
                   ld.param.u64 %ra, [a_ptr];\n\
                   ld.param.u64 %rb, [b_ptr];\n\
               \n\
                   add.u64 %ra, %ra, %off;\n\
                   add.u64 %rb, %rb, %off;\n\
               \n\
                   ld.global.f32 %fa, [%ra];\n\
                   ld.global.f32 %fb, [%rb];\n\
                   add.f32 %fr, %fa, %fb;\n\
                   st.global.f32 [%rb], %fr;\n\
               \n\
                   ret;\n\
               }\n\0";

    let ptx_off: usize = 0x0100;
    let bind_off: usize = 0x1100;
    let mem_size: usize = 0x1200;

    // CLIF: upload A to buf 0, B to buf 1, launch, download buf 1 to out_ptr
    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload_ptr sig2
    fn3 = %cl_cuda_download_ptr sig2
    fn4 = %cl_cuda_launch sig3
    fn5 = %cl_cuda_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x10
    v3 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)

    v91 = load.i64 notrap aligned v0+0

    ; create 2 buffers
    v4 = iconst.i64 {data_bytes}
    v5 = call fn1(v91, v4)
    v6 = call fn1(v91, v4)

    ; upload A from data_ptr to buf 0
    v7 = call fn2(v91, v5, v1, v4)

    ; upload B from data_ptr + data_bytes to buf 1
    v8 = iadd v1, v4
    v9 = call fn2(v91, v6, v8, v4)

    ; launch: grid(1,1,1) block(64,1,1)
    v10 = iadd_imm v0, {ptx_off}
    v11 = iconst.i32 2
    v12 = iadd_imm v0, {bind_off}
    v13 = iconst.i32 1
    v14 = iconst.i32 64
    v15 = call fn4(v91, v10, v11, v12, v13, v13, v13, v14, v13, v13)

    ; download buf 1 (result) to out_ptr
    v16 = call fn3(v91, v6, v3, v4)

    call fn5(v90)
    return
}}"#,
        data_bytes = data_bytes,
        ptx_off = ptx_off,
        bind_off = bind_off,
    );

    let mut memory = vec![0u8; mem_size];
    let ptx_bytes = ptx.as_bytes();
    memory[ptx_off..ptx_off + ptx_bytes.len()].copy_from_slice(ptx_bytes);
    // bind desc: buf_id=0 (A), buf_id=1 (B)
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    memory[bind_off + 4..bind_off + 8].copy_from_slice(&1i32.to_le_bytes());

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: memory,
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    // First execute: A=[1..64], B=[100..100]
    let mut payload1 = vec![0u8; n * 4 * 2];
    for i in 0..n {
        let a_val = (i + 1) as f32;
        let b_val = 100.0f32;
        payload1[i * 4..i * 4 + 4].copy_from_slice(&a_val.to_le_bytes());
        payload1[n * 4 + i * 4..n * 4 + i * 4 + 4].copy_from_slice(&b_val.to_le_bytes());
    }
    let mut out1 = vec![0u8; n * 4];
    base.execute_into(&alg, &payload1, &mut out1).unwrap();

    for i in 0..n {
        let actual = f32::from_le_bytes(out1[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 + 100.0;
        assert!(
            (actual - expected).abs() < 0.01,
            "Run 1, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    // Second execute: A=[200..200], B=[1..64]
    let mut payload2 = vec![0u8; n * 4 * 2];
    for i in 0..n {
        let a_val = 200.0f32;
        let b_val = (i + 1) as f32;
        payload2[i * 4..i * 4 + 4].copy_from_slice(&a_val.to_le_bytes());
        payload2[n * 4 + i * 4..n * 4 + i * 4 + 4].copy_from_slice(&b_val.to_le_bytes());
    }
    let mut out2 = vec![0u8; n * 4];
    base.execute_into(&alg, &payload2, &mut out2).unwrap();

    for i in 0..n {
        let actual = f32::from_le_bytes(out2[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = 200.0 + (i + 1) as f32;
        assert!(
            (actual - expected).abs() < 0.01,
            "Run 2, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_cublas_sgemm_strided_batched_reuse() {
    // Exercises the cl_cublas_sgemm_strided_batched FFI directly with a small
    // batched GEMV-shaped workload:
    //   for each batch i: y_i = A_i @ x_i
    // where A_i is 2x3 row-major and x_i is length-3.
    //
    // Reuses the same Base instance across two execute_into calls to ensure the
    // wrapper behaves correctly across repeated executions.
    let batch_count: usize = 2;
    let m: usize = 2;
    let k: usize = 3;
    let a_elems: usize = batch_count * m * k;
    let x_elems: usize = batch_count * k;
    let y_elems: usize = batch_count * m;
    let a_bytes: usize = a_elems * 4;
    let x_bytes: usize = x_elems * 4;
    let y_bytes: usize = y_elems * 4;

    let mem_size: usize = 0x0800;

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i32, i32, i32, i32, i32, i32, i32, i64, i32, i64, i32, i32, i64, i32) -> i32 system_v
    sig4 = (i64) -> i32 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload_ptr sig2
    fn3 = %cl_cuda_download_ptr sig2
    fn4 = %cl_cublas_sgemm_strided_batched sig3
    fn5 = %cl_cuda_sync sig4
    fn6 = %cl_cuda_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)

    v91 = load.i64 notrap aligned v0+0

    ; create A, x, y buffers
    v10 = iconst.i64 {a_bytes}
    v11 = iconst.i64 {x_bytes}
    v12 = iconst.i64 {y_bytes}
    v13 = call fn1(v91, v10)
    v14 = call fn1(v91, v11)
    v15 = call fn1(v91, v12)

    ; upload A from data_ptr
    v16 = call fn2(v91, v13, v1, v10)

    ; upload x from data_ptr + a_bytes
    v17 = iadd v1, v10
    v18 = call fn2(v91, v14, v17, v11)

    ; batched GEMV using SGEMM-strided-batched
    ; row-major A (2x3) => transa=1, transb=0, m=2, n=1, k=3
    ; stride_a=6, stride_b=3, stride_c=2 elements, batch_count=2
    v20 = iconst.i32 1
    v21 = iconst.i32 0
    v22 = iconst.i32 {m}
    v23 = iconst.i32 1
    v24 = iconst.i32 {k}
    v25 = iconst.i32 0x3f800000
    v26 = iconst.i64 {stride_a}
    v27 = iconst.i64 {stride_b}
    v28 = iconst.i64 {stride_c}
    v29 = iconst.i32 {batch_count}
    v30 = call fn4(v91, v20, v21, v22, v23, v24, v25, v13, v26, v14, v27, v21, v15, v28, v29)

    v31 = call fn5(v91)
    v32 = call fn3(v91, v15, v2, v12)

    call fn6(v90)
    return
}}"#,
        a_bytes = a_bytes,
        x_bytes = x_bytes,
        y_bytes = y_bytes,
        m = m,
        k = k,
        stride_a = m * k,
        stride_b = k,
        stride_c = m,
        batch_count = batch_count,
    );

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![0u8; mem_size],
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    let a1: [f32; 12] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let x1: [f32; 6] = [1.0, 1.0, 1.0, 2.0, 0.0, -1.0];
    let expected1: [f32; 4] = [6.0, 15.0, 5.0, 8.0];

    let mut payload1 = Vec::with_capacity(a_bytes + x_bytes);
    for v in a1 {
        payload1.extend_from_slice(&v.to_le_bytes());
    }
    for v in x1 {
        payload1.extend_from_slice(&v.to_le_bytes());
    }
    let mut out1 = vec![0u8; y_bytes];
    base.execute_into(&alg, &payload1, &mut out1).unwrap();

    for (i, expected) in expected1.iter().enumerate() {
        let actual = f32::from_le_bytes(out1[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - expected).abs() < 0.01,
            "Run 1, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    let a2: [f32; 12] = [
        -1.0, 0.0, 1.0, 2.0, -2.0, 0.5, 3.0, 1.0, -4.0, 0.0, 2.0, 5.0,
    ];
    let x2: [f32; 6] = [3.0, -1.0, 2.0, -2.0, 4.0, 1.0];
    let expected2: [f32; 4] = [-1.0, 9.0, -6.0, 13.0];

    let mut payload2 = Vec::with_capacity(a_bytes + x_bytes);
    for v in a2 {
        payload2.extend_from_slice(&v.to_le_bytes());
    }
    for v in x2 {
        payload2.extend_from_slice(&v.to_le_bytes());
    }
    let mut out2 = vec![0u8; y_bytes];
    base.execute_into(&alg, &payload2, &mut out2).unwrap();

    for (i, expected) in expected2.iter().enumerate() {
        let actual = f32::from_le_bytes(out2[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - expected).abs() < 0.01,
            "Run 2, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_cuda_upload_ptr_offset_reuse() {
    // Verifies cl_cuda_upload_ptr_offset can update a subrange of an existing
    // device buffer across repeated execute_into calls.
    let total_bytes: usize = 16; // 4 f32s
    let mem_size: usize = 0x0400;

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64, i64) -> i32 system_v
    sig3 = (i64, i32, i64, i64) -> i32 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload_ptr_offset sig2
    fn3 = %cl_cuda_download_ptr sig3
    fn4 = %cl_cuda_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)

    v91 = load.i64 notrap aligned v0+0

    v10 = iconst.i64 {total_bytes}
    v11 = call fn1(v91, v10)

    ; upload first 2 floats to offset 0
    v12 = iconst.i64 8
    v13 = iconst.i64 0
    v14 = call fn2(v91, v11, v13, v1, v12)

    ; upload second 2 floats to offset 8
    v15 = iadd v1, v12
    v16 = call fn2(v91, v11, v12, v15, v12)

    ; download full 4-float buffer
    v17 = call fn3(v91, v11, v2, v10)

    call fn4(v90)
    return
}}"#,
        total_bytes = total_bytes,
    );

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![0u8; mem_size],
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    let payload1: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut bytes1 = Vec::with_capacity(total_bytes);
    for v in payload1 {
        bytes1.extend_from_slice(&v.to_le_bytes());
    }
    let mut out1 = vec![0u8; total_bytes];
    base.execute_into(&alg, &bytes1, &mut out1).unwrap();
    for (i, expected) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
        let actual = f32::from_le_bytes(out1[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - expected).abs() < 0.001,
            "Run 1, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    let payload2: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
    let mut bytes2 = Vec::with_capacity(total_bytes);
    for v in payload2 {
        bytes2.extend_from_slice(&v.to_le_bytes());
    }
    let mut out2 = vec![0u8; total_bytes];
    base.execute_into(&alg, &bytes2, &mut out2).unwrap();
    for (i, expected) in [10.0f32, 20.0, 30.0, 40.0].iter().enumerate() {
        let actual = f32::from_le_bytes(out2[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - expected).abs() < 0.001,
            "Run 2, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_cuda_launch_named_reuses_named_kernel() {
    // Verifies cl_cuda_launch_named can launch a non-"main" entry point and
    // that repeated named launches in the same CUDA context remain correct.
    let n: usize = 8;
    let data_bytes: usize = n * 4;
    let ptx_off: usize = 0x0100;
    let name_off: usize = 0x0600;
    let bind_off: usize = 0x0700;
    let mem_size: usize = 0x0800;

    let ptx = ".version 7.0\n\
               .target sm_50\n\
               .address_size 64\n\
               \n\
               .visible .entry add_one(\n\
                   .param .u64 data_ptr\n\
               )\n\
               {\n\
                   .reg .u32 %r0;\n\
                   .reg .u64 %rd, %off;\n\
                   .reg .f32 %fv, %fc;\n\
                   mov.u32 %r0, %tid.x;\n\
                   cvt.u64.u32 %off, %r0;\n\
                   shl.b64 %off, %off, 2;\n\
                   ld.param.u64 %rd, [data_ptr];\n\
                   add.u64 %rd, %rd, %off;\n\
                   ld.global.f32 %fv, [%rd];\n\
                   mov.f32 %fc, 0f3F800000;\n\
                   add.f32 %fv, %fv, %fc;\n\
                   st.global.f32 [%rd], %fv;\n\
                   ret;\n\
               }\n\0";

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload_ptr sig2
    fn3 = %cl_cuda_download_ptr sig2
    fn4 = %cl_cuda_launch_named sig3
    fn5 = %cl_cuda_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)

    v91 = load.i64 notrap aligned v0+0

    v10 = iconst.i64 {data_bytes}
    v11 = call fn1(v91, v10)
    v12 = call fn2(v91, v11, v1, v10)

    ; launch named kernel twice: x -> x+1 -> x+2
    v13 = iadd_imm v0, {ptx_off}
    v14 = iadd_imm v0, {name_off}
    v15 = iconst.i32 1
    v16 = iadd_imm v0, {bind_off}
    v17 = iconst.i32 1
    v18 = iconst.i32 {n}
    v19 = call fn4(v91, v13, v14, v15, v16, v17, v17, v17, v18, v17, v17)
    v20 = call fn4(v91, v13, v14, v15, v16, v17, v17, v17, v18, v17, v17)

    v21 = call fn3(v91, v11, v2, v10)

    call fn5(v90)
    return
}}"#,
        data_bytes = data_bytes,
        ptx_off = ptx_off,
        name_off = name_off,
        bind_off = bind_off,
        n = n,
    );

    let mut memory = vec![0u8; mem_size];
    memory[ptx_off..ptx_off + ptx.len()].copy_from_slice(ptx.as_bytes());
    memory[name_off..name_off + 8].copy_from_slice(b"add_one\0");
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: memory,
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    let payload1: Vec<f32> = (1..=n).map(|x| x as f32).collect();
    let mut bytes1 = Vec::with_capacity(data_bytes);
    for v in &payload1 {
        bytes1.extend_from_slice(&v.to_le_bytes());
    }
    let mut out1 = vec![0u8; data_bytes];
    base.execute_into(&alg, &bytes1, &mut out1).unwrap();
    for (i, input) in payload1.iter().enumerate() {
        let actual = f32::from_le_bytes(out1[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = *input + 2.0;
        assert!(
            (actual - expected).abs() < 0.001,
            "Run 1, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    let payload2: Vec<f32> = vec![5.0; n];
    let mut bytes2 = Vec::with_capacity(data_bytes);
    for v in &payload2 {
        bytes2.extend_from_slice(&v.to_le_bytes());
    }
    let mut out2 = vec![0u8; data_bytes];
    base.execute_into(&alg, &bytes2, &mut out2).unwrap();
    for i in 0..n {
        let actual = f32::from_le_bytes(out2[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - 7.0).abs() < 0.001,
            "Run 2, element {}: expected 7.0, got {}",
            i,
            actual
        );
    }
}

#[test]
fn test_cublas_sgemv_reuse() {
    // Directly exercises cl_cublas_sgemv with a small row-major 2x3 matrix.
    let rows: usize = 2;
    let cols: usize = 3;
    let a_elems: usize = rows * cols;
    let x_elems: usize = cols;
    let y_elems: usize = rows;
    let a_bytes: usize = a_elems * 4;
    let x_bytes: usize = x_elems * 4;
    let y_bytes: usize = y_elems * 4;
    let mem_size: usize = 0x0400;

    let clif_ir = format!(
        r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i32, i32, i32, i32, i32, i32, i32, i32) -> i32 system_v
    sig4 = (i64) -> i32 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload_ptr sig2
    fn3 = %cl_cuda_download_ptr sig2
    fn4 = %cl_cublas_sgemv sig3
    fn5 = %cl_cuda_sync sig4
    fn6 = %cl_cuda_cleanup sig0

block0(v0: i64):
    v1 = load.i64 notrap aligned v0+0x08
    v2 = load.i64 notrap aligned v0+0x18
    v90 = iadd_imm v0, 0
    call fn0(v90)

    v91 = load.i64 notrap aligned v0+0

    v10 = iconst.i64 {a_bytes}
    v11 = iconst.i64 {x_bytes}
    v12 = iconst.i64 {y_bytes}
    v13 = call fn1(v91, v10)
    v14 = call fn1(v91, v11)
    v15 = call fn1(v91, v12)

    v16 = call fn2(v91, v13, v1, v10)
    v17 = iadd v1, v10
    v18 = call fn2(v91, v14, v17, v11)

    ; row-major A[rows, cols] -> sgemv(trans=1, m=cols, n=rows)
    v19 = iconst.i32 1
    v20 = iconst.i32 {cols}
    v21 = iconst.i32 {rows}
    v22 = iconst.i32 0x3f800000
    v23 = iconst.i32 0
    v24 = call fn4(v91, v19, v20, v21, v22, v13, v14, v23, v15)

    v25 = call fn5(v91)
    v26 = call fn3(v91, v15, v2, v12)

    call fn6(v90)
    return
}}"#,
        a_bytes = a_bytes,
        x_bytes = x_bytes,
        y_bytes = y_bytes,
        cols = cols,
        rows = rows,
    );

    let config = Setup {
        cranelift_ir: clif_ir,
        memory_size: mem_size,
        runtime_header: legacy_runtime_header(),
        initial_memory: vec![0u8; mem_size],
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        fn_idx: 1,
        output: vec![],
    };

    let a1: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x1: [f32; 3] = [1.0, 1.0, 1.0];
    let expected1: [f32; 2] = [6.0, 15.0];
    let mut payload1 = Vec::with_capacity(a_bytes + x_bytes);
    for v in a1 {
        payload1.extend_from_slice(&v.to_le_bytes());
    }
    for v in x1 {
        payload1.extend_from_slice(&v.to_le_bytes());
    }
    let mut out1 = vec![0u8; y_bytes];
    base.execute_into(&alg, &payload1, &mut out1).unwrap();
    for (i, expected) in expected1.iter().enumerate() {
        let actual = f32::from_le_bytes(out1[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - expected).abs() < 0.01,
            "Run 1, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    let a2: [f32; 6] = [-1.0, 0.0, 2.0, 3.0, -2.0, 1.0];
    let x2: [f32; 3] = [2.0, -1.0, 4.0];
    let expected2: [f32; 2] = [6.0, 12.0];
    let mut payload2 = Vec::with_capacity(a_bytes + x_bytes);
    for v in a2 {
        payload2.extend_from_slice(&v.to_le_bytes());
    }
    for v in x2 {
        payload2.extend_from_slice(&v.to_le_bytes());
    }
    let mut out2 = vec![0u8; y_bytes];
    base.execute_into(&alg, &payload2, &mut out2).unwrap();
    for (i, expected) in expected2.iter().enumerate() {
        let actual = f32::from_le_bytes(out2[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - expected).abs() < 0.01,
            "Run 2, element {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

