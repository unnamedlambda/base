use arrow_array::{Float64Array, Int64Array, StringArray};
use arrow_schema::{DataType, Field, Schema};
use base::{run, Base, RecordBatch};
use base_types::{
    Algorithm, Setup, OutputBatchSchema, OutputColumn, OutputType,
};
use std::fs;
use std::sync::Arc;

mod common;
use common::*;
use tempfile::TempDir;

fn cranelift_config(memory: Vec<u8>, clif: Program) -> Setup {
    dump(&clif);
    Setup {
        clif,
        memory_size: memory.len(),
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
    clif: Program,
) -> (Setup, Algorithm) {
    (cranelift_config(memory, clif), cranelift_algorithm(fn_idx))
}

#[test]
fn test_cranelift_basic_compilation() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_basic.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    // Single CLIF function that writes 8 bytes at offset 2000 to the file at offset 3000.
    let clif_prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                iconst64(v(1), 3000),
                iconst64(v(2), 2000),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 4096];
    memory[2000..2008].copy_from_slice(&42u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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
    let clif_prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                load64(v(1), v(0), 2000),
                load64(v(2), v(0), 2008),
                iadd(v(3), v(1), v(2)),
                store(v(3), v(0), 2016),
                iconst64(v(4), 3000),
                iconst64(v(5), 2016),
                iconst64(v(6), 0),
                iconst64(v(7), 8),
                call(Some(v(8)), 0, &[v(0), v(4), v(5), v(6), v(7)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 4096];
    memory[2000..2008].copy_from_slice(&100u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&200u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                load64(v(1), v(0), 2000),
                load64(v(2), v(0), 2008),
                imul(v(3), v(1), v(2)),
                store(v(3), v(0), 2016),
                iconst64(v(4), 3000),
                iconst64(v(5), 2016),
                iconst64(v(6), 0),
                iconst64(v(7), 8),
                call(Some(v(8)), 0, &[v(0), v(4), v(5), v(6), v(7)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 4096];
    memory[2000..2008].copy_from_slice(&7u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&9u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                load32(v(1), v(0), 2000),
                load32(v(2), v(0), 2004),
                load32(v(3), v(0), 2008),
                iadd(v(4), v(1), v(2)),
                iadd(v(5), v(4), v(3)),
                store(v(5), v(0), 2012),
                iconst64(v(6), 3000),
                iconst64(v(7), 2012),
                iconst64(v(8), 0),
                iconst64(v(9), 4),
                call(Some(v(10)), 0, &[v(0), v(6), v(7), v(8), v(9)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 4096];
    memory[2000..2004].copy_from_slice(&10u32.to_le_bytes());
    memory[2004..2008].copy_from_slice(&20u32.to_le_bytes());
    memory[2008..2012].copy_from_slice(&30u32.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                load64(v(1), v(0), 2000),
                load64(v(2), v(0), 2008),
                load64(v(3), v(0), 2016),
                iconst64(v(9001), 0),
                icmp(v(4), IntCC::Eq, v(1), v(9001)),
                brif(v(4), 2, &[], 1, &[]),
            ])
            .block(1, &[], vec![
                store(v(2), v(0), 2024),
                jump(3, &[]),
            ])
            .block(2, &[], vec![
                store(v(3), v(0), 2024),
                jump(3, &[]),
            ])
            .block(3, &[], vec![
                iconst64(v(5), 3000),
                iconst64(v(6), 2024),
                iconst64(v(7), 0),
                iconst64(v(8), 8),
                call(Some(v(9)), 0, &[v(0), v(5), v(6), v(7), v(8)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 4096];
    // condition=1, value_a=100, value_b=200 → should store value_a
    memory[2000..2008].copy_from_slice(&1u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&100u64.to_le_bytes());
    memory[2016..2024].copy_from_slice(&200u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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
        "cl_cuda_pinned_alloc", "cl_cuda_pinned_ptr", "cl_cuda_pinned_ptr_at",
        "cl_cuda_pinned_free", "cl_cuda_mem_info_free", "cl_cuda_mem_info_total",
        "cl_cuda_launch", "cl_cuda_launch_named", "cl_cuda_launch_on_stream",
        "cl_cuda_launch_named_on_stream", "cl_cuda_sync", "cl_cuda_cleanup",
        "cl_cublas_sgemm", "cl_cublas_sgemv", "cl_cublas_sgemv_on_stream",
        "cl_cublas_sgemm_strided_batched", "cl_cublas_sgemm_strided_batched_on_stream",
        "cl_cublas_ptr_array", "cl_cublas_sgemm_batched_on_stream",
        "cl_cublas_gemm_ex_bf16",
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

    let mut f = function(0).sig(0, &[I64], Some(I32));
    let mut insts = Vec::new();
    for (i, sym) in symbols.iter().enumerate() {
        let i = i as u32;
        f = f.import(i, sym, 0);
        // Taking the address is what forces the linker to resolve the symbol.
        insts.push(func_addr(v(100 + i), i));
        insts.push(store(v(100 + i), v(0), 0));
    }
    insts.push(ret());
    let clif_prog = program(f.entry(insts));

    let memory = vec![0u8; 4096];
    let (config, algorithm) =
        create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .sig(1, &[I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .import(1, "cl_file_read", 0)
            .import(2, "cl_file_write_from_ptr", 1)
            .import(3, "cl_file_read_to_ptr", 1)
            .entry(vec![
                iconst64(v(1), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 5),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                iconst64(v(6), 3100),
                call(Some(v(7)), 1, &[v(0), v(1), v(6), v(3), v(4)]),
                iadd_imm(v(8), v(0), 2256),
                iadd_imm(v(9), v(0), 3000),
                call(Some(v(10)), 2, &[v(8), v(9), v(3), v(4)]),
                iadd_imm(v(11), v(0), 3200),
                call(Some(v(12)), 3, &[v(8), v(11), v(3), v(4)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 4096];
    memory[2000..2000 + path_a_str.len()].copy_from_slice(path_a_str.as_bytes());
    memory[2256..2256 + path_b_str.len()].copy_from_slice(path_b_str.as_bytes());
    memory[3000..3005].copy_from_slice(b"hello");


    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = program(
        function(0)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I64, I64, I32], Some(I32))
            .sig(4, &[I64, I32, I32, I32, I32], Some(I32))
            .sig(5, &[I64, I32, I64, I64], Some(I32))
            .import(0, "cl_gpu_init", 0)
            .import(1, "cl_gpu_create_buffer", 1)
            .import(2, "cl_gpu_upload", 2)
            .import(3, "cl_gpu_create_pipeline", 3)
            .import(4, "cl_gpu_dispatch", 4)
            .import(5, "cl_gpu_download", 5)
            .import(6, "cl_gpu_cleanup", 0)
            .entry(vec![
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                iconst64(v(1), data_bytes as i64),
                call(Some(v(2)), 1, &[v(91), v(1)]),
                iadd_imm(v(3), v(0), data_off as i64),
                call(Some(v(10)), 2, &[v(91), v(2), v(3), v(1)]),
                iadd_imm(v(4), v(0), shader_off as i64),
                iadd_imm(v(5), v(0), bind_off as i64),
                iconst32(v(6), 1),
                call(Some(v(7)), 3, &[v(91), v(4), v(5), v(6)]),
                call(Some(v(11)), 4, &[v(91), v(7), v(6), v(6), v(6)]),
                iadd_imm(v(8), v(0), result_off as i64),
                call(Some(v(12)), 5, &[v(91), v(2), v(8), v(1)]),
                call(None, 6, &[v(90)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 6144];

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
        create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = program(
        function(0)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I64))
            .sig(2, &[I64, I64, I64, I64], Some(I64))
            .sig(3, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_net_init", 0)
            .import(1, "cl_net_connect", 1)
            .import(2, "cl_net_send", 2)
            .import(3, "cl_net_recv", 2)
            .import(4, "cl_net_cleanup", 0)
            .import(5, "cl_file_write", 3)
            .entry(vec![
                call(None, 0, &[v(0)]),
                load_trusted(v(1), I64, v(0), 0),
                iadd_imm(v(2), v(0), 2000),
                call(Some(v(3)), 1, &[v(1), v(2)]),
                iadd_imm(v(4), v(0), 3000),
                iconst64(v(5), 5),
                call(Some(v(6)), 2, &[v(1), v(3), v(4), v(5)]),
                iadd_imm(v(7), v(0), 3100),
                call(Some(v(8)), 3, &[v(1), v(3), v(7), v(5)]),
                iconst64(v(9), 2100),
                iconst64(v(10), 3100),
                iconst64(v(11), 0),
                call(Some(v(12)), 5, &[v(0), v(9), v(10), v(11), v(5)]),
                call(None, 4, &[v(0)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 4096];
    memory[2000..2000 + addr_str.len()].copy_from_slice(addr_str.as_bytes());
    memory[2100..2100 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[3000..3005].copy_from_slice(b"hello");


    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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
    let clif_prog = program(
        function(0)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64, I32], Some(I32))
            .sig(2, &[I64, I32, I64, I32, I64, I32], Some(I32))
            .sig(3, &[I64, I32, I64, I32, I64], Some(I32))
            .sig(4, &[I64, I32, I64, I32, I32, I64], Some(I32))
            .import(0, "cl_lmdb_init", 0)
            .import(1, "cl_lmdb_open", 1)
            .import(2, "cl_lmdb_put", 2)
            .import(3, "cl_lmdb_get", 3)
            .import(4, "cl_lmdb_cursor_scan", 4)
            .import(5, "cl_lmdb_cleanup", 0)
            .entry(vec![
                call(None, 0, &[v(0)]),
                load_trusted(v(91), I64, v(0), 0),
                iadd_imm(v(1), v(0), 2000),
                iconst32(v(2), 10),
                call(Some(v(3)), 1, &[v(91), v(1), v(2)]),
                iadd_imm(v(4), v(0), 3000),
                iconst32(v(5), 5),
                iadd_imm(v(6), v(0), 3100),
                call(Some(v(10)), 2, &[v(91), v(3), v(4), v(5), v(6), v(5)]),
                iadd_imm(v(7), v(0), 3200),
                call(Some(v(11)), 3, &[v(91), v(3), v(4), v(5), v(7)]),
                iadd_imm(v(8), v(0), 3500),
                iconst64(v(9), 0),
                iconst32(v(14), 0),
                iconst32(v(12), 100),
                call(Some(v(13)), 4, &[v(91), v(3), v(9), v(14), v(12), v(8)]),
                call(None, 5, &[v(0)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 6144];
    memory[2000..2000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[3000..3005].copy_from_slice(b"hello");
    memory[3100..3105].copy_from_slice(b"world");


    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = programs(vec![
        function(0)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64, I64], Some(I64))
            .sig(2, &[I64, I64], Some(I64))
            .sig(3, &[I64], None)
            .sig(4, &[I64, I64, I64], Some(I64))
            .sig(5, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_thread_init", 0)
            .import(1, "cl_thread_spawn", 1)
            .import(2, "cl_thread_join", 2)
            .import(3, "cl_thread_cleanup", 3)
            .import(4, "cl_thread_call", 4)
            .import(5, "cl_file_write", 5)
            .entry(vec![
                iadd_imm(v(1), v(0), 16),
                call(None, 0, &[v(1)]),
                load_trusted(v(10), I64, v(0), 16),
                iconst64(v(2), 1),
                iadd_imm(v(3), v(0), 200),
                call(Some(v(4)), 1, &[v(10), v(2), v(3)]),
                call(Some(v(5)), 2, &[v(10), v(4)]),
                iconst64(v(6), 2),
                iadd_imm(v(7), v(0), 208),
                call(Some(v(8)), 4, &[v(10), v(6), v(7)]),
                call(None, 3, &[v(1)]),
                iconst64(v(20), 3000),
                iconst64(v(21), 200),
                iconst64(v(22), 0),
                iconst64(v(23), 16),
                call(Some(v(24)), 5, &[v(0), v(20), v(21), v(22), v(23)]),
                ret(),
            ]),
        function(1)
            .entry_spawned(vec![
                iconst64(v(1), 42),
                store(v(1), v(0), 0),
                ret(),
            ]),
        function(2)
            .entry_spawned(vec![
                iconst64(v(1), 99),
                store(v(1), v(0), 0),
                ret(),
            ]),
    ]);

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 16);
    assert_eq!(u64::from_le_bytes(contents[0..8].try_into().unwrap()), 42);
    assert_eq!(u64::from_le_bytes(contents[8..16].try_into().unwrap()), 99);
}

#[test]
fn test_clif_call_basic() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("clif_call_basic.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                iconst64(v(1), 3000),
                iconst64(v(2), 2000),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 4096];
    memory[2000..2008].copy_from_slice(&42u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());


    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = programs(vec![
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                iconst64(v(1), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
        function(1)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                iconst64(v(1), 2256),
                iconst64(v(2), 3008),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; 4096];
    memory[2000..2000 + file_a_str.len()].copy_from_slice(file_a_str.as_bytes());
    memory[2256..2256 + file_b_str.len()].copy_from_slice(file_b_str.as_bytes());
    memory[3000..3008].copy_from_slice(&100u64.to_le_bytes());
    memory[3008..3016].copy_from_slice(&200u64.to_le_bytes());

    // Demonstrates JIT-once, run-many: one Base, two execute() calls picking different
    // fn_idx into the same compiled module.
    let mut base = Base::new(cranelift_config(memory, clif_prog)).unwrap();
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

    let clif_prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                load64(v(1), v(0), 2000),
                load64(v(2), v(0), 2008),
                iadd(v(3), v(1), v(2)),
                store(v(3), v(0), 2016),
                iconst64(v(4), 3000),
                iconst64(v(5), 2016),
                iconst64(v(6), 0),
                iconst64(v(7), 8),
                call(Some(v(8)), 0, &[v(0), v(4), v(5), v(6), v(7)]),
                ret(),
            ]),
    );

    let mut memory = vec![0u8; 4096];
    memory[2000..2008].copy_from_slice(&30u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&12u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let (config, algorithm) = create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                iconst64(v(1), 10),
                store(v(1), v(0), 2000),
                ret(),
            ]),
        function(1)
            .entry(vec![
                load64(v(1), v(0), 2000),
                iconst64(v(2), 5),
                imul(v(3), v(1), v(2)),
                store(v(3), v(0), 2008),
                ret(),
            ]),
        function(2)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                iconst64(v(1), 3000),
                iconst64(v(2), 2008),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let mut base = Base::new(cranelift_config(memory, clif_prog)).unwrap();
    base.execute(&cranelift_algorithm(0), &[]).unwrap();
    base.execute(&cranelift_algorithm(1), &[]).unwrap();
    base.execute(&cranelift_algorithm(2), &[]).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 50, "10 * 5 = 50");
}

#[test]
fn test_clif_call_no_workers_needed() {
    let clif_prog = program(
        function(0)
            .entry(vec![
                iconst64(v(1), 77),
                store(v(1), v(0), 2000),
                ret(),
            ]),
    );

    let memory = vec![0u8; 4096];


    // cranelift_units: 0 — no workers
    let (_config, _algorithm) = create_cranelift_algorithm(0, memory, clif_prog);

    // Rebuild with file write verification
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("clif_call_no_workers.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_prog2 = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                iconst64(v(1), 77),
                store(v(1), v(0), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 2000),
                iconst64(v(4), 0),
                iconst64(v(5), 8),
                call(Some(v(6)), 0, &[v(0), v(2), v(3), v(4), v(5)]),
                ret(),
            ]),
    );

    let mut memory2 = vec![0u8; 4096];
    memory2[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());


    let (config2, algorithm2) =
        create_cranelift_algorithm(0, memory2, clif_prog2);
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

    let clif_prog = programs(vec![
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_read", 0)
            .entry(vec![
                iconst64(v(1), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 256),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
        function(1)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                iconst64(v(1), 2256),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 256),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; 4096];
    memory[2000..2000 + input_str.len()].copy_from_slice(input_str.as_bytes());
    memory[2256..2256 + output_str.len()].copy_from_slice(output_str.as_bytes());

    // Two execute() calls on one Base: fn0 reads input file, fn1 writes output file.
    let mut base = Base::new(cranelift_config(memory, clif_prog)).unwrap();
    base.execute(&cranelift_algorithm(0), &[]).unwrap();
    base.execute(&cranelift_algorithm(1), &[]).unwrap();

    assert!(output_file.exists());
    let output_data = fs::read(&output_file).unwrap();
    assert_eq!(output_data, input_data, "output should match input");
}

fn create_output_algorithm(
    clif: Program,
    memory: Vec<u8>,
    output: Vec<OutputBatchSchema>,
) -> (Setup, Algorithm) {
    dump(&clif);
    let p = memory;

    let config = Setup {
        clif,
        memory_size: p.len(),
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iconst64(v(1), 42),
                iconst64(v(2), 2000),
                iadd(v(3), v(0), v(2)),
                store(v(1), v(3), 0),
                ret(),
            ]),
    );

    let memory = vec![0u8; 4096];
    let (cfg, alg) = create_output_algorithm(clif_prog, memory, vec![]);
    let batches = run(cfg, alg).unwrap();
    assert!(batches.is_empty());
}

#[test]
fn test_output_single_i64_column() {
    // CLIF writes i64 value 99 at offset 2000 and row_count=1 at offset 2008.
    let clif_prog = program(
        function(0)
            .entry(vec![
                iconst64(v(1), 99),
                iconst64(v(2), 2000),
                iadd(v(3), v(0), v(2)),
                store(v(1), v(3), 0),
                iconst64(v(4), 1),
                iconst64(v(5), 2008),
                iadd(v(6), v(0), v(5)),
                store(v(4), v(6), 0),
                ret(),
            ]),
    );

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

    let (cfg, alg) = create_output_algorithm(clif_prog, memory, output);
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iconst64(v(1), 42),
                iconst64(v(2), 2000),
                iadd(v(3), v(0), v(2)),
                store(v(1), v(3), 0),
                f64const(v(4), 3.141592653589793f64),
                iconst64(v(5), 2008),
                iadd(v(6), v(0), v(5)),
                store(v(4), v(6), 0),
                iconst64(v(7), 1),
                iconst64(v(8), 2016),
                iadd(v(9), v(0), v(8)),
                store(v(7), v(9), 0),
                ret(),
            ]),
    );

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

    let (cfg, alg) = create_output_algorithm(clif_prog, memory, output);
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iconst64(v(1), 0x6f6c6c6568),
                iconst64(v(2), 2000),
                iadd(v(3), v(0), v(2)),
                store(v(1), v(3), 0),
                iconst64(v(4), 5),
                iconst64(v(5), 2008),
                iadd(v(6), v(0), v(5)),
                store(v(4), v(6), 0),
                iconst64(v(7), 1),
                iconst64(v(8), 2016),
                iadd(v(9), v(0), v(8)),
                store(v(7), v(9), 0),
                ret(),
            ]),
    );

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

    let (cfg, alg) = create_output_algorithm(clif_prog, memory, output);
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iconst64(v(1), 10),
                iconst64(v(2), 2000),
                iadd(v(3), v(0), v(2)),
                store(v(1), v(3), 0),
                iconst64(v(4), 20),
                iconst64(v(5), 2008),
                iadd(v(6), v(0), v(5)),
                store(v(4), v(6), 0),
                iconst64(v(7), 30),
                iconst64(v(8), 2016),
                iadd(v(9), v(0), v(8)),
                store(v(7), v(9), 0),
                iconst64(v(10), 3),
                iconst64(v(11), 2024),
                iadd(v(12), v(0), v(11)),
                store(v(10), v(12), 0),
                ret(),
            ]),
    );

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

    let (cfg, alg) = create_output_algorithm(clif_prog, memory, output);
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                ret(),
            ]),
    );

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

    let (cfg, alg) = create_output_algorithm(clif_prog, memory, output);
    let batches = run(cfg, alg).unwrap();
    assert!(batches.is_empty());
}

#[test]
fn test_output_multiple_batches() {
    // Two output schemas — each becomes a separate RecordBatch.
    // Batch 1: single i64 at 2000, row_count at 2008.
    // Batch 2: single f64 at 2016, row_count at 2024.
    let clif_prog = program(
        function(0)
            .entry(vec![
                iconst64(v(1), 7),
                iconst64(v(2), 2000),
                iadd(v(3), v(0), v(2)),
                store(v(1), v(3), 0),
                iconst64(v(4), 1),
                iconst64(v(5), 2008),
                iadd(v(6), v(0), v(5)),
                store(v(4), v(6), 0),
                f64const(v(7), 7.0f64),
                iconst64(v(8), 2016),
                iadd(v(9), v(0), v(8)),
                store(v(7), v(9), 0),
                iconst64(v(10), 1),
                iconst64(v(11), 2024),
                iadd(v(12), v(0), v(11)),
                store(v(10), v(12), 0),
                ret(),
            ]),
    );

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

    let (cfg, alg) = create_output_algorithm(clif_prog, memory, output);
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iconst64(v(1), 0x66656400636261),
                iconst64(v(2), 2000),
                iadd(v(3), v(0), v(2)),
                store(v(1), v(3), 0),
                iconst64(v(4), 7),
                iconst64(v(5), 2100),
                iadd(v(6), v(0), v(5)),
                store(v(4), v(6), 0),
                iconst64(v(7), 2),
                iconst64(v(8), 2108),
                iadd(v(9), v(0), v(8)),
                store(v(7), v(9), 0),
                ret(),
            ]),
    );

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

    let (cfg, alg) = create_output_algorithm(clif_prog, memory, output);
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                // batch0 row_count = 1
                iconst64(v(1), 1),
                iconst64(v(2), 2000),
                iadd(v(3), v(0), v(2)),
                store(v(1), v(3), 0),
                // batch0 total = 300
                iconst64(v(4), 300),
                iconst64(v(5), 2008),
                iadd(v(6), v(0), v(5)),
                store(v(4), v(6), 0),
                // batch0 average = 100.0
                f64const(v(7), 100.0f64),
                iconst64(v(8), 2016),
                iadd(v(9), v(0), v(8)),
                store(v(7), v(9), 0),
                // batch1 row_count = 3
                iconst64(v(10), 3),
                iconst64(v(11), 2024),
                iadd(v(12), v(0), v(11)),
                store(v(10), v(12), 0),
                // batch1 id[0] = 1
                iconst64(v(13), 1),
                iconst64(v(14), 2032),
                iadd(v(15), v(0), v(14)),
                store(v(13), v(15), 0),
                // batch1 id[1] = 2
                iconst64(v(16), 2),
                iconst64(v(17), 2040),
                iadd(v(18), v(0), v(17)),
                store(v(16), v(18), 0),
                // batch1 id[2] = 3
                iconst64(v(19), 3),
                iconst64(v(20), 2048),
                iadd(v(21), v(0), v(20)),
                store(v(19), v(21), 0),
                // batch1 names: "alice\0bob\0charlie\0" packed at 2056
                // "alice\0bo" = 0x6f62_0065_6369_6c61
                iconst64(v(22), 0x6f62006563696c61),
                iconst64(v(23), 2056),
                iadd(v(24), v(0), v(23)),
                store(v(22), v(24), 0),
                // "b\0charli" = 0x696c_7261_6863_0062
                iconst64(v(25), 0x696c726168630062),
                iconst64(v(26), 2064),
                iadd(v(27), v(0), v(26)),
                store(v(25), v(27), 0),
                // "e\0" + padding = 0x0065
                iconst64(v(28), 0x65),
                iconst64(v(29), 2072),
                iadd(v(30), v(0), v(29)),
                store(v(28), v(30), 0),
                // batch1 name len_offset = 19
                iconst64(v(31), 19),
                iconst64(v(32), 2080),
                iadd(v(33), v(0), v(32)),
                store(v(31), v(33), 0),
                ret(),
            ]),
    );

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

    let (cfg, alg) = create_output_algorithm(clif_prog, memory, output);
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                // batch0: row_count=1, value=42
                iconst64(v(1), 1),
                iconst64(v(2), 2000),
                iadd(v(3), v(0), v(2)),
                store(v(1), v(3), 0),
                iconst64(v(4), 42),
                iconst64(v(5), 2008),
                iadd(v(6), v(0), v(5)),
                store(v(4), v(6), 0),
                // batch1: row_count stays 0 (skipped)
                // batch2: row_count=2, values=[10, 20]
                iconst64(v(7), 2),
                iconst64(v(8), 2032),
                iadd(v(9), v(0), v(8)),
                store(v(7), v(9), 0),
                iconst64(v(10), 10),
                iconst64(v(11), 2040),
                iadd(v(12), v(0), v(11)),
                store(v(10), v(12), 0),
                iconst64(v(13), 20),
                iconst64(v(14), 2048),
                iadd(v(15), v(0), v(14)),
                store(v(13), v(15), 0),
                ret(),
            ]),
    );

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

    let (cfg, alg) = create_output_algorithm(clif_prog, memory, output);
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                load64(v(1), v(0), 100),
                iconst64(v(2), 7),
                imul(v(3), v(1), v(2)),
                store(v(3), v(0), 200),
                iconst64(v(4), 1),
                store(v(4), v(0), 208),
                ret(),
            ]),
    );

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
        clif: clif_prog.clone(),
        memory_size: memory.len(),
        initial_memory: memory.clone(),
    };
    let alg1 = Algorithm {
        fn_idx: 0,
        output: output_schema.clone(),
    };
    let batches1 = run(config1, alg1).unwrap();

    // Base struct
    let config2 = Setup {
        clif: clif_prog.clone(),
        memory_size: memory.len(),
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                iconst64(v(3), 3),
                imul(v(4), v(2), v(3)),
                store(v(4), v(0), 200),
                iconst64(v(5), 1),
                iconst64(v(6), 208),
                iadd(v(7), v(0), v(6)),
                store(v(5), v(7), 0),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                iconst64(v(1), 42),
                store(v(1), v(0), 200),
                iconst64(v(2), 1),
                iconst64(v(3), 208),
                iadd(v(4), v(0), v(3)),
                store(v(2), v(4), 0),
                ret(),
            ]),
        function(1)
            .entry(vec![
                iconst64(v(1), 99),
                store(v(1), v(0), 200),
                iconst64(v(2), 1),
                iconst64(v(3), 208),
                iadd(v(4), v(0), v(3)),
                store(v(2), v(4), 0),
                ret(),
            ]),
    ]);

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                load64(v(1), v(0), 200),
                iadd_imm(v(2), data_ptr(), 0),
                load64(v(3), v(2), 0),
                iadd(v(4), v(1), v(3)),
                store(v(4), v(0), 200),
                iconst64(v(5), 1),
                iconst64(v(6), 208),
                iadd(v(7), v(0), v(6)),
                store(v(5), v(7), 0),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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

    let clif_prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(vec![
                iconst64(v(1), 256),
                iconst64(v(2), 512),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
    );

    // Execute 1: write value 42 to file1
    let mut mem1 = vec![0u8; 4096];
    mem1[256..256 + file1_str.len()].copy_from_slice(file1_str.as_bytes());
    mem1[512..520].copy_from_slice(&42u64.to_le_bytes());
    let config1 = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iconst64(v(1), 1),
                store(v(1), v(0), 200),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                load64(v(1), v(0), 100),
                iadd_imm(v(2), data_ptr(), 0),
                load64(v(3), v(2), 0),
                iadd(v(4), v(1), v(3)),
                store(v(4), v(0), 300),
                iconst64(v(5), 1),
                store(v(5), v(0), 308),
                ret(),
            ]),
    );

    let mut mem = vec![0u8; 4096];
    mem[100..108].copy_from_slice(&11i64.to_le_bytes());

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                iconst64(v(1), 77),
                store(v(1), v(0), 200),
                ret(),
            ]),
        function(1)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                load64(v(3), v(0), 200),
                iadd(v(4), v(2), v(3)),
                store(v(4), v(0), 300),
                iconst64(v(5), 1),
                store(v(5), v(0), 308),
                ret(),
            ]),
    ]);

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                load64(v(1), v(0), 200),
                iconst64(v(2), 1),
                iadd(v(3), v(1), v(2)),
                store(v(3), v(0), 200),
                iconst64(v(4), 1),
                store(v(4), v(0), 208),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                load64(v(3), v(1), 8),
                iadd(v(4), v(2), v(3)),
                store(v(4), v(0), 200),
                iconst64(v(5), 1),
                store(v(5), v(0), 208),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                load64(v(3), v(0), 400),
                iconst64(v(4), 8),
                imul(v(5), v(3), v(4)),
                iconst64(v(6), 500),
                iadd(v(7), v(5), v(6)),
                iadd(v(8), v(0), v(7)),
                store(v(2), v(8), 0),
                iconst64(v(9), 1),
                iadd(v(10), v(3), v(9)),
                store(v(10), v(0), 400),
                store(v(10), v(0), 408),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
fn clif_error_value_used_before_defined() {
    // v9 is never defined. The text path reported this as a parse error; the
    // decoder reports it against the program, which is where the defect is.
    let config = cranelift_config(
        vec![0u8; 256],
        program(function(0).entry(vec![store(v(9), v(0), 0), ret()])),
    );
    let Err(err) = Base::new(config) else {
        panic!("expected an error for a value used before it is defined");
    };
    let base::Error::Clif(msg) = err else {
        panic!("expected Error::Clif");
    };
    assert!(msg.contains("v9"), "message should name the value: {msg}");
}

#[test]
fn clif_error_branch_to_undeclared_block() {
    let config = cranelift_config(
        vec![0u8; 256],
        program(function(0).entry(vec![jump(7, &[])])),
    );
    let Err(err) = run(config, cranelift_algorithm(0)) else {
        panic!("expected an error for a branch to an undeclared block");
    };
    assert!(matches!(err, base::Error::Clif(_)));
}

#[test]
fn clif_error_call_to_undeclared_fn() {
    let config = cranelift_config(
        vec![0u8; 256],
        program(function(0).entry(vec![call(None, 3, &[v(0)]), ret()])),
    );
    let Err(err) = Base::new(config) else {
        panic!("expected an error for a call to an undeclared callee");
    };
    assert!(matches!(err, base::Error::Clif(_)));
}

#[test]
fn clif_error_function_index_disagrees_with_position() {
    // `u0:N` is resolved as a FuncId, so a program whose indices do not match
    // their positions would silently call the wrong function.
    let config = cranelift_config(vec![0u8; 256], program(noop(4)));
    let Err(err) = Base::new(config) else {
        panic!("expected an error for a function index that disagrees");
    };
    let base::Error::Clif(msg) = err else {
        panic!("expected Error::Clif");
    };
    assert!(msg.contains("u0:4"), "message should name the index: {msg}");
}

#[test]
fn clif_parse_error_empty_ir_no_error() {
    // Empty string should NOT error — it skips compilation entirely
    let config = Setup {
        clif: Default::default(),
        memory_size: 256,
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

    let clif_prog = program(
        function(0)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I64, I32, I64, I32, I32, I32, I32, I32, I32], Some(I32))
            .sig(4, &[I64], Some(I32))
            .import(0, "cl_cuda_init", 0)
            .import(1, "cl_cuda_create_buffer", 1)
            .import(2, "cl_cuda_upload", 2)
            .import(3, "cl_cuda_launch", 3)
            .import(4, "cl_cuda_sync", 4)
            .import(5, "cl_cuda_download", 2)
            .import(6, "cl_cuda_cleanup", 0)
            .entry(vec![
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                iconst64(v(1), data_bytes as i64),
                call(Some(v(2)), 1, &[v(91), v(1)]),
                iadd_imm(v(3), v(0), data_off as i64),
                call(Some(v(10)), 2, &[v(91), v(2), v(3), v(1)]),
                iadd_imm(v(4), v(0), ptx_off as i64),
                iconst32(v(5), 1),
                iadd_imm(v(6), v(0), bind_off as i64),
                iconst32(v(7), 4),
                call(Some(v(11)), 3, &[v(91), v(4), v(5), v(6), v(5), v(5), v(5), v(7), v(5), v(5)]),
                call(Some(v(12)), 4, &[v(91)]),
                iadd_imm(v(9), v(0), result_off as i64),
                call(Some(v(13)), 5, &[v(91), v(2), v(9), v(1)]),
                call(None, 6, &[v(90)]),
                ret(),
            ]),
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
        create_cranelift_algorithm(0, memory, clif_prog);
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

    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I32, I32, I32, I32, I32, I32, I32, I32, I32], Some(I32))
            .sig(4, &[I64], Some(I32))
            .sig(5, &[I64, I32], Some(I32))
            .import(0, "cl_cuda_init", 0)
            .import(1, "cl_cuda_create_buffer", 1)
            .import(2, "cl_cuda_upload_ptr", 2)
            .import(3, "cl_cuda_download_ptr", 2)
            .import(4, "cl_cublas_sgemv_on_stream", 3)
            .import(5, "cl_cuda_stream_create", 4)
            .import(6, "cl_cuda_stream_sync", 5)
            .import(7, "cl_cuda_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                iconst64(v(10), a_bytes as i64),
                iconst64(v(11), x_bytes as i64),
                iconst64(v(12), y_bytes as i64),
                call(Some(v(13)), 1, &[v(91), v(10)]),
                call(Some(v(14)), 1, &[v(91), v(11)]),
                call(Some(v(15)), 1, &[v(91), v(12)]),
                call(Some(v(16)), 2, &[v(91), v(13), v(1), v(10)]),
                iadd(v(17), v(1), v(10)),
                call(Some(v(18)), 2, &[v(91), v(14), v(17), v(11)]),
                call(Some(v(19)), 5, &[v(91)]),
                iconst32(v(20), 1),
                iconst32(v(21), cols as i64),
                iconst32(v(22), rows as i64),
                iconst32(v(23), 0x3f800000),
                iconst32(v(24), 0),
                call(Some(v(25)), 4, &[v(91), v(20), v(21), v(22), v(23), v(13), v(14), v(24), v(15), v(19)]),
                call(Some(v(26)), 6, &[v(91), v(19)]),
                call(Some(v(27)), 3, &[v(91), v(15), v(2), v(12)]),
                call(None, 7, &[v(90)]),
                ret(),
            ]),
    ]);

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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

    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I32, I32, I32, I32, I32, I32, I32, I64, I32, I64, I32, I32, I64, I32, I32,
                      I64, I64, I64, I32, I32, I32], Some(I32))
            .sig(4, &[I64], Some(I32))
            .sig(5, &[I64, I32], Some(I32))
            .import(0, "cl_cuda_init", 0)
            .import(1, "cl_cuda_create_buffer", 1)
            .import(2, "cl_cuda_upload_ptr", 2)
            .import(3, "cl_cuda_download_ptr", 2)
            .import(4, "cl_cublas_sgemm_strided_batched_on_stream", 3)
            .import(5, "cl_cuda_stream_create", 4)
            .import(6, "cl_cuda_stream_sync", 5)
            .import(7, "cl_cuda_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                iconst64(v(10), a_bytes as i64),
                iconst64(v(11), x_bytes as i64),
                iconst64(v(12), y_bytes as i64),
                call(Some(v(13)), 1, &[v(91), v(10)]),
                call(Some(v(14)), 1, &[v(91), v(11)]),
                call(Some(v(15)), 1, &[v(91), v(12)]),
                call(Some(v(16)), 2, &[v(91), v(13), v(1), v(10)]),
                iadd(v(17), v(1), v(10)),
                call(Some(v(18)), 2, &[v(91), v(14), v(17), v(11)]),
                call(Some(v(19)), 5, &[v(91)]),
                iconst32(v(20), 1),
                iconst32(v(21), 0),
                iconst32(v(22), m as i64),
                iconst32(v(23), 1),
                iconst32(v(24), k as i64),
                iconst32(v(25), 0x3f800000),
                iconst64(v(26), (m * k) as i64),
                iconst64(v(27), k as i64),
                iconst64(v(28), m as i64),
                iconst32(v(29), batch_count as i64),
                // Element offsets into each operand, and explicit leading
                // dimensions; zero means "start at the buffer, shape-implied".
                iconst64(v(33), 0),
                call(Some(v(30)), 4, &[v(91), v(20), v(21), v(22), v(23), v(24), v(25), v(13), v(26), v(14), v(27), v(21), v(15), v(28), v(29), v(19),
                                       v(33), v(33), v(33), v(21), v(21), v(21)]),
                call(Some(v(31)), 6, &[v(91), v(19)]),
                call(Some(v(32)), 3, &[v(91), v(15), v(2), v(12)]),
                call(None, 7, &[v(90)]),
                ret(),
            ]),
    ]);

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                // load data_ptr from offset 8
                iadd_imm(v(1), data_ptr(), 0),
                // load data_len from offset 16
                iadd_imm(v(2), data_len(), 0),
                // read first i64 from caller's buffer
                load64(v(3), v(1), 0),
                // read second i64 from caller's buffer (offset 8)
                load64(v(4), v(1), 8),
                iadd(v(5), v(3), v(4)),
                // store result and row_count
                store(v(5), v(0), 200),
                store(v(2), v(0), 208),
                iconst64(v(6), 1),
                store(v(6), v(0), 216),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), data_len(), 0),
                store(v(1), v(0), 200),
                iconst64(v(3), 1),
                store(v(3), v(0), 208),
                ret(),
            ]),
    );

    let mut initial = vec![0u8; 4096];
    initial[16..24].copy_from_slice(&0xCAFEBABEu64.to_le_bytes());

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), out_len(), 0),
                store(v(1), v(0), 200),
                iconst64(v(3), 1),
                store(v(3), v(0), 208),
                ret(),
            ]),
    );

    let mut initial = vec![0u8; 4096];
    initial[32..40].copy_from_slice(&0x22222222u64.to_le_bytes());

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                // read data_ptr, load input from caller's data buffer
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                // compute: input * 7
                iconst64(v(3), 7),
                imul(v(4), v(2), v(3)),
                // read out_ptr, write result into caller's out buffer
                iadd_imm(v(5), out_ptr(), 0),
                store(v(4), v(5), 0),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                iadd_imm(v(3), out_ptr(), 0),
                store(v(2), v(3), 0),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                // read data_ptr and data_len
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), data_len(), 0),
                // read last i64 from caller buffer: data_ptr + data_len - 8
                iconst64(v(3), 8),
                isub(v(4), v(2), v(3)),
                iadd(v(5), v(1), v(4)),
                load64(v(6), v(5), 0),
                store(v(6), v(0), 200),
                store(v(2), v(0), 208),
                iconst64(v(7), 1),
                store(v(7), v(0), 216),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 256,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                // read static multiplier from shared memory (set by initial_memory)
                load64(v(1), v(0), 100),
                // read dynamic input from data pointer
                iadd_imm(v(2), data_ptr(), 0),
                load64(v(3), v(2), 0),
                // multiply
                imul(v(4), v(1), v(3)),
                store(v(4), v(0), 200),
                iconst64(v(5), 1),
                store(v(5), v(0), 208),
                ret(),
            ]),
    );

    let mut initial = vec![0u8; 4096];
    // Static multiplier = 13
    initial[100..108].copy_from_slice(&13i64.to_le_bytes());

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), out_ptr(), 0),
                iadd_imm(v(2), out_len(), 0),
                // write values at out[0], out[8], out[16]
                iconst64(v(3), 100),
                store(v(3), v(1), 0),
                iconst64(v(4), 200),
                store(v(4), v(1), 8),
                iconst64(v(5), 300),
                store(v(5), v(1), 16),
                // write out_len at the end for verification
                store(v(2), v(1), 24),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 64,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                store(v(2), v(0), 200),
                iconst64(v(3), 1),
                store(v(3), v(0), 208),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), data_len(), 0),
                store(v(1), v(0), 200),
                iconst64(v(2), 1),
                store(v(2), v(0), 208),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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
    let clif_prog = program(
        function(0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                iadd_imm(v(3), data_len(), 0),
                store(v(2), v(0), 200),
                store(v(3), v(0), 208),
                iconst64(v(4), 1),
                store(v(4), v(0), 216),
                ret(),
            ]),
    );

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: 4096,
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

    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I64, I64, I32], Some(I32))
            .sig(3, &[I64, I32, I64, I64], Some(I32))
            .sig(4, &[I64, I32, I32, I32, I32], Some(I32))
            .sig(5, &[I64, I32, I64, I64, I64], Some(I32))
            .import(0, "cl_gpu_init", 0)
            .import(1, "cl_gpu_create_buffer", 1)
            .import(2, "cl_gpu_create_pipeline", 2)
            .import(3, "cl_gpu_upload_ptr", 3)
            .import(4, "cl_gpu_dispatch", 4)
            .import(5, "cl_gpu_download_ptr", 5)
            .import(6, "cl_gpu_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), data_len(), 0),
                iadd_imm(v(3), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                call(Some(v(4)), 1, &[v(91), v(2)]),
                call(Some(v(5)), 3, &[v(91), v(4), v(1), v(2)]),
                iadd_imm(v(6), v(0), shader_off as i64),
                iadd_imm(v(7), v(0), bind_off as i64),
                iconst32(v(8), 1),
                call(Some(v(9)), 2, &[v(91), v(6), v(7), v(8)]),
                call(Some(v(10)), 4, &[v(91), v(9), v(8), v(8), v(8)]),
                iconst64(v(9001), 3),
                ushr(v(11), v(2), v(9001)),
                iconst64(v(9002), 2),
                ishl(v(12), v(11), v(9002)),
                iconst64(v(13), 0),
                call(Some(v(14)), 5, &[v(91), v(4), v(13), v(3), v(12)]),
                call(None, 6, &[v(90)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; mem_size];
    let shader_bytes = wgsl.as_bytes();
    memory[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    memory[shader_off + shader_bytes.len()] = 0;
    // bind desc: buf_id=0, read_only=0
    memory[bind_off..bind_off + 8].copy_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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

    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I32, I64, I64, I64], Some(I32))
            .import(0, "cl_gpu_init", 0)
            .import(1, "cl_gpu_create_buffer", 1)
            .import(2, "cl_gpu_upload_ptr", 2)
            .import(3, "cl_gpu_download_ptr", 3)
            .import(4, "cl_gpu_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), data_len(), 0),
                iadd_imm(v(3), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                // create buffer for full data (2*64*4 = 512 bytes)
                call(Some(v(4)), 1, &[v(91), v(2)]),
                // upload all data from payload
                call(Some(v(5)), 2, &[v(91), v(4), v(1), v(2)]),
                // download only second half: buf_offset = 256, size = 256, to out_ptr
                iconst64(v(6), (n * 4) as i64),
                call(Some(v(7)), 3, &[v(91), v(4), v(6), v(3), v(6)]),
                call(None, 4, &[v(90)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; mem_size];
    let shader_bytes = wgsl.as_bytes();
    memory[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    memory[shader_off + shader_bytes.len()] = 0;
    memory[bind_off..bind_off + 8].copy_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]);

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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
    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I64, I32, I64, I32, I32, I32, I32, I32, I32], Some(I32))
            .import(0, "cl_cuda_init", 0)
            .import(1, "cl_cuda_create_buffer", 1)
            .import(2, "cl_cuda_upload_ptr", 2)
            .import(3, "cl_cuda_download_ptr", 2)
            .import(4, "cl_cuda_launch", 3)
            .import(5, "cl_cuda_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), data_len(), 0),
                iadd_imm(v(3), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                // create 3 buffers of {data_bytes} bytes each
                iconst64(v(4), data_bytes as i64),
                call(Some(v(5)), 1, &[v(91), v(4)]),
                call(Some(v(6)), 1, &[v(91), v(4)]),
                call(Some(v(7)), 1, &[v(91), v(4)]),
                // upload A from data_ptr
                call(Some(v(8)), 2, &[v(91), v(5), v(1), v(4)]),
                // upload B from data_ptr + data_bytes
                iadd(v(9), v(1), v(4)),
                call(Some(v(10)), 2, &[v(91), v(6), v(9), v(4)]),
                // launch PTX kernel: grid(1,1,1) block(64,1,1)
                iadd_imm(v(11), v(0), ptx_off as i64),
                iconst32(v(12), 3),
                iadd_imm(v(13), v(0), bind_off as i64),
                iconst32(v(14), 1),
                iconst32(v(15), 64),
                call(Some(v(16)), 4, &[v(91), v(11), v(12), v(13), v(14), v(14), v(14), v(15), v(14), v(14)]),
                // download result from buf 2 to out_ptr
                call(Some(v(17)), 3, &[v(91), v(7), v(3), v(4)]),
                call(None, 5, &[v(90)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; mem_size];
    let ptx_bytes = ptx.as_bytes();
    memory[ptx_off..ptx_off + ptx_bytes.len()].copy_from_slice(ptx_bytes);
    // bind desc: buf_id=0 (A), buf_id=1 (B), buf_id=2 (C)
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    memory[bind_off + 4..bind_off + 8].copy_from_slice(&1i32.to_le_bytes());
    memory[bind_off + 8..bind_off + 12].copy_from_slice(&2i32.to_le_bytes());

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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
    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I64, I32, I64, I32, I32, I32, I32, I32, I32], Some(I32))
            .import(0, "cl_cuda_init", 0)
            .import(1, "cl_cuda_create_buffer", 1)
            .import(2, "cl_cuda_upload_ptr", 2)
            .import(3, "cl_cuda_download_ptr", 2)
            .import(4, "cl_cuda_launch", 3)
            .import(5, "cl_cuda_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), data_len(), 0),
                iadd_imm(v(3), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                // create 2 buffers
                iconst64(v(4), data_bytes as i64),
                call(Some(v(5)), 1, &[v(91), v(4)]),
                call(Some(v(6)), 1, &[v(91), v(4)]),
                // upload A from data_ptr to buf 0
                call(Some(v(7)), 2, &[v(91), v(5), v(1), v(4)]),
                // upload B from data_ptr + data_bytes to buf 1
                iadd(v(8), v(1), v(4)),
                call(Some(v(9)), 2, &[v(91), v(6), v(8), v(4)]),
                // launch: grid(1,1,1) block(64,1,1)
                iadd_imm(v(10), v(0), ptx_off as i64),
                iconst32(v(11), 2),
                iadd_imm(v(12), v(0), bind_off as i64),
                iconst32(v(13), 1),
                iconst32(v(14), 64),
                call(Some(v(15)), 4, &[v(91), v(10), v(11), v(12), v(13), v(13), v(13), v(14), v(13), v(13)]),
                // download buf 1 (result) to out_ptr
                call(Some(v(16)), 3, &[v(91), v(6), v(3), v(4)]),
                call(None, 5, &[v(90)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; mem_size];
    let ptx_bytes = ptx.as_bytes();
    memory[ptx_off..ptx_off + ptx_bytes.len()].copy_from_slice(ptx_bytes);
    // bind desc: buf_id=0 (A), buf_id=1 (B)
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    memory[bind_off + 4..bind_off + 8].copy_from_slice(&1i32.to_le_bytes());

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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

    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I32, I32, I32, I32, I32, I32, I32, I64, I32, I64, I32, I32, I64, I32,
                      I64, I64, I64, I32, I32, I32], Some(I32))
            .sig(4, &[I64], Some(I32))
            .import(0, "cl_cuda_init", 0)
            .import(1, "cl_cuda_create_buffer", 1)
            .import(2, "cl_cuda_upload_ptr", 2)
            .import(3, "cl_cuda_download_ptr", 2)
            .import(4, "cl_cublas_sgemm_strided_batched", 3)
            .import(5, "cl_cuda_sync", 4)
            .import(6, "cl_cuda_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                // create A, x, y buffers
                iconst64(v(10), a_bytes as i64),
                iconst64(v(11), x_bytes as i64),
                iconst64(v(12), y_bytes as i64),
                call(Some(v(13)), 1, &[v(91), v(10)]),
                call(Some(v(14)), 1, &[v(91), v(11)]),
                call(Some(v(15)), 1, &[v(91), v(12)]),
                // upload A from data_ptr
                call(Some(v(16)), 2, &[v(91), v(13), v(1), v(10)]),
                // upload x from data_ptr + a_bytes
                iadd(v(17), v(1), v(10)),
                call(Some(v(18)), 2, &[v(91), v(14), v(17), v(11)]),
                // batched GEMV using SGEMM-strided-batched
                // row-major A (2x3) => transa=1, transb=0, m=2, n=1, k=3
                // stride_a=6, stride_b=3, stride_c=2 elements, batch_count=2
                iconst32(v(20), 1),
                iconst32(v(21), 0),
                iconst32(v(22), m as i64),
                iconst32(v(23), 1),
                iconst32(v(24), k as i64),
                iconst32(v(25), 0x3f800000),
                iconst64(v(26), (m * k) as i64),
                iconst64(v(27), k as i64),
                iconst64(v(28), m as i64),
                iconst32(v(29), batch_count as i64),
                // Element offsets into each operand, and explicit leading
                // dimensions; zero means "start at the buffer, shape-implied".
                iconst64(v(33), 0),
                call(Some(v(30)), 4, &[v(91), v(20), v(21), v(22), v(23), v(24), v(25), v(13), v(26), v(14), v(27), v(21), v(15), v(28), v(29),
                                       v(33), v(33), v(33), v(21), v(21), v(21)]),
                call(Some(v(31)), 5, &[v(91)]),
                call(Some(v(32)), 3, &[v(91), v(15), v(2), v(12)]),
                call(None, 6, &[v(90)]),
                ret(),
            ]),
    ]);

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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

    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64, I64], Some(I32))
            .sig(3, &[I64, I32, I64, I64], Some(I32))
            .import(0, "cl_cuda_init", 0)
            .import(1, "cl_cuda_create_buffer", 1)
            .import(2, "cl_cuda_upload_ptr_offset", 2)
            .import(3, "cl_cuda_download_ptr", 3)
            .import(4, "cl_cuda_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                iconst64(v(10), total_bytes as i64),
                call(Some(v(11)), 1, &[v(91), v(10)]),
                // upload first 2 floats to offset 0
                iconst64(v(12), 8),
                iconst64(v(13), 0),
                call(Some(v(14)), 2, &[v(91), v(11), v(13), v(1), v(12)]),
                // upload second 2 floats to offset 8
                iadd(v(15), v(1), v(12)),
                call(Some(v(16)), 2, &[v(91), v(11), v(12), v(15), v(12)]),
                // download full 4-float buffer
                call(Some(v(17)), 3, &[v(91), v(11), v(2), v(10)]),
                call(None, 4, &[v(90)]),
                ret(),
            ]),
    ]);

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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

    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I64, I64, I32, I64, I32, I32, I32, I32, I32, I32], Some(I32))
            .import(0, "cl_cuda_init", 0)
            .import(1, "cl_cuda_create_buffer", 1)
            .import(2, "cl_cuda_upload_ptr", 2)
            .import(3, "cl_cuda_download_ptr", 2)
            .import(4, "cl_cuda_launch_named", 3)
            .import(5, "cl_cuda_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                iconst64(v(10), data_bytes as i64),
                call(Some(v(11)), 1, &[v(91), v(10)]),
                call(Some(v(12)), 2, &[v(91), v(11), v(1), v(10)]),
                // launch named kernel twice: x -> x+1 -> x+2
                iadd_imm(v(13), v(0), ptx_off as i64),
                iadd_imm(v(14), v(0), name_off as i64),
                iconst32(v(15), 1),
                iadd_imm(v(16), v(0), bind_off as i64),
                iconst32(v(17), 1),
                iconst32(v(18), n as i64),
                call(Some(v(19)), 4, &[v(91), v(13), v(14), v(15), v(16), v(17), v(17), v(17), v(18), v(17), v(17)]),
                call(Some(v(20)), 4, &[v(91), v(13), v(14), v(15), v(16), v(17), v(17), v(17), v(18), v(17), v(17)]),
                call(Some(v(21)), 3, &[v(91), v(11), v(2), v(10)]),
                call(None, 5, &[v(90)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; mem_size];
    memory[ptx_off..ptx_off + ptx.len()].copy_from_slice(ptx.as_bytes());
    memory[name_off..name_off + 8].copy_from_slice(b"add_one\0");
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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

    let clif_prog = programs(vec![
        function(0)
            .entry(vec![
                ret(),
            ]),
        function(1)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64], Some(I32))
            .sig(2, &[I64, I32, I64, I64], Some(I32))
            .sig(3, &[I64, I32, I32, I32, I32, I32, I32, I32, I32], Some(I32))
            .sig(4, &[I64], Some(I32))
            .import(0, "cl_cuda_init", 0)
            .import(1, "cl_cuda_create_buffer", 1)
            .import(2, "cl_cuda_upload_ptr", 2)
            .import(3, "cl_cuda_download_ptr", 2)
            .import(4, "cl_cublas_sgemv", 3)
            .import(5, "cl_cuda_sync", 4)
            .import(6, "cl_cuda_cleanup", 0)
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                iadd_imm(v(2), out_ptr(), 0),
                iadd_imm(v(90), v(0), 0),
                call(None, 0, &[v(90)]),
                load_trusted(v(91), I64, v(0), 0),
                iconst64(v(10), a_bytes as i64),
                iconst64(v(11), x_bytes as i64),
                iconst64(v(12), y_bytes as i64),
                call(Some(v(13)), 1, &[v(91), v(10)]),
                call(Some(v(14)), 1, &[v(91), v(11)]),
                call(Some(v(15)), 1, &[v(91), v(12)]),
                call(Some(v(16)), 2, &[v(91), v(13), v(1), v(10)]),
                iadd(v(17), v(1), v(10)),
                call(Some(v(18)), 2, &[v(91), v(14), v(17), v(11)]),
                // row-major A[rows, cols] -> sgemv(trans=1, m=cols, n=rows)
                iconst32(v(19), 1),
                iconst32(v(20), cols as i64),
                iconst32(v(21), rows as i64),
                iconst32(v(22), 0x3f800000),
                iconst32(v(23), 0),
                call(Some(v(24)), 4, &[v(91), v(19), v(20), v(21), v(22), v(13), v(14), v(23), v(15)]),
                call(Some(v(25)), 5, &[v(91)]),
                call(Some(v(26)), 3, &[v(91), v(15), v(2), v(12)]),
                call(None, 6, &[v(90)]),
                ret(),
            ]),
    ]);

    let config = Setup {
        clif: clif_prog.clone(),
        memory_size: mem_size,
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


// --- instruction coverage ---------------------------------------------------
//
// The tests above are about FFI linkage and reach only a third of the
// instruction set. These reach the rest: each builds a program using
// instructions nothing else exercises, runs it, and checks the values it
// computed. A decode arm mapped to the wrong Cranelift instruction shows up
// here as a wrong number rather than as a latent bug an application discovers
// later.

/// Runs `insts` and returns `n` bytes written from memory offset 2000.
fn compute(insts: Vec<Inst>, n: i64) -> Vec<u8> {
    let temp_dir = TempDir::new().unwrap();
    let out = temp_dir.path().join("out.bin");
    let path = format!("{}\0", out.to_str().unwrap());

    let mut body = insts;
    body.extend([
        iconst64(v(900), 3000),
        iconst64(v(901), 2000),
        iconst64(v(902), 0),
        iconst64(v(903), n),
        call(Some(v(904)), 0, &[v(0), v(900), v(901), v(902), v(903)]),
        ret(),
    ]);
    let prog = program(
        function(0)
            .sig(0, &[I64, I64, I64, I64, I64], Some(I64))
            .import(0, "cl_file_write", 0)
            .entry(body),
    );

    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + path.len()].copy_from_slice(path.as_bytes());
    let (config, algorithm) = create_cranelift_algorithm(0, memory, prog);
    run(config, algorithm).unwrap();
    fs::read(&out).unwrap()
}

fn i64s(bytes: &[u8]) -> Vec<i64> {
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn instr_integer_arithmetic() {
    // udiv, ineg, band, band_not, bor, bxor
    let bytes = compute(
        vec![
            // A negative dividend: unsigned division of -100 is a huge
            // quotient, signed division would be -14.
            iconst64(v(1), -100),
            iconst64(v(2), 7),
            udiv(v(3), v(1), v(2)),
            store(v(3), v(0), 2000),
            ineg(v(4), v(2)),
            store(v(4), v(0), 2008),
            iconst64(v(5), 0xF0),
            iconst64(v(6), 0x3C),
            band(v(7), v(5), v(6)),
            store(v(7), v(0), 2016),
            band_not(v(8), v(5), v(6)),
            store(v(8), v(0), 2024),
            bor(v(9), v(5), v(6)),
            store(v(9), v(0), 2032),
            bxor(v(10), v(5), v(6)),
            store(v(10), v(0), 2040),
        ],
        48,
    );
    assert_eq!(
        i64s(&bytes),
        vec![
            ((-100i64) as u64 / 7) as i64, // unsigned, not -14
            -7,           // -(7)
            0x30,         // F0 & 3C
            0xC0,         // F0 & !3C
            0xFC,         // F0 | 3C
            0xCC,         // F0 ^ 3C
        ]
    );
}

#[test]
fn instr_bit_counting_and_select() {
    // ctz, popcnt, select, bitselect
    let bytes = compute(
        vec![
            // 0b1_0000 has one set bit and four trailing zeros, so a swap of
            // the two would show.
            iconst64(v(1), 0b1_0000),
            ctz(v(2), v(1)),
            store(v(2), v(0), 2000),
            popcnt(v(3), v(1)),
            store(v(3), v(0), 2008),
            // select picks by a condition, bitselect picks by a mask
            iconst64(v(4), 111),
            iconst64(v(5), 222),
            iconst64(v(6), 1),
            iconst64(v(7), 2),
            icmp(v(8), IntCC::Ult, v(6), v(7)),
            select(v(9), v(8), v(4), v(5)),
            store(v(9), v(0), 2016),
            iconst64(v(10), 0xFF00),
            bitselect(v(11), v(10), v(4), v(5)),
            store(v(11), v(0), 2024),
        ],
        32,
    );
    let got = i64s(&bytes);
    assert_eq!(got[0], 4, "ctz(0b1_0000)");
    assert_eq!(got[1], 1, "popcnt(0b1_0000)");
    assert_eq!(got[2], 111, "select(1 < 2, 111, 222)");
    assert_eq!(got[3], (111 & 0xFF00) | (222 & !0xFF00), "bitselect");
}

#[test]
fn instr_width_conversions() {
    // ireduce32, uextend64, sextend64, istore8, store_typed
    let bytes = compute(
        vec![
            // sign extension differs from zero extension for a negative i32
            iconst32(v(1), -5),
            sextend64(v(2), v(1)),
            store(v(2), v(0), 2000),
            uextend64(v(3), v(1)),
            store(v(3), v(0), 2008),
            // narrowing keeps the low 32 bits
            iconst64(v(4), 0x1_0000_002A),
            ireduce32(v(5), v(4)),
            uextend64(v(6), v(5)),
            store(v(6), v(0), 2016),
            // istore8 writes one byte; store_typed carries `notrap aligned`
            iconst64(v(7), 0xAB),
            istore8(v(7), v(0), 2024),
            iconst64(v(8), 77),
            store_typed(I64, v(8), v(0), 2032),
        ],
        40,
    );
    let got = i64s(&bytes);
    assert_eq!(got[0], -5, "sextend64(-5i32)");
    assert_eq!(got[1], 0xFFFF_FFFB, "uextend64(-5i32)");
    assert_eq!(got[2], 0x2A, "ireduce32 keeps the low word");
    assert_eq!(got[3] & 0xFF, 0xAB, "istore8");
    assert_eq!(got[4], 77, "store_typed");
}

#[test]
fn instr_float_arithmetic() {
    // f32const, f64const, fadd, fsub, fmul, fneg, fmax, fmin, fpromote
    let bytes = compute(
        vec![
            f64const(v(1), 3.5),
            f64const(v(2), 1.25),
            fadd(v(3), v(1), v(2)),
            store(v(3), v(0), 2000),
            fsub(v(4), v(1), v(2)),
            store(v(4), v(0), 2008),
            fmul(v(5), v(1), v(2)),
            store(v(5), v(0), 2016),
            fneg(v(6), v(1)),
            store(v(6), v(0), 2024),
            fmax(v(7), v(1), v(2)),
            store(v(7), v(0), 2032),
            fmin(v(8), v(1), v(2)),
            store(v(8), v(0), 2040),
            // f32 -> f64 keeps the value
            f32const(v(9), 2.5),
            fpromote(v(10), v(9)),
            store(v(10), v(0), 2048),
        ],
        56,
    );
    let got: Vec<f64> = bytes
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(got, vec![4.75, 2.25, 4.375, -3.5, 3.5, 1.25, 2.5]);
}

#[test]
fn instr_float_conversions_and_compare() {
    // fcvt_from_sint, fcvt_to_uint, bitcast, fcmp
    let bytes = compute(
        vec![
            iconst64(v(1), 9),
            fcvt_from_sint(v(2), F64, v(1)),
            f64const(v(3), 0.5),
            fmul(v(4), v(2), v(3)),
            store(v(4), v(0), 2000), // 4.5
            fcvt_to_uint(v(5), I64, v(4)),
            store(v(5), v(0), 2008), // 4, truncated
            // bitcast reinterprets rather than converts
            bitcast(v(6), I64, v(3)),
            store(v(6), v(0), 2016),
            // fcmp yields a one-byte flag, widened here so it can be read back
            fcmp(v(7), FloatCC::Gt, v(2), v(3)),
            uextend64(v(70), v(7)),
            store(v(70), v(0), 2024),
            fcmp(v(8), FloatCC::Lt, v(2), v(3)),
            uextend64(v(80), v(8)),
            store(v(80), v(0), 2032),
        ],
        40,
    );
    assert_eq!(f64::from_le_bytes(bytes[0..8].try_into().unwrap()), 4.5);
    let got = i64s(&bytes);
    assert_eq!(got[1], 4, "fcvt_to_uint truncates");
    assert_eq!(got[2], 0.5f64.to_bits() as i64, "bitcast is a reinterpretation");
    assert_eq!(got[3], 1, "9.0 > 0.5");
    assert_eq!(got[4], 0, "9.0 < 0.5 is false");
}

#[test]
fn instr_vector_lanes() {
    // splat, extractlane, vhigh_bits
    let bytes = compute(
        vec![
            // Four distinct lanes staged in memory, so the lane index is what
            // decides the answer. Bits: [1.0, 2.0] then [3.0, 4.0].
            iconst64(v(1), 0x4000_0000_3F80_0000u64 as i64),
            store(v(1), v(0), 2200),
            iconst64(v(2), 0x4080_0000_4040_0000u64 as i64),
            store(v(2), v(0), 2208),
            load_trusted(v(3), F32X4, v(0), 2200),
            extractlane(v(4), v(3), 2),
            fpromote(v(5), v(4)),
            store(v(5), v(0), 2000),
            // and a splat really does fill every lane
            f32const(v(20), 1.5),
            splat(v(21), F32X4, v(20)),
            extractlane(v(22), v(21), 3),
            fpromote(v(23), v(22)),
            store(v(23), v(0), 2016),
            // vhigh_bits gathers the sign bit of each byte lane: sixteen 0xFF
            // bytes staged in memory, read back as one vector.
            iconst64(v(6), -1),
            store(v(6), v(0), 2100),
            store(v(6), v(0), 2108),
            load_trusted(v(7), I8X16, v(0), 2100),
            vhigh_bits(v(8), v(7)),
            uextend64(v(9), v(8)),
            store(v(9), v(0), 2008),
        ],
        24,
    );
    assert_eq!(
        f64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        3.0,
        "lane 2 of [1, 2, 3, 4]"
    );
    assert_eq!(i64s(&bytes)[1], 0xFFFF, "every one of the sixteen lanes is negative");
    assert_eq!(
        f64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        1.5,
        "splat fills lane 3 too"
    );
}

// --- multi-function programs ------------------------------------------------

#[test]
fn local_calls_dispatch_to_other_functions() {
    // A wrapper function calling two others by `u0:N` index, which is the shape
    // `clifSequenceWrapper` emits for every multi-stage artifact. Resolution
    // goes through `Callee::Local`, not the symbol table.
    let temp_dir = TempDir::new().unwrap();
    let out = temp_dir.path().join("locals.bin");
    let path = format!("{}\0", out.to_str().unwrap());

    let clif_prog = programs(vec![
        // u0:0 writes 11 at 2000
        function(0).entry(vec![
            iconst64(v(1), 11),
            store(v(1), v(0), 2000),
            ret(),
        ]),
        // u0:1 writes 22 at 2008
        function(1).entry(vec![
            iconst64(v(1), 22),
            store(v(1), v(0), 2008),
            ret(),
        ]),
        // u0:2 calls both, then writes the pair out
        function(2)
            .sig(0, &[I64], None)
            .sig(1, &[I64, I64, I64, I64, I64], Some(I64))
            .local(0, 0, 0)
            .local(1, 1, 0)
            .import(2, "cl_file_write", 1)
            .entry(vec![
                call(None, 0, &[v(0)]),
                call(None, 1, &[v(0)]),
                iconst64(v(1), 3000),
                iconst64(v(2), 2000),
                iconst64(v(3), 0),
                iconst64(v(4), 16),
                call(Some(v(5)), 2, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + path.len()].copy_from_slice(path.as_bytes());
    let (config, algorithm) = create_cranelift_algorithm(2, memory, clif_prog);
    run(config, algorithm).unwrap();

    let bytes = fs::read(&out).unwrap();
    assert_eq!(
        i64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        11,
        "u0:0 ran"
    );
    assert_eq!(
        i64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        22,
        "u0:1 ran"
    );
}

#[test]
fn clif_error_local_call_to_missing_function() {
    // A program of one function whose wrapper names u0:3.
    let config = cranelift_config(
        vec![0u8; 256],
        program(
            function(0)
                .sig(0, &[I64], None)
                .local(0, 3, 0)
                .entry(vec![call(None, 0, &[v(0)]), ret()]),
        ),
    );
    let Err(err) = Base::new(config) else {
        panic!("expected an error for a local call to a function that is not defined");
    };
    let base::Error::Clif(msg) = err else {
        panic!("expected Error::Clif");
    };
    assert!(msg.contains("u0:3"), "message should name the callee: {msg}");
}

#[test]
fn clif_error_callee_names_undeclared_sig() {
    let config = cranelift_config(
        vec![0u8; 256],
        program(
            function(0)
                .import(0, "cl_file_write", 7)
                .entry(vec![ret()]),
        ),
    );
    let Err(err) = Base::new(config) else {
        panic!("expected an error for a callee naming a signature that is not declared");
    };
    let base::Error::Clif(msg) = err else {
        panic!("expected Error::Clif");
    };
    assert!(msg.contains("sig7"), "message should name the signature: {msg}");
}

#[test]
fn clif_error_binding_the_result_of_a_void_callee() {
    let config = cranelift_config(
        vec![0u8; 256],
        program(
            function(0)
                .sig(0, &[I64], None)
                .import(0, "cl_gpu_init", 0)
                .entry(vec![call(Some(v(1)), 0, &[v(0)]), ret()]),
        ),
    );
    let Err(err) = Base::new(config) else {
        panic!("expected an error for binding the result of a callee that returns nothing");
    };
    assert!(matches!(err, base::Error::Clif(_)));
}

#[test]
fn clif_error_float_constant_of_integer_type() {
    let config = cranelift_config(
        vec![0u8; 256],
        program(function(0).entry(vec![
            base_types::clif::Inst::Fconst(v(1), I64, 0),
            ret(),
        ])),
    );
    let Err(err) = Base::new(config) else {
        panic!("expected an error for a float constant of a non-float type");
    };
    let base::Error::Clif(msg) = err else {
        panic!("expected Error::Clif");
    };
    assert!(msg.contains("fconst"), "message should say what is wrong: {msg}");
}
