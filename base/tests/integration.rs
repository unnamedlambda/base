use base::{Artifact, Base};
use std::fs;

mod common;
use common::*;
use tempfile::TempDir;

/// The artifact running `functions` in `memory`.
///
/// A function the test did not name is exported as its `u0:N`, so a test can
/// call any of them by index; the ones about naming name their own.
fn cranelift_config(memory: Vec<u8>, functions: Vec<Function>) -> Artifact {
    let functions = exported(functions);
    dump(&functions);
    Artifact {
        functions,
        memory_size: memory.len() as u64,
        data: image(memory),
    }
}

/// One segment holding the whole of `bytes`, for a test that writes an image
/// out as a flat vector.
fn image(bytes: Vec<u8>) -> Vec<base_types::Segment> {
    if bytes.is_empty() {
        return vec![];
    }
    vec![base_types::Segment { offset: 0, bytes }]
}

/// The i64 a program left at `offset` of its memory.
fn read_i64(base: &Base, offset: usize) -> i64 {
    i64::from_le_bytes(base.memory()[offset..offset + 8].try_into().unwrap())
}

/// The artifact, and the name its function at `fn_idx` is called by.
fn create_cranelift_algorithm(
    fn_idx: u32,
    memory: Vec<u8>,
    functions: Vec<Function>,
) -> (Artifact, String) {
    let config = cranelift_config(memory, functions);
    let name = config.functions[fn_idx as usize].export_name.clone().unwrap();
    (config, name)
}

/// Compile `artifact` and call `name` once.
fn run(artifact: Artifact, name: impl AsRef<str>) -> Result<i64, base::Error> {
    base::run(artifact, name.as_ref())
}

/// `functions`, each one the test did not name exported as its `u0:N`.
fn exported(mut functions: Vec<Function>) -> Vec<Function> {
    for (i, f) in functions.iter_mut().enumerate() {
        f.export_name.get_or_insert_with(|| at(i as u32));
    }
    functions
}

/// The name `exported` gives an unnamed function at `i`.
fn at(i: u32) -> String {
    format!("u0:{i}")
}

#[test]
fn test_cranelift_basic_compilation() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_basic.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    // Single CLIF function that writes 8 bytes at offset 2000 to the file at offset 3000.
    let clif_prog = program(
        function()
            .import(0, "cl_file_write")
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
        function()
            .import(0, "cl_file_write")
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
        function()
            .import(0, "cl_file_write")
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
        function()
            .import(0, "cl_file_write")
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
        function()
            .import(0, "cl_file_write")
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
fn test_clif_ffi_file_smoke() {
    // Runtime smoke: exercises cl_file_read, cl_file_write, cl_file_read_to_ptr,
    // cl_file_write_from_ptr via a real round-trip.
    let temp_dir = TempDir::new().unwrap();
    let path_a = temp_dir.path().join("smoke_a.bin");
    let path_b = temp_dir.path().join("smoke_b.bin");
    let path_a_str = format!("{}\0", path_a.to_str().unwrap());
    let path_b_str = format!("{}\0", path_b.to_str().unwrap());

    let clif_prog = program(
        function()
            .import(0, "cl_file_write")
            .import(1, "cl_file_read")
            .import(2, "cl_file_write_from_ptr")
            .import(3, "cl_file_read_to_ptr")
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
        function()
            .import(0, "cl_gpu_init")
            .import(1, "cl_gpu_create_buffer")
            .import(2, "cl_gpu_upload")
            .import(3, "cl_gpu_create_pipeline")
            .import(4, "cl_gpu_dispatch")
            .import(5, "cl_gpu_download")
            .import(6, "cl_gpu_cleanup")
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
        function()
            .import(0, "cl_net_init")
            .import(1, "cl_net_connect")
            .import(2, "cl_net_send")
            .import(3, "cl_net_recv")
            .import(4, "cl_net_cleanup")
            .import(5, "cl_file_write")
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
        function()
            .import(0, "cl_lmdb_init")
            .import(1, "cl_lmdb_open")
            .import(2, "cl_lmdb_put")
            .import(3, "cl_lmdb_get")
            .import(4, "cl_lmdb_cursor_scan")
            .import(5, "cl_lmdb_cleanup")
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
        function()
            .import(0, "cl_thread_init")
            .import(1, "cl_thread_spawn")
            .import(2, "cl_thread_join")
            .import(3, "cl_thread_cleanup")
            .import(4, "cl_thread_call")
            .import(5, "cl_file_write")
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
        function()
            .entry_spawned(vec![
                iconst64(v(1), 42),
                store(v(1), v(0), 0),
                ret(),
            ]),
        function()
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
        function()
            .import(0, "cl_file_write")
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
        function()
            .import(0, "cl_file_write")
            .entry(vec![
                iconst64(v(1), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
        function()
            .import(0, "cl_file_write")
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
    base.execute(&at(0), &[], &mut []).unwrap();
    base.execute(&at(1), &[], &mut []).unwrap();

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
        function()
            .import(0, "cl_file_write")
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
        function()
            .entry(vec![
                iconst64(v(1), 10),
                store(v(1), v(0), 2000),
                ret(),
            ]),
        function()
            .entry(vec![
                load64(v(1), v(0), 2000),
                iconst64(v(2), 5),
                imul(v(3), v(1), v(2)),
                store(v(3), v(0), 2008),
                ret(),
            ]),
        function()
            .import(0, "cl_file_write")
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
    base.execute(&at(0), &[], &mut []).unwrap();
    base.execute(&at(1), &[], &mut []).unwrap();
    base.execute(&at(2), &[], &mut []).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 50, "10 * 5 = 50");
}

#[test]
fn test_clif_call_no_workers_needed() {
    let clif_prog = program(
        function()
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
        function()
            .import(0, "cl_file_write")
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
        function()
            .import(0, "cl_file_read")
            .entry(vec![
                iconst64(v(1), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 256),
                call(Some(v(5)), 0, &[v(0), v(1), v(2), v(3), v(4)]),
                ret(),
            ]),
        function()
            .import(0, "cl_file_write")
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
    base.execute(&at(0), &[], &mut []).unwrap();
    base.execute(&at(1), &[], &mut []).unwrap();

    assert!(output_file.exists());
    let output_data = fs::read(&output_file).unwrap();
    assert_eq!(output_data, input_data, "output should match input");
}

#[test]
fn test_base_multi_execute_different_data() {
    // Compile once, execute twice with different input data via pointer.
    // CLIF reads i64 from data pointer, multiplies by 3, stores result at 200, row_count=1 at 208.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // First execute: input = 10, expect 30
    let data1 = 10i64.to_le_bytes();
    base.execute(&at(0), &data1, &mut [])
        .unwrap();
    assert_eq!(read_i64(&base, 200), 30);

    // Second execute: input = 100, expect 300
    let data2 = 100i64.to_le_bytes();
    base.execute(&at(0), &data2, &mut [])
        .unwrap();
    assert_eq!(read_i64(&base, 200), 300);
}

#[test]
fn test_base_multi_execute_different_actions() {
    // Compile once with two CLIF functions, execute with different action sequences.
    // fn0: stores 42 at offset 200, row_count=1 at 208
    // fn1: stores 99 at offset 200, row_count=1 at 208
    let clif_prog = programs(vec![
        function()
            .entry(vec![
                iconst64(v(1), 42),
                store(v(1), v(0), 200),
                iconst64(v(2), 1),
                iconst64(v(3), 208),
                iadd(v(4), v(0), v(3)),
                store(v(2), v(4), 0),
                ret(),
            ]),
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // First execute: call fn0 only
    let alg1 = at(0);
    base.execute(&alg1, &vec![0u8; 4096], &mut []).unwrap();
    assert_eq!(read_i64(&base, 200), 42);

    // Second execute: call fn1 only
    let alg2 = at(1);
    base.execute(&alg2, &vec![0u8; 4096], &mut []).unwrap();
    assert_eq!(read_i64(&base, 200), 99);
}

#[test]
fn test_base_multi_execute_accumulates_in_memory() {
    // Accumulator in shared memory persists across executes.
    // CLIF: load accumulator from v0+200, add input from data pointer, store back.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Execute 1: add 10 → total = 10
    let d1 = 10i64.to_le_bytes();
    base.execute(&at(0), &d1, &mut [])
        .unwrap();
    assert_eq!(read_i64(&base, 200), 10);

    // Execute 2: add 25 → total = 35
    let d2 = 25i64.to_le_bytes();
    base.execute(&at(0), &d2, &mut [])
        .unwrap();
    assert_eq!(read_i64(&base, 200), 35);

    // Execute 3: add 5 → total = 40
    let d3 = 5i64.to_le_bytes();
    base.execute(&at(0), &d3, &mut [])
        .unwrap();
    assert_eq!(read_i64(&base, 200), 40);
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
        function()
            .import(0, "cl_file_write")
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
    let config1 = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: image(mem1),
    };
    let mut base = Base::new(config1).unwrap();
    base.execute(&at(0), &[], &mut [])
    .unwrap();
    assert!(file1.exists());
    let data1 = fs::read(&file1).unwrap();
    assert_eq!(u64::from_le_bytes(data1[..8].try_into().unwrap()), 42);

    // Execute 2: write value 99 to file2 — new Base with different initial_memory
    let mut mem2 = vec![0u8; 4096];
    mem2[256..256 + file2_str.len()].copy_from_slice(file2_str.as_bytes());
    mem2[512..520].copy_from_slice(&99u64.to_le_bytes());
    let config2 = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: image(mem2),
    };
    let mut base2 = Base::new(config2).unwrap();
    base2
        .execute(&at(0), &[], &mut [])
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
        function()
            .entry(vec![
                iconst64(v(1), 1),
                store(v(1), v(0), 200),
                ret(),
            ]),
    );

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Execute with 0 units
    base.execute(&at(0), &vec![0u8; 4096], &mut [])
    .unwrap();

    // Execute with 2 units
    base.execute(&at(0), &vec![0u8; 4096], &mut [])
    .unwrap();

    // Execute with 4 units
    base.execute(&at(0), &vec![0u8; 4096], &mut [])
    .unwrap();
}

#[test]
fn test_base_initial_memory_and_data_pointer_coexist() {
    // initial_memory provides static config at v0+100, data pointer provides dynamic input.
    // CLIF reads both and adds them.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: image(mem),
    };
    let mut base = Base::new(config).unwrap();

    let data = 99i64.to_le_bytes();
    base.execute(&at(0), &data, &mut [])
        .unwrap();

    assert_eq!(read_i64(&base, 300), 110); // 11 + 99
}

#[test]
fn test_base_persistent_memory_survives_across_executes() {
    // Shared memory persists across executes. fn0 seeds a value, fn1 reads it.
    // CLIF fn0: stores 77 at offset 200
    // CLIF fn1: reads data pointer input + offset 200 → stores at 300
    let clif_prog = programs(vec![
        function()
            .entry(vec![
                iconst64(v(1), 77),
                store(v(1), v(0), 200),
                ret(),
            ]),
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Execute 1: seed 77 at offset 200
    base.execute(&at(0), &[], &mut [])
    .unwrap();

    // Execute 2: input=5 via pointer, read persistent 77 from offset 200
    let data = 5i64.to_le_bytes();
    base.execute(&at(1), &data, &mut [])
        .unwrap();

    // 5 (data pointer) + 77 (persistent) = 82
    assert_eq!(read_i64(&base, 300), 82);
}

#[test]
fn test_base_empty_data_leaves_memory_intact() {
    // Empty memory don't touch memory at all — persistent state survives.
    // CLIF: accumulate into offset 200 (read, add 1, store back). row_count at 208.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Three executes with empty memory — counter should increment each time
    for expected in 1..=3 {
        base.execute(&at(0), &[], &mut [])
            .unwrap();
        assert_eq!(read_i64(&base, 200), expected);
    }
}

#[test]
fn test_base_data_pointer_updates_each_execute() {
    // Data pointer is updated each execute call with fresh caller buffer.
    // CLIF reads two i64s from data pointer and adds them.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Execute 1: 10 + 20 = 30
    let mut d1 = vec![0u8; 16];
    d1[0..8].copy_from_slice(&10i64.to_le_bytes());
    d1[8..16].copy_from_slice(&20i64.to_le_bytes());
    base.execute(&at(0), &d1, &mut [])
        .unwrap();
    assert_eq!(read_i64(&base, 200), 30);

    // Execute 2: 100 + 200 = 300 — pointer should update to new buffer
    let mut d2 = vec![0u8; 16];
    d2[0..8].copy_from_slice(&100i64.to_le_bytes());
    d2[8..16].copy_from_slice(&200i64.to_le_bytes());
    base.execute(&at(0), &d2, &mut [])
        .unwrap();
    assert_eq!(read_i64(&base, 200), 300);
}

#[test]
fn test_base_output_in_persistent_region() {
    // Shared memory persists across executes. CLIF appends values from data pointer
    // into a growing buffer at offset 500+.
    // fn0: reads input from data_ptr, reads count from offset 400, stores at 500+8*count, increments count.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Execute 3 times with values 100, 200, 300
    for &val in &[100i64, 200, 300] {
        let d = val.to_le_bytes();
        base.execute(&at(0), &d, &mut [])
        .unwrap();
    }

    // Final read: count at 400 should be 3, values at 500/508/516 should be 100/200/300
    // One more execute to read output — pass a dummy input
    let d = 999i64.to_le_bytes();
    base.execute(&at(0), &d, &mut [])
        .unwrap();

    // count is now 4 (we did 4 executes), values: 100, 200, 300, 999
    assert_eq!(read_i64(&base, 408) as usize, 4);
    assert_eq!(read_i64(&base, 500), 100);
    assert_eq!(read_i64(&base, 508), 200);
    assert_eq!(read_i64(&base, 516), 300);
    assert_eq!(read_i64(&base, 524), 999);
}

#[test]
fn a_program_answers_the_status_it_returns() {
    // `return 7` — the signature says the function answers because its `Ret`
    // carries a value, so nothing else has to declare it.
    let (cfg, alg) = create_cranelift_algorithm(
        0,
        vec![0u8; 256],
        program(function().entry(vec![iconst64(v(1), 7), ret_status(v(1))])),
    );
    let mut base = Base::new(cfg).unwrap();
    assert_eq!(base.execute(&alg, &[], &mut []).unwrap(), 7);
}

/// A program that returns nothing has status 0, so a host reading a status
/// never has to ask which kind of program it called.
#[test]
fn a_program_that_returns_nothing_has_status_zero() {
    let (cfg, alg) = create_cranelift_algorithm(
        0,
        vec![0u8; 256],
        program(function().entry(vec![ret()])),
    );
    let mut base = Base::new(cfg).unwrap();
    assert_eq!(base.execute(&alg, &[], &mut []).unwrap(), 0);
}

/// Whether a function answers is one fact about it, so returns that disagree
/// are a malformed body rather than a signature base has to guess at.
#[test]
fn clif_error_returns_disagree() {
    let config = cranelift_config(
        vec![0u8; 256],
        program(
            function()
                .entry(vec![iconst64(v(1), 1), brif(v(1), 1, &[], 2, &[])])
                .block(1, &[], vec![ret_status(v(1))])
                .block(2, &[], vec![ret()]),
        ),
    );
    let Err(base::Error::Clif(msg)) = Base::new(config) else {
        panic!("expected a function whose returns disagree to be refused");
    };
    assert!(msg.contains("some paths"), "message should say what is wrong: {msg}");
}

#[test]
fn clif_error_value_used_before_defined() {
    // v9 is never defined. The text path reported this as a parse error; the
    // decoder reports it against the program, which is where the defect is.
    let config = cranelift_config(
        vec![0u8; 256],
        program(function().entry(vec![store(v(9), v(0), 0), ret()])),
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
        program(function().entry(vec![jump(7, &[])])),
    );
    let Err(err) = run(config, at(0)) else {
        panic!("expected an error for a branch to an undeclared block");
    };
    assert!(matches!(err, base::Error::Clif(_)));
}

#[test]
fn clif_error_call_to_undeclared_fn() {
    let config = cranelift_config(
        vec![0u8; 256],
        program(function().entry(vec![call(None, 3, &[v(0)]), ret()])),
    );
    let Err(err) = Base::new(config) else {
        panic!("expected an error for a call to an undeclared callee");
    };
    assert!(matches!(err, base::Error::Clif(_)));
}

#[test]
fn clif_parse_error_empty_ir_no_error() {
    // Empty string should NOT error — it skips compilation entirely
    let config = Artifact {
        functions: Default::default(),
        memory_size: 256,
        data: vec![],
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
        function()
            .import(0, "cl_cuda_init")
            .import(1, "cl_cuda_create_buffer")
            .import(2, "cl_cuda_upload")
            .import(3, "cl_cuda_launch")
            .import(4, "cl_cuda_sync")
            .import(5, "cl_cuda_download")
            .import(6, "cl_cuda_cleanup")
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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_cuda_init")
            .import(1, "cl_cuda_create_buffer")
            .import(2, "cl_cuda_upload_ptr")
            .import(3, "cl_cuda_download_ptr")
            .import(4, "cl_cublas_sgemv_on_stream")
            .import(5, "cl_cuda_stream_create")
            .import(6, "cl_cuda_stream_sync")
            .import(7, "cl_cuda_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(vec![0u8; mem_size]),
    };
    let mut base = Base::new(config).unwrap();
    let alg = at(1);

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
    base.execute(&alg, &payload1, &mut out1).unwrap();
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
    base.execute(&alg, &payload2, &mut out2).unwrap();
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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_cuda_init")
            .import(1, "cl_cuda_create_buffer")
            .import(2, "cl_cuda_upload_ptr")
            .import(3, "cl_cuda_download_ptr")
            .import(4, "cl_cublas_sgemm_strided_batched_on_stream")
            .import(5, "cl_cuda_stream_create")
            .import(6, "cl_cuda_stream_sync")
            .import(7, "cl_cuda_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(vec![0u8; mem_size]),
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(1);

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
    base.execute(&alg, &payload1, &mut out1).unwrap();

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
    base.execute(&alg, &payload2, &mut out2).unwrap();

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
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let mut data = vec![0u8; 16];
    data[0..8].copy_from_slice(&100i64.to_le_bytes());
    data[8..16].copy_from_slice(&200i64.to_le_bytes());

    let alg = at(0);

    base.execute(&alg, &data, &mut []).unwrap();
    assert_eq!(
        read_i64(&base, 200),
        300,
        "should read 100+200 from caller buffer via pointer"
    );
    assert_eq!(read_i64(&base, 208), 16, "data_len should be 16");
}

#[test]
fn test_data_ptr_written_even_when_data_empty() {
    // Offsets 8-16 are always written — even with empty data.
    // Seed those offsets with sentinels to verify they get overwritten.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: image(initial),
    };

    let alg = at(0);

    let mut base = Base::new(config).unwrap();
    base.execute(&alg, &[], &mut []).unwrap();
    assert_eq!(
        read_i64(&base, 200),
        0,
        "data_len should be 0 for empty data, sentinel overwritten"
    );
}

#[test]
fn test_out_ptr_written_even_when_out_empty() {
    // Offsets 24-32 are always written — even with empty out.
    // Seed those offsets with sentinels to verify they get overwritten.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: image(initial),
    };

    let alg = at(0);

    let mut base = Base::new(config).unwrap();
    base.execute(&alg, &[], &mut []).unwrap();
    assert_eq!(
        read_i64(&base, 200),
        0,
        "out_len should be 0 for empty out, sentinel overwritten"
    );
}

#[test]
fn test_execute_into_clif_writes_to_caller_out_buffer() {
    // CLIF reads out_ptr from offset 24, writes a computed value into caller's out buffer.
    // This tests the full zero-copy output path.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let mut data = vec![0u8; 8];
    data[0..8].copy_from_slice(&6i64.to_le_bytes());

    let mut out = vec![0u8; 8];

    let alg = at(0);

    base.execute(&alg, &data, &mut out).unwrap();
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
        function()
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                iadd_imm(v(3), out_ptr(), 0),
                store(v(2), v(3), 0),
                ret(),
            ]),
    );

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(0);

    // Call 1: data=111
    let data1 = 111i64.to_le_bytes().to_vec();
    let mut out1 = vec![0u8; 8];
    base.execute(&alg, &data1, &mut out1).unwrap();
    assert_eq!(i64::from_le_bytes(out1[0..8].try_into().unwrap()), 111);

    // Call 2: data=222, different buffers
    let data2 = 222i64.to_le_bytes().to_vec();
    let mut out2 = vec![0u8; 8];
    base.execute(&alg, &data2, &mut out2).unwrap();
    assert_eq!(i64::from_le_bytes(out2[0..8].try_into().unwrap()), 222);

    // out1 should be unchanged from call 2
    assert_eq!(i64::from_le_bytes(out1[0..8].try_into().unwrap()), 111);
}

#[test]
fn test_data_ptr_with_large_buffer_no_shared_mem_copy() {
    // Data buffer is larger than memory_size. The data pointer gives CLIF
    // access to the full buffer without copying it into shared memory.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 256,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    // Data is 1KB — much larger than memory_size (256)
    let mut data = vec![0u8; 1024];
    // Write sentinel at the very end
    data[1016..1024].copy_from_slice(&999i64.to_le_bytes());

    let alg = at(0);

    base.execute(&alg, &data, &mut []).unwrap();
    assert_eq!(
        read_i64(&base, 200),
        999,
        "CLIF should read last value from caller buffer via pointer"
    );
    assert_eq!(read_i64(&base, 208), 1024, "data_len should be full buffer size");
}

#[test]
fn test_initial_memory_and_data_coexist() {
    // initial_memory sets up static config (e.g., a multiplier at offset 100).
    // data provides dynamic input via pointer.
    // CLIF reads multiplier from shared memory AND input from data pointer.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: image(initial),
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(0);

    // Dynamic input = 7
    let data = 7i64.to_le_bytes().to_vec();
    base.execute(&alg, &data, &mut []).unwrap();
    assert_eq!(
        read_i64(&base, 200),
        91,
        "13 * 7 = 91: static config from initial_memory, dynamic input via pointer"
    );
}

#[test]
fn test_execute_into_out_buffer_larger_than_memory() {
    // Out buffer can be any size — it's caller-owned, not bounded by memory_size.
    // CLIF writes multiple values into a large out buffer.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 64,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(0);

    // Tiny shared memory (64 bytes) but large out buffer
    let data = vec![0u8; 8]; // need non-empty data so pointers at 8-16 get written, but we need out ptrs
    let mut out = vec![0u8; 32];
    base.execute(&alg, &data, &mut out).unwrap();

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
        function()
            .entry(vec![
                iadd_imm(v(1), data_ptr(), 0),
                load64(v(2), v(1), 0),
                store(v(2), v(0), 200),
                iconst64(v(3), 1),
                store(v(3), v(0), 208),
                ret(),
            ]),
    );

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };

    let alg = at(0);

    let data = 777i64.to_le_bytes().to_vec();
    let mut base = Base::new(config).unwrap();
    base.execute(&alg, &data, &mut []).unwrap();
    assert_eq!(
        read_i64(&base, 200),
        777,
        "execute() should pass data pointer through to CLIF"
    );
}

#[test]
fn test_data_single_byte_still_writes_pointer() {
    // Even a 1-byte data buffer should write the pointer.
    // Edge case: smallest possible non-empty data.
    let clif_prog = program(
        function()
            .entry(vec![
                iadd_imm(v(1), data_len(), 0),
                store(v(1), v(0), 200),
                iconst64(v(2), 1),
                store(v(2), v(0), 208),
                ret(),
            ]),
    );

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };

    let alg = at(0);

    let data = vec![42u8]; // single byte
    let mut base = Base::new(config).unwrap();
    base.execute(&alg, &data, &mut []).unwrap();
    assert_eq!(read_i64(&base, 200), 1, "data_len should be 1 for single-byte data");
}

#[test]
fn test_data_ptr_survives_across_multi_execute() {
    // Multiple execute calls with data — each call gets fresh pointers.
    // Verify that stale pointers from previous calls don't leak.
    let clif_prog = program(
        function()
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: 4096,
        data: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(0);

    // Call 1: 8-byte buffer
    let data1 = 11i64.to_le_bytes().to_vec();
    base.execute(&alg, &data1, &mut []).unwrap();
    assert_eq!(read_i64(&base, 200), 11);
    assert_eq!(read_i64(&base, 208), 8);

    // Call 2: 16-byte buffer (different size!)
    let mut data2 = vec![0u8; 16];
    data2[0..8].copy_from_slice(&22i64.to_le_bytes());
    base.execute(&alg, &data2, &mut []).unwrap();
    assert_eq!(read_i64(&base, 200), 22);
    assert_eq!(read_i64(&base, 208), 16, "data_len should reflect new buffer size");
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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_gpu_init")
            .import(1, "cl_gpu_create_buffer")
            .import(2, "cl_gpu_create_pipeline")
            .import(3, "cl_gpu_upload_ptr")
            .import(4, "cl_gpu_dispatch")
            .import(5, "cl_gpu_download_ptr")
            .import(6, "cl_gpu_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(memory),
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
    let alg = at(1);

    base.execute(&alg, &payload, &mut out).unwrap();

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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_gpu_init")
            .import(1, "cl_gpu_create_buffer")
            .import(2, "cl_gpu_upload_ptr")
            .import(3, "cl_gpu_download_ptr")
            .import(4, "cl_gpu_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(memory),
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
    let alg = at(1);

    base.execute(&alg, &payload, &mut out).unwrap();

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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_cuda_init")
            .import(1, "cl_cuda_create_buffer")
            .import(2, "cl_cuda_upload_ptr")
            .import(3, "cl_cuda_download_ptr")
            .import(4, "cl_cuda_launch")
            .import(5, "cl_cuda_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(memory),
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
    let alg = at(1);

    base.execute(&alg, &payload, &mut out).unwrap();

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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_cuda_init")
            .import(1, "cl_cuda_create_buffer")
            .import(2, "cl_cuda_upload_ptr")
            .import(3, "cl_cuda_download_ptr")
            .import(4, "cl_cuda_launch")
            .import(5, "cl_cuda_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(memory),
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(1);

    // First execute: A=[1..64], B=[100..100]
    let mut payload1 = vec![0u8; n * 4 * 2];
    for i in 0..n {
        let a_val = (i + 1) as f32;
        let b_val = 100.0f32;
        payload1[i * 4..i * 4 + 4].copy_from_slice(&a_val.to_le_bytes());
        payload1[n * 4 + i * 4..n * 4 + i * 4 + 4].copy_from_slice(&b_val.to_le_bytes());
    }
    let mut out1 = vec![0u8; n * 4];
    base.execute(&alg, &payload1, &mut out1).unwrap();

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
    base.execute(&alg, &payload2, &mut out2).unwrap();

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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_cuda_init")
            .import(1, "cl_cuda_create_buffer")
            .import(2, "cl_cuda_upload_ptr")
            .import(3, "cl_cuda_download_ptr")
            .import(4, "cl_cublas_sgemm_strided_batched")
            .import(5, "cl_cuda_sync")
            .import(6, "cl_cuda_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(vec![0u8; mem_size]),
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(1);

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
    base.execute(&alg, &payload1, &mut out1).unwrap();

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
    base.execute(&alg, &payload2, &mut out2).unwrap();

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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_cuda_init")
            .import(1, "cl_cuda_create_buffer")
            .import(2, "cl_cuda_upload_ptr_offset")
            .import(3, "cl_cuda_download_ptr")
            .import(4, "cl_cuda_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(vec![0u8; mem_size]),
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(1);

    let payload1: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut bytes1 = Vec::with_capacity(total_bytes);
    for v in payload1 {
        bytes1.extend_from_slice(&v.to_le_bytes());
    }
    let mut out1 = vec![0u8; total_bytes];
    base.execute(&alg, &bytes1, &mut out1).unwrap();
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
    base.execute(&alg, &bytes2, &mut out2).unwrap();
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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_cuda_init")
            .import(1, "cl_cuda_create_buffer")
            .import(2, "cl_cuda_upload_ptr")
            .import(3, "cl_cuda_download_ptr")
            .import(4, "cl_cuda_launch_named")
            .import(5, "cl_cuda_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(memory),
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(1);

    let payload1: Vec<f32> = (1..=n).map(|x| x as f32).collect();
    let mut bytes1 = Vec::with_capacity(data_bytes);
    for v in &payload1 {
        bytes1.extend_from_slice(&v.to_le_bytes());
    }
    let mut out1 = vec![0u8; data_bytes];
    base.execute(&alg, &bytes1, &mut out1).unwrap();
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
    base.execute(&alg, &bytes2, &mut out2).unwrap();
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
        function()
            .entry(vec![
                ret(),
            ]),
        function()
            .import(0, "cl_cuda_init")
            .import(1, "cl_cuda_create_buffer")
            .import(2, "cl_cuda_upload_ptr")
            .import(3, "cl_cuda_download_ptr")
            .import(4, "cl_cublas_sgemv")
            .import(5, "cl_cuda_sync")
            .import(6, "cl_cuda_cleanup")
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

    let config = Artifact {
        functions: exported(clif_prog.clone()),
        memory_size: mem_size as u64,
        data: image(vec![0u8; mem_size]),
    };
    let mut base = Base::new(config).unwrap();

    let alg = at(1);

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
    base.execute(&alg, &payload1, &mut out1).unwrap();
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
    base.execute(&alg, &payload2, &mut out2).unwrap();
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
        function()
            .import(0, "cl_file_write")
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
        function().entry(vec![
            iconst64(v(1), 11),
            store(v(1), v(0), 2000),
            ret(),
        ]),
        // u0:1 writes 22 at 2008
        function().entry(vec![
            iconst64(v(1), 22),
            store(v(1), v(0), 2008),
            ret(),
        ]),
        // u0:2 calls both, then writes the pair out
        function()
            .local(0, 0)
            .local(1, 1)
            .import(2, "cl_file_write")
            .entry(vec![
                call(None, 0, &[v(0), data_ptr(), data_len(), out_ptr(), out_len()]),
                call(None, 1, &[v(0), data_ptr(), data_len(), out_ptr(), out_len()]),
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
            function()
                .local(0, 3)
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

/// The error `Base::new` answers for `functions`, which must be refused.
fn refusal(functions: Vec<Function>) -> String {
    match Base::new(cranelift_config(vec![0u8; 256], functions)) {
        Err(base::Error::Clif(msg)) => msg,
        Err(e) => panic!("expected a build error, got {e:?}"),
        Ok(_) => panic!("expected the program to be refused"),
    }
}

/// A name base does not provide is refused, even one the process has loaded:
/// `abort` is in libc, and the JIT would otherwise have found it there.
#[test]
fn clif_error_import_base_does_not_provide() {
    let msg = refusal(program(
        function()
            .import(0, "abort")
            .entry(vec![call(None, 0, &[]), ret()]),
    ));
    assert!(msg.contains("abort") && msg.contains("does not provide"), "{msg}");
}

/// A program calling `cl_sinf` with an integer would pass it in a register the
/// function never reads. The artifact declares no signature, so the call is
/// checked against the one base's table gives that import.
#[test]
fn clif_error_import_at_the_wrong_signature() {
    let msg = refusal(program(
        function()
            .import(0, "cl_sinf")
            .entry(vec![call(None, 0, &[v(0)]), ret()]),
    ));
    assert!(msg.contains("cl_sinf") && msg.contains("takes"), "{msg}");
}

/// A program's own functions are reached by index, never by the name the JIT
/// happens to give them: that name is a numbering, and would move with it.
#[test]
fn clif_error_import_naming_an_own_function() {
    let msg = refusal(programs(vec![
        noop(),
        function()
            .import(0, "fn_0")
            .entry(vec![call(None, 0, &[v(0), data_ptr(), data_len(), out_ptr(), out_len()]), ret()]),
    ]));
    assert!(msg.contains("fn_0") && msg.contains("does not provide"), "{msg}");
}

/// A call passing fewer arguments than its callee takes would leave the rest to
/// whatever the registers held. What the callee takes is read off its own entry
/// block, so a call cannot be checked against anything else.
#[test]
fn clif_error_local_call_at_the_wrong_signature() {
    let msg = refusal(programs(vec![
        noop(),
        function()
            .local(0, 0)
            .entry(vec![call(None, 0, &[v(0)]), ret()]),
    ]));
    assert!(msg.contains("u0:0") && msg.contains("takes"), "{msg}");
}

// --- export names ------------------------------------------------------------

/// An artifact whose functions are exactly `functions`, named as they say.
fn named(functions: Vec<Function>) -> Artifact {
    Artifact { functions, memory_size: 256, data: vec![] }
}

/// Answers `k` as its status.
fn answering(k: i64) -> Func {
    function().entry(vec![iconst64(v(1), k), ret_status(v(1))])
}

#[test]
fn an_entry_point_is_called_by_its_name() {
    let mut base = Base::new(named(programs(vec![
        noop(),
        answering(7).export("seven"),
        answering(9).export("nine"),
    ])))
    .unwrap();
    assert_eq!(base.execute("nine", &[], &mut []).unwrap(), 9);
    assert_eq!(base.execute("seven", &[], &mut []).unwrap(), 7);
    assert_eq!(base::run(named(programs(vec![answering(3).export("main")])), "main").unwrap(), 3);
}

/// A function nobody named is the program's own, and no name reaches it: not
/// even the `u0:N` a host might guess from the artifact.
#[test]
fn an_unexported_function_is_not_callable() {
    let mut base = Base::new(named(programs(vec![answering(1), answering(2).export("two")])))
        .unwrap();
    for name in ["u0:0", "one", ""] {
        let Err(base::Error::Execution(msg)) = base.execute(name, &[], &mut []) else {
            panic!("{name:?} is not exported and should not be callable");
        };
        assert!(msg.contains(&format!("{name:?}")), "{msg}");
    }
    assert_eq!(base.execute("two", &[], &mut []).unwrap(), 2);
}

#[test]
fn clif_error_two_functions_exported_under_one_name() {
    let Err(base::Error::Clif(msg)) =
        Base::new(named(programs(vec![answering(1).export("x"), answering(2).export("x")])))
    else {
        panic!("a name exported twice should be refused");
    };
    assert!(msg.contains("u0:0") && msg.contains("u0:1") && msg.contains("\"x\""), "{msg}");
}

/// A worker takes the one pointer `cl_thread_spawn` hands it; the two-argument
/// function here is shaped like neither that nor an entry point, so a name for
/// it would promise a call base cannot make.
#[test]
fn clif_error_exported_function_not_shaped_like_an_entry() {
    let two = function()
        .export("pair")
        .block(0, &[(v(0), I64), (v(1), I64)], vec![ret()]);
    let Err(base::Error::Clif(msg)) = Base::new(named(programs(vec![two]))) else {
        panic!("a two-parameter function should not be exportable");
    };
    assert!(msg.contains("pair") && msg.contains("2 parameters"), "{msg}");
}

/// The functions in a new order, each local call renumbered to follow its
/// callee: what a generator is free to do, say to put hot code together.
/// `order[k]` is the old position of the function placed at `k`.
fn reordered(functions: &[Function], order: &[u32]) -> Vec<Function> {
    let new_index = |old: u32| order.iter().position(|&o| o == old).unwrap() as u32;
    order
        .iter()
        .map(|&old| {
            let mut f = functions[old as usize].clone();
            for decl in &mut f.fns {
                if let Callee::Local(i) = decl.callee {
                    decl.callee = Callee::Local(new_index(i));
                }
            }
            f
        })
        .collect()
}

/// A host calls by name, so the generator can move functions without a host
/// noticing, even when those functions call each other. The positions do
/// move: a host holding one would now call something else.
#[test]
fn names_survive_a_reordering() {
    // `tens` calls the unexported `times_ten` with 4.
    let times_ten = function().block(
        0,
        &[(v(0), I64)],
        vec![iconst64(v(1), 10), imul(v(2), v(0), v(1)), ret_status(v(2))],
    );
    let tens = function()
        .export("tens")
        .local(0, 1)
        .entry(vec![iconst64(v(1), 4), call(Some(v(2)), 0, &[v(1)]), ret_status(v(2))]);
    let before = programs(vec![noop(), times_ten, tens, answering(7).export("seven")]);
    let after = reordered(&before, &[3, 2, 0, 1]);

    let Callee::Local(callee) = after[1].fns[0].callee else { panic!("a local call") };
    assert_eq!(callee, 3, "the call follows its callee");
    assert_eq!(before[3].export_name.as_deref(), Some("seven"));
    assert_eq!(after[3].export_name, None, "position 3 is now the unexported helper");

    for functions in [before, after] {
        let mut base = Base::new(named(functions)).unwrap();
        assert_eq!(base.execute("tens", &[], &mut []).unwrap(), 40);
        assert_eq!(base.execute("seven", &[], &mut []).unwrap(), 7);
    }
}

/// A callee naming a signature the function never declared was a refusal of
/// its own. It has no test because it has no spelling: an artifact declares no
/// signatures, so a callee cannot name one that is missing — base's table gives
/// an import its signature and a local callee's own entry block gives its.

#[test]
fn clif_error_binding_the_result_of_a_void_callee() {
    let config = cranelift_config(
        vec![0u8; 256],
        program(
            function()
                .import(0, "cl_gpu_init")
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
        program(function().entry(vec![
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
