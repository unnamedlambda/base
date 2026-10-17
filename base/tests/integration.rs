use base::{Artifact, Driver};
use std::fs;

mod common;
use common::*;
use tempfile::TempDir;

/// The artifact running `functions` in `memory`.
///
/// A function the test did not name is exported as its `u0:N`, so a test can
/// call any of them by index; the ones about naming name their own.
fn cranelift_config(memory: Vec<u8>, functions: Vec<Function>) -> Artifact {
    let size = memory.len() as u64;
    let a = artifact(exported(functions), size, image(memory));
    dump(&a);
    a
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
fn read_i64(base: &Driver, offset: usize) -> i64 {
    let mut word = [0u8; 8];
    base.read(offset, &mut word).unwrap();
    i64::from_le_bytes(word)
}

/// The artifact, and the name its function at `fn_idx` is called by.
fn create_cranelift_algorithm(
    fn_idx: u32,
    memory: Vec<u8>,
    functions: Vec<Function>,
) -> (Artifact, String) {
    let config = cranelift_config(memory, functions);
    let name = config.functions[fn_idx as usize].name.clone().unwrap();
    (config, name)
}

/// Compile `artifact` and call `name` once.
fn run(artifact: Artifact, name: impl AsRef<str>) -> Result<i64, base::Error> {
    unsafe { base::run(artifact, name.as_ref()) }
}

/// `functions`, each one the test did not name exported as its `u0:N`.
fn exported(mut functions: Vec<Function>) -> Vec<Function> {
    for (i, f) in functions.iter_mut().enumerate() {
        f.name.get_or_insert_with(|| at(i as u32));
    }
    functions
}

/// A file's length is set and read back through the program's own calls: one
/// made 5 bytes long answers 5.
#[test]
fn a_file_is_made_a_length_and_reports_it() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path().join("len.bin");
    let path_str = format!("{}\0", path.to_str().unwrap());
    let f = function()
        .entry(vec![
            iconst64(v(1), 1000),
            iadd(v(2), v(0), v(1)),
            iconst64(v(3), 5),
            iconst64(v(9600), 256),
            call(Some(v(4)), imp("cl_file_set_len"), &[v(2), v(9600), v(3)]),
            iconst64(v(9601), 256),
            call(Some(v(5)), imp("cl_file_size"), &[v(2), v(9601)]),
            ret_status(v(5)),
        ])
        .export("main");
    let mut memory = vec![0u8; 2048];
    memory[1000..1000 + path_str.len()].copy_from_slice(path_str.as_bytes());
    assert_eq!(run(cranelift_config(memory, program(f)), "main").unwrap(), 5);
    assert_eq!(fs::read(&path).unwrap(), [0u8; 5]);
}

/// The memory is the size the artifact states: an image that runs past it is
/// refused, not made room for.
#[test]
fn a_segment_past_the_stated_memory_is_refused() {
    let data = vec![base_types::Segment { offset: 200, bytes: vec![1; 100] }];
    let Err(e) = Driver::load(artifact(vec![], 256, data)) else {
        panic!("expected the artifact to be refused");
    };
    assert!(e.to_string().contains("ends past the 256 bytes"), "{e}");
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
            .entry(vec![
                iconst64(v(1), 3000),
                iconst64(v(2), 2000),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                iadd(v(11), v(0), v(1)),
                iadd(v(12), v(0), v(2)),
                iconst64(v(9602), 256),
                call(Some(v(5)), imp("cl_file_write_from_ptr"), &[v(11), v(9602), v(12), v(3), v(4)]),
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
            .entry(vec![
                load64(v(1), v(0), 2000),
                load64(v(2), v(0), 2008),
                iadd(v(3), v(1), v(2)),
                store(v(3), v(0), 2016),
                iconst64(v(4), 3000),
                iconst64(v(5), 2016),
                iconst64(v(6), 0),
                iconst64(v(7), 8),
                iadd(v(13), v(0), v(4)),
                iadd(v(14), v(0), v(5)),
                iconst64(v(9603), 256),
                call(Some(v(8)), imp("cl_file_write_from_ptr"), &[v(13), v(9603), v(14), v(6), v(7)]),
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
            .entry(vec![
                load64(v(1), v(0), 2000),
                load64(v(2), v(0), 2008),
                imul(v(3), v(1), v(2)),
                store(v(3), v(0), 2016),
                iconst64(v(4), 3000),
                iconst64(v(5), 2016),
                iconst64(v(6), 0),
                iconst64(v(7), 8),
                iadd(v(9505), v(0), v(4)),
                iadd(v(9506), v(0), v(5)),
                iconst64(v(9604), 256),
                call(Some(v(8)), imp("cl_file_write_from_ptr"), &[v(9505), v(9604), v(9506), v(6), v(7)]),
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
                iadd(v(9507), v(0), v(6)),
                iadd(v(9508), v(0), v(7)),
                iconst64(v(9605), 256),
                call(Some(v(10)), imp("cl_file_write_from_ptr"), &[v(9507), v(9605), v(9508), v(8), v(9)]),
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
            .entry(vec![
                load64(v(1), v(0), 2000),
                load64(v(2), v(0), 2008),
                load64(v(3), v(0), 2016),
                iconst64(v(9001), 0),
                icmp(v(4), "eq", v(1), v(9001)),
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
                iadd(v(9509), v(0), v(5)),
                iadd(v(9510), v(0), v(6)),
                iconst64(v(9606), 256),
                call(Some(v(9)), imp("cl_file_write_from_ptr"), &[v(9509), v(9606), v(9510), v(7), v(8)]),
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
    // Runtime smoke: exercises cl_file_read_to_ptr and cl_file_write_from_ptr
    // via a real round-trip.
    let temp_dir = TempDir::new().unwrap();
    let path_a = temp_dir.path().join("smoke_a.bin");
    let path_b = temp_dir.path().join("smoke_b.bin");
    let path_a_str = format!("{}\0", path_a.to_str().unwrap());
    let path_b_str = format!("{}\0", path_b.to_str().unwrap());

    let clif_prog = program(
        function()
            .entry(vec![
                iconst64(v(1), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 5),
                iadd(v(9511), v(0), v(1)),
                iadd(v(9512), v(0), v(2)),
                iconst64(v(9607), 256),
                call(Some(v(5)), imp("cl_file_write_from_ptr"), &[v(9511), v(9607), v(9512), v(3), v(4)]),
                iconst64(v(6), 3100),
                iadd(v(9513), v(0), v(1)),
                iadd(v(9514), v(0), v(6)),
                iconst64(v(9608), 256),
                call(Some(v(7)), imp("cl_file_read_to_ptr"), &[v(9513), v(9608), v(9514), v(3), v(4)]),
                iadd_imm(v(8), v(0), 2256),
                iadd_imm(v(9), v(0), 3000),
                iconst64(v(9609), 256),
                call(Some(v(10)), imp("cl_file_write_from_ptr"), &[v(8), v(9609), v(9), v(3), v(4)]),
                iadd_imm(v(11), v(0), 3200),
                iconst64(v(9610), 256),
                call(Some(v(12)), imp("cl_file_read_to_ptr"), &[v(8), v(9610), v(11), v(3), v(4)]),
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
fn test_clif_call_basic() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("clif_call_basic.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_prog = program(
        function()
            .entry(vec![
                iconst64(v(1), 3000),
                iconst64(v(2), 2000),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                iadd(v(9515), v(0), v(1)),
                iadd(v(9516), v(0), v(2)),
                iconst64(v(9611), 256),
                call(Some(v(5)), imp("cl_file_write_from_ptr"), &[v(9515), v(9611), v(9516), v(3), v(4)]),
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
            .entry(vec![
                iconst64(v(1), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                iadd(v(9517), v(0), v(1)),
                iadd(v(9518), v(0), v(2)),
                iconst64(v(9612), 256),
                call(Some(v(5)), imp("cl_file_write_from_ptr"), &[v(9517), v(9612), v(9518), v(3), v(4)]),
                ret(),
            ]),
        function()
            .entry(vec![
                iconst64(v(1), 2256),
                iconst64(v(2), 3008),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                iadd(v(9519), v(0), v(1)),
                iadd(v(9520), v(0), v(2)),
                iconst64(v(9613), 256),
                call(Some(v(5)), imp("cl_file_write_from_ptr"), &[v(9519), v(9613), v(9520), v(3), v(4)]),
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
    let base = Driver::load(cranelift_config(memory, clif_prog)).unwrap();
    unsafe { base.execute(&at(0), &[], &mut []) }.unwrap();
    unsafe { base.execute(&at(1), &[], &mut []) }.unwrap();

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
            .entry(vec![
                load64(v(1), v(0), 2000),
                load64(v(2), v(0), 2008),
                iadd(v(3), v(1), v(2)),
                store(v(3), v(0), 2016),
                iconst64(v(4), 3000),
                iconst64(v(5), 2016),
                iconst64(v(6), 0),
                iconst64(v(7), 8),
                iadd(v(9521), v(0), v(4)),
                iadd(v(9522), v(0), v(5)),
                iconst64(v(9614), 256),
                call(Some(v(8)), imp("cl_file_write_from_ptr"), &[v(9521), v(9614), v(9522), v(6), v(7)]),
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
            .entry(vec![
                iconst64(v(1), 3000),
                iconst64(v(2), 2008),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                iadd(v(9523), v(0), v(1)),
                iadd(v(9524), v(0), v(2)),
                iconst64(v(9615), 256),
                call(Some(v(5)), imp("cl_file_write_from_ptr"), &[v(9523), v(9615), v(9524), v(3), v(4)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let base = Driver::load(cranelift_config(memory, clif_prog)).unwrap();
    unsafe { base.execute(&at(0), &[], &mut []) }.unwrap();
    unsafe { base.execute(&at(1), &[], &mut []) }.unwrap();
    unsafe { base.execute(&at(2), &[], &mut []) }.unwrap();

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
            .entry(vec![
                iconst64(v(1), 77),
                store(v(1), v(0), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 2000),
                iconst64(v(4), 0),
                iconst64(v(5), 8),
                iadd(v(9525), v(0), v(2)),
                iadd(v(9526), v(0), v(3)),
                iconst64(v(9616), 256),
                call(Some(v(6)), imp("cl_file_write_from_ptr"), &[v(9525), v(9616), v(9526), v(4), v(5)]),
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
            .entry(vec![
                iconst64(v(1), 2000),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 256),
                iadd(v(9527), v(0), v(1)),
                iadd(v(9528), v(0), v(2)),
                iconst64(v(9617), 256),
                call(Some(v(5)), imp("cl_file_read_to_ptr"), &[v(9527), v(9617), v(9528), v(3), v(4)]),
                ret(),
            ]),
        function()
            .entry(vec![
                iconst64(v(1), 2256),
                iconst64(v(2), 3000),
                iconst64(v(3), 0),
                iconst64(v(4), 256),
                iadd(v(9529), v(0), v(1)),
                iadd(v(9530), v(0), v(2)),
                iconst64(v(9618), 256),
                call(Some(v(5)), imp("cl_file_write_from_ptr"), &[v(9529), v(9618), v(9530), v(3), v(4)]),
                ret(),
            ]),
    ]);

    let mut memory = vec![0u8; 4096];
    memory[2000..2000 + input_str.len()].copy_from_slice(input_str.as_bytes());
    memory[2256..2256 + output_str.len()].copy_from_slice(output_str.as_bytes());

    // Two execute() calls on one Base: fn0 reads input file, fn1 writes output file.
    let base = Driver::load(cranelift_config(memory, clif_prog)).unwrap();
    unsafe { base.execute(&at(0), &[], &mut []) }.unwrap();
    unsafe { base.execute(&at(1), &[], &mut []) }.unwrap();

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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    // First execute: input = 10, expect 30
    let data1 = 10i64.to_le_bytes();
    unsafe { base.execute(&at(0), &data1, &mut []) }
        .unwrap();
    assert_eq!(read_i64(&base, 200), 30);

    // Second execute: input = 100, expect 300
    let data2 = 100i64.to_le_bytes();
    unsafe { base.execute(&at(0), &data2, &mut []) }
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    // First execute: call fn0 only
    let alg1 = at(0);
    unsafe { base.execute(&alg1, &vec![0u8; 4096], &mut []) }.unwrap();
    assert_eq!(read_i64(&base, 200), 42);

    // Second execute: call fn1 only
    let alg2 = at(1);
    unsafe { base.execute(&alg2, &vec![0u8; 4096], &mut []) }.unwrap();
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    // Execute 1: add 10 → total = 10
    let d1 = 10i64.to_le_bytes();
    unsafe { base.execute(&at(0), &d1, &mut []) }
        .unwrap();
    assert_eq!(read_i64(&base, 200), 10);

    // Execute 2: add 25 → total = 35
    let d2 = 25i64.to_le_bytes();
    unsafe { base.execute(&at(0), &d2, &mut []) }
        .unwrap();
    assert_eq!(read_i64(&base, 200), 35);

    // Execute 3: add 5 → total = 40
    let d3 = 5i64.to_le_bytes();
    unsafe { base.execute(&at(0), &d3, &mut []) }
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
            .entry(vec![
                iconst64(v(1), 256),
                iconst64(v(2), 512),
                iconst64(v(3), 0),
                iconst64(v(4), 8),
                iadd(v(9531), v(0), v(1)),
                iadd(v(9532), v(0), v(2)),
                iconst64(v(9619), 256),
                call(Some(v(5)), imp("cl_file_write_from_ptr"), &[v(9531), v(9619), v(9532), v(3), v(4)]),
                ret(),
            ]),
    );

    // Execute 1: write value 42 to file1
    let mut mem1 = vec![0u8; 4096];
    mem1[256..256 + file1_str.len()].copy_from_slice(file1_str.as_bytes());
    mem1[512..520].copy_from_slice(&42u64.to_le_bytes());
    let config1 = artifact(exported(clif_prog.clone()), 4096, image(mem1));
    let base = Driver::load(config1).unwrap();
    unsafe { base.execute(&at(0), &[], &mut []) }
    .unwrap();
    assert!(file1.exists());
    let data1 = fs::read(&file1).unwrap();
    assert_eq!(u64::from_le_bytes(data1[..8].try_into().unwrap()), 42);

    // Execute 2: write value 99 to file2 — new Base with different initial_memory
    let mut mem2 = vec![0u8; 4096];
    mem2[256..256 + file2_str.len()].copy_from_slice(file2_str.as_bytes());
    mem2[512..520].copy_from_slice(&99u64.to_le_bytes());
    let config2 = artifact(exported(clif_prog.clone()), 4096, image(mem2));
    let base2 = Driver::load(config2).unwrap();
    unsafe { base2.execute(&at(0), &[], &mut []) }.unwrap();
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    // Execute with 0 units
    unsafe { base.execute(&at(0), &vec![0u8; 4096], &mut []) }
    .unwrap();

    // Execute with 2 units
    unsafe { base.execute(&at(0), &vec![0u8; 4096], &mut []) }
    .unwrap();

    // Execute with 4 units
    unsafe { base.execute(&at(0), &vec![0u8; 4096], &mut []) }
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

    let config = artifact(exported(clif_prog.clone()), 4096, image(mem));
    let base = Driver::load(config).unwrap();

    let data = 99i64.to_le_bytes();
    unsafe { base.execute(&at(0), &data, &mut []) }
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    // Execute 1: seed 77 at offset 200
    unsafe { base.execute(&at(0), &[], &mut []) }
    .unwrap();

    // Execute 2: input=5 via pointer, read persistent 77 from offset 200
    let data = 5i64.to_le_bytes();
    unsafe { base.execute(&at(1), &data, &mut []) }
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    // Three executes with empty memory — counter should increment each time
    for expected in 1..=3 {
        unsafe { base.execute(&at(0), &[], &mut []) }
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    // Execute 1: 10 + 20 = 30
    let mut d1 = vec![0u8; 16];
    d1[0..8].copy_from_slice(&10i64.to_le_bytes());
    d1[8..16].copy_from_slice(&20i64.to_le_bytes());
    unsafe { base.execute(&at(0), &d1, &mut []) }
        .unwrap();
    assert_eq!(read_i64(&base, 200), 30);

    // Execute 2: 100 + 200 = 300 — pointer should update to new buffer
    let mut d2 = vec![0u8; 16];
    d2[0..8].copy_from_slice(&100i64.to_le_bytes());
    d2[8..16].copy_from_slice(&200i64.to_le_bytes());
    unsafe { base.execute(&at(0), &d2, &mut []) }
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    // Execute 3 times with values 100, 200, 300
    for &val in &[100i64, 200, 300] {
        let d = val.to_le_bytes();
        unsafe { base.execute(&at(0), &d, &mut []) }
        .unwrap();
    }

    // Final read: count at 400 should be 3, values at 500/508/516 should be 100/200/300
    // One more execute to read output — pass a dummy input
    let d = 999i64.to_le_bytes();
    unsafe { base.execute(&at(0), &d, &mut []) }
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
    let base = Driver::load(cfg).unwrap();
    assert_eq!(unsafe { base.execute(&alg, &[], &mut []) }.unwrap(), 7);
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
    let base = Driver::load(cfg).unwrap();
    assert_eq!(unsafe { base.execute(&alg, &[], &mut []) }.unwrap(), 0);
}

/// What a function answers is its signature's, so a return answering anything
/// else is a malformed body, which the verifier refuses. An exported one
/// answers a status by what exporting it means, so a return that answers
/// nothing is refused the same way.
#[test]
fn clif_error_returns_disagree() {
    let disagreeing = |params: &[(Val, ClifTy)]| {
        function()
            .block(0, params, vec![iconst64(v(1), 1), brif(v(1), 1, &[], 2, &[])])
            .block(1, &[], vec![ret_status(v(1))])
            .block(2, &[], vec![ret()])
    };
    let Err(base::Error::Clif(msg)) =
        Driver::load(named(programs(vec![answering(0).export("main"), disagreeing(&[(v(5), I64)])])))
    else {
        panic!("expected a function whose returns disagree to be refused");
    };
    assert!(msg.contains("return must match"), "message should say what is wrong: {msg}");
    let Err(base::Error::Clif(msg)) = Driver::load(named(programs(vec![disagreeing(&[(v(5), I64), (v(6), I64), (v(7), I64), (v(8), I64)]).export("main")])))
    else {
        panic!("expected an entry point that returns nothing to be refused");
    };
    assert!(msg.contains("so a host calls it") && msg.contains("return"), "{msg}");
}

#[test]
fn clif_error_value_used_before_defined() {
    // v9 is never defined. The text path reported this as a parse error; the
    // decoder reports it against the program, which is where the defect is.
    let config = cranelift_config(
        vec![0u8; 256],
        program(function().entry(vec![store(v(9), v(0), 0), ret()])),
    );
    let Err(err) = Driver::load(config) else {
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

// A call naming a callee the function never declared was a refusal of its own.
// It has no test because it has no spelling: a call carries its callee, so
// there is nothing to declare it against. What is still refusable is a callee
// that names nothing — an import base does not provide, or a local index past
// the artifact's functions — and those have their own tests below.

#[test]
fn clif_parse_error_empty_ir_no_error() {
    // Empty string should NOT error — it skips compilation entirely
    let config = artifact(Default::default(), 256, vec![]);
    let base = Driver::load(config);
    assert!(base.is_ok());
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    let mut data = vec![0u8; 16];
    data[0..8].copy_from_slice(&100i64.to_le_bytes());
    data[8..16].copy_from_slice(&200i64.to_le_bytes());

    let alg = at(0);

    unsafe { base.execute(&alg, &data, &mut []) }.unwrap();
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

    let config = artifact(exported(clif_prog.clone()), 4096, image(initial));

    let alg = at(0);

    let base = Driver::load(config).unwrap();
    unsafe { base.execute(&alg, &[], &mut []) }.unwrap();
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

    let config = artifact(exported(clif_prog.clone()), 4096, image(initial));

    let alg = at(0);

    let base = Driver::load(config).unwrap();
    unsafe { base.execute(&alg, &[], &mut []) }.unwrap();
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    let mut data = vec![0u8; 8];
    data[0..8].copy_from_slice(&6i64.to_le_bytes());

    let mut out = vec![0u8; 8];

    let alg = at(0);

    unsafe { base.execute(&alg, &data, &mut out) }.unwrap();
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    let alg = at(0);

    // Call 1: data=111
    let data1 = 111i64.to_le_bytes().to_vec();
    let mut out1 = vec![0u8; 8];
    unsafe { base.execute(&alg, &data1, &mut out1) }.unwrap();
    assert_eq!(i64::from_le_bytes(out1[0..8].try_into().unwrap()), 111);

    // Call 2: data=222, different buffers
    let data2 = 222i64.to_le_bytes().to_vec();
    let mut out2 = vec![0u8; 8];
    unsafe { base.execute(&alg, &data2, &mut out2) }.unwrap();
    assert_eq!(i64::from_le_bytes(out2[0..8].try_into().unwrap()), 222);

    // out1 should be unchanged from call 2
    assert_eq!(i64::from_le_bytes(out1[0..8].try_into().unwrap()), 111);
}

#[test]
fn test_data_ptr_with_large_buffer_no_shared_mem_copy() {
    // Data buffer is larger than the arena. The data pointer gives CLIF
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

    let config = artifact(exported(clif_prog.clone()), 256, vec![]);
    let base = Driver::load(config).unwrap();

    // Data is 1KB — much larger than the arena (256)
    let mut data = vec![0u8; 1024];
    // Write sentinel at the very end
    data[1016..1024].copy_from_slice(&999i64.to_le_bytes());

    let alg = at(0);

    unsafe { base.execute(&alg, &data, &mut []) }.unwrap();
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

    let config = artifact(exported(clif_prog.clone()), 4096, image(initial));
    let base = Driver::load(config).unwrap();

    let alg = at(0);

    // Dynamic input = 7
    let data = 7i64.to_le_bytes().to_vec();
    unsafe { base.execute(&alg, &data, &mut []) }.unwrap();
    assert_eq!(
        read_i64(&base, 200),
        91,
        "13 * 7 = 91: static config from initial_memory, dynamic input via pointer"
    );
}

#[test]
fn test_execute_into_out_buffer_larger_than_memory() {
    // Out buffer can be any size — it's caller-owned, not bounded by the arena.
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

    let config = artifact(exported(clif_prog.clone()), 64, vec![]);
    let base = Driver::load(config).unwrap();

    let alg = at(0);

    // Tiny shared memory (64 bytes) but large out buffer
    let data = vec![0u8; 8]; // need non-empty data so pointers at 8-16 get written, but we need out ptrs
    let mut out = vec![0u8; 32];
    unsafe { base.execute(&alg, &data, &mut out) }.unwrap();

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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);

    let alg = at(0);

    let data = 777i64.to_le_bytes().to_vec();
    let base = Driver::load(config).unwrap();
    unsafe { base.execute(&alg, &data, &mut []) }.unwrap();
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);

    let alg = at(0);

    let data = vec![42u8]; // single byte
    let base = Driver::load(config).unwrap();
    unsafe { base.execute(&alg, &data, &mut []) }.unwrap();
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

    let config = artifact(exported(clif_prog.clone()), 4096, vec![]);
    let base = Driver::load(config).unwrap();

    let alg = at(0);

    // Call 1: 8-byte buffer
    let data1 = 11i64.to_le_bytes().to_vec();
    unsafe { base.execute(&alg, &data1, &mut []) }.unwrap();
    assert_eq!(read_i64(&base, 200), 11);
    assert_eq!(read_i64(&base, 208), 8);

    // Call 2: 16-byte buffer (different size!)
    let mut data2 = vec![0u8; 16];
    data2[0..8].copy_from_slice(&22i64.to_le_bytes());
    unsafe { base.execute(&alg, &data2, &mut []) }.unwrap();
    assert_eq!(read_i64(&base, 200), 22);
    assert_eq!(read_i64(&base, 208), 16, "data_len should reflect new buffer size");
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
        iadd(v(9533), v(0), v(900)),
        iadd(v(9534), v(0), v(901)),
        iconst64(v(9620), 256),
        call(Some(v(904)), imp("cl_file_write_from_ptr"), &[v(9533), v(9620), v(9534), v(902), v(903)]),
        ret(),
    ]);
    let prog = program(
        function()
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
            icmp(v(8), "ult", v(6), v(7)),
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
    // ireduce32, uextend64, sextend64, istore8, an aligned store
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
            // istore8 writes one byte; an aligned store states its alignment
            iconst64(v(7), 0xAB),
            istore8(v(7), v(0), 2024),
            iconst64(v(8), 77),
            store_aligned(v(8), v(0), 2032),
        ],
        40,
    );
    let got = i64s(&bytes);
    assert_eq!(got[0], -5, "sextend64(-5i32)");
    assert_eq!(got[1], 0xFFFF_FFFB, "uextend64(-5i32)");
    assert_eq!(got[2], 0x2A, "ireduce32 keeps the low word");
    assert_eq!(got[3] & 0xFF, 0xAB, "istore8");
    assert_eq!(got[4], 77, "aligned store");
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
            fcmp(v(7), "gt", v(2), v(3)),
            uextend64(v(70), v(7)),
            store(v(70), v(0), 2024),
            fcmp(v(8), "lt", v(2), v(3)),
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
    // goes through `Callee::Function`, not the symbol table.
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
            .entry(vec![
                call(Some(v(30)), loc(0), &[data_ptr(), data_len(), out_ptr(), out_len()]),
                call(Some(v(31)), loc(1), &[data_ptr(), data_len(), out_ptr(), out_len()]),
                iconst64(v(1), 3000),
                iconst64(v(2), 2000),
                iconst64(v(3), 0),
                iconst64(v(4), 16),
                iadd(v(9535), v(0), v(1)),
                iadd(v(9536), v(0), v(2)),
                iconst64(v(9621), 256),
                call(Some(v(5)), imp("cl_file_write_from_ptr"), &[v(9535), v(9621), v(9536), v(3), v(4)]),
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
                .entry(vec![call(None, loc(3), &[v(0)]), ret()]),
        ),
    );
    let Err(err) = Driver::load(config) else {
        panic!("expected an error for a local call to a function that is not defined");
    };
    let base::Error::Clif(msg) = err else {
        panic!("expected Error::Clif");
    };
    assert!(msg.contains("u0:3"), "message should name the callee: {msg}");
}

/// The error `Driver::load` answers for `functions`, which must be refused.
fn refusal(functions: Vec<Function>) -> String {
    match Driver::load(cranelift_config(vec![0u8; 256], functions)) {
        Err(base::Error::Clif(msg)) => msg,
        Err(e) => panic!("expected a build error, got {e:?}"),
        Ok(_) => panic!("expected the program to be refused"),
    }
}

/// A symbol `base` does not have answers -1, as a missing library's do, and
/// is never looked up in what else the process has loaded: `abort` is in libc,
/// and `fn_0` is what the JIT names the program's first function.
#[test]
fn a_symbol_base_does_not_have_answers_minus_one() {
    for symbol in ["abort", "fn_0"] {
        let f = function()
            .entry(vec![call(Some(v(10)), base_fn(symbol, &[], Some(I64)), &[]), ret_status(v(10))])
            .export("main");
        assert_eq!(run(named(program(f)), "main").unwrap(), -1, "{symbol}");
    }
}

/// A call is checked against the signature its callee declares: passing
/// `cl_native_arch`, declared to take nothing, an argument is refused.
#[test]
fn clif_error_call_at_the_wrong_signature() {
    let msg = refusal(program(
        function()
            .entry(vec![call(None, imp("cl_native_arch"), &[v(0)]), ret()]),
    ));
    assert!(msg.contains("cl_native_arch") && msg.contains("takes"), "{msg}");
}

/// A call passing fewer arguments than its callee takes would leave the rest to
/// whatever the registers held, so it is checked against the callee's own
/// signature.
#[test]
fn clif_error_local_call_at_the_wrong_signature() {
    let msg = refusal(programs(vec![
        noop(),
        function()
            .entry(vec![call(None, loc(0), &[v(0)]), ret()]),
    ]));
    assert!(msg.contains("u0:0") && msg.contains("takes"), "{msg}");
}

// --- export names ------------------------------------------------------------

/// An artifact whose functions are exactly `functions`, named as they say.
fn named(functions: Vec<Function>) -> Artifact {
    artifact(functions, 256, vec![])
}

/// Answers `k` as its status.
fn answering(k: i64) -> Func {
    function().entry(vec![iconst64(v(1), k), ret_status(v(1))])
}

#[test]
fn an_entry_point_is_called_by_its_name() {
    let base = Driver::load(named(programs(vec![
        noop(),
        answering(7).export("seven"),
        answering(9).export("nine"),
    ])))
    .unwrap();
    assert_eq!(unsafe { base.execute("nine", &[], &mut []) }.unwrap(), 9);
    assert_eq!(unsafe { base.execute("seven", &[], &mut []) }.unwrap(), 7);
    assert_eq!(unsafe { base::run(named(programs(vec![answering(3).export("main")])), "main") }.unwrap(), 3);
}

/// A function nobody named is the program's own, and no name reaches it: not
/// even the `u0:N` a host might guess from the artifact.
#[test]
fn an_unexported_function_is_not_callable() {
    let base = Driver::load(named(programs(vec![answering(1), answering(2).export("two")])))
        .unwrap();
    for name in ["u0:0", "one", ""] {
        let Err(base::Error::NoSuchFunction(msg)) = (unsafe { base.execute(name, &[], &mut []) }) else {
            panic!("{name:?} is not exported and should not be callable");
        };
        assert!(msg.contains(&format!("{name:?}")), "{msg}");
    }
    assert_eq!(unsafe { base.execute("two", &[], &mut []) }.unwrap(), 2);
}

#[test]
fn clif_error_two_functions_exported_under_one_name() {
    let Err(base::Error::Clif(msg)) =
        Driver::load(named(programs(vec![answering(1).export("x"), answering(2).export("x")])))
    else {
        panic!("a name exported twice should be refused");
    };
    assert!(msg.contains("u0:0") && msg.contains("u0:1") && msg.contains("\"x\""), "{msg}");
}

/// A worker takes the one pointer `cl_thread_start` hands it; the two-argument
/// function here is shaped like neither that nor an entry point, so a name for
/// it would promise a call base cannot make.
#[test]
fn clif_error_exported_function_not_shaped_like_an_entry() {
    let two = function()
        .export("pair")
        .block(0, &[(v(0), I64), (v(1), I64)], vec![ret()]);
    let Err(base::Error::Clif(msg)) = Driver::load(named(programs(vec![two]))) else {
        panic!("a two-parameter function should not be exportable");
    };
    assert!(msg.contains("u0:0 has a name, so a host calls it, and its signature is not BaseFunction's"), "{msg}");
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
            for b in &mut f.blocks {
                for inst in &mut b.insts {
                    let c = match inst {
                        Inst::Call(_, _, _, c, _) | Inst::FuncAddr(_, _, _, c) => c,
                        _ => continue,
                    };
                    if let Callee::Function(i) = *c {
                        *c = Callee::Function(new_index(i));
                    }
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
        .entry(vec![iconst64(v(1), 4), call(Some(v(2)), loc(1), &[v(1)]), ret_status(v(2))]);
    let before = programs(vec![noop(), times_ten, tens, answering(7).export("seven")]);
    let after = reordered(&before, &[3, 2, 0, 1]);

    // After the entry's `arena`, the constant, then the call.
    let Inst::Call(_, _, _, Callee::Function(callee), _) = after[1].blocks[0].insts[2] else {
        panic!("a local call")
    };
    assert_eq!(callee, 3, "the call follows its callee");
    assert_eq!(before[3].name.as_deref(), Some("seven"));
    assert_eq!(after[3].name, None, "position 3 is now the unexported helper");

    for functions in [before, after] {
        let base = Driver::load(named(functions)).unwrap();
        assert_eq!(unsafe { base.execute("tens", &[], &mut []) }.unwrap(), 40);
        assert_eq!(unsafe { base.execute("seven", &[], &mut []) }.unwrap(), 7);
    }
}

/// An instruction is a format and an opcode of it, at a type CLIF defines it
/// at, with the type named exactly where CLIF does not take it from an operand.
/// Anything else is refused at load with what is wrong, never built: building
/// an instruction at a type its opcode is not defined at can panic inside
/// Cranelift.
#[test]
fn an_instruction_cranelift_does_not_define_is_refused() {
    let refused = |insts: Vec<Inst>, says: &str| {
        let Err(base::Error::Clif(msg)) = Driver::load(cranelift_config(vec![0u8; 64], program(function().entry(insts))))
        else {
            panic!("expected a refusal saying {says:?}");
        };
        assert!(msg.contains(says), "{msg}");
    };
    // No such opcode.
    refused(vec![Inst::Binary(vec![v(1)], "iadd_frobnicate".into(), None, v(0), v(0)), ret()], "not a CLIF opcode");
    // An opcode of another format.
    refused(vec![Inst::Binary(vec![v(1)], "ineg".into(), None, v(0), v(0)), ret()], "Unary instruction");
    // A type CLIF does not take from an operand, left out.
    refused(vec![Inst::Unary(vec![v(1)], "uextend".into(), None, v(0)), ret()], "names none");
    // A type CLIF takes from an operand, named.
    refused(vec![Inst::Binary(vec![v(1)], "iadd".into(), Some(I64), v(0), v(0)), ret()], "takes its type from an operand");
    // A type the opcode is not defined at.
    refused(vec![Inst::Binary(vec![v(1)], "fadd".into(), None, v(0), v(0)), ret()], "not defined at i64");
    // A type not as CLIF spells it.
    refused(vec![iconst(v(1), ClifTy("i32x01".into()), 0), ret()], "not a CLIF type");
    // Results not named, or named for an instruction that has none.
    refused(vec![Inst::Binary(vec![], "iadd".into(), None, v(0), v(0)), ret()], "answers 1 result");
    refused(vec![Inst::Store(vec![v(1)], "store".into(), None, vec![], v(0), v(0), 0), ret()], "answers 0 results");
    // A value defined twice.
    refused(vec![iconst64(v(1), 1), iconst64(v(1), 2), ret()], "v1 is defined twice");
    // Flags: one CLIF does not have, one that is the engine's, two out of CLIF's order.
    let load_with = |flags: &[&str]| Inst::Load(vec![v(1)], "load".into(), Some(I64), flags.iter().map(|f| f.to_string()).collect(), v(0), 0);
    refused(vec![load_with(&["fast"]), ret()], "not a CLIF memory flag");
    refused(vec![load_with(&["notrap"]), ret()], "the engine's to say");
    refused(vec![load_with(&["readonly", "aligned"]), ret()], "CLIF writes as aligned readonly");
    // A parameter attribute CLIF does not write.
    let normal = Signature { convention: Convention::Host, params: vec![AbiParam(I64, vec!["normal".into()])], returns: vec![] };
    refused(vec![call_indirect(None, normal, v(0), &[v(0)]), ret()], "not a parameter as CLIF writes one");
}

/// Everything an artifact declares is declared once and found where it is
/// named: a primitive past the table, one declared twice, and two blocks of
/// one name are refused.
#[test]
fn what_an_artifact_names_resolves_once() {
    let refusal_of = |a: Artifact| match Driver::load(a) {
        Err(base::Error::Clif(msg)) => msg,
        _ => panic!("expected a refusal"),
    };
    let calls = |c: Callee| program(function().entry(vec![call(None, c, &[]), ret()]));
    let msg = refusal_of(artifact(exported(calls(Callee::Primitive(1000))), 64, vec![]));
    assert!(msg.contains("primitive 1000, which the program does not declare"), "{msg}");

    let f = calls(base_fn("cl_twice", &[], None));
    let mut a = artifact(exported(f), 64, vec![]);
    a.primitives.push(a.primitives[0].clone());
    assert!(refusal_of(a).contains("declared twice"));

    let f = function().entry(vec![jump(1, &[])]).block(1, &[], vec![ret()]).block(1, &[], vec![ret()]);
    let msg = refusal_of(artifact(exported(program(f)), 64, vec![]));
    assert!(msg.contains("names two blocks block1"), "{msg}");
}

#[test]
fn clif_error_a_float_constant_names_no_type() {
    let config = cranelift_config(
        vec![0u8; 256],
        program(function().entry(vec![
            base_types::clif::Inst::UnaryIeee64(vec![v(1)], "f64const".into(), Some(I64), 0),
            ret(),
        ])),
    );
    let Err(err) = Driver::load(config) else {
        panic!("expected an error for a float constant naming a type");
    };
    let base::Error::Clif(msg) = err else {
        panic!("expected Error::Clif");
    };
    assert!(msg.contains("f64const"), "message should say what is wrong: {msg}");
}

/// Machine code carried as data: the program keeps `a + b` as bytes in its own
/// memory, asks the runtime to make them executable, and calls the address it
/// gets back itself (`call_indirect` at the architecture's convention), all as CLIF. The same
/// program answers the same way on every OS of the architecture, because the
/// code is called under one convention per architecture.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn a_program_runs_machine_code_it_carries_as_data() {
    #[cfg(target_arch = "x86_64")]
    let (arch, add): (i64, &[u8]) = (1, &[0x48, 0x89, 0xf8, 0x48, 0x01, 0xf0, 0xc3]); // mov rax,rdi; add rax,rsi; ret
    #[cfg(target_arch = "aarch64")]
    let (arch, add): (i64, &[u8]) = (2, &[0x00, 0x00, 0x01, 0x8b, 0xc0, 0x03, 0x5f, 0xd6]); // add x0,x0,x1; ret

    let mut memory = vec![0u8; 4096];
    memory[0x100..0x100 + add.len()].copy_from_slice(add);
    let clif_prog = program(function().entry(vec![
        call(Some(v(1)), imp("cl_native_arch"), &[]),
        store(v(1), v(0), 300),
        iadd_imm(v(2), v(0), 0x100),
        iconst64(v(3), add.len() as i64),
        call(Some(v(4)), imp("cl_native_load"), &[v(2), v(3)]),
        iconst64(v(5), 40),
        iconst64(v(6), 2),
        iconst64(v(7), 0),
        call_indirect(Some(v(8)), sig(Convention::Arch, &[I64; 4], Some(I64)), v(4), &[v(5), v(6), v(7), v(7)]),
        store(v(8), v(0), 200),
        call(None, imp("cl_native_free"), &[v(4)]),
        ret(),
    ]));
    let mut base = Driver::load(cranelift_config(memory, clif_prog)).unwrap();
    unsafe { base.execute(&at(0), &[], &mut []) }.unwrap();
    assert_eq!(read_i64(&base, 200), 42);
    assert_eq!(base.arena()[300] as i64, arch);
}

/// A callback: the host hands the program a function of its own, an entry
/// point's shape, by its address in the input, and the program calls it
/// (`call_indirect` at the host's convention) on the rest of the input and its own output, answering
/// what it answered. Nothing of the engine's runs between the two.
#[test]
fn a_program_calls_back_a_host_function_it_is_handed() {
    unsafe extern "C" fn double(input: *const u8, len: usize, out: *mut u8, cap: usize) -> i64 {
        let n = len.min(cap);
        for i in 0..n {
            unsafe { *out.add(i) = (*input.add(i)).wrapping_mul(2) };
        }
        n as i64 + 100
    }
    let clif_prog = program(function().entry(vec![
        load64(v(1), data_ptr(), 0),
        iadd_imm(v(2), data_ptr(), 8),
        iadd_imm(v(3), data_len(), -8),
        call_indirect(Some(v(4)), sig(Convention::Host, &[I64; 4], Some(I64)), v(1), &[v(2), v(3), out_ptr(), out_len()]),
        ret_status(v(4)),
    ]));
    let base = Driver::load(cranelift_config(vec![0u8; 64], clif_prog)).unwrap();
    let f: base::FunctionPtr = double;
    let mut input = (f as usize as u64).to_le_bytes().to_vec();
    input.extend_from_slice(&[1, 2, 3]);
    let mut out = [0u8; 3];
    assert_eq!(unsafe { base.execute(&at(0), &input, &mut out) }.unwrap(), 103);
    assert_eq!(out, [2, 4, 6]);
}

/// An indirect call is checked like any other: an address and the arguments its
/// signature declares, or the load says what is wrong instead of compiling a
/// call with arguments in the wrong registers.
#[test]
fn a_native_call_with_the_wrong_arguments_is_refused() {
    let clif_prog = program(function().entry(vec![
        iconst64(v(1), 0),
        call_indirect(Some(v(2)), sig(Convention::Arch, &[I64; 4], Some(I64)), v(1), &[v(1)]),
        ret(),
    ]));
    let Err(err) = Driver::load(cranelift_config(vec![0u8; 4096], clif_prog)) else {
        panic!("a native call with one argument loaded");
    };
    let base::Error::Clif(msg) = err else {
        panic!("expected Error::Clif");
    };
    assert!(msg.contains("indirect callee") && msg.contains("1 argument"), "{msg}");
}
