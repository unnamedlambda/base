use base::{run, Base, RecordBatch};
use base_types::{Action, Algorithm, BaseConfig, Kind, OutputBatchSchema, OutputColumn, OutputType};
use std::fs;
use std::sync::Arc;
use arrow_array::{Int64Array, Float64Array, StringArray};
use arrow_schema::{Schema, Field, DataType};
use tempfile::TempDir;

fn create_cranelift_algorithm(
    actions: Vec<Action>,
    memory: Vec<u8>,
    cranelift_units: usize,
    cranelift_ir: String,
) -> (BaseConfig, Algorithm) {
    let config = BaseConfig {
        cranelift_ir,
        memory_size: memory.len(),
        context_offset: 0,
        initial_memory: memory,
    };
    let algorithm = Algorithm {
        actions,
        cranelift_units,
        timeout_ms: Some(5000),
        output: vec![],
    };
    (config, algorithm)
}

#[test]
fn test_integration_conditional_jump() {
    let temp_dir = TempDir::new().unwrap();
    let test_file_a = temp_dir.path().join("result_a.txt");
    let test_file_b = temp_dir.path().join("result_b.txt");
    let file_a_str = format!("{}\0", test_file_a.to_str().unwrap());
    let file_b_str = format!("{}\0", test_file_b.to_str().unwrap());

    // Two CLIF functions: fn0 writes to file A, fn1 writes to file B
    // Both write 8 bytes from offset 3016 (data value 42)
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3016
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
    v2 = iconst.i64 3016
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + file_a_str.len()].copy_from_slice(file_a_str.as_bytes());
    memory[2256..2256 + file_b_str.len()].copy_from_slice(file_b_str.as_bytes());
    // Conditions: 1 (true) and 0 (false)
    memory[3000..3008].copy_from_slice(&1u64.to_le_bytes());
    memory[3008..3016].copy_from_slice(&0u64.to_le_bytes());
    // Data value
    memory[3016..3024].copy_from_slice(&42u64.to_le_bytes());

    let flag_a = 1024u32;
    let flag_b = 1032u32;

    let actions = vec![
        // Action 0: CLIF fn0 writes file A (dispatched by src=0)
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        // Action 1: CLIF fn1 writes file B (dispatched by src=1)
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        // Action 2: ConditionalJump — condition true → jump to 5 (skip A dispatch)
        Action { kind: Kind::ConditionalJump, src: 3000, dst: 5, offset: 0, size: 0 },
        // Action 3: ClifCallAsync CLIF fn0 (SKIPPED)
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: flag_a, size: 1 },
        // Action 4: Wait (SKIPPED)
        Action { kind: Kind::Wait, dst: flag_a, src: 0, offset: 0, size: 0 },
        // Action 5: ConditionalJump — condition false → fall through
        Action { kind: Kind::ConditionalJump, src: 3008, dst: 99, offset: 0, size: 0 },
        // Action 6: ClifCallAsync CLIF fn1 (EXECUTED)
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 1, offset: flag_b, size: 1 },
        // Action 7: Wait
        Action { kind: Kind::Wait, dst: flag_b, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    assert!(!test_file_a.exists());
    assert!(test_file_b.exists());
    let contents = fs::read(&test_file_b).unwrap();
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(value, 42);
}

#[test]
fn test_integration_conditional_jump_variable_size() {
    // size=4 checks only first 4 bytes; size=8 checks all 8
    let temp_dir = TempDir::new().unwrap();
    let test_file_4byte = temp_dir.path().join("result_4byte.txt");
    let test_file_8byte = temp_dir.path().join("result_8byte.txt");
    let file_4_str = format!("{}\0", test_file_4byte.to_str().unwrap());
    let file_8_str = format!("{}\0", test_file_8byte.to_str().unwrap());

    // Two CLIF functions: fn0 writes to 4byte file, fn1 writes to 8byte file
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3016
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
    v2 = iconst.i64 3016
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + file_4_str.len()].copy_from_slice(file_4_str.as_bytes());
    memory[2256..2256 + file_8_str.len()].copy_from_slice(file_8_str.as_bytes());
    // Condition at 3000: first 4 bytes zero, next 4 non-zero
    memory[3000..3004].copy_from_slice(&0u32.to_le_bytes());
    memory[3004..3008].copy_from_slice(&0xFFu32.to_le_bytes());
    // Data value
    memory[3016..3024].copy_from_slice(&99u64.to_le_bytes());

    let flag_4 = 1024u32;
    let flag_8 = 1032u32;

    let actions = vec![
        // Action 0: CLIF fn0 writes 4byte file
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        // Action 1: CLIF fn1 writes 8byte file
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        // Action 2: ConditionalJump size=4 (first 4 bytes zero → no jump)
        Action { kind: Kind::ConditionalJump, src: 3000, dst: 5, offset: 0, size: 4 },
        // Action 3: ClifCallAsync CLIF fn0 (EXECUTED)
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: flag_4, size: 1 },
        // Action 4: Wait
        Action { kind: Kind::Wait, dst: flag_4, src: 0, offset: 0, size: 0 },
        // Action 5: ConditionalJump size=8 (bytes 4-7 non-zero → jump)
        Action { kind: Kind::ConditionalJump, src: 3000, dst: 8, offset: 0, size: 8 },
        // Action 6: ClifCallAsync CLIF fn1 (SKIPPED)
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 1, offset: flag_8, size: 1 },
        // Action 7: Wait (SKIPPED)
        Action { kind: Kind::Wait, dst: flag_8, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    assert!(test_file_4byte.exists(), "4-byte check should NOT jump when first 4 bytes are zero");
    let contents = fs::read(&test_file_4byte).unwrap();
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(value, 99);

    assert!(!test_file_8byte.exists(), "8-byte check SHOULD jump when any of 8 bytes are non-zero");
}

#[test]
fn test_cranelift_basic_compilation() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_basic.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    // No-op CLIF + verify fn that writes data_offset(8 bytes) to file
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
block0(v0: i64):
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2000
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&42u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 1, offset: 1032, size: 1 },
        Action { kind: Kind::Wait, dst: 1032, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
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

    // fn0: add, fn1: write result to file
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = iadd v1, v2
    store.i64 v3, v0+16
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2016
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&100u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&200u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 2000, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 1, offset: 1032, size: 1 },
        Action { kind: Kind::Wait, dst: 1032, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
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
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = imul v1, v2
    store.i64 v3, v0+16
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2016
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&7u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&9u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 2000, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 1, offset: 1032, size: 1 },
        Action { kind: Kind::Wait, dst: 1032, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
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
block0(v0: i64):
    v1 = load.i32 v0
    v2 = load.i32 v0+4
    v3 = load.i32 v0+8
    v4 = iadd v1, v2
    v5 = iadd v4, v3
    store.i32 v5, v0+12
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2012
    v3 = iconst.i64 0
    v4 = iconst.i64 4
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2004].copy_from_slice(&10u32.to_le_bytes());
    memory[2004..2008].copy_from_slice(&20u32.to_le_bytes());
    memory[2008..2012].copy_from_slice(&30u32.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 2000, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 1, offset: 1032, size: 1 },
        Action { kind: Kind::Wait, dst: 1032, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(result, 60);
}

#[test]
fn test_cranelift_multiple_units() {
    let temp_dir = TempDir::new().unwrap();
    let test_file1 = temp_dir.path().join("unit1_add.txt");
    let test_file2 = temp_dir.path().join("unit2_mul.txt");
    let file1_str = format!("{}\0", test_file1.to_str().unwrap());
    let file2_str = format!("{}\0", test_file2.to_str().unwrap());

    // Single CLIF source with 4 functions:
    //   fn0: add (for worker 0)
    //   fn1: write to file1 (for worker 0)
    //   fn2: multiply (for worker 1)
    //   fn3: write to file2 (for worker 1)
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = iadd v1, v2
    store.i64 v3, v0+16
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2016
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}

function u0:2(i64) system_v {{
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = imul v1, v2
    store.i64 v3, v0+16
    return
}}

function u0:3(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3200
    v2 = iconst.i64 2116
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 8192];

    // Unit 0 data: 5 + 3 = 8 at offset 2000
    memory[2000..2008].copy_from_slice(&5u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&3u64.to_le_bytes());
    // Unit 1 data: 4 * 6 = 24 at offset 2100
    memory[2100..2108].copy_from_slice(&4u64.to_le_bytes());
    memory[2108..2116].copy_from_slice(&6u64.to_le_bytes());
    // Filenames
    memory[3000..3000 + file1_str.len()].copy_from_slice(file1_str.as_bytes());
    memory[3200..3200 + file2_str.len()].copy_from_slice(file2_str.as_bytes());

    let actions = vec![
        // Action 0: worker 0 compute (fn0: add)
        Action { kind: Kind::Describe, dst: 2000, src: 0, offset: 0, size: 0 },
        // Action 1: worker 1 compute (fn2: multiply)
        Action { kind: Kind::Describe, dst: 2100, src: 2, offset: 0, size: 0 },
        // Action 2: worker 0 write (fn1: write file1)
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        // Action 3: worker 1 write (fn3: write file2)
        Action { kind: Kind::Describe, dst: 0, src: 3, offset: 0, size: 0 },
        // Dispatch compute
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::ClifCallAsync, dst: 1, src: 1, offset: 1032, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 1032, src: 0, offset: 0, size: 0 },
        // Dispatch writes
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 2, offset: 1040, size: 1 },
        Action { kind: Kind::ClifCallAsync, dst: 1, src: 3, offset: 1048, size: 1 },
        Action { kind: Kind::Wait, dst: 1040, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Wait, dst: 1048, src: 0, offset: 0, size: 0 },
    ];

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: memory.len(),
        context_offset: 0,
        initial_memory: memory,
    };
    let algorithm = Algorithm {
        actions,
        cranelift_units: 2,
        timeout_ms: Some(5000),
        output: vec![],
    };
    run(config, algorithm).unwrap();

    let contents1 = fs::read(&test_file1).unwrap();
    assert_eq!(u64::from_le_bytes(contents1[0..8].try_into().unwrap()), 8);
    let contents2 = fs::read(&test_file2).unwrap();
    assert_eq!(u64::from_le_bytes(contents2[0..8].try_into().unwrap()), 24);
}

#[test]
fn test_cranelift_conditional_logic() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("cranelift_cond.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
block0(v0: i64):
    v1 = load.i64 v0
    v2 = load.i64 v0+8
    v3 = load.i64 v0+16
    v4 = icmp_imm eq v1, 0
    brif v4, block2, block1

block1:
    store.i64 v2, v0+24
    return

block2:
    store.i64 v3, v0+24
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2024
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    // condition=1, value_a=100, value_b=200 → should store value_a
    memory[2000..2008].copy_from_slice(&1u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&100u64.to_le_bytes());
    memory[2016..2024].copy_from_slice(&200u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 2000, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 1, offset: 1032, size: 1 },
        Action { kind: Kind::Wait, dst: 1032, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 100);
}


#[test]
fn test_integration_wake_then_park_progresses() {
    // Wake increments wake word 0→1, Park sees word != expected(0) and passes immediately.
    // CLIF FFI writes wake word + status to file for verification.
    let tmp_dir = TempDir::new().unwrap();
    let test_file = tmp_dir.path().join("park_wake.bin");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    // CLIF fn: write 24 bytes from offset 3000 (wake_addr) to file at offset 2000
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 24
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + file_str.len()].copy_from_slice(file_str.as_bytes());
    // 3000..3008 = wake word (starts 0)
    // 3008..3016 = expected value (0)
    // 3016..3024 = status output
    memory[3008..3016].copy_from_slice(&0u64.to_le_bytes());

    let clif_flag = 1024u32;

    let actions = vec![
        // 0: placeholder for CLIF dispatch
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        // 1: Wake — increments wake word 0→1
        Action { kind: Kind::Wake, dst: 3000, src: 0, offset: 0, size: 0 },
        // 2: Park — wake word is already 1 != expected(0), passes immediately
        Action { kind: Kind::Park, dst: 3000, src: 3008, offset: 3016, size: 0 },
        // 3: ClifCallAsync CLIF (writes to file)
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: clif_flag, size: 1 },
        // 4: Wait
        Action { kind: Kind::Wait, dst: clif_flag, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(5000);
    run(config, algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let wake_val = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(wake_val, 1, "wake word should be 1 after Wake");
    let status_val = u64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(status_val, 1, "status should be 1 (woken)");
}

#[test]
fn test_integration_park_times_out_without_wake() {
    // Park with 10ms timeout, no wake fires. Status should be 0 (timed out).
    let tmp_dir = TempDir::new().unwrap();
    let test_file = tmp_dir.path().join("park_timeout.bin");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    // CLIF fn: write 32 bytes from offset 3000 to file at offset 2000
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 32
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + file_str.len()].copy_from_slice(file_str.as_bytes());
    // 3000..3008 = wake word (stays 0)
    // 3008..3016 = expected value (0)
    // 3016..3024 = status output
    // 3024..3032 = expected status (0, for WaitUntil)
    memory[3008..3016].copy_from_slice(&0u64.to_le_bytes());
    memory[3024..3032].copy_from_slice(&0u64.to_le_bytes());

    let clif_flag = 1024u32;

    let actions = vec![
        // 0: placeholder for CLIF dispatch
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        // 1: Park with 10ms timeout — wake word 0 == expected(0), times out
        Action { kind: Kind::Park, dst: 3000, src: 3008, offset: 3016, size: 10 },
        // 2: WaitUntil — status == expected_status (0 == 0)
        Action { kind: Kind::WaitUntil, dst: 3016, src: 3024, offset: 0, size: 0 },
        // 3: ClifCallAsync CLIF (writes to file)
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: clif_flag, size: 1 },
        // 4: Wait
        Action { kind: Kind::Wait, dst: clif_flag, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(5000);
    run(config, algorithm).unwrap();

    assert!(test_file.exists());
    let contents = fs::read(&test_file).unwrap();
    let wake_val = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(wake_val, 0, "wake word should still be 0");
    let status_val = u64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(status_val, 0, "status should be 0 (timed out)");
}

#[test]
fn test_clif_ffi_file_write_and_read() {
    // CLIF IR writes "Hello, CLIF!" to a file via cl_file_write,
    // then reads it back via cl_file_read into a different memory region.
    // Verified by a subsequent FileWrite action that dumps the read-back data.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("clif_ffi_data.bin");
    let verify_file = temp_dir.path().join("clif_ffi_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   0..~600:    CLIF IR (null-terminated)
    //  1024..1032:  cranelift completion flag
    //  1032..1040:  file completion flag
    //  2000..2256:  data file path (null-terminated)
    //  2256..2512:  verify file path (null-terminated)
    //  3000..3012:  source data "Hello, CLIF!" (12 bytes)
    //  3100..3112:  read-back destination (12 bytes)
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0

block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 12
    ; write 12 bytes to data file
    v5 = call fn0(v0, v1, v2, v3, v4)
    ; read 12 bytes back into offset 3100
    v6 = iconst.i64 3100
    v7 = call fn1(v0, v1, v6, v3, v4)
    ; write read-back to verify file
    v8 = iconst.i64 2256
    v9 = call fn0(v0, v8, v6, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    memory[2256..2256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[3000..3012].copy_from_slice(b"Hello, CLIF!");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..12], b"Hello, CLIF!");
}

#[test]
fn test_clif_ffi_file_write_read_with_offset() {
    // Tests cl_file_write at a file offset, then cl_file_read at that offset.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("clif_offset.bin");
    let verify_file = temp_dir.path().join("clif_offset_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0

block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 8
    v4 = iconst.i64 5
    ; write 5 bytes at file_offset=8
    v5 = call fn0(v0, v1, v2, v3, v4)
    ; read 5 bytes at file_offset=8 into offset 3100
    v6 = iconst.i64 3100
    v7 = call fn1(v0, v1, v6, v3, v4)
    ; write read-back to verify file
    v8 = iconst.i64 2256
    v9 = iconst.i64 0
    v10 = call fn0(v0, v8, v6, v9, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    memory[2256..2256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[3000..3005].copy_from_slice(b"ABCDE");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..5], b"ABCDE");

    let raw = fs::read(&data_file).unwrap();
    assert_eq!(&raw[..8], &[0u8; 8]);
    assert_eq!(&raw[8..13], b"ABCDE");
}

#[test]
fn test_clif_ffi_file_binary_data() {
    // Tests that binary data with embedded null bytes round-trips correctly.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("clif_binary.bin");
    let verify_file = temp_dir.path().join("clif_binary_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0

block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    v6 = iconst.i64 3100
    v7 = call fn1(v0, v1, v6, v3, v4)
    ; write read-back to verify file
    v8 = iconst.i64 2256
    v9 = call fn0(v0, v8, v6, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    memory[2256..2256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[3000..3008].copy_from_slice(&[0xFF, 0x00, 0x01, 0x00, 0xAB, 0xCD, 0x00, 0xEF]);

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..8], &[0xFF, 0x00, 0x01, 0x00, 0xAB, 0xCD, 0x00, 0xEF]);
}

#[test]
fn test_clif_ffi_gpu_vec_add() {
    // Tests the full GPU pipeline via CLIF FFI:
    // init → create buffers → upload → create pipeline → dispatch → download → cleanup.
    // Adds two 64-element f32 vectors element-wise.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("gpu_vec_add_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4; // 256 bytes per buffer

    // WGSL shader for element-wise add: result[i] = a[i] + b[i]
    let wgsl = "@group(0) @binding(0) var<storage, read> a: array<f32>;\n\
                @group(0) @binding(1) var<storage, read> b: array<f32>;\n\
                @group(0) @binding(2) var<storage, read_write> result: array<f32>;\n\
                @compute @workgroup_size(64)\n\
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                    let i = gid.x;\n\
                    if (i < arrayLength(&a)) {\n\
                        result[i] = a[i] + b[i];\n\
                    }\n\
                }\n";

    // Memory layout (must be 8-byte aligned for i64 loads):
    //    0..~1500:  CLIF IR
    // 2000..2256:   verify file path
    // 3000..3400:   shader source (null-terminated, ~300 bytes)
    // 3400..3424:   binding descriptors: 3 bindings × 8 bytes = 24 bytes
    //              [buf_id:i32, read_only:i32] × 3
    // 4000..4256:   buffer A data (64 f32s)
    // 4256..4512:   buffer B data (64 f32s)
    // 4512..4768:   result download area (64 f32s)
    let shader_off = 3000;
    let bind_off = 3400;
    let a_off = 4000;
    let b_off = 4256;
    let result_off = 4512;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v
    sig5 = (i64, i32, i64, i64) -> i32 system_v
    sig6 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload sig2
    fn3 = %cl_gpu_create_pipeline sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig5
    fn6 = %cl_gpu_cleanup sig0
    fn7 = %cl_file_write sig6

block0(v0: i64):
    call fn0(v0)

    ; create 3 buffers (a, b, result) each {data_bytes} bytes
    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)
    v3 = call fn1(v0, v1)
    v4 = call fn1(v0, v1)

    ; upload A to buf0
    v5 = iconst.i64 {a_off}
    v16 = call fn2(v0, v2, v5, v1)

    ; upload B to buf1
    v6 = iconst.i64 {b_off}
    v17 = call fn2(v0, v3, v6, v1)

    ; create pipeline with 3 bindings: [buf0 read, buf1 read, buf2 rw]
    v7 = iconst.i64 {shader_off}
    v8 = iconst.i64 {bind_off}
    v9 = iconst.i32 3
    v10 = call fn3(v0, v7, v8, v9)

    ; dispatch 1 workgroup of 64 threads
    v11 = iconst.i32 1
    v18 = call fn4(v0, v10, v11, v11, v11)

    ; download result buffer to offset {result_off}
    v12 = iconst.i64 {result_off}
    v19 = call fn5(v0, v4, v12, v1)

    ; write result to verify file
    v13 = iconst.i64 {file_off}
    v14 = iconst.i64 0
    v15 = call fn7(v0, v13, v12, v14, v1)

    call fn6(v0)
    return
}}"#,
        data_bytes = data_bytes,
        a_off = a_off,
        b_off = b_off,
        shader_off = shader_off,
        bind_off = bind_off,
        result_off = result_off,
        file_off = 2000,
    );

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    assert!(clif_bytes.len() < 2000, "CLIF IR too large: {} bytes", clif_bytes.len());
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    // Verify file path
    memory[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    // Shader source (null-terminated)
    let shader_bytes = wgsl.as_bytes();
    memory[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    memory[shader_off + shader_bytes.len()] = 0;

    // Binding descriptors: [buf_id:i32, read_only:i32] × 3
    // buf0 read_only=1, buf1 read_only=1, buf2 read_only=0
    let bind = &mut memory[bind_off..];
    bind[0..4].copy_from_slice(&0i32.to_le_bytes()); // buf0
    bind[4..8].copy_from_slice(&1i32.to_le_bytes()); // read_only
    bind[8..12].copy_from_slice(&1i32.to_le_bytes()); // buf1
    bind[12..16].copy_from_slice(&1i32.to_le_bytes()); // read_only
    bind[16..20].copy_from_slice(&2i32.to_le_bytes()); // buf2
    bind[20..24].copy_from_slice(&0i32.to_le_bytes()); // read_write

    // Fill buffer A: [1.0, 2.0, 3.0, ..., 64.0]
    for i in 0..n {
        let val = (i + 1) as f32;
        memory[a_off + i * 4..a_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }
    // Fill buffer B: [100.0, 100.0, ..., 100.0]
    for i in 0..n {
        let val = 100.0f32;
        memory[b_off + i * 4..b_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(15000); // GPU init can be slow
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes, "Result file should be {} bytes", data_bytes);

    // Verify: result[i] = (i+1) + 100.0
    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 + 100.0;
        assert!(
            (actual - expected).abs() < 0.01,
            "Mismatch at index {}: got {}, expected {}", i, actual, expected
        );
    }
}

#[test]
fn test_clif_ffi_file_return_values() {
    // Verify that cl_file_write and cl_file_read return correct byte counts,
    // and cl_file_read on a nonexistent file returns -1.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("retval.bin");
    let missing_file = temp_dir.path().join("does_not_exist.bin");
    let verify_file = temp_dir.path().join("retval_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let missing_file_str = format!("{}\0", missing_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Layout: data file path @ 2000, missing file path @ 2200,
    //         verify file path @ 2400, source data @ 3000,
    //         results @ 3100 (3 x i64: write_ret, read_ret, missing_read_ret)
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 7
    v5 = call fn0(v0, v1, v2, v3, v4)
    store.i64 v5, v0+3100
    v6 = iconst.i64 3050
    v7 = call fn1(v0, v1, v6, v3, v4)
    store.i64 v7, v0+3108
    v8 = iconst.i64 2200
    v9 = iconst.i64 3060
    v10 = call fn1(v0, v8, v9, v3, v4)
    store.i64 v10, v0+3116
    v11 = iconst.i64 2400
    v12 = iconst.i64 3100
    v13 = iconst.i64 24
    v14 = call fn0(v0, v11, v12, v3, v13)
    return
}
"#;

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    memory[2200..2200 + missing_file_str.len()].copy_from_slice(missing_file_str.as_bytes());
    memory[2400..2400 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[3000..3007].copy_from_slice(b"RETVALS");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 24);
    let write_ret = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    let read_ret = i64::from_le_bytes(contents[8..16].try_into().unwrap());
    let missing_ret = i64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(write_ret, 7, "cl_file_write should return bytes written");
    assert_eq!(read_ret, 7, "cl_file_read should return bytes read");
    assert_eq!(missing_ret, -1, "cl_file_read on missing file should return -1");
}

#[test]
fn test_clif_ffi_file_read_dynamic_size() {
    // cl_file_read with size=0 should read the entire file.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("dynamic.bin");
    let verify_file = temp_dir.path().join("dynamic_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Pre-create the file with known content (23 bytes)
    fs::write(&data_file, b"dynamic size read test!").unwrap();

    // CLIF: read with size=0 → should read entire file, return 23
    // Then write the result + return value to verify file
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_read sig0
    fn1 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = call fn0(v0, v1, v2, v3, v3)
    store.i64 v4, v0+3100
    v5 = iconst.i64 2200
    v6 = iconst.i64 3100
    v7 = iconst.i64 8
    v8 = call fn1(v0, v5, v6, v3, v7)
    v9 = iconst.i64 8
    v10 = iconst.i64 23
    v11 = call fn1(v0, v5, v2, v9, v10)
    return
}
"#;

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    memory[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 31, "Expected at least 31 bytes, got {}", contents.len());
    let bytes_read = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(bytes_read, 23, "size=0 should read entire 23-byte file");
    assert_eq!(&contents[8..31], b"dynamic size read test!");
}

#[test]
fn test_clif_ffi_file_write_cstring_mode() {
    // cl_file_write with size=0 writes a C-string (stops at null byte).
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("cstring.bin");
    let verify_file = temp_dir.path().join("cstring_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Source at offset 3000: "CSTR\0extra" — size=0 should write only "CSTR" (4 bytes)
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = call fn0(v0, v1, v2, v3, v3)
    store.i64 v4, v0+3100
    v5 = iconst.i64 2200
    v6 = iconst.i64 3100
    v7 = iconst.i64 8
    v8 = call fn0(v0, v5, v6, v3, v7)
    return
}
"#;

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    memory[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[3000..3009].copy_from_slice(b"CSTR\0xtra");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let verify = fs::read(&verify_file).unwrap();
    let write_ret = i64::from_le_bytes(verify[0..8].try_into().unwrap());
    assert_eq!(write_ret, 4, "size=0 should write 4 bytes (up to null)");

    let raw = fs::read(&data_file).unwrap();
    assert_eq!(&raw, b"CSTR", "File should contain only 'CSTR'");
}

#[test]
fn test_clif_ffi_file_overwrite_shorter() {
    // Write long data, then overwrite with shorter data at offset 0 (truncates).
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("overwrite.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 10
    v5 = call fn0(v0, v1, v2, v3, v4)
    v6 = iconst.i64 3100
    v7 = iconst.i64 3
    v8 = call fn0(v0, v1, v6, v3, v7)
    return
}
"#;

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    memory[3000..3010].copy_from_slice(b"LONGDATA!!");
    memory[3100..3103].copy_from_slice(b"SML");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    // Second write with file_offset=0 uses File::create() which truncates
    let raw = fs::read(&data_file).unwrap();
    assert_eq!(&raw, b"SML", "Second write should truncate file to 3 bytes");
}

#[test]
fn test_clif_ffi_file_partial_reads() {
    // Write 100 bytes, then read in two 50-byte chunks at different offsets.
    let temp_dir = TempDir::new().unwrap();
    let data_file = temp_dir.path().join("partial.bin");
    let verify_file = temp_dir.path().join("partial_verify.bin");
    let data_file_str = format!("{}\0", data_file.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
    fn1 = %cl_file_read sig0
block0(v0: i64):
    v1 = iconst.i64 2000
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 100
    v5 = call fn0(v0, v1, v2, v3, v4)
    v6 = iconst.i64 3200
    v7 = iconst.i64 50
    v8 = call fn1(v0, v1, v6, v3, v7)
    v9 = iconst.i64 3250
    v10 = call fn1(v0, v1, v9, v7, v7)
    v11 = iconst.i64 2200
    v12 = iconst.i64 3200
    v13 = call fn0(v0, v11, v12, v3, v4)
    return
}
"#;

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + data_file_str.len()].copy_from_slice(data_file_str.as_bytes());
    memory[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    // Fill 100 bytes at offset 3000 with pattern 0..99
    for i in 0..100u8 {
        memory[3000 + i as usize] = i;
    }

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 100);
    // First 50 bytes should be 0..49, second 50 bytes should be 50..99
    for i in 0..100u8 {
        assert_eq!(contents[i as usize], i, "Byte {} mismatch in reassembled read", i);
    }
}

#[test]
fn test_clif_ffi_gpu_multiple_dispatches_before_download() {
    // Tests the pending_encoder batching: dispatch pipeline 3 times, then download once.
    // Uses a shader that multiplies by 2 each dispatch: data * 2 * 2 * 2 = data * 8.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("gpu_multi_dispatch.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4;

    let wgsl = "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
                @compute @workgroup_size(64)\n\
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                    let i = gid.x;\n\
                    if (i < arrayLength(&data)) {\n\
                        data[i] = data[i] * 2.0;\n\
                    }\n\
                }\n";

    let shader_off = 3000;
    let bind_off = 3400;
    let data_off = 4000;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v
    sig5 = (i64, i32, i64, i64) -> i32 system_v
    sig6 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload sig2
    fn3 = %cl_gpu_create_pipeline sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig5
    fn6 = %cl_gpu_cleanup sig0
    fn7 = %cl_file_write sig6
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)
    v3 = iconst.i64 {data_off}
    v12 = call fn2(v0, v2, v3, v1)
    v4 = iconst.i64 {shader_off}
    v5 = iconst.i64 {bind_off}
    v6 = iconst.i32 1
    v7 = call fn3(v0, v4, v5, v6)
    v13 = call fn4(v0, v7, v6, v6, v6)
    v14 = call fn4(v0, v7, v6, v6, v6)
    v15 = call fn4(v0, v7, v6, v6, v6)
    v8 = iconst.i64 {result_off}
    v16 = call fn5(v0, v2, v8, v1)
    v9 = iconst.i64 2000
    v10 = iconst.i64 0
    v11 = call fn7(v0, v9, v8, v10, v1)
    call fn6(v0)
    return
}}"#,
        data_bytes = data_bytes,
        data_off = data_off,
        shader_off = shader_off,
        bind_off = bind_off,
        result_off = 4500,
    );

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let shader_bytes = wgsl.as_bytes();
    memory[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    memory[shader_off + shader_bytes.len()] = 0;

    // 1 binding: buf0 read_write
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    memory[bind_off + 4..bind_off + 8].copy_from_slice(&0i32.to_le_bytes());

    // Fill data: [1.0, 2.0, ..., 64.0]
    for i in 0..n {
        let val = (i + 1) as f32;
        memory[data_off + i * 4..data_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(15000);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes);
    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 * 8.0; // *2 three times
        assert!(
            (actual - expected).abs() < 0.01,
            "Index {}: got {}, expected {} (after 3x multiply by 2)", i, actual, expected
        );
    }
}

#[test]
fn test_clif_ffi_gpu_buffer_reuse() {
    // Upload data A, dispatch, download result A, then upload data B, dispatch, download result B.
    // Verifies that buffer state is correctly updated on re-upload.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("gpu_reuse.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4;

    // Shader: data[i] = data[i] + 1.0
    let wgsl = "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n\
                @compute @workgroup_size(64)\n\
                fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
                    let i = gid.x;\n\
                    if (i < arrayLength(&data)) {\n\
                        data[i] = data[i] + 1.0;\n\
                    }\n\
                }\n";

    let shader_off = 3000;
    let bind_off = 3400;
    let data_a_off = 4000;
    let data_b_off = 4300;
    let result_a_off = 4600;
    let result_b_off = result_a_off + data_bytes;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v
    sig5 = (i64, i32, i64, i64) -> i32 system_v
    sig6 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload sig2
    fn3 = %cl_gpu_create_pipeline sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig5
    fn6 = %cl_gpu_cleanup sig0
    fn7 = %cl_file_write sig6
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)
    v3 = iconst.i64 {shader_off}
    v4 = iconst.i64 {bind_off}
    v5 = iconst.i32 1
    v6 = call fn3(v0, v3, v4, v5)
    v7 = iconst.i64 {data_a_off}
    v15 = call fn2(v0, v2, v7, v1)
    v16 = call fn4(v0, v6, v5, v5, v5)
    v8 = iconst.i64 {result_a_off}
    v17 = call fn5(v0, v2, v8, v1)
    v9 = iconst.i64 {data_b_off}
    v18 = call fn2(v0, v2, v9, v1)
    v19 = call fn4(v0, v6, v5, v5, v5)
    v10 = iconst.i64 {result_b_off}
    v20 = call fn5(v0, v2, v10, v1)
    v11 = iconst.i64 2000
    v12 = iconst.i64 0
    v13 = iconst.i64 {two_data_bytes}
    v14 = call fn7(v0, v11, v8, v12, v13)
    call fn6(v0)
    return
}}"#,
        data_bytes = data_bytes,
        shader_off = shader_off,
        bind_off = bind_off,
        data_a_off = data_a_off,
        data_b_off = data_b_off,
        result_a_off = result_a_off,
        result_b_off = result_b_off,
        two_data_bytes = data_bytes * 2,
    );

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let shader_bytes = wgsl.as_bytes();
    memory[shader_off..shader_off + shader_bytes.len()].copy_from_slice(shader_bytes);
    memory[shader_off + shader_bytes.len()] = 0;

    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());
    memory[bind_off + 4..bind_off + 8].copy_from_slice(&0i32.to_le_bytes());

    // Data A: all 10.0
    for i in 0..n {
        memory[data_a_off + i * 4..data_a_off + i * 4 + 4].copy_from_slice(&10.0f32.to_le_bytes());
    }
    // Data B: all 100.0
    for i in 0..n {
        memory[data_b_off + i * 4..data_b_off + i * 4 + 4].copy_from_slice(&100.0f32.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(15000);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes * 2);

    // Result A: each element should be 10.0 + 1.0 = 11.0
    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - 11.0).abs() < 0.01,
            "Result A index {}: got {}, expected 11.0", i, actual
        );
    }
    // Result B: each element should be 100.0 + 1.0 = 101.0
    for i in 0..n {
        let off = data_bytes + i * 4;
        let actual = f32::from_le_bytes(contents[off..off + 4].try_into().unwrap());
        assert!(
            (actual - 101.0).abs() < 0.01,
            "Result B index {}: got {}, expected 101.0", i, actual
        );
    }
}

///   - create_pipeline with out-of-range buf_id in binding returns -1
/// All error codes are stored and written to a verify file.
#[test]
fn test_clif_ffi_gpu_error_codes() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("gpu_errors.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   0..~2100:  CLIF IR (null-terminated)
    //   2500..xx:  verify file path
    //   3500..3700: dummy shader (valid WGSL, needed for init)
    //   3700..3708: binding descriptor [buf_id=99, read_only=0] (invalid buf_id)
    //   4500..4540: return values (5 i32s stored as i64: create_buf_rc, upload_rc, dispatch_rc, download_rc, pipeline_bad_bind_rc)

    let wgsl = "@group(0) @binding(0) var<storage, read_write> d: array<f32>;\n\
                @compute @workgroup_size(1)\n\
                fn main() { d[0] = 1.0; }\n";

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i32, i32, i32) -> i32 system_v
    sig5 = (i64) system_v
    sig6 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_gpu_init sig0
    fn1 = %cl_gpu_create_buffer sig1
    fn2 = %cl_gpu_upload sig2
    fn3 = %cl_gpu_create_pipeline sig3
    fn4 = %cl_gpu_dispatch sig4
    fn5 = %cl_gpu_download sig2
    fn6 = %cl_gpu_cleanup sig5
    fn7 = %cl_file_write sig6
block0(v0: i64):
    call fn0(v0)

    ; create_buffer with size=0 → should return -1
    v1 = iconst.i64 0
    v2 = call fn1(v0, v1)
    v20 = sextend.i64 v2
    store.i64 v20, v0+4500

    ; upload with buf_id=99 (no buffers exist yet) → should return -1
    v3 = iconst.i32 99
    v4 = iconst.i64 3500
    v5 = iconst.i64 64
    v6 = call fn2(v0, v3, v4, v5)
    v21 = sextend.i64 v6
    store.i64 v21, v0+4508

    ; dispatch with pipeline_id=99 (no pipelines exist) → should return -1
    v7 = iconst.i32 99
    v8 = iconst.i32 1
    v9 = call fn4(v0, v7, v8, v8, v8)
    v22 = sextend.i64 v9
    store.i64 v22, v0+4516

    ; download with buf_id=99 → should return -1
    v10 = iconst.i64 3500
    v11 = iconst.i64 64
    v12 = call fn5(v0, v3, v10, v11)
    v23 = sextend.i64 v12
    store.i64 v23, v0+4524

    ; create a valid buffer so we can test pipeline with bad binding
    v13 = iconst.i64 256
    v14 = call fn1(v0, v13)

    ; create_pipeline with binding that references buf_id=99 → should return -1
    v15 = iconst.i64 3500
    v16 = iconst.i64 3700
    v17 = iconst.i32 1
    v18 = call fn3(v0, v15, v16, v17)
    v24 = sextend.i64 v18
    store.i64 v24, v0+4532

    ; write 40 bytes of return values to verify file
    v25 = iconst.i64 2500
    v26 = iconst.i64 4500
    v27 = iconst.i64 0
    v28 = iconst.i64 40
    v29 = call fn7(v0, v25, v26, v27, v28)

    call fn6(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2500..2500 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    // Valid shader at 3500
    let shader_bytes = wgsl.as_bytes();
    memory[3500..3500 + shader_bytes.len()].copy_from_slice(shader_bytes);
    memory[3500 + shader_bytes.len()] = 0;

    // Binding descriptor at 3700: buf_id=99 (invalid), read_only=0
    memory[3700..3704].copy_from_slice(&99i32.to_le_bytes());
    memory[3704..3708].copy_from_slice(&0i32.to_le_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(10000);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 40, "Expected 40 bytes of return values, got {}", contents.len());

    let create_buf_rc = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    let upload_rc = i64::from_le_bytes(contents[8..16].try_into().unwrap());
    let dispatch_rc = i64::from_le_bytes(contents[16..24].try_into().unwrap());
    let download_rc = i64::from_le_bytes(contents[24..32].try_into().unwrap());
    let pipeline_bad_bind_rc = i64::from_le_bytes(contents[32..40].try_into().unwrap());

    assert_eq!(create_buf_rc, -1, "create_buffer(size=0) should return -1");
    assert_eq!(upload_rc, -1, "upload(buf_id=99) should return -1");
    assert_eq!(dispatch_rc, -1, "dispatch(pipeline_id=99) should return -1");
    assert_eq!(download_rc, -1, "download(buf_id=99) should return -1");
    assert_eq!(pipeline_bad_bind_rc, -1, "create_pipeline with invalid buf_id binding should return -1");
}

#[test]
fn test_clif_ffi_net_echo() {
    // CLIF IR: init network → connect to a Rust echo server → send "hello" → recv response → cleanup.
    // Verification: the echoed data is written to a file via cl_file_write from within CLIF.
    use std::net::TcpListener;
    use std::io::{Read, Write};

    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("net_echo_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Find a free port
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let addr_str = format!("127.0.0.1:{}\0", port);

    // Spawn echo server thread
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf).unwrap();
        stream.write_all(&buf[..n]).unwrap();
    });

    // Memory layout:
    //   2000..2100:  server address (null-terminated)
    //   2100..2200:  verify file path (null-terminated)
    //   3000..3005:  send data "hello"
    //   3100..3105:  recv buffer
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
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

    ; connect
    v1 = iconst.i64 {addr_off}
    v2 = call fn1(v0, v1)

    ; send 5 bytes from offset 3000
    v3 = iconst.i64 3000
    v4 = iconst.i64 5
    v5 = call fn2(v0, v2, v3, v4)

    ; recv into offset 3100
    v6 = iconst.i64 3100
    v7 = call fn3(v0, v2, v6, v4)

    ; write received data to verify file via cl_file_write
    v8 = iconst.i64 {file_off}
    v9 = iconst.i64 0
    v10 = call fn5(v0, v8, v6, v9, v4)

    call fn4(v0)
    return
}}"#, addr_off = 2000, file_off = 2100);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + addr_str.len()].copy_from_slice(addr_str.as_bytes());
    memory[2100..2100 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[3000..3005].copy_from_slice(b"hello");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();
    handle.join().unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..5], b"hello");
}

#[test]
fn test_clif_ffi_net_listen_accept() {
    // CLIF IR: init → listen → accept → recv → send (echo) → cleanup.
    // A Rust thread connects and sends "ping!", expects echo back.
    use std::net::TcpStream;
    use std::io::{Read, Write};

    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("net_listen_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);

    let addr_str = format!("127.0.0.1:{}\0", port);

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i64 system_v
    sig2 = (i64, i64) -> i64 system_v
    sig3 = (i64, i64, i64, i64) -> i64 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_net_init sig0
    fn1 = %cl_net_listen sig1
    fn2 = %cl_net_accept sig2
    fn3 = %cl_net_send sig3
    fn4 = %cl_net_recv sig3
    fn5 = %cl_net_cleanup sig0
    fn6 = %cl_file_write sig4

block0(v0: i64):
    call fn0(v0)

    ; listen
    v1 = iconst.i64 {addr_off}
    v2 = call fn1(v0, v1)

    ; accept a connection
    v3 = call fn2(v0, v2)

    ; recv 5 bytes into offset 3100
    v4 = iconst.i64 3100
    v5 = iconst.i64 5
    v6 = call fn4(v0, v3, v4, v5)

    ; echo: send those 5 bytes back
    v7 = call fn3(v0, v3, v4, v5)

    ; write received data to verify file
    v8 = iconst.i64 {file_off}
    v9 = iconst.i64 0
    v10 = call fn6(v0, v8, v4, v9, v5)

    call fn5(v0)
    return
}}"#, addr_off = 2000, file_off = 2100);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + addr_str.len()].copy_from_slice(addr_str.as_bytes());
    memory[2100..2100 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    // Spawn client thread that connects and sends "ping!"
    let port_copy = port;
    let client_handle = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let mut stream = TcpStream::connect(format!("127.0.0.1:{}", port_copy)).unwrap();
        stream.write_all(b"ping!").unwrap();
        let mut buf = [0u8; 5];
        stream.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"ping!");
    });

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(10000);
    run(config, algorithm).unwrap();
    client_handle.join().unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(&contents[..5], b"ping!");
}

#[test]
fn test_clif_ffi_net_invalid_handle() {
    // Send/recv on handle 0 (never created) should return errors.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("net_invalid_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i64, i64) -> i64 system_v
    sig2 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_net_init sig0
    fn1 = %cl_net_send sig1
    fn2 = %cl_net_recv sig1
    fn3 = %cl_net_cleanup sig0
    fn4 = %cl_file_write sig2
block0(v0: i64):
    call fn0(v0)
    ; send on handle 0 (never created)
    v1 = iconst.i64 0
    v2 = iconst.i64 3000
    v3 = iconst.i64 5
    v4 = call fn1(v0, v1, v2, v3)
    store.i64 v4, v0+3100
    ; recv on handle 0
    v5 = iconst.i64 3050
    v6 = call fn2(v0, v1, v5, v3)
    store.i64 v6, v0+3108
    ; send on handle 42 (never created)
    v8 = iconst.i64 42
    v9 = call fn1(v0, v8, v2, v3)
    store.i64 v9, v0+3116
    ; write results to file
    v10 = iconst.i64 2200
    v11 = iconst.i64 3100
    v12 = iconst.i64 0
    v13 = iconst.i64 24
    v14 = call fn4(v0, v10, v11, v12, v13)
    call fn3(v0)
    return
}
"#;

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[3000..3005].copy_from_slice(b"hello");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 24);
    let send_ret = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    let recv_ret = i64::from_le_bytes(contents[8..16].try_into().unwrap());
    let send42_ret = i64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(send_ret, -1, "send on handle 0 should return -1");
    assert_eq!(recv_ret, -1, "recv on handle 0 should return -1");
    assert_eq!(send42_ret, -1, "send on handle 42 (never created) should return -1");
}

#[test]
fn test_clif_ffi_net_connect_bad_address() {
    // Connect to an address that can't be reached, verify handle=0.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("net_badaddr_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let bad_addr = "127.0.0.1:1\0";

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i64 system_v
    sig2 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_net_init sig0
    fn1 = %cl_net_connect sig1
    fn2 = %cl_net_cleanup sig0
    fn3 = %cl_file_write sig2
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 2000
    v2 = call fn1(v0, v1)
    store.i64 v2, v0+3000
    v3 = iconst.i64 2200
    v4 = iconst.i64 3000
    v5 = iconst.i64 0
    v6 = iconst.i64 8
    v7 = call fn3(v0, v3, v4, v5, v6)
    call fn2(v0)
    return
}
"#;

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + bad_addr.len()].copy_from_slice(bad_addr.as_bytes());
    memory[2200..2200 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(10000);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let handle = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(handle, 0, "connect to unreachable address should return handle 0");
}

#[test]
fn test_clif_ffi_net_recv_writes_at_offset() {
    // Verify cl_net_recv writes data at the correct shared memory offset.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("recv_offset.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = probe.local_addr().unwrap().port();
    drop(probe);
    let addr_str = format!("127.0.0.1:{}\0", port);

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i64 system_v
    sig2 = (i64, i64) -> i64 system_v
    sig3 = (i64, i64, i64, i64) -> i64 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_net_init sig0
    fn1 = %cl_net_listen sig1
    fn2 = %cl_net_accept sig2
    fn3 = %cl_net_recv sig3
    fn4 = %cl_net_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)

    ; listen
    v1 = iconst.i64 2000
    v2 = call fn1(v0, v1)

    ; accept
    v3 = call fn2(v0, v2)

    ; recv 16 bytes into offset 3000
    v4 = iconst.i64 3000
    v5 = iconst.i64 16
    v6 = call fn3(v0, v3, v4, v5)

    ; store recv return value at 5000
    store.i64 v6, v0+5000

    ; write recv buffer (16 bytes from offset 3000) to verify file
    v7 = iconst.i64 2100
    v8 = iconst.i64 0
    v9 = call fn5(v0, v7, v4, v8, v5)

    ; append recv return value (8 bytes from offset 5000) at file offset 16
    v10 = iconst.i64 5000
    v11 = iconst.i64 16
    v12 = iconst.i64 8
    v13 = call fn5(v0, v7, v10, v11, v12)

    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + addr_str.len()].copy_from_slice(addr_str.as_bytes());
    memory[2100..2100 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let pattern: [u8; 16] = [0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let client_port = port;
    let client = std::thread::spawn(move || {
        use std::io::Write;
        for _ in 0..40 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            if let Ok(mut stream) = std::net::TcpStream::connect(format!("127.0.0.1:{}", client_port)) {
                stream.write_all(&pattern).unwrap();
                return;
            }
        }
        panic!("client could not connect");
    });

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(10000);
    run(config, algorithm).unwrap();
    client.join().unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 24, "expected 24 bytes, got {}", contents.len());

    let recv_data = &contents[0..16];
    assert_eq!(recv_data, &[0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

    let recv_ret = i64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(recv_ret, 16, "recv should return 16 bytes read");
}

#[test]
fn test_clif_ffi_lmdb_put_get_delete() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_basic");
    let verify_file = temp_dir.path().join("lmdb_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i32, i64, i32) -> i32 system_v
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_delete sig4
    fn5 = %cl_lmdb_cleanup sig0
    fn6 = %cl_file_write sig5
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 3
    v6 = iconst.i64 5010
    v7 = iconst.i32 6
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    v9 = iconst.i64 5100
    v10 = call fn3(v0, v3, v4, v5, v9)
    store.i32 v10, v0+5300
    v11 = call fn4(v0, v3, v4, v5)
    v12 = iconst.i64 5200
    v13 = call fn3(v0, v3, v4, v5, v12)
    store.i32 v13, v0+5304
    v14 = iconst.i64 4256
    v15 = iconst.i64 5300
    v16 = iconst.i64 0
    v17 = iconst.i64 8
    v18 = call fn6(v0, v14, v15, v16, v17)
    v19 = iconst.i64 8
    v20 = iconst.i64 6
    v22 = iconst.i64 5104
    v21 = call fn6(v0, v14, v22, v19, v20)
    call fn5(v0)
    return
}}"#);

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000..5003].copy_from_slice(b"foo");
    memory[5010..5016].copy_from_slice(b"barbaz");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 14);
    let first_len = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(first_len, 6, "First get should return length 6");
    let second_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(second_ret, -1, "Get after delete should return -1");
    assert_eq!(&contents[8..14], b"barbaz");
}

#[test]
fn test_clif_ffi_lmdb_batch_write() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_batch");
    let verify_file = temp_dir.path().join("lmdb_batch_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i32) -> i32 system_v
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_begin_write_txn sig4
    fn5 = %cl_lmdb_commit_write_txn sig4
    fn6 = %cl_lmdb_cleanup sig0
    fn7 = %cl_file_write sig5
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = call fn4(v0, v3)
    v5 = iconst.i32 2
    v6 = iconst.i64 5000
    v7 = iconst.i64 5030
    v8 = call fn2(v0, v3, v6, v5, v7, v5)
    v9 = iconst.i64 5010
    v10 = iconst.i64 5040
    v11 = call fn2(v0, v3, v9, v5, v10, v5)
    v12 = iconst.i64 5020
    v13 = iconst.i64 5050
    v14 = call fn2(v0, v3, v12, v5, v13, v5)
    v15 = call fn5(v0, v3)
    v16 = iconst.i64 5100
    v17 = call fn3(v0, v3, v6, v5, v16)
    store.i32 v17, v0+5400
    v18 = iconst.i64 5200
    v19 = call fn3(v0, v3, v9, v5, v18)
    store.i32 v19, v0+5404
    v20 = iconst.i64 5300
    v21 = call fn3(v0, v3, v12, v5, v20)
    store.i32 v21, v0+5408
    v22 = iconst.i64 4256
    v23 = iconst.i64 5400
    v24 = iconst.i64 0
    v25 = iconst.i64 12
    v26 = call fn7(v0, v22, v23, v24, v25)
    v27 = iconst.i64 12
    v28 = iconst.i64 2
    v34 = iconst.i64 5104
    v29 = call fn7(v0, v22, v34, v27, v28)
    v30 = iconst.i64 14
    v35 = iconst.i64 5204
    v31 = call fn7(v0, v22, v35, v30, v28)
    v32 = iconst.i64 16
    v36 = iconst.i64 5304
    v33 = call fn7(v0, v22, v36, v32, v28)
    call fn6(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000..5002].copy_from_slice(b"k1");
    memory[5010..5012].copy_from_slice(b"k2");
    memory[5020..5022].copy_from_slice(b"k3");
    memory[5030..5032].copy_from_slice(b"v1");
    memory[5040..5042].copy_from_slice(b"v2");
    memory[5050..5052].copy_from_slice(b"v3");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 18);
    let len1 = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let len2 = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    let len3 = i32::from_le_bytes(contents[8..12].try_into().unwrap());
    assert_eq!(len1, 2);
    assert_eq!(len2, 2);
    assert_eq!(len3, 2);
    assert_eq!(&contents[12..14], b"v1");
    assert_eq!(&contents[14..16], b"v2");
    assert_eq!(&contents[16..18], b"v3");
}

#[test]
fn test_clif_ffi_lmdb_cursor_scan() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_scan");
    let verify_file = temp_dir.path().join("lmdb_scan_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_cursor_scan sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i32 1
    v5 = iconst.i64 5000
    v6 = iconst.i64 5030
    v7 = call fn2(v0, v3, v5, v4, v6, v4)
    v8 = iconst.i64 5010
    v9 = call fn2(v0, v3, v8, v4, v6, v4)
    v10 = iconst.i64 5020
    v11 = call fn2(v0, v3, v10, v4, v6, v4)
    v12 = iconst.i64 5040
    v13 = iconst.i32 0
    v14 = iconst.i32 100
    v15 = iconst.i64 5100
    v16 = call fn3(v0, v3, v12, v13, v14, v15)
    store.i32 v16, v0+5500
    v17 = iconst.i64 4256
    v18 = iconst.i64 5500
    v19 = iconst.i64 0
    v20 = iconst.i64 4
    v21 = call fn5(v0, v17, v18, v19, v20)
    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'a';
    memory[5010] = b'b';
    memory[5020] = b'c';
    memory[5030] = b'x';

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 4);
    let count = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(count, 3, "Cursor scan should find 3 entries");
}

#[test]
fn test_clif_ffi_lmdb_get_nonexistent_key() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_nokey");
    let verify_file = temp_dir.path().join("lmdb_nokey_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = iconst.i64 5010
    v7 = iconst.i32 3
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    store.i32 v8, v0+5100
    v9 = iconst.i64 5001
    v10 = iconst.i64 5200
    v11 = call fn3(v0, v3, v9, v5, v10)
    store.i32 v11, v0+5104
    v12 = iconst.i64 4256
    v13 = iconst.i64 5100
    v14 = iconst.i64 0
    v15 = iconst.i64 8
    v16 = call fn5(v0, v12, v13, v14, v15)
    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'a';
    memory[5001] = b'b';
    memory[5010..5013].copy_from_slice(b"val");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let put_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let get_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(put_ret, 0, "put should succeed");
    assert_eq!(get_ret, -1, "get on non-existent key should return -1");
}

#[test]
fn test_clif_ffi_lmdb_put_overwrite() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_overwrite");
    let verify_file = temp_dir.path().join("lmdb_overwrite_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = iconst.i64 5010
    v7 = iconst.i32 3
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    v9 = iconst.i64 5020
    v10 = call fn2(v0, v3, v4, v5, v9, v7)
    v11 = iconst.i64 5100
    v12 = call fn3(v0, v3, v4, v5, v11)
    store.i32 v12, v0+5200
    v13 = iconst.i64 4256
    v14 = iconst.i64 5200
    v15 = iconst.i64 0
    v16 = iconst.i64 4
    v17 = call fn5(v0, v13, v14, v15, v16)
    v18 = iconst.i64 4
    v19 = iconst.i64 3
    v20 = iconst.i64 5104
    v21 = call fn5(v0, v13, v20, v18, v19)
    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'k';
    memory[5010..5013].copy_from_slice(b"old");
    memory[5020..5023].copy_from_slice(b"new");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert!(contents.len() >= 7);
    let get_len = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(get_len, 3);
    assert_eq!(&contents[4..7], b"new");
}

#[test]
fn test_clif_ffi_lmdb_commit_without_begin() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_nobegin");
    let verify_file = temp_dir.path().join("lmdb_nobegin_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32) -> i32 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_commit_write_txn sig2
    fn3 = %cl_lmdb_cleanup sig0
    fn4 = %cl_file_write sig3
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = call fn2(v0, v3)
    store.i32 v4, v0+5000
    v5 = iconst.i64 4256
    v6 = iconst.i64 5000
    v7 = iconst.i64 0
    v8 = iconst.i64 4
    v9 = call fn4(v0, v5, v6, v7, v8)
    call fn3(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let commit_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(commit_ret, -1, "commit without begin should return -1");
}

#[test]
fn test_clif_ffi_lmdb_cursor_scan_empty_db() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_empty");
    let verify_file = temp_dir.path().join("lmdb_empty_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_cursor_scan sig2
    fn3 = %cl_lmdb_cleanup sig0
    fn4 = %cl_file_write sig3
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 0
    v6 = iconst.i32 100
    v7 = iconst.i64 5100
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    store.i32 v8, v0+5200
    v9 = iconst.i64 4256
    v10 = iconst.i64 5200
    v11 = iconst.i64 0
    v12 = iconst.i64 4
    v13 = call fn4(v0, v9, v10, v11, v12)
    call fn3(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let count = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(count, 0, "cursor scan on empty db should return 0");
}

#[test]
fn test_clif_ffi_lmdb_cursor_scan_with_start_key() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_rangescan");
    let verify_file = temp_dir.path().join("lmdb_range_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_cursor_scan sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i32 1
    v20 = iconst.i64 5060
    v5 = iconst.i64 5000
    v6 = call fn2(v0, v3, v5, v4, v20, v4)
    v7 = iconst.i64 5001
    v8 = call fn2(v0, v3, v7, v4, v20, v4)
    v9 = iconst.i64 5002
    v10 = call fn2(v0, v3, v9, v4, v20, v4)
    v11 = iconst.i64 5003
    v12 = call fn2(v0, v3, v11, v4, v20, v4)
    v13 = iconst.i64 5004
    v14 = call fn2(v0, v3, v13, v4, v20, v4)
    v15 = iconst.i32 100
    v16 = iconst.i64 5200
    v17 = call fn3(v0, v3, v9, v4, v15, v16)
    store.i32 v17, v0+5300
    v18 = iconst.i64 4256
    v19 = iconst.i64 5300
    v21 = iconst.i64 0
    v22 = iconst.i64 4
    v23 = call fn5(v0, v18, v19, v21, v22)
    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'a';
    memory[5001] = b'b';
    memory[5002] = b'c';
    memory[5003] = b'd';
    memory[5004] = b'e';
    memory[5060] = b'x';

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let count = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(count, 3, "scan from 'c' should find c, d, e = 3 entries");
}

#[test]
fn test_clif_ffi_lmdb_sync() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_sync");
    let verify_file = temp_dir.path().join("lmdb_sync_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_sync sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = iconst.i64 5010
    v7 = call fn2(v0, v3, v4, v5, v6, v5)
    v8 = call fn3(v0, v3)
    store.i32 v8, v0+5100
    v9 = iconst.i64 4256
    v10 = iconst.i64 5100
    v11 = iconst.i64 0
    v12 = iconst.i64 4
    v13 = call fn5(v0, v9, v10, v11, v12)
    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'k';
    memory[5010] = b'v';

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let sync_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(sync_ret, 0, "sync should return 0");
}

#[test]
fn test_clif_ffi_lmdb_delete_nonexistent_key() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_delnone");
    let verify_file = temp_dir.path().join("lmdb_delnone_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Delete a key that was never inserted (auto-commit path — exercises txn abort fix).
    // Then put a key and get it to prove the db still works after the failed delete.
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_delete sig2
    fn3 = %cl_lmdb_put sig3
    fn4 = %cl_lmdb_get sig4
    fn5 = %cl_lmdb_cleanup sig0
    fn6 = %cl_file_write sig5
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    ; delete key "x" which doesn't exist
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = call fn2(v0, v3, v4, v5)
    store.i32 v6, v0+5100
    ; now put key "a" = "ok" to prove db still works
    v7 = iconst.i64 5010
    v8 = iconst.i64 5020
    v9 = iconst.i32 2
    v10 = call fn3(v0, v3, v7, v5, v8, v9)
    store.i32 v10, v0+5104
    ; get key "a"
    v11 = iconst.i64 5200
    v12 = call fn4(v0, v3, v7, v5, v11)
    store.i32 v12, v0+5108
    ; write [del_ret, put_ret, get_ret, value]
    v13 = iconst.i64 4256
    v14 = iconst.i64 5100
    v15 = iconst.i64 0
    v16 = iconst.i64 12
    v17 = call fn6(v0, v13, v14, v15, v16)
    v18 = iconst.i64 12
    v19 = iconst.i64 2
    v20 = iconst.i64 5204
    v21 = call fn6(v0, v13, v20, v18, v19)
    call fn5(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'x';
    memory[5010] = b'a';
    memory[5020..5022].copy_from_slice(b"ok");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 14);
    let del_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let put_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    let get_ret = i32::from_le_bytes(contents[8..12].try_into().unwrap());
    assert_eq!(del_ret, -1, "delete nonexistent key should return -1");
    assert_eq!(put_ret, 0, "put after failed delete should succeed");
    assert_eq!(get_ret, 2, "get should return value length 2");
    assert_eq!(&contents[12..14], b"ok");
}

#[test]
fn test_clif_ffi_lmdb_invalid_handle() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_invhandle");
    let verify_file = temp_dir.path().join("lmdb_invhandle_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Open a db (handle 0), then do put/get/delete/scan/sync on handle 99.
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i32, i64, i32) -> i32 system_v
    sig5 = (i64, i32) -> i32 system_v
    sig6 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig7 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_delete sig4
    fn5 = %cl_lmdb_sync sig5
    fn6 = %cl_lmdb_cursor_scan sig6
    fn7 = %cl_lmdb_begin_write_txn sig5
    fn8 = %cl_lmdb_commit_write_txn sig5
    fn9 = %cl_lmdb_cleanup sig0
    fn10 = %cl_file_write sig7
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    ; use bogus handle 99 for everything
    v4 = iconst.i32 99
    v5 = iconst.i64 5000
    v6 = iconst.i32 1
    v7 = iconst.i64 5010
    v8 = iconst.i32 2
    ; put on bad handle
    v9 = call fn2(v0, v4, v5, v6, v7, v8)
    store.i32 v9, v0+5100
    ; get on bad handle
    v10 = iconst.i64 5200
    v11 = call fn3(v0, v4, v5, v6, v10)
    store.i32 v11, v0+5104
    ; delete on bad handle
    v12 = call fn4(v0, v4, v5, v6)
    store.i32 v12, v0+5108
    ; sync on bad handle
    v13 = call fn5(v0, v4)
    store.i32 v13, v0+5112
    ; cursor_scan on bad handle
    v14 = iconst.i32 100
    v15 = iconst.i64 5300
    v16 = call fn6(v0, v4, v5, v6, v14, v15)
    store.i32 v16, v0+5116
    ; begin_write_txn on bad handle
    v17 = call fn7(v0, v4)
    store.i32 v17, v0+5120
    ; commit_write_txn on bad handle
    v18 = call fn8(v0, v4)
    store.i32 v18, v0+5124
    ; write 28 bytes of results
    v19 = iconst.i64 4256
    v20 = iconst.i64 5100
    v21 = iconst.i64 0
    v22 = iconst.i64 28
    v23 = call fn10(v0, v19, v20, v21, v22)
    call fn9(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'k';
    memory[5010..5012].copy_from_slice(b"vv");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 28);
    let put_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let get_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    let del_ret = i32::from_le_bytes(contents[8..12].try_into().unwrap());
    let sync_ret = i32::from_le_bytes(contents[12..16].try_into().unwrap());
    let scan_ret = i32::from_le_bytes(contents[16..20].try_into().unwrap());
    let begin_ret = i32::from_le_bytes(contents[20..24].try_into().unwrap());
    let commit_ret = i32::from_le_bytes(contents[24..28].try_into().unwrap());
    assert_eq!(put_ret, -1, "put on invalid handle");
    assert_eq!(get_ret, -1, "get on invalid handle");
    assert_eq!(del_ret, -1, "delete on invalid handle");
    assert_eq!(sync_ret, -1, "sync on invalid handle");
    assert_eq!(scan_ret, 0, "scan on invalid handle returns 0 entries");
    assert_eq!(begin_ret, -1, "begin_write_txn on invalid handle");
    assert_eq!(commit_ret, -1, "commit_write_txn on invalid handle");
}

#[test]
fn test_clif_ffi_lmdb_double_begin() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_dblbegin");
    let verify_file = temp_dir.path().join("lmdb_dblbegin_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // begin_write_txn, put k1=v1, begin_write_txn again (aborts first),
    // put k2=v2, commit. k1 should be gone, k2 should exist.
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig4 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig5 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_begin_write_txn sig2
    fn3 = %cl_lmdb_put sig3
    fn4 = %cl_lmdb_commit_write_txn sig2
    fn5 = %cl_lmdb_get sig4
    fn6 = %cl_lmdb_cleanup sig0
    fn7 = %cl_file_write sig5
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    ; first begin
    v4 = call fn2(v0, v3)
    ; put k1=v1 in first txn
    v5 = iconst.i64 5000
    v6 = iconst.i32 2
    v7 = iconst.i64 5010
    v8 = call fn3(v0, v3, v5, v6, v7, v6)
    ; second begin — should abort the first txn (k1=v1 lost)
    v9 = call fn2(v0, v3)
    ; put k2=v2 in second txn
    v10 = iconst.i64 5020
    v11 = iconst.i64 5030
    v12 = call fn3(v0, v3, v10, v6, v11, v6)
    ; commit second txn
    v13 = call fn4(v0, v3)
    ; get k1 — should fail (-1)
    v14 = iconst.i64 5100
    v15 = call fn5(v0, v3, v5, v6, v14)
    store.i32 v15, v0+5200
    ; get k2 — should succeed
    v16 = iconst.i64 5300
    v17 = call fn5(v0, v3, v10, v6, v16)
    store.i32 v17, v0+5204
    ; write results
    v18 = iconst.i64 4256
    v19 = iconst.i64 5200
    v20 = iconst.i64 0
    v21 = iconst.i64 8
    v22 = call fn7(v0, v18, v19, v20, v21)
    ; write the k2 value
    v23 = iconst.i64 8
    v24 = iconst.i64 2
    v25 = iconst.i64 5304
    v26 = call fn7(v0, v18, v25, v23, v24)
    call fn6(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000..5002].copy_from_slice(b"k1");
    memory[5010..5012].copy_from_slice(b"v1");
    memory[5020..5022].copy_from_slice(b"k2");
    memory[5030..5032].copy_from_slice(b"v2");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 10);
    let get_k1 = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let get_k2 = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(get_k1, -1, "k1 should be lost (first txn aborted by second begin)");
    assert_eq!(get_k2, 2, "k2 should exist with length 2");
    assert_eq!(&contents[8..10], b"v2");
}

#[test]
fn test_clif_ffi_lmdb_empty_batch() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_emptybatch");
    let verify_file = temp_dir.path().join("lmdb_emptybatch_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // begin_write_txn then immediately commit with no puts
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32) -> i32 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_begin_write_txn sig2
    fn3 = %cl_lmdb_commit_write_txn sig2
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig3
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = call fn2(v0, v3)
    store.i32 v4, v0+5000
    v5 = call fn3(v0, v3)
    store.i32 v5, v0+5004
    v6 = iconst.i64 4256
    v7 = iconst.i64 5000
    v8 = iconst.i64 0
    v9 = iconst.i64 8
    v10 = call fn5(v0, v6, v7, v8, v9)
    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let begin_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let commit_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(begin_ret, 0, "begin empty batch should succeed");
    assert_eq!(commit_ret, 0, "commit empty batch should succeed");
}

#[test]
fn test_clif_ffi_lmdb_put_empty_value() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_emptyval");
    let verify_file = temp_dir.path().join("lmdb_emptyval_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Put key="k" with val_len=0, then get it back
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 1
    v6 = iconst.i64 5010
    v7 = iconst.i32 0
    ; put key="k" with empty value
    v8 = call fn2(v0, v3, v4, v5, v6, v7)
    store.i32 v8, v0+5100
    ; get it back
    v9 = iconst.i64 5200
    v10 = call fn3(v0, v3, v4, v5, v9)
    store.i32 v10, v0+5104
    ; write [put_ret, get_ret]
    v11 = iconst.i64 4256
    v12 = iconst.i64 5100
    v13 = iconst.i64 0
    v14 = iconst.i64 8
    v15 = call fn5(v0, v11, v12, v13, v14)
    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'k';

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let put_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let get_ret = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(put_ret, 0, "put with empty value should succeed");
    assert_eq!(get_ret, 0, "get should return length 0 for empty value");
}

#[test]
fn test_clif_ffi_lmdb_multiple_databases() {
    let temp_dir = TempDir::new().unwrap();
    let db1_path = temp_dir.path().join("clif_lmdb_multi1");
    let db2_path = temp_dir.path().join("clif_lmdb_multi2");
    let verify_file = temp_dir.path().join("lmdb_multi_verify.bin");
    let db1_path_str = format!("{}\0", db1_path.to_str().unwrap());
    let db2_path_str = format!("{}\0", db2_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Open two databases. Put key="k" in db1 with val="d1", in db2 with val="d2".
    // Get from each to verify isolation.
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_get sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    ; open db1
    v3 = call fn1(v0, v1, v2)
    ; open db2
    v4 = iconst.i64 4200
    v5 = call fn1(v0, v4, v2)
    ; key "k" at 5000, val "d1" at 5010, val "d2" at 5020
    v6 = iconst.i64 5000
    v7 = iconst.i32 1
    v8 = iconst.i64 5010
    v9 = iconst.i32 2
    ; put "k"="d1" in db1
    v10 = call fn2(v0, v3, v6, v7, v8, v9)
    ; put "k"="d2" in db2
    v11 = iconst.i64 5020
    v12 = call fn2(v0, v5, v6, v7, v11, v9)
    ; get from db1
    v13 = iconst.i64 5100
    v14 = call fn3(v0, v3, v6, v7, v13)
    store.i32 v14, v0+5200
    ; get from db2
    v15 = iconst.i64 5300
    v16 = call fn3(v0, v5, v6, v7, v15)
    store.i32 v16, v0+5204
    ; write [len1, len2, val1, val2]
    v17 = iconst.i64 4400
    v18 = iconst.i64 5200
    v19 = iconst.i64 0
    v20 = iconst.i64 8
    v21 = call fn5(v0, v17, v18, v19, v20)
    v22 = iconst.i64 8
    v23 = iconst.i64 2
    v24 = iconst.i64 5104
    v25 = call fn5(v0, v17, v24, v22, v23)
    v26 = iconst.i64 10
    v27 = iconst.i64 5304
    v28 = call fn5(v0, v17, v27, v26, v23)
    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db1_path_str.len()].copy_from_slice(db1_path_str.as_bytes());
    memory[4200..4200 + db2_path_str.len()].copy_from_slice(db2_path_str.as_bytes());
    memory[4400..4400 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'k';
    memory[5010..5012].copy_from_slice(b"d1");
    memory[5020..5022].copy_from_slice(b"d2");

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 12);
    let len1 = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    let len2 = i32::from_le_bytes(contents[4..8].try_into().unwrap());
    assert_eq!(len1, 2);
    assert_eq!(len2, 2);
    assert_eq!(&contents[8..10], b"d1", "db1 should have val d1");
    assert_eq!(&contents[10..12], b"d2", "db2 should have val d2");
}

#[test]
fn test_clif_ffi_lmdb_cursor_scan_max_entries_limit() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_scanlimit");
    let verify_file = temp_dir.path().join("lmdb_scanlimit_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Insert 5 keys (a-e), scan with max_entries=2
    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i32, i64) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_put sig2
    fn3 = %cl_lmdb_cursor_scan sig3
    fn4 = %cl_lmdb_cleanup sig0
    fn5 = %cl_file_write sig4
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i32 1
    v20 = iconst.i64 5060
    v5 = iconst.i64 5000
    v6 = call fn2(v0, v3, v5, v4, v20, v4)
    v7 = iconst.i64 5001
    v8 = call fn2(v0, v3, v7, v4, v20, v4)
    v9 = iconst.i64 5002
    v10 = call fn2(v0, v3, v9, v4, v20, v4)
    v11 = iconst.i64 5003
    v12 = call fn2(v0, v3, v11, v4, v20, v4)
    v13 = iconst.i64 5004
    v14 = call fn2(v0, v3, v13, v4, v20, v4)
    ; scan all but limit to 2
    v15 = iconst.i32 0
    v16 = iconst.i32 2
    v17 = iconst.i64 5200
    v18 = iconst.i64 5100
    v19 = call fn3(v0, v3, v18, v15, v16, v17)
    store.i32 v19, v0+5300
    v21 = iconst.i64 4256
    v22 = iconst.i64 5300
    v23 = iconst.i64 0
    v24 = iconst.i64 4
    v25 = call fn5(v0, v21, v22, v23, v24)
    call fn4(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory[5000] = b'a';
    memory[5001] = b'b';
    memory[5002] = b'c';
    memory[5003] = b'd';
    memory[5004] = b'e';
    memory[5060] = b'x';

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let count = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(count, 2, "scan with max_entries=2 should return exactly 2");
}

#[test]
fn test_clif_ffi_lmdb_uncommitted_batch_cleanup() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("clif_lmdb_uncommitted");
    let verify_file = temp_dir.path().join("lmdb_uncommitted_verify.bin");
    let db_path_str = format!("{}\0", db_path.to_str().unwrap());
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    // begin_write_txn, put a key, then cleanup WITHOUT committing.
    // Drop should abort the txn. Then reopen the db and verify key is missing.
    // We do this in two separate execute() calls.
    let clif_ir_1 =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32) -> i32 system_v
    sig3 = (i64, i32, i64, i32, i64, i32) -> i32 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_begin_write_txn sig2
    fn3 = %cl_lmdb_put sig3
    fn4 = %cl_lmdb_cleanup sig0
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = call fn2(v0, v3)
    v5 = iconst.i64 5000
    v6 = iconst.i32 3
    v7 = iconst.i64 5010
    v8 = call fn3(v0, v3, v5, v6, v7, v6)
    ; cleanup without commit — Drop should abort the active write txn
    call fn4(v0)
    return
}
"#;

    let mut memory1 = vec![0u8; 8192];
    let clif_bytes1 = format!("{}\0", clif_ir_1).into_bytes();
    memory1[0..clif_bytes1.len()].copy_from_slice(&clif_bytes1);
    memory1[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory1[5000..5003].copy_from_slice(b"key");
    memory1[5010..5013].copy_from_slice(b"val");

    let actions1 = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config1, algorithm1) = create_cranelift_algorithm(actions1, memory1, 1, clif_ir_1.to_string());
    run(config1, algorithm1).unwrap();

    // Second run: reopen db, try to get the key — should not exist
    let clif_ir_2 =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64, i32) -> i32 system_v
    sig2 = (i64, i32, i64, i32, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_lmdb_init sig0
    fn1 = %cl_lmdb_open sig1
    fn2 = %cl_lmdb_get sig2
    fn3 = %cl_lmdb_cleanup sig0
    fn4 = %cl_file_write sig3
block0(v0: i64):
    call fn0(v0)
    v1 = iconst.i64 4000
    v2 = iconst.i32 10
    v3 = call fn1(v0, v1, v2)
    v4 = iconst.i64 5000
    v5 = iconst.i32 3
    v6 = iconst.i64 5100
    v7 = call fn2(v0, v3, v4, v5, v6)
    store.i32 v7, v0+5200
    v8 = iconst.i64 4256
    v9 = iconst.i64 5200
    v10 = iconst.i64 0
    v11 = iconst.i64 4
    v12 = call fn4(v0, v8, v9, v10, v11)
    call fn3(v0)
    return
}
"#;

    let mut memory2 = vec![0u8; 8192];
    let clif_bytes2 = format!("{}\0", clif_ir_2).into_bytes();
    memory2[0..clif_bytes2.len()].copy_from_slice(&clif_bytes2);
    memory2[4000..4000 + db_path_str.len()].copy_from_slice(db_path_str.as_bytes());
    memory2[4256..4256 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());
    memory2[5000..5003].copy_from_slice(b"key");

    let actions2 = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config2, algorithm2) = create_cranelift_algorithm(actions2, memory2, 1, clif_ir_2.to_string());
    run(config2, algorithm2).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 4);
    let get_ret = i32::from_le_bytes(contents[0..4].try_into().unwrap());
    assert_eq!(get_ret, -1, "uncommitted key should not persist after cleanup");
}

#[test]
fn test_clif_ffi_thread_spawn_and_join() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("thread_result.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   0-7:     HT context (set by runtime)
    //   16-23:   thread context pointer
    //   200-207: worker writes here (value 42)
    //   3000+:   file path
    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    // fn0: init thread ctx, spawn fn1, join, cleanup
    // fn1: worker writes 42 at offset 0 (relative to its ptr = base+200)
    // fn2: write memory[200..208] to file for verification
    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64, i64) -> i64 system_v
    fn2 = %cl_thread_join sig2
    sig3 = (i64) system_v
    fn3 = %cl_thread_cleanup sig3
block0(v0: i64):
    v1 = iadd_imm v0, 16
    call fn0(v1)
    v2 = iconst.i64 1
    v3 = iadd_imm v0, 200
    v4 = call fn1(v1, v2, v3)
    v5 = call fn2(v1, v4)
    call fn3(v1)
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 42
    store.i64 v1, v0
    return
}

function u0:2(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 200
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}"#;

    let clif_off = 3200usize;
    memory[clif_off..clif_off + clif_ir.len()].copy_from_slice(clif_ir.as_bytes());
    memory[clif_off + clif_ir.len()] = 0;

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 2, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 2016, size: 2 },
        Action { kind: Kind::Wait, dst: 2016, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(value, 42, "spawned thread should have written 42");
}

#[test]
fn test_clif_ffi_thread_multiple_workers() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("thread_multi.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   16-23:    thread context
    //   200-207:  worker 0 writes here (value 99)
    //   208-215:  worker 1 writes here (value 99)
    //   216-223:  worker 2 writes here (value 99)
    //   3000+:    file path
    let mut memory = vec![0u8; 8192];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    // fn0: spawn 3 workers, join all
    // fn1: worker reads its base ptr offset, writes a value based on position
    //      Each worker gets base+200, base+208, base+216 respectively.
    //      Worker writes (offset_from_200 / 8 + 1) * 100 at offset 0.
    // fn2: write 24 bytes (3 u64s) from offset 200 to file
    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64, i64) -> i64 system_v
    fn2 = %cl_thread_join sig2
    sig3 = (i64) system_v
    fn3 = %cl_thread_cleanup sig3
block0(v0: i64):
    v1 = iadd_imm v0, 16
    call fn0(v1)
    v2 = iconst.i64 1
    v3 = iadd_imm v0, 200
    v4 = call fn1(v1, v2, v3)
    v5 = iadd_imm v0, 208
    v6 = call fn1(v1, v2, v5)
    v7 = iadd_imm v0, 216
    v8 = call fn1(v1, v2, v7)
    v9 = call fn2(v1, v4)
    v10 = call fn2(v1, v6)
    v11 = call fn2(v1, v8)
    call fn3(v1)
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 99
    store.i64 v1, v0
    return
}

function u0:2(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 200
    v3 = iconst.i64 0
    v4 = iconst.i64 24
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}"#;

    let clif_off = 3200usize;
    memory[clif_off..clif_off + clif_ir.len()].copy_from_slice(clif_ir.as_bytes());
    memory[clif_off + clif_ir.len()] = 0;

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 2, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 2016, size: 2 },
        Action { kind: Kind::Wait, dst: 2016, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 24);
    let v0 = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    let v1 = u64::from_le_bytes(contents[8..16].try_into().unwrap());
    let v2 = u64::from_le_bytes(contents[16..24].try_into().unwrap());
    assert_eq!(v0, 99, "worker 0 should have written 99");
    assert_eq!(v1, 99, "worker 1 should have written 99");
    assert_eq!(v2, 99, "worker 2 should have written 99");
}

#[test]
fn test_clif_ffi_thread_join_invalid_handle() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("thread_invalid.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    // fn0: init, join invalid handle (999), store return value, cleanup
    // fn1: write the return value to file
    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64) -> i64 system_v
    fn1 = %cl_thread_join sig1
    sig2 = (i64) system_v
    fn2 = %cl_thread_cleanup sig2
block0(v0: i64):
    v1 = iadd_imm v0, 16
    call fn0(v1)
    v2 = iconst.i64 999
    v3 = call fn1(v1, v2)
    store.i64 v3, v0+200
    call fn2(v1)
    return
}

function u0:1(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 200
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}"#;

    let clif_off = 3200usize;
    memory[clif_off..clif_off + clif_ir.len()].copy_from_slice(clif_ir.as_bytes());
    memory[clif_off + clif_ir.len()] = 0;

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 2016, size: 2 },
        Action { kind: Kind::Wait, dst: 2016, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let ret = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(ret, -1, "joining invalid handle should return -1");
}

#[test]
fn test_clif_ffi_thread_spawn_oob_fn_index() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("thread_oob.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   16-23:    thread context
    //   200-207:  return value from spawn with fn_index=999 (out of bounds)
    //   208-215:  return value from spawn with fn_index=-1 (negative wraps to huge)
    //   3000+:    file path
    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    // fn0: init, try spawn with oob index 999 and negative -1, store returns, cleanup
    // fn1: write results to file
    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64) system_v
    fn2 = %cl_thread_cleanup sig2
block0(v0: i64):
    v1 = iadd_imm v0, 16
    call fn0(v1)
    v2 = iconst.i64 999
    v3 = iconst.i64 0
    v4 = call fn1(v1, v2, v3)
    store.i64 v4, v0+200
    v5 = iconst.i64 -1
    v6 = call fn1(v1, v5, v3)
    store.i64 v6, v0+208
    call fn2(v1)
    return
}

function u0:1(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 200
    v3 = iconst.i64 0
    v4 = iconst.i64 16
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}"#;

    let clif_off = 3200usize;
    memory[clif_off..clif_off + clif_ir.len()].copy_from_slice(clif_ir.as_bytes());
    memory[clif_off + clif_ir.len()] = 0;

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 2016, size: 2 },
        Action { kind: Kind::Wait, dst: 2016, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 16);
    let ret_oob = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    let ret_neg = i64::from_le_bytes(contents[8..16].try_into().unwrap());
    assert_eq!(ret_oob, -1, "spawn with out-of-bounds fn_index should return -1");
    assert_eq!(ret_neg, -1, "spawn with negative fn_index should return -1");
}

#[test]
fn test_clif_ffi_thread_cleanup_joins_unjoined() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("thread_cleanup.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   16-23:    thread context
    //   200-207:  worker 0 writes 77 here
    //   208-215:  worker 1 writes 88 here
    //   3000+:    file path
    let mut memory = vec![0u8; 8192];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    // fn0: init, spawn 2 workers WITHOUT joining, then cleanup (should join them)
    //      After cleanup, sleep briefly via a busy loop, then write results to file
    // fn1: worker writes 77 at its ptr
    // fn2: worker writes 88 at its ptr
    // fn3: write 16 bytes from offset 200 to file
    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64) system_v
    fn2 = %cl_thread_cleanup sig2
block0(v0: i64):
    v1 = iadd_imm v0, 16
    call fn0(v1)
    v2 = iconst.i64 1
    v3 = iadd_imm v0, 200
    v4 = call fn1(v1, v2, v3)
    v5 = iconst.i64 2
    v6 = iadd_imm v0, 208
    v7 = call fn1(v1, v5, v6)
    call fn2(v1)
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 77
    store.i64 v1, v0
    return
}

function u0:2(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 88
    store.i64 v1, v0
    return
}

function u0:3(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 200
    v3 = iconst.i64 0
    v4 = iconst.i64 16
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}"#;

    let clif_off = 3200usize;
    memory[clif_off..clif_off + clif_ir.len()].copy_from_slice(clif_ir.as_bytes());
    memory[clif_off + clif_ir.len()] = 0;

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 3, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 2016, size: 2 },
        Action { kind: Kind::Wait, dst: 2016, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 16);
    let v0 = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    let v1 = u64::from_le_bytes(contents[8..16].try_into().unwrap());
    assert_eq!(v0, 77, "cleanup should have joined worker 0 which wrote 77");
    assert_eq!(v1, 88, "cleanup should have joined worker 1 which wrote 88");
}

#[test]
fn test_clif_ffi_thread_double_join() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("thread_double.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   16-23:    thread context
    //   200-207:  first join return (should be 0)
    //   208-215:  second join same handle (should be -1)
    //   3000+:    file path
    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    // fn0: init, spawn fn1, join twice, store both returns, cleanup
    // fn1: noop worker
    // fn2: write results to file
    let clif_ir = r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64, i64) -> i64 system_v
    fn2 = %cl_thread_join sig2
    sig3 = (i64) system_v
    fn3 = %cl_thread_cleanup sig3
block0(v0: i64):
    v1 = iadd_imm v0, 16
    call fn0(v1)
    v2 = iconst.i64 1
    v3 = iconst.i64 0
    v4 = call fn1(v1, v2, v3)
    v5 = call fn2(v1, v4)
    store.i64 v5, v0+200
    v6 = call fn2(v1, v4)
    store.i64 v6, v0+208
    call fn3(v1)
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    return
}

function u0:2(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 200
    v3 = iconst.i64 0
    v4 = iconst.i64 16
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}"#;

    let clif_off = 3200usize;
    memory[clif_off..clif_off + clif_ir.len()].copy_from_slice(clif_ir.as_bytes());
    memory[clif_off + clif_ir.len()] = 0;

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 2, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 2016, size: 2 },
        Action { kind: Kind::Wait, dst: 2016, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 16);
    let first_join = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    let second_join = i64::from_le_bytes(contents[8..16].try_into().unwrap());
    assert_eq!(first_join, 0, "first join should succeed with 0");
    assert_eq!(second_join, -1, "second join of same handle should return -1");
}

#[test]
fn test_clif_ffi_thread_spawned_thread_spawns() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("thread_recursive.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   16-23:    thread context (main)
    //   32-39:    thread context (child, at base+32 relative to child's ptr)
    //   200-207:  grandchild writes 55 here
    //   3000+:    file path
    //
    // fn0 (main): init ctx at base+16, spawn fn1 with ptr=base, join, cleanup
    // fn1 (child): init ctx at base+32, spawn fn2 with ptr=base+200, join, cleanup
    // fn2 (grandchild): writes 55 at its ptr
    // fn3: write results to file
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
block0(v0: i64):
    v1 = iadd_imm v0, 16
    call fn0(v1)
    v2 = iconst.i64 1
    v3 = call fn1(v1, v2, v0)
    v4 = call fn2(v1, v3)
    call fn3(v1)
    return
}

function u0:1(i64) system_v {
    sig0 = (i64) system_v
    fn0 = %cl_thread_init sig0
    sig1 = (i64, i64, i64) -> i64 system_v
    fn1 = %cl_thread_spawn sig1
    sig2 = (i64, i64) -> i64 system_v
    fn2 = %cl_thread_join sig2
    sig3 = (i64) system_v
    fn3 = %cl_thread_cleanup sig3
block0(v0: i64):
    v1 = iadd_imm v0, 32
    call fn0(v1)
    v2 = iconst.i64 2
    v3 = iadd_imm v0, 200
    v4 = call fn1(v1, v2, v3)
    v5 = call fn2(v1, v4)
    call fn3(v1)
    return
}

function u0:2(i64) system_v {
block0(v0: i64):
    v1 = iconst.i64 55
    store.i64 v1, v0
    return
}

function u0:3(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 200
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}"#;

    let clif_off = 3200usize;
    memory[clif_off..clif_off + clif_ir.len()].copy_from_slice(clif_ir.as_bytes());
    memory[clif_off + clif_ir.len()] = 0;

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 3, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 2016, size: 2 },
        Action { kind: Kind::Wait, dst: 2016, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 8);
    let value = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(value, 55, "grandchild thread should have written 55");
}

#[test]
fn test_clif_atomic_rmw_add() {
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("atomic_rmw.bin");
    let file_str = format!("{}\0", verify_file.to_str().unwrap());

    // Memory layout:
    //   64:       accumulator (u64, init 0)
    //   200-215:  descriptor { addend: u64=10, acc_rel: i64 }
    //   216-231:  descriptor { addend: u64=32, acc_rel: i64 }
    //   3000+:    file path
    let mut memory = vec![0u8; 4096];
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    // accumulator = 0
    memory[64..72].copy_from_slice(&0u64.to_le_bytes());

    // desc0 at 200: addend=10, acc_rel = 64 - 200 = -136
    memory[200..208].copy_from_slice(&10u64.to_le_bytes());
    memory[208..216].copy_from_slice(&(-136i64).to_le_bytes());

    // desc1 at 216: addend=32, acc_rel = 64 - 216 = -152
    memory[216..224].copy_from_slice(&32u64.to_le_bytes());
    memory[224..232].copy_from_slice(&(-152i64).to_le_bytes());

    // fn0: noop
    // fn1: atomic add worker — reads addend + rel offset, does atomic_rmw add
    // fn2: write accumulator (8 bytes at offset 64) to file
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    return
}

function u0:1(i64) system_v {
block0(v0: i64):
    v1 = load.i64 notrap aligned v0
    v2 = load.i64 notrap aligned v0+8
    v3 = iadd v0, v2
    v4 = atomic_rmw.i64 little add v3, v1
    return
}

function u0:2(i64) system_v {
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 64
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}"#;

    let clif_off = 3200usize;
    memory[clif_off..clif_off + clif_ir.len()].copy_from_slice(clif_ir.as_bytes());
    memory[clif_off + clif_ir.len()] = 0;

    // Actions:
    //   0: Describe (fn1, ptr=base+200 → first descriptor)
    //   1: Describe (fn1, ptr=base+216 → second descriptor)
    //   2: Describe (fn2, ptr=base → write file)
    //   3: ClifCallAsync (dispatches actions 0..3)
    //   4: Wait
    let actions = vec![
        Action { kind: Kind::Describe, dst: 200, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 216, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::Describe, dst: 0, src: 2, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 2016, size: 3 },
        Action { kind: Kind::Wait, dst: 2016, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
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
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&42u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 0, clif_ir.to_string());
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
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + file_a_str.len()].copy_from_slice(file_a_str.as_bytes());
    memory[2256..2256 + file_b_str.len()].copy_from_slice(file_b_str.as_bytes());
    memory[3000..3008].copy_from_slice(&100u64.to_le_bytes());
    memory[3008..3016].copy_from_slice(&200u64.to_le_bytes());

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCall, dst: 0, src: 1, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 0, clif_ir.to_string());
    run(config, algorithm).unwrap();

    assert!(test_file_a.exists());
    let contents_a = fs::read(&test_file_a).unwrap();
    assert_eq!(u64::from_le_bytes(contents_a[0..8].try_into().unwrap()), 100);

    assert!(test_file_b.exists());
    let contents_b = fs::read(&test_file_b).unwrap();
    assert_eq!(u64::from_le_bytes(contents_b[0..8].try_into().unwrap()), 200);
}

#[test]
fn test_clif_call_arithmetic() {
    // ClifCall runs a function that does arithmetic and stores result in shared memory.
    // fn0: load two values, add them, store result back.
    // fn1: write result to file for verification.
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("clif_call_arith.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
block0(v0: i64):
    v1 = load.i64 v0+2000
    v2 = load.i64 v0+2008
    v3 = iadd v1, v2
    store.i64 v3, v0+2016
    return
}}

function u0:1(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 3000
    v2 = iconst.i64 2016
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2008].copy_from_slice(&30u64.to_le_bytes());
    memory[2008..2016].copy_from_slice(&12u64.to_le_bytes());
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCall, dst: 0, src: 1, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 0, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 42, "30 + 12 = 42");
}

#[test]
fn test_clif_call_sequential_mutations() {
    // Multiple ClifCall actions run sequentially, each mutating shared memory.
    // fn0: store 10 at offset 2000
    // fn1: load offset 2000, multiply by 5, store at 2008
    // fn2: write offset 2008 to file
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
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCall, dst: 0, src: 1, offset: 0, size: 0 },
        Action { kind: Kind::ClifCall, dst: 0, src: 2, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 0, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 50, "10 * 5 = 50");
}

#[test]
fn test_clif_call_with_conditional_jump() {
    // ClifCall combined with ConditionalJump.
    // fn0: sets condition to nonzero
    // ConditionalJump skips fn1 (writes "bad"), falls through to fn2 (writes "good")
    let temp_dir = TempDir::new().unwrap();
    let test_file_bad = temp_dir.path().join("clif_call_bad.txt");
    let test_file_good = temp_dir.path().join("clif_call_good.txt");
    let file_bad_str = format!("{}\0", test_file_bad.to_str().unwrap());
    let file_good_str = format!("{}\0", test_file_good.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
block0(v0: i64):
    v1 = iconst.i64 1
    store.i64 v1, v0+1000
    return
}}

function u0:1(i64) system_v {{
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

function u0:2(i64) system_v {{
    sig0 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %cl_file_write sig0
block0(v0: i64):
    v1 = iconst.i64 2256
    v2 = iconst.i64 3000
    v3 = iconst.i64 0
    v4 = iconst.i64 8
    v5 = call fn0(v0, v1, v2, v3, v4)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + file_bad_str.len()].copy_from_slice(file_bad_str.as_bytes());
    memory[2256..2256 + file_good_str.len()].copy_from_slice(file_good_str.as_bytes());
    memory[3000..3008].copy_from_slice(&99u64.to_le_bytes());

    let actions = vec![
        // Action 0: ClifCall fn0 — sets condition at offset 1000 to 1
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
        // Action 1: ConditionalJump — condition at 1000 is nonzero → jump to action 3
        Action { kind: Kind::ConditionalJump, src: 1000, dst: 3, offset: 0, size: 0 },
        // Action 2: ClifCall fn1 — writes "bad" file (SKIPPED)
        Action { kind: Kind::ClifCall, dst: 0, src: 1, offset: 0, size: 0 },
        // Action 3: ClifCall fn2 — writes "good" file (EXECUTED)
        Action { kind: Kind::ClifCall, dst: 0, src: 2, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 0, clif_ir.to_string());
    run(config, algorithm).unwrap();

    assert!(!test_file_bad.exists(), "fn1 should be skipped by conditional jump");
    assert!(test_file_good.exists(), "fn2 should be executed");
    let contents = fs::read(&test_file_good).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 99);
}

#[test]
fn test_clif_call_no_workers_needed() {
    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
block0(v0: i64):
    v1 = iconst.i64 77
    store.i64 v1, v0+2000
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    // cranelift_units: 0 — no workers
    let (_config, _algorithm) = create_cranelift_algorithm(actions, memory, 0, clif_ir.to_string());

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
}}"#);

    let mut memory2 = vec![0u8; 4096];
    let clif_bytes2 = format!("{}\0", clif_ir2).into_bytes();
    memory2[0..clif_bytes2.len()].copy_from_slice(&clif_bytes2);
    memory2[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions2 = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    let (config2, algorithm2) = create_cranelift_algorithm(actions2, memory2, 0, clif_ir2.to_string());
    run(config2, algorithm2).unwrap();

    let contents = fs::read(&test_file).unwrap();
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 77);
}

#[test]
fn test_clif_call_mixed_with_async() {
    // ClifCall and ClifCallAsync can coexist in the same algorithm.
    // ClifCall runs fn0 synchronously, then ClifCallAsync dispatches fn1 to worker.
    let temp_dir = TempDir::new().unwrap();
    let test_file_sync = temp_dir.path().join("clif_call_sync.txt");
    let test_file_async = temp_dir.path().join("clif_call_async.txt");
    let file_sync_str = format!("{}\0", test_file_sync.to_str().unwrap());
    let file_async_str = format!("{}\0", test_file_async.to_str().unwrap());

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
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + file_sync_str.len()].copy_from_slice(file_sync_str.as_bytes());
    memory[2256..2256 + file_async_str.len()].copy_from_slice(file_async_str.as_bytes());
    memory[3000..3008].copy_from_slice(&111u64.to_le_bytes());
    memory[3008..3016].copy_from_slice(&222u64.to_le_bytes());

    let flag_async = 1024u32;

    let actions = vec![
        // Describe for the async path
        Action { kind: Kind::Describe, dst: 0, src: 1, offset: 0, size: 0 },
        // Action 1: ClifCall fn0 synchronously
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
        // Action 2: ClifCallAsync dispatches descriptor 0 (fn1) to worker
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: flag_async, size: 1 },
        // Action 3: Wait for async completion
        Action { kind: Kind::Wait, dst: flag_async, src: 0, offset: 0, size: 0 },
    ];

    // Need 1 worker unit for the async dispatch
    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    run(config, algorithm).unwrap();

    assert!(test_file_sync.exists(), "sync ClifCall should write file");
    let sync_val = u64::from_le_bytes(fs::read(&test_file_sync).unwrap()[0..8].try_into().unwrap());
    assert_eq!(sync_val, 111);

    assert!(test_file_async.exists(), "async ClifCallAsync should write file");
    let async_val = u64::from_le_bytes(fs::read(&test_file_async).unwrap()[0..8].try_into().unwrap());
    assert_eq!(async_val, 222);
}

#[test]
fn test_clif_call_hash_table() {
    // ClifCall can use hash table FFI (ht_create, ht_insert, ht_lookup).
    // fn0: create HT, insert key "abc" → value 42, lookup "abc", write result to file.
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("clif_call_ht.txt");
    let file_str = format!("{}\0", test_file.to_str().unwrap());

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) -> i32 system_v
    sig1 = (i64, i64, i32, i64, i32) system_v
    sig2 = (i64, i64, i32, i64) -> i32 system_v
    sig3 = (i64, i64, i64, i64, i64) -> i64 system_v
    fn0 = %ht_create sig0
    fn1 = %ht_insert sig1
    fn2 = %ht_lookup sig2
    fn3 = %cl_file_write sig3
block0(v0: i64):
    ; Load HT context from offset 0
    v1 = load.i64 v0

    ; Create hash table
    v2 = call fn0(v1)

    ; Insert key "abc" (at offset 2000, len 3) → value 42 (at offset 2008, len 8)
    v3 = iconst.i64 2000
    v4 = iadd v0, v3
    v5 = iconst.i32 3
    v6 = iconst.i64 2008
    v7 = iadd v0, v6
    v8 = iconst.i32 8
    call fn1(v1, v4, v5, v7, v8)

    ; Lookup key "abc" → result at offset 2100
    v9 = iconst.i64 2100
    v10 = iadd v0, v9
    v11 = call fn2(v1, v4, v5, v10)

    ; Write result (8 bytes at offset 2100) to file (path at offset 3000)
    v12 = iconst.i64 3000
    v13 = iconst.i64 2100
    v14 = iconst.i64 0
    v15 = iconst.i64 8
    v16 = call fn3(v0, v12, v13, v14, v15)
    return
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    // Key "abc" at offset 2000
    memory[2000..2003].copy_from_slice(b"abc");
    // Value 42 at offset 2008
    memory[2008..2016].copy_from_slice(&42u64.to_le_bytes());
    // File path at offset 3000
    memory[3000..3000 + file_str.len()].copy_from_slice(file_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 0, clif_ir.to_string());
    run(config, algorithm).unwrap();

    let contents = fs::read(&test_file).unwrap();
    assert_eq!(contents.len(), 8);
    let result = u64::from_le_bytes(contents[0..8].try_into().unwrap());
    assert_eq!(result, 42, "HT lookup should return inserted value");
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
}}"#);

    let mut memory = vec![0u8; 4096];
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    memory[0..clif_bytes.len()].copy_from_slice(&clif_bytes);
    memory[2000..2000 + input_str.len()].copy_from_slice(input_str.as_bytes());
    memory[2256..2256 + output_str.len()].copy_from_slice(output_str.as_bytes());

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCall, dst: 0, src: 1, offset: 0, size: 0 },
    ];

    let (config, algorithm) = create_cranelift_algorithm(actions, memory, 0, clif_ir.to_string());
    run(config, algorithm).unwrap();

    assert!(output_file.exists());
    let output_data = fs::read(&output_file).unwrap();
    assert_eq!(output_data, input_data, "output should match input");
}

fn create_output_algorithm(
    clif_ir: &str,
    memory: Vec<u8>,
    output: Vec<OutputBatchSchema>,
) -> (BaseConfig, Algorithm) {
    let mut p = memory;
    let clif_bytes = format!("{}\0", clif_ir).into_bytes();
    if p.len() < clif_bytes.len() {
        p.resize(clif_bytes.len().max(p.len()), 0);
    }
    p[0..clif_bytes.len()].copy_from_slice(&clif_bytes);

    let config = BaseConfig {
        cranelift_ir: clif_ir.to_string(),
        memory_size: p.len(),
        context_offset: 0,
        initial_memory: p,
    };
    let algorithm = Algorithm {
        actions: vec![
            Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
        ],
        cranelift_units: 1,
        timeout_ms: Some(5000),
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
        Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![99i64]))],
    ).unwrap();
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
    ).unwrap();
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
        Arc::new(Schema::new(vec![Field::new("greeting", DataType::Utf8, false)])),
        vec![Arc::new(StringArray::from(vec!["hello"]))],
    ).unwrap();
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
        Arc::new(Schema::new(vec![Field::new("values", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![10i64, 20, 30]))],
    ).unwrap();
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
        Arc::new(Schema::new(vec![Field::new("integer_val", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![7i64]))],
    ).unwrap();
    assert_eq!(batches[0], expected_0);

    let expected_1 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("float_val", DataType::Float64, false)])),
        vec![Arc::new(Float64Array::from(vec![7.0f64]))],
    ).unwrap();
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
        Arc::new(Schema::new(vec![Field::new("words", DataType::Utf8, false)])),
        vec![Arc::new(StringArray::from(vec!["abc", "def"]))],
    ).unwrap();
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
    ).unwrap();
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
    ).unwrap();
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
            row_count_offset: 2016,  // stays 0 — skipped
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
    assert_eq!(batches.len(), 2, "middle batch with row_count=0 should be skipped");

    let expected_0 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![42i64]))],
    ).unwrap();
    assert_eq!(batches[0], expected_0);

    let expected_1 = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("c", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![10i64, 20]))],
    ).unwrap();
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
}"#.to_string();

    let mut memory = vec![0u8; 4096];
    memory[100..108].copy_from_slice(&6i64.to_le_bytes());

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];
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
    let config1 = BaseConfig { cranelift_ir: clif_ir.clone(), memory_size: memory.len(), context_offset: 0, initial_memory: memory.clone() };
    let alg1 = Algorithm {
        actions: actions.clone(),
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema.clone(),
    };
    let batches1 = run(config1, alg1).unwrap();

    // Base struct
    let config2 = BaseConfig { cranelift_ir: clif_ir, memory_size: memory.len(), context_offset: 0, initial_memory: memory };
    let alg2 = Algorithm {
        actions,
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };
    let mut base = Base::new(config2).unwrap();
    let batches2 = base.execute(&alg2, &[]).unwrap();

    // Both should produce 6 * 7 = 42
    assert_eq!(batches1.len(), 1);
    assert_eq!(batches2.len(), 1);
    let col1 = batches1[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let col2 = batches2[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
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
}"#.to_string();

    let config = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 0, initial_memory: vec![] };
    let mut base = Base::new(config).unwrap();

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];
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
    let batches1 = base.execute(&Algorithm {
        actions: actions.clone(),
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema.clone(),
    }, &data1).unwrap();
    assert_eq!(batches1.len(), 1);
    let col1 = batches1[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col1.value(0), 30);

    // Second execute: input = 100, expect 300
    let data2 = 100i64.to_le_bytes();
    let batches2 = base.execute(&Algorithm {
        actions,
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    }, &data2).unwrap();
    assert_eq!(batches2.len(), 1);
    let col2 = batches2[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
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
}"#.to_string();

    let config = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 4096, initial_memory: vec![] };
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
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema.clone(),
    };
    let batches1 = base.execute(&alg1, &vec![0u8; 4096]).unwrap();
    let col1 = batches1[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col1.value(0), 42);

    // Second execute: call fn1 only
    let alg2 = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 1, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };
    let batches2 = base.execute(&alg2, &vec![0u8; 4096]).unwrap();
    let col2 = batches2[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
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
}"#.to_string();

    let config = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 0, initial_memory: vec![] };
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

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    // Execute 1: add 10 → total = 10
    let d1 = 10i64.to_le_bytes();
    let batches = base.execute(&Algorithm {
        actions: actions.clone(),
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema.clone(),
    }, &d1).unwrap();
    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col.value(0), 10);

    // Execute 2: add 25 → total = 35
    let d2 = 25i64.to_le_bytes();
    let batches = base.execute(&Algorithm {
        actions: actions.clone(),
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema.clone(),
    }, &d2).unwrap();
    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col.value(0), 35);

    // Execute 3: add 5 → total = 40
    let d3 = 5i64.to_le_bytes();
    let batches = base.execute(&Algorithm {
        actions,
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    }, &d3).unwrap();
    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
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
}"#.to_string();

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    // Execute 1: write value 42 to file1
    let mut mem1 = vec![0u8; 4096];
    mem1[256..256 + file1_str.len()].copy_from_slice(file1_str.as_bytes());
    mem1[512..520].copy_from_slice(&42u64.to_le_bytes());
    let config1 = BaseConfig { cranelift_ir: clif_ir.clone(), memory_size: 4096, context_offset: 0, initial_memory: mem1 };
    let mut base = Base::new(config1).unwrap();
    base.execute(&Algorithm {
        actions: actions.clone(),
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: vec![],
    }, &[]).unwrap();
    assert!(file1.exists());
    let data1 = fs::read(&file1).unwrap();
    assert_eq!(u64::from_le_bytes(data1[..8].try_into().unwrap()), 42);

    // Execute 2: write value 99 to file2 — new Base with different initial_memory
    let mut mem2 = vec![0u8; 4096];
    mem2[256..256 + file2_str.len()].copy_from_slice(file2_str.as_bytes());
    mem2[512..520].copy_from_slice(&99u64.to_le_bytes());
    let config2 = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 0, initial_memory: mem2 };
    let mut base2 = Base::new(config2).unwrap();
    base2.execute(&Algorithm {
        actions,
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: vec![],
    }, &[]).unwrap();
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
}"#.to_string();

    let config = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 4096, initial_memory: vec![] };
    let mut base = Base::new(config).unwrap();

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    // Execute with 0 units
    base.execute(&Algorithm {
        actions: actions.clone(),
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: vec![],
    }, &vec![0u8; 4096]).unwrap();

    // Execute with 2 units
    base.execute(&Algorithm {
        actions: actions.clone(),
        cranelift_units: 2,
        timeout_ms: Some(5000),
        output: vec![],
    }, &vec![0u8; 4096]).unwrap();

    // Execute with 4 units
    base.execute(&Algorithm {
        actions,
        cranelift_units: 4,
        timeout_ms: Some(5000),
        output: vec![],
    }, &vec![0u8; 4096]).unwrap();
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
}"#.to_string();

    let mut mem = vec![0u8; 4096];
    mem[100..108].copy_from_slice(&11i64.to_le_bytes());

    let config = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 0, initial_memory: mem };
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
    let batches = base.execute(&Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    }, &data).unwrap();

    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
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
}"#.to_string();

    let config = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 0, initial_memory: vec![] };
    let mut base = Base::new(config).unwrap();

    // Execute 1: seed 77 at offset 200
    base.execute(&Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: vec![],
    }, &[]).unwrap();

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
    let batches = base.execute(&Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 1, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    }, &data).unwrap();

    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
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
}"#.to_string();

    let config = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 128, initial_memory: vec![] };
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
    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    // Three executes with empty memory — counter should increment each time
    for expected in 1..=3 {
        let batches = base.execute(&Algorithm {
            actions: actions.clone(),
            cranelift_units: 0,
            timeout_ms: Some(5000),
            output: output_schema.clone(),
        }, &[]).unwrap();
        let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
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
}"#.to_string();

    let config = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 0, initial_memory: vec![] };
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
    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    // Execute 1: 10 + 20 = 30
    let mut d1 = vec![0u8; 16];
    d1[0..8].copy_from_slice(&10i64.to_le_bytes());
    d1[8..16].copy_from_slice(&20i64.to_le_bytes());
    let batches = base.execute(&Algorithm {
        actions: actions.clone(),
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema.clone(),
    }, &d1).unwrap();
    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col.value(0), 30);

    // Execute 2: 100 + 200 = 300 — pointer should update to new buffer
    let mut d2 = vec![0u8; 16];
    d2[0..8].copy_from_slice(&100i64.to_le_bytes());
    d2[8..16].copy_from_slice(&200i64.to_le_bytes());
    let batches = base.execute(&Algorithm {
        actions,
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    }, &d2).unwrap();
    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
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
}"#.to_string();

    let config = BaseConfig { cranelift_ir: clif_ir, memory_size: 4096, context_offset: 0, initial_memory: vec![] };
    let mut base = Base::new(config).unwrap();

    let actions = vec![
        Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 },
    ];

    // Execute 3 times with values 100, 200, 300
    for &val in &[100i64, 200, 300] {
        let d = val.to_le_bytes();
        base.execute(&Algorithm {
            actions: actions.clone(),
            cranelift_units: 0,
            timeout_ms: Some(5000),
            output: vec![],
        }, &d).unwrap();
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
    let batches = base.execute(&Algorithm {
        actions,
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    }, &d).unwrap();

    // count is now 4 (we did 4 executes), values: 100, 200, 300, 999
    assert_eq!(batches.len(), 1);
    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col.len(), 4);
    assert_eq!(col.value(0), 100);
    assert_eq!(col.value(1), 200);
    assert_eq!(col.value(2), 300);
    assert_eq!(col.value(3), 999);
}

#[test]
fn clif_parse_error_garbage_ir() {
    let config = BaseConfig {
        cranelift_ir: "this is not valid CLIF".to_string(),
        memory_size: 256,
        context_offset: 0,
        initial_memory: vec![],
    };
    let Err(err) = Base::new(config) else {
        panic!("expected ClifParse error for garbage IR");
    };
    assert!(matches!(err, base::Error::ClifParse(_)));
}

#[test]
fn clif_parse_error_via_run() {
    let config = BaseConfig {
        cranelift_ir: "not valid clif at all {}[]".to_string(),
        memory_size: 256,
        context_offset: 0,
        initial_memory: vec![],
    };
    let algorithm = Algorithm {
        actions: vec![],
        cranelift_units: 0,
        timeout_ms: Some(1000),
        output: vec![],
    };
    let Err(err) = run(config, algorithm) else {
        panic!("expected ClifParse error for invalid CLIF via run()");
    };
    assert!(matches!(err, base::Error::ClifParse(_)));
}

#[test]
fn clif_parse_error_incomplete_function() {
    let config = BaseConfig {
        cranelift_ir: "function %f0(i64) {\n".to_string(),
        memory_size: 256,
        context_offset: 0,
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
    let config = BaseConfig {
        cranelift_ir: String::new(),
        memory_size: 256,
        context_offset: 0,
        initial_memory: vec![],
    };
    let base = Base::new(config);
    assert!(base.is_ok());
}

#[test]
fn test_clif_ffi_cuda_vec_add() {
    // Full CUDA pipeline: init → create buffers → upload → launch PTX → download → verify.
    // Adds two 64-element f32 vectors element-wise using a PTX kernel.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("cuda_vec_add_verify.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4; // 256 bytes per buffer

    // PTX kernel: result[tid] = a[tid] + b[tid]
    let ptx = "\
.version 7.0
.target sm_50
.address_size 64

.visible .entry main(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 result_ptr
)
{
    .reg .u32 %r0;
    .reg .u64 %ra, %rb, %rc, %off;
    .reg .f32 %fa, %fb, %fr;

    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %off, %r0;
    shl.b64 %off, %off, 2;

    ld.param.u64 %ra, [a_ptr];
    ld.param.u64 %rb, [b_ptr];
    ld.param.u64 %rc, [result_ptr];

    add.u64 %ra, %ra, %off;
    add.u64 %rb, %rb, %off;
    add.u64 %rc, %rc, %off;

    ld.global.f32 %fa, [%ra];
    ld.global.f32 %fb, [%rb];
    add.f32 %fr, %fa, %fb;
    st.global.f32 [%rc], %fr;

    ret;
}\0";

    // Memory layout:
    //    0..2000:  CLIF IR (in BaseConfig, not in initial_memory)
    // 2000..2256:  verify file path
    // 3000..3800:  PTX source (null-terminated)
    // 3800..3812:  binding descriptors: 3 × i32 buf_ids = 12 bytes
    // 4000..4256:  buffer A data (64 f32s)
    // 4256..4512:  buffer B data (64 f32s)
    // 4512..4768:  result download area (64 f32s)
    let ptx_off = 3000;
    let bind_off = 3800;
    let a_off = 4000;
    let b_off = 4256;
    let result_off = 4512;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload sig2
    fn3 = %cl_cuda_download sig2
    fn4 = %cl_cuda_launch sig3
    fn5 = %cl_cuda_cleanup sig0
    fn6 = %cl_file_write sig4

block0(v0: i64):
    call fn0(v0)

    ; create 3 buffers of {data_bytes} bytes each
    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)
    v3 = call fn1(v0, v1)
    v4 = call fn1(v0, v1)

    ; upload A from offset {a_off}
    v5 = iconst.i64 {a_off}
    v20 = call fn2(v0, v2, v5, v1)

    ; upload B from offset {b_off}
    v6 = iconst.i64 {b_off}
    v21 = call fn2(v0, v3, v6, v1)

    ; launch PTX kernel: ptx at {ptx_off}, 3 bufs, binds at {bind_off}
    ; grid(1,1,1) block(64,1,1)
    v7 = iconst.i64 {ptx_off}
    v8 = iconst.i32 3
    v9 = iconst.i64 {bind_off}
    v10 = iconst.i32 1
    v11 = iconst.i32 1
    v12 = iconst.i32 1
    v13 = iconst.i32 64
    v14 = iconst.i32 1
    v15 = iconst.i32 1
    v22 = call fn4(v0, v7, v8, v9, v10, v11, v12, v13, v14, v15)

    ; download result to offset {result_off}
    v16 = iconst.i64 {result_off}
    v23 = call fn3(v0, v4, v16, v1)

    ; write result to verify file
    v17 = iconst.i64 {file_off}
    v18 = iconst.i64 0
    v19 = call fn6(v0, v17, v16, v18, v1)

    call fn5(v0)
    return
}}"#,
        data_bytes = data_bytes,
        a_off = a_off,
        b_off = b_off,
        ptx_off = ptx_off,
        bind_off = bind_off,
        result_off = result_off,
        file_off = 2000,
    );

    let mut memory = vec![0u8; 8192];

    // Verify file path
    memory[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    // PTX source (already null-terminated in the string literal)
    let ptx_bytes = ptx.as_bytes();
    memory[ptx_off..ptx_off + ptx_bytes.len()].copy_from_slice(ptx_bytes);

    // Binding descriptors: packed i32 buf_ids [0, 1, 2]
    let bind = &mut memory[bind_off..];
    bind[0..4].copy_from_slice(&0i32.to_le_bytes());
    bind[4..8].copy_from_slice(&1i32.to_le_bytes());
    bind[8..12].copy_from_slice(&2i32.to_le_bytes());

    // Fill buffer A: [1.0, 2.0, ..., 64.0]
    for i in 0..n {
        let val = (i + 1) as f32;
        memory[a_off + i * 4..a_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }
    // Fill buffer B: [100.0, 100.0, ..., 100.0]
    for i in 0..n {
        let val = 100.0f32;
        memory[b_off + i * 4..b_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) =
        create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(15000);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes, "Result file should be {} bytes", data_bytes);

    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 + 100.0;
        assert!(
            (actual - expected).abs() < 0.01,
            "Mismatch at index {}: got {}, expected {}",
            i, actual, expected
        );
    }
}

#[test]
fn test_clif_ffi_cuda_scalar_multiply() {
    // In-place kernel: single buffer, multiply each f32 element by 3.0.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("cuda_scalar_mul.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4;

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
    mov.f32 %fc, 0f40400000;
    mul.f32 %fv, %fv, %fc;
    st.global.f32 [%rd], %fv;

    ret;
}\0";

    let ptx_off = 3000;
    let bind_off = 3800;
    let data_off = 4000;
    let result_off = 4512;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload sig2
    fn3 = %cl_cuda_download sig2
    fn4 = %cl_cuda_launch sig3
    fn5 = %cl_cuda_cleanup sig0
    fn6 = %cl_file_write sig4

block0(v0: i64):
    call fn0(v0)

    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)

    ; upload data
    v3 = iconst.i64 {data_off}
    v20 = call fn2(v0, v2, v3, v1)

    ; launch: 1 buffer binding
    v4 = iconst.i64 {ptx_off}
    v5 = iconst.i32 1
    v6 = iconst.i64 {bind_off}
    v7 = iconst.i32 1
    v8 = iconst.i32 64
    v21 = call fn4(v0, v4, v5, v6, v7, v7, v7, v8, v7, v7)

    ; download
    v9 = iconst.i64 {result_off}
    v22 = call fn3(v0, v2, v9, v1)

    ; write to file
    v10 = iconst.i64 2000
    v11 = iconst.i64 0
    v23 = call fn6(v0, v10, v9, v11, v1)

    call fn5(v0)
    return
}}"#,
        data_bytes = data_bytes,
        data_off = data_off,
        ptx_off = ptx_off,
        bind_off = bind_off,
        result_off = result_off,
    );

    let mut memory = vec![0u8; 8192];
    memory[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let ptx_bytes = ptx.as_bytes();
    memory[ptx_off..ptx_off + ptx_bytes.len()].copy_from_slice(ptx_bytes);

    // 1 binding: buf 0
    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());

    // Data: [1.0, 2.0, ..., 64.0]
    for i in 0..n {
        let val = (i + 1) as f32;
        memory[data_off + i * 4..data_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) =
        create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(15000);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes);

    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 * 3.0;
        assert!(
            (actual - expected).abs() < 0.01,
            "Mismatch at index {}: got {}, expected {}",
            i, actual, expected
        );
    }
}

#[test]
fn test_clif_ffi_cuda_multiple_launches() {
    // Launch a multiply-by-2 kernel three times on the same buffer.
    // Result: data * 2 * 2 * 2 = data * 8.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("cuda_multi_launch.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4;

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
    .reg .f32 %fv;

    mov.u32 %r0, %tid.x;
    cvt.u64.u32 %off, %r0;
    shl.b64 %off, %off, 2;

    ld.param.u64 %rd, [data_ptr];
    add.u64 %rd, %rd, %off;

    ld.global.f32 %fv, [%rd];
    add.f32 %fv, %fv, %fv;
    st.global.f32 [%rd], %fv;

    ret;
}\0";

    let ptx_off = 3000;
    let bind_off = 3800;
    let data_off = 4000;
    let result_off = 4512;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload sig2
    fn3 = %cl_cuda_download sig2
    fn4 = %cl_cuda_launch sig3
    fn5 = %cl_cuda_cleanup sig0
    fn6 = %cl_file_write sig4

block0(v0: i64):
    call fn0(v0)

    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)

    v3 = iconst.i64 {data_off}
    v20 = call fn2(v0, v2, v3, v1)

    ; launch 3 times
    v4 = iconst.i64 {ptx_off}
    v5 = iconst.i32 1
    v6 = iconst.i64 {bind_off}
    v7 = iconst.i32 1
    v8 = iconst.i32 64
    v21 = call fn4(v0, v4, v5, v6, v7, v7, v7, v8, v7, v7)
    v22 = call fn4(v0, v4, v5, v6, v7, v7, v7, v8, v7, v7)
    v23 = call fn4(v0, v4, v5, v6, v7, v7, v7, v8, v7, v7)

    v9 = iconst.i64 {result_off}
    v24 = call fn3(v0, v2, v9, v1)

    v10 = iconst.i64 2000
    v11 = iconst.i64 0
    v25 = call fn6(v0, v10, v9, v11, v1)

    call fn5(v0)
    return
}}"#,
        data_bytes = data_bytes,
        data_off = data_off,
        ptx_off = ptx_off,
        bind_off = bind_off,
        result_off = result_off,
    );

    let mut memory = vec![0u8; 8192];
    memory[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let ptx_bytes = ptx.as_bytes();
    memory[ptx_off..ptx_off + ptx_bytes.len()].copy_from_slice(ptx_bytes);

    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());

    for i in 0..n {
        let val = (i + 1) as f32;
        memory[data_off + i * 4..data_off + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) =
        create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(15000);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes);

    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        let expected = (i + 1) as f32 * 8.0;
        assert!(
            (actual - expected).abs() < 0.01,
            "Mismatch at index {}: got {}, expected {} (after 3x multiply by 2)",
            i, actual, expected
        );
    }
}

#[test]
fn test_clif_ffi_cuda_buffer_reuse() {
    // Upload data A → launch → download result A, then re-upload data B → launch → download result B.
    // Verifies buffer state is correctly updated on re-upload.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("cuda_reuse.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let n: usize = 64;
    let data_bytes = n * 4;

    // Kernel: data[tid] += 1.0
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
    mov.f32 %fc, 0f3F800000;
    add.f32 %fv, %fv, %fc;
    st.global.f32 [%rd], %fv;

    ret;
}\0";

    let ptx_off = 3000;
    let bind_off = 3800;
    let data_a_off = 4000;
    let data_b_off = 4300;
    let result_a_off = 4600;
    let result_b_off = result_a_off + data_bytes;

    let clif_ir = format!(
r#"function u0:0(i64) system_v {{
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload sig2
    fn3 = %cl_cuda_download sig2
    fn4 = %cl_cuda_launch sig3
    fn5 = %cl_cuda_cleanup sig0
    fn6 = %cl_file_write sig4

block0(v0: i64):
    call fn0(v0)

    v1 = iconst.i64 {data_bytes}
    v2 = call fn1(v0, v1)

    ; round 1: upload A, launch, download
    v3 = iconst.i64 {data_a_off}
    v20 = call fn2(v0, v2, v3, v1)
    v4 = iconst.i64 {ptx_off}
    v5 = iconst.i32 1
    v6 = iconst.i64 {bind_off}
    v7 = iconst.i32 1
    v8 = iconst.i32 64
    v21 = call fn4(v0, v4, v5, v6, v7, v7, v7, v8, v7, v7)
    v9 = iconst.i64 {result_a_off}
    v22 = call fn3(v0, v2, v9, v1)

    ; round 2: re-upload B, launch, download
    v10 = iconst.i64 {data_b_off}
    v23 = call fn2(v0, v2, v10, v1)
    v24 = call fn4(v0, v4, v5, v6, v7, v7, v7, v8, v7, v7)
    v11 = iconst.i64 {result_b_off}
    v25 = call fn3(v0, v2, v11, v1)

    ; write both results to file
    v12 = iconst.i64 2000
    v13 = iconst.i64 0
    v14 = iconst.i64 {two_data_bytes}
    v26 = call fn6(v0, v12, v9, v13, v14)

    call fn5(v0)
    return
}}"#,
        data_bytes = data_bytes,
        data_a_off = data_a_off,
        data_b_off = data_b_off,
        ptx_off = ptx_off,
        bind_off = bind_off,
        result_a_off = result_a_off,
        result_b_off = result_b_off,
        two_data_bytes = data_bytes * 2,
    );

    let mut memory = vec![0u8; 8192];
    memory[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    let ptx_bytes = ptx.as_bytes();
    memory[ptx_off..ptx_off + ptx_bytes.len()].copy_from_slice(ptx_bytes);

    memory[bind_off..bind_off + 4].copy_from_slice(&0i32.to_le_bytes());

    // Data A: all 10.0
    for i in 0..n {
        memory[data_a_off + i * 4..data_a_off + i * 4 + 4].copy_from_slice(&10.0f32.to_le_bytes());
    }
    // Data B: all 100.0
    for i in 0..n {
        memory[data_b_off + i * 4..data_b_off + i * 4 + 4].copy_from_slice(&100.0f32.to_le_bytes());
    }

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) =
        create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(15000);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), data_bytes * 2);

    // Result A: 10.0 + 1.0 = 11.0
    for i in 0..n {
        let actual = f32::from_le_bytes(contents[i * 4..i * 4 + 4].try_into().unwrap());
        assert!(
            (actual - 11.0).abs() < 0.01,
            "Result A index {}: got {}, expected 11.0", i, actual
        );
    }
    // Result B: 100.0 + 1.0 = 101.0
    for i in 0..n {
        let off = data_bytes + i * 4;
        let actual = f32::from_le_bytes(contents[off..off + 4].try_into().unwrap());
        assert!(
            (actual - 101.0).abs() < 0.01,
            "Result B index {}: got {}, expected 101.0", i, actual
        );
    }
}

#[test]
fn test_clif_ffi_cuda_error_codes() {
    // Verify FFI functions return -1 for invalid arguments.
    let temp_dir = TempDir::new().unwrap();
    let verify_file = temp_dir.path().join("cuda_errors.bin");
    let verify_file_str = format!("{}\0", verify_file.to_str().unwrap());

    let clif_ir =
r#"function u0:0(i64) system_v {
    sig0 = (i64) system_v
    sig1 = (i64, i64) -> i32 system_v
    sig2 = (i64, i32, i64, i64) -> i32 system_v
    sig3 = (i64, i64, i32, i64, i32, i32, i32, i32, i32, i32) -> i32 system_v
    sig4 = (i64, i64, i64, i64, i64) -> i64 system_v

    fn0 = %cl_cuda_init sig0
    fn1 = %cl_cuda_create_buffer sig1
    fn2 = %cl_cuda_upload sig2
    fn3 = %cl_cuda_download sig2
    fn4 = %cl_cuda_launch sig3
    fn5 = %cl_cuda_cleanup sig0
    fn6 = %cl_file_write sig4

block0(v0: i64):
    call fn0(v0)

    ; create_buffer with size=0 → should return -1
    v1 = iconst.i64 0
    v2 = call fn1(v0, v1)
    v20 = sextend.i64 v2
    store.i64 v20, v0+4500

    ; upload with buf_id=99 → should return -1
    v3 = iconst.i32 99
    v4 = iconst.i64 3000
    v5 = iconst.i64 64
    v6 = call fn2(v0, v3, v4, v5)
    v21 = sextend.i64 v6
    store.i64 v21, v0+4508

    ; download with buf_id=99 → should return -1
    v7 = call fn3(v0, v3, v4, v5)
    v22 = sextend.i64 v7
    store.i64 v22, v0+4516

    ; launch with n_bufs=0 is valid but grid_x=0 → should return -1
    v8 = iconst.i64 3500
    v9 = iconst.i32 0
    v10 = iconst.i64 3800
    v11 = iconst.i32 1
    v12 = call fn4(v0, v8, v9, v10, v9, v11, v11, v11, v11, v11)
    v23 = sextend.i64 v12
    store.i64 v23, v0+4524

    ; create_buffer with size=-1 → should return -1
    v13 = iconst.i64 -1
    v14 = call fn1(v0, v13)
    v24 = sextend.i64 v14
    store.i64 v24, v0+4532

    ; write 40 bytes of return values to verify file
    v15 = iconst.i64 2000
    v16 = iconst.i64 4500
    v17 = iconst.i64 0
    v18 = iconst.i64 40
    v19 = call fn6(v0, v15, v16, v17, v18)

    call fn5(v0)
    return
}
"#;

    let mut memory = vec![0u8; 8192];
    memory[2000..2000 + verify_file_str.len()].copy_from_slice(verify_file_str.as_bytes());

    // Dummy null-terminated string at 3500 for the PTX source in launch test
    memory[3500] = b'x';
    memory[3501] = 0;

    let actions = vec![
        Action { kind: Kind::Describe, dst: 0, src: 0, offset: 0, size: 0 },
        Action { kind: Kind::ClifCallAsync, dst: 0, src: 0, offset: 1024, size: 1 },
        Action { kind: Kind::Wait, dst: 1024, src: 0, offset: 0, size: 0 },
    ];

    let (config, mut algorithm) =
        create_cranelift_algorithm(actions, memory, 1, clif_ir.to_string());
    algorithm.timeout_ms = Some(15000);
    run(config, algorithm).unwrap();

    let contents = fs::read(&verify_file).unwrap();
    assert_eq!(contents.len(), 40);

    let create_zero_rc = i64::from_le_bytes(contents[0..8].try_into().unwrap());
    let upload_bad_rc = i64::from_le_bytes(contents[8..16].try_into().unwrap());
    let download_bad_rc = i64::from_le_bytes(contents[16..24].try_into().unwrap());
    let launch_bad_grid_rc = i64::from_le_bytes(contents[24..32].try_into().unwrap());
    let create_neg_rc = i64::from_le_bytes(contents[32..40].try_into().unwrap());

    assert_eq!(create_zero_rc, -1, "create_buffer(size=0) should return -1");
    assert_eq!(upload_bad_rc, -1, "upload(buf_id=99) should return -1");
    assert_eq!(download_bad_rc, -1, "download(buf_id=99) should return -1");
    assert_eq!(launch_bad_grid_rc, -1, "launch(grid_x=0) should return -1");
    assert_eq!(create_neg_rc, -1, "create_buffer(size=-1) should return -1");
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
}"#.to_string();

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        context_offset: 128,
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let mut data = vec![0u8; 16];
    data[0..8].copy_from_slice(&100i64.to_le_bytes());
    data[8..16].copy_from_slice(&200i64.to_le_bytes());

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 216,
        columns: vec![
            OutputColumn { name: "sum".to_string(), dtype: OutputType::I64, data_offset: 200, len_offset: 0 },
            OutputColumn { name: "len".to_string(), dtype: OutputType::I64, data_offset: 208, len_offset: 0 },
        ],
    }];
    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };

    let batches = base.execute(&alg, &data).unwrap();
    let sum = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let len = batches[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(sum.value(0), 300, "should read 100+200 from caller buffer via pointer");
    assert_eq!(len.value(0), 16, "data_len should be 16");
}

#[test]
fn test_data_ptr_not_written_when_data_empty() {
    // When data is empty, offsets 8-16 should NOT be overwritten.
    // Seed those offsets with a known value in initial_memory,
    // then verify CLIF sees the original values.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+8
    v2 = load.i64 v0+16
    store v1, v0+200
    store v2, v0+208
    v3 = iconst.i64 1
    store v3, v0+216
    return
}"#.to_string();

    let mut initial = vec![0u8; 4096];
    // Seed offsets 8 and 16 with sentinel values
    initial[8..16].copy_from_slice(&0xDEADBEEFu64.to_le_bytes());
    initial[16..24].copy_from_slice(&0xCAFEBABEu64.to_le_bytes());

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        context_offset: 0,
        initial_memory: initial,
    };

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 216,
        columns: vec![
            OutputColumn { name: "off8".to_string(), dtype: OutputType::I64, data_offset: 200, len_offset: 0 },
            OutputColumn { name: "off16".to_string(), dtype: OutputType::I64, data_offset: 208, len_offset: 0 },
        ],
    }];
    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };

    // Pass empty data — should NOT clobber offsets 8, 16
    let batches = run(config, alg).unwrap();
    let off8 = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let off16 = batches[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(off8.value(0), 0xDEADBEEF_i64, "offset 8 should retain sentinel when data is empty");
    assert_eq!(off16.value(0), 0xCAFEBABE_i64, "offset 16 should retain sentinel when data is empty");
}

#[test]
fn test_out_ptr_not_written_when_out_empty() {
    // When out is empty, offsets 24-32 should NOT be overwritten.
    // Seed offsets 24, 32 with sentinels in initial_memory,
    // pass non-empty data (which DOES write 8-16) but empty out.
    let clif_ir = r#"function u0:0(i64) system_v {
block0(v0: i64):
    v1 = load.i64 v0+24
    v2 = load.i64 v0+32
    store v1, v0+200
    store v2, v0+208
    v3 = iconst.i64 1
    store v3, v0+216
    return
}"#.to_string();

    let mut initial = vec![0u8; 4096];
    initial[24..32].copy_from_slice(&0x11111111u64.to_le_bytes());
    initial[32..40].copy_from_slice(&0x22222222u64.to_le_bytes());

    // Data is non-empty, so offsets 8-16 will get the data pointer written,
    // but offsets 24/32 should remain untouched.
    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        context_offset: 0,
        initial_memory: initial,
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 216,
        columns: vec![
            OutputColumn { name: "off24".to_string(), dtype: OutputType::I64, data_offset: 200, len_offset: 0 },
            OutputColumn { name: "off32".to_string(), dtype: OutputType::I64, data_offset: 208, len_offset: 0 },
        ],
    }];
    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };

    // Pass non-empty data but execute (not execute_into) — out is empty
    let data = vec![0u8; 8];
    let batches = base.execute(&alg, &data).unwrap();
    let off24 = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let off32 = batches[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(off24.value(0), 0x11111111_i64, "offset 24 should retain sentinel when out is empty");
    assert_eq!(off32.value(0), 0x22222222_i64, "offset 32 should retain sentinel when out is empty");
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
}"#.to_string();

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        context_offset: 64,
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let mut data = vec![0u8; 8];
    data[0..8].copy_from_slice(&6i64.to_le_bytes());

    let mut out = vec![0u8; 8];

    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: vec![],
    };

    base.execute_into(&alg, &data, &mut out).unwrap();
    let result = i64::from_le_bytes(out[0..8].try_into().unwrap());
    assert_eq!(result, 42, "CLIF should write 6*7=42 into caller's out buffer");
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
}"#.to_string();

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        context_offset: 64,
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: vec![],
    };

    // Call 1: data=111
    let mut data1 = 111i64.to_le_bytes().to_vec();
    let mut out1 = vec![0u8; 8];
    base.execute_into(&alg, &data1, &mut out1).unwrap();
    assert_eq!(i64::from_le_bytes(out1[0..8].try_into().unwrap()), 111);

    // Call 2: data=222, different buffers
    let mut data2 = 222i64.to_le_bytes().to_vec();
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
}"#.to_string();

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 256,
        context_offset: 64,
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
            OutputColumn { name: "last_val".to_string(), dtype: OutputType::I64, data_offset: 200, len_offset: 0 },
            OutputColumn { name: "len".to_string(), dtype: OutputType::I64, data_offset: 208, len_offset: 0 },
        ],
    }];
    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };

    let batches = base.execute(&alg, &data).unwrap();
    let last_val = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let len = batches[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(last_val.value(0), 999, "CLIF should read last value from caller buffer via pointer");
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
}"#.to_string();

    let mut initial = vec![0u8; 4096];
    // Static multiplier = 13
    initial[100..108].copy_from_slice(&13i64.to_le_bytes());

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        context_offset: 64,
        initial_memory: initial,
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![
            OutputColumn { name: "product".to_string(), dtype: OutputType::I64, data_offset: 200, len_offset: 0 },
        ],
    }];
    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };

    // Dynamic input = 7
    let data = 7i64.to_le_bytes().to_vec();
    let batches = base.execute(&alg, &data).unwrap();
    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col.value(0), 91, "13 * 7 = 91: static config from initial_memory, dynamic input via pointer");
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
}"#.to_string();

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 64,
        context_offset: 64,
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
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
}"#.to_string();

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        context_offset: 128,
        initial_memory: vec![],
    };

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![
            OutputColumn { name: "val".to_string(), dtype: OutputType::I64, data_offset: 200, len_offset: 0 },
        ],
    }];
    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };

    let data = 777i64.to_le_bytes().to_vec();
    let mut base = Base::new(config).unwrap();
    let batches = base.execute(&alg, &data).unwrap();
    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col.value(0), 777, "execute() should pass data pointer through to CLIF");
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
}"#.to_string();

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        context_offset: 128,
        initial_memory: vec![],
    };

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 208,
        columns: vec![
            OutputColumn { name: "len".to_string(), dtype: OutputType::I64, data_offset: 200, len_offset: 0 },
        ],
    }];
    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };

    let data = vec![42u8]; // single byte
    let mut base = Base::new(config).unwrap();
    let batches = base.execute(&alg, &data).unwrap();
    let col = batches[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
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
}"#.to_string();

    let config = BaseConfig {
        cranelift_ir: clif_ir,
        memory_size: 4096,
        context_offset: 64,
        initial_memory: vec![],
    };
    let mut base = Base::new(config).unwrap();

    let output_schema = vec![OutputBatchSchema {
        row_count_offset: 216,
        columns: vec![
            OutputColumn { name: "val".to_string(), dtype: OutputType::I64, data_offset: 200, len_offset: 0 },
            OutputColumn { name: "len".to_string(), dtype: OutputType::I64, data_offset: 208, len_offset: 0 },
        ],
    }];
    let alg = Algorithm {
        actions: vec![Action { kind: Kind::ClifCall, dst: 0, src: 0, offset: 0, size: 0 }],
        cranelift_units: 0,
        timeout_ms: Some(5000),
        output: output_schema,
    };

    // Call 1: 8-byte buffer
    let data1 = 11i64.to_le_bytes().to_vec();
    let b1 = base.execute(&alg, &data1).unwrap();
    let v1 = b1[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let l1 = b1[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(v1.value(0), 11);
    assert_eq!(l1.value(0), 8);

    // Call 2: 16-byte buffer (different size!)
    let mut data2 = vec![0u8; 16];
    data2[0..8].copy_from_slice(&22i64.to_le_bytes());
    let b2 = base.execute(&alg, &data2).unwrap();
    let v2 = b2[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let l2 = b2[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(v2.value(0), 22);
    assert_eq!(l2.value(0), 16, "data_len should reflect new buffer size");

    // Call 3: empty data — pointers should NOT be updated, old values stay
    let b3 = base.execute(&alg, &[]).unwrap();
    let v3 = b3[0].column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let l3 = b3[0].column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    // Pointers still point to data2's (now potentially dangling) location,
    // but importantly len should still be 16 from previous call
    assert_eq!(l3.value(0), 16, "empty data should not update data_len — stale from previous call");
}
