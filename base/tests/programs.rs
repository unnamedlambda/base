//! Programs calling outside themselves: a primitive, a symbol that is not
//! there, machine code they carry, and a function the host hands them.

use base_types::Segment;

use crate::common::*;

#[test]
fn a_primitive_acts_on_the_world() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("len.bin");
    let d = driver(artifact(
        functions(vec![function().named("main").entry(vec![
            iconst64(v(1), 256),
            iconst32(v(2), 6),
            call(Some(v(3)), lib_fn("fs", "base_fs_open", &[I64, I64, I32], Some(I64)), &[v(0), v(1), v(2)]),
            iconst64(v(4), 5),
            call(Some(v(5)), lib_fn("fs", "base_fs_set_len", &[I64, I64], Some(I64)), &[v(3), v(4)]),
            call(Some(v(6)), lib_fn("fs", "base_fs_size", &[I64], Some(I64)), &[v(3)]),
            call(None, lib_fn("fs", "base_fs_close", &[I64], None), &[v(3)]),
            ret_status(v(6)),
        ])]),
        256,
        vec![Segment { offset: 0, bytes: path.to_str().unwrap().as_bytes().to_vec() }],
    ));
    assert_eq!(call_with(&d, "main", &[], &mut []), 5);
    assert_eq!(std::fs::read(&path).unwrap(), [0u8; 5]);
}

/// A symbol base does not have answers -1, and is never looked up among what
/// else the process has loaded.
#[test]
fn a_missing_symbol_answers_minus_one() {
    for symbol in ["abort", "fn_0"] {
        let d = driver(main_of(vec![call(Some(v(1)), base_fn(symbol, &[], Some(I64)), &[]), ret_status(v(1))]));
        assert_eq!(call_with(&d, "main", &[], &mut []), -1, "{symbol}");
    }
}

/// Machine code carried as data, made executable and called through its
/// address at the architecture's convention.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn a_program_runs_machine_code_it_carries() {
    #[cfg(target_arch = "x86_64")]
    let (arch, add): (i64, &[u8]) = (1, &[0x48, 0x89, 0xf8, 0x48, 0x01, 0xf0, 0xc3]); // mov rax,rdi; add rax,rsi; ret
    #[cfg(target_arch = "aarch64")]
    let (arch, add): (i64, &[u8]) = (2, &[0x00, 0x00, 0x01, 0x8b, 0xc0, 0x03, 0x5f, 0xd6]); // add x0,x0,x1; ret
    let d = driver(artifact(
        functions(vec![function().named("main").entry(vec![
            call(Some(v(1)), base_primitive("cl_native_arch"), &[]),
            uextend64(v(9), v(1)),
            store(v(9), v(0), 300),
            iadd_imm(v(2), v(0), 0x100),
            iconst64(v(3), add.len() as i64),
            call(Some(v(4)), base_primitive("cl_native_load"), &[v(2), v(3)]),
            iconst64(v(5), 40),
            iconst64(v(6), 2),
            iconst64(v(7), 0),
            call_indirect(Some(v(8)), sig(Convention::Arch, &[I64; 4], Some(I64)), v(4), &[v(5), v(6), v(7), v(7)]),
            store(v(8), v(0), 200),
            call(None, base_primitive("cl_native_free"), &[v(4)]),
            ret(),
        ])]),
        4096,
        vec![Segment { offset: 0x100, bytes: add.to_vec() }],
    ));
    call_with(&d, "main", &[], &mut []);
    assert_eq!(read_i64(&d, 200), 42);
    assert_eq!(read_i64(&d, 300), arch);
}

/// The host hands the program a function of its own by address, and the
/// program calls it at the host's convention, answering what it answered.
#[test]
fn a_program_calls_back_a_host_function() {
    unsafe extern "C" fn double(input: *const u8, len: usize, out: *mut u8, cap: usize) -> i64 {
        let n = len.min(cap);
        for i in 0..n {
            unsafe { *out.add(i) = (*input.add(i)).wrapping_mul(2) };
        }
        n as i64 + 100
    }
    let d = driver(main_of(vec![
        load64(v(1), input(), 0),
        iadd_imm(v(2), input(), 8),
        iadd_imm(v(3), input_len(), -8),
        call_indirect(Some(v(4)), sig(Convention::Host, &[I64; 4], Some(I64)), v(1), &[v(2), v(3), output(), output_len()]),
        ret_status(v(4)),
    ]));
    let f: base::BaseFunction = double;
    let mut input = (f as usize as u64).to_le_bytes().to_vec();
    input.extend_from_slice(&[1, 2, 3]);
    let mut out = [0u8; 3];
    assert_eq!(call_with(&d, "main", &input, &mut out), 103);
    assert_eq!(out, [2, 4, 6]);
}
