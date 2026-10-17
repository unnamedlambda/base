//! The C API: what a host outside Rust calls.

use base::capi::*;
use base::Driver;
use base_types::{Artifact, Segment};

/// An artifact with no functions: enough to exercise the ABI without a JIT.
fn empty_artifact() -> Vec<u8> {
    Artifact {
        primitives: vec![],
        functions: vec![],
        arena_size: 64,
        data: vec![Segment { offset: 0, bytes: vec![1, 2, 3, 4] }],
    }
    .to_bytes()
}

/// The driver, or why there is none.
fn try_load(bytes: &[u8]) -> Result<*mut Driver, String> {
    let mut err = [0xffu8; 256];
    let driver = unsafe { base_driver_load(bytes.as_ptr(), bytes.len(), err.as_mut_ptr(), err.len()) };
    if driver.is_null() {
        let end = err.iter().position(|&b| b == 0).expect("NUL-terminated");
        Err(String::from_utf8(err[..end].to_vec()).unwrap())
    } else {
        Ok(driver)
    }
}

fn load(bytes: &[u8]) -> *mut Driver {
    try_load(bytes).unwrap_or_else(|e| panic!("{e}"))
}

/// A field the format does not have is refused, by name: a writer left
/// behind by a change to the format fails to load, rather than handing
/// over an artifact whose program starts from an arena it did not fill.
#[test]
fn an_artifact_naming_a_field_the_format_does_not_have_is_refused() {
    // `{"primitives": [], "functions": [], "memory_size": 64, "data": []}`
    let text = |s: &str| [&[0x60 + s.len() as u8][..], s.as_bytes()].concat();
    let stale = [
        &[0xa4][..],
        &text("primitives"),
        &[0x80],
        &text("functions"),
        &[0x80],
        &text("memory_size"),
        &[0x18, 64],
        &text("data"),
        &[0x80],
    ]
    .concat();
    let e = try_load(&stale).expect_err("a stale artifact should not build");
    assert!(e.contains("memory_size"), "the message should name the field: {e}");
}

#[test]
fn an_artifact_builds_and_frees() {
    let driver = load(&empty_artifact());
    let mut len = 0usize;
    assert!(!unsafe { base_driver_arena(driver, &mut len) }.is_null());
    assert_eq!(len, 64);
    unsafe { base_driver_free(driver) };
}

/// The data segments are what the program starts from, so reading them back
/// is how a host confirms it got the artifact it sent.
#[test]
fn the_arena_answers_the_bytes_the_artifact_starts_with() {
    let driver = load(&empty_artifact());
    let mut len = 0usize;
    let memory = unsafe { base_driver_arena(driver, &mut len) };
    assert_eq!(len, 64);
    assert_eq!(unsafe { std::slice::from_raw_parts(memory, 4) }, [1, 2, 3, 4]);
    assert_eq!(unsafe { std::slice::from_raw_parts(memory, len) }[4..], [0u8; 60]);
    unsafe { base_driver_free(driver) };
}

/// Every function answers its failure value on a null driver rather than
/// dereferencing it.
#[test]
fn a_null_driver_is_refused_everywhere() {
    let name = "main";
    assert!(unsafe { base_driver_function(std::ptr::null(), name.as_ptr(), name.len()) }.is_none());
    let mut len = 7usize;
    assert!(unsafe { base_driver_arena(std::ptr::null(), &mut len) }.is_null());
    assert_eq!(len, 0);
    // Freeing null is a no-op, not a double free.
    unsafe { base_driver_free(std::ptr::null_mut()) };
}

/// An arena no host holds, and segments past it or out of order, are errors.
#[test]
fn an_arena_too_large_to_hold_is_refused() {
    let huge = Artifact { primitives: vec![], functions: vec![], arena_size: u64::MAX, data: vec![] };
    let e = try_load(&huge.to_bytes()).expect_err("should not build");
    assert!(e.contains("more than this host can hold"), "{e}");
    let segments = |data| Artifact { primitives: vec![], functions: vec![], arena_size: 8, data }.to_bytes();
    let e = try_load(&segments(vec![Segment { offset: u64::MAX, bytes: vec![1] }])).expect_err("should not build");
    assert!(e.contains("ends past the 8 bytes"), "{e}");
    // Segments out of order, or overlapping, are refused: the image has one
    // spelling.
    for data in [
        vec![Segment { offset: 4, bytes: vec![1] }, Segment { offset: 0, bytes: vec![2] }],
        vec![Segment { offset: 0, bytes: vec![1, 2] }, Segment { offset: 1, bytes: vec![3] }],
    ] {
        let e = try_load(&segments(data)).expect_err("should not build");
        assert!(e.contains("out of order or overlaps"), "{e}");
    }
}

/// A name the artifact does not export has no function.
#[test]
fn a_name_no_function_has_answers_null() {
    let driver = load(&empty_artifact());
    let name = "infer";
    assert!(unsafe { base_driver_function(driver, name.as_ptr(), name.len()) }.is_none());
    unsafe { base_driver_free(driver) };
}

/// An entry point is a plain C function: called with the four host
/// arguments, answering the program's status.
#[test]
fn a_function_is_a_plain_c_function() {
    use base_types::clif::{Block, BlockRef, ClifTy, Function, Inst, Signature, Val};
    let entry = |name: &str, arity: u32, ret: Option<i64>| Function {
        name: Some(name.into()),
        signature: Signature::base_function(),
        stack_slots: vec![],
        blocks: vec![Block {
            reference: BlockRef(0),
            params: (0..arity).map(|k| (Val(k), ClifTy::I64)).collect(),
            insts: match ret {
                Some(k) => vec![
                    Inst::UnaryImm(vec![Val(9)], "iconst".into(), Some(ClifTy::I64), k),
                    Inst::MultiAry(vec![], "return".into(), None, vec![Val(9)]),
                ],
                None => vec![Inst::MultiAry(vec![], "return".into(), None, vec![])],
            },
        }],
    };
    let artifact = |fs| Artifact { primitives: vec![], functions: fs, arena_size: 64, data: vec![] }.to_bytes();
    let driver = load(&artifact(vec![entry("seven", 4, Some(7)), entry("three", 4, Some(3))]));
    for (name, status) in [("seven", 7), ("three", 3)] {
        let f = unsafe { base_driver_function(driver, name.as_ptr(), name.len()) }
            .unwrap_or_else(|| panic!("{name} is not exported"));
        let mut out = [0u8; 4];
        assert_eq!(unsafe { f(b"in".as_ptr(), 2, out.as_mut_ptr(), out.len()) }, status, "{name}");
    }
    unsafe { base_driver_free(driver) };
    // A function not shaped like one is not exported as one: it would be
    // called with arguments it does not take, or answer nothing.
    for f in [entry("five", 5, Some(5)), entry("quiet", 4, None)] {
        let e = try_load(&artifact(vec![f])).expect_err("not an entry point");
        assert!(e.contains("so a host calls it"), "{e}");
    }
    // Nor is one that states another signature.
    let mut f = entry("seven", 4, Some(7));
    f.signature.params.pop();
    let e = try_load(&artifact(vec![f])).expect_err("not an entry point");
    assert!(e.contains("not BaseFunction's"), "{e}");
}

/// A program reaches the arena through the symbol the engine links it
/// under: the address it takes is the one the host is handed, and what it
/// reads there is what the artifact started it with.
#[test]
fn a_program_takes_the_arena_by_symbol() {
    use base_types::clif::{Block, BlockRef, ClifTy, Function, Global, Inst, Signature, Val};
    let entry = |name: &str, insts: Vec<Inst>| Function {
        name: Some(name.into()),
        signature: Signature::base_function(),
        stack_slots: vec![],
        blocks: vec![Block { reference: BlockRef(0), params: (1..5).map(|k| (Val(k), ClifTy::I64)).collect(), insts }],
    };
    let arena = Inst::UnaryGlobalValue(vec![Val(9)], "symbol_value".into(), Some(ClifTy::I64), Global::Arena);
    let ret = |v| Inst::MultiAry(vec![], "return".into(), None, vec![Val(v)]);
    let artifact = Artifact {
        primitives: vec![],
        functions: vec![
            entry("addr", vec![arena.clone(), ret(9)]),
            entry("word", vec![
                arena,
                Inst::Load(vec![Val(10)], "load".into(), Some(ClifTy::I64), vec![], Val(9), 8),
                ret(10),
            ]),
        ],
        arena_size: 64,
        data: vec![Segment { offset: 8, bytes: 0x1122_3344_5566_7788u64.to_le_bytes().to_vec() }],
    }
    .to_bytes();
    let driver = load(&artifact);
    let call = |name: &str| {
        let f = unsafe { base_driver_function(driver, name.as_ptr(), name.len()) }.unwrap();
        unsafe { f(std::ptr::null(), 0, std::ptr::null_mut(), 0) }
    };
    let mut len = 0usize;
    let memory = unsafe { base_driver_arena(driver, &mut len) };
    assert_eq!(call("addr") as usize, memory as usize);
    assert_eq!(call("word") as u64, 0x1122_3344_5566_7788);
    unsafe { base_driver_free(driver) };
}

/// A null pointer carrying a length is a caller bug, and is refused before
/// it reaches `from_raw_parts`.
#[test]
fn a_null_pointer_with_a_length_is_refused() {
    let mut err = [0u8; 64];
    assert!(unsafe { base_driver_load(std::ptr::null(), 16, err.as_mut_ptr(), err.len()) }.is_null());
    assert!(String::from_utf8_lossy(&err).contains("non-zero length"));
    // And with nowhere to say why, still null.
    assert!(unsafe { base_driver_load(std::ptr::null(), 16, std::ptr::null_mut(), 0) }.is_null());
}

/// A cut-off file and a JSON file are errors with a message, not panics.
#[test]
fn a_malformed_artifact_is_an_error_and_not_a_panic() {
    let whole = empty_artifact();
    for bad in [&whole[..whole.len() - 1], b"{\"functions\": []}".as_slice()] {
        let e = try_load(bad).expect_err("malformed");
        assert!(e.starts_with("not an artifact"), "{e}");
    }
}


/// A message longer than the buffer is cut to fit, and still terminated.
#[test]
fn the_message_is_cut_to_fit() {
    let mut err = [0xffu8; 4];
    assert!(unsafe { base_driver_load(std::ptr::null(), 16, err.as_mut_ptr(), err.len()) }.is_null());
    assert_eq!(&err, b"art\0");
}
