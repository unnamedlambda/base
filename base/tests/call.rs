//! Calling a driver's functions: the status, the caller's buffers, the arena
//! they share, and finding them by name.

use base::Driver;
use base_types::Segment;

use crate::common::*;

fn answering(k: i64) -> Func {
    function().entry(vec![iconst64(v(1), k), ret_status(v(1))])
}

#[test]
fn the_status_is_what_the_function_returns() {
    let d = driver(artifact(functions(vec![answering(7).named("seven"), function().named("none").entry(vec![ret()])]), 64, vec![]));
    assert_eq!(call_with(&d, "seven", &[], &mut []), 7);
    assert_eq!(call_with(&d, "none", &[], &mut []), 0);
}

/// The input is the caller's own memory, of any size: here larger than the
/// arena, read at its last word, and its length as given.
#[test]
fn the_input_is_read_in_place() {
    let d = driver(artifact(
        functions(vec![function().named("main").entry(vec![
            iadd(v(1), input(), input_len()),
            load64(v(2), v(1), -8),
            store(v(2), v(0), 0),
            store(input_len(), v(0), 8),
            ret(),
        ])]),
        64,
        vec![],
    ));
    for len in [8usize, 1024] {
        let mut input = vec![0u8; len];
        input[len - 8..].copy_from_slice(&(len as i64 * 3).to_le_bytes());
        call_with(&d, "main", &input, &mut []);
        assert_eq!(read_i64(&d, 0), len as i64 * 3);
        assert_eq!(read_i64(&d, 8), len as i64);
    }
    let empty = driver(main_of(vec![store(input_len(), v(0), 0), store(output_len(), v(0), 8), ret()]));
    call_with(&empty, "main", &[], &mut []);
    assert_eq!((read_i64(&empty, 0), read_i64(&empty, 8)), (0, 0));
}

/// The output is written in place, of any size, and each call writes only the
/// buffer it was handed.
#[test]
fn the_output_is_written_in_place() {
    let d = driver(artifact(
        functions(vec![function().named("main").entry(vec![
            load64(v(1), input(), 0),
            store(v(1), output(), 0),
            store(output_len(), output(), 8),
            ret(),
        ])]),
        64,
        vec![],
    ));
    let mut first = [0u8; 16];
    let mut second = [0u8; 256];
    call_with(&d, "main", &11i64.to_le_bytes(), &mut first);
    call_with(&d, "main", &22i64.to_le_bytes(), &mut second);
    assert_eq!(i64::from_le_bytes(first[..8].try_into().unwrap()), 11);
    assert_eq!(i64::from_le_bytes(first[8..].try_into().unwrap()), 16);
    assert_eq!(i64::from_le_bytes(second[..8].try_into().unwrap()), 22);
    assert_eq!(i64::from_le_bytes(second[8..16].try_into().unwrap()), 256);
}

/// The arena starts as the artifact states, zero where no segment reaches; a
/// program reaches it at the address the host is handed.
#[test]
fn the_arena_starts_as_the_artifact_states() {
    let mut d = driver(artifact(
        functions(vec![function().named("addr").entry(vec![ret_status(v(0))])]),
        64,
        vec![Segment { offset: 8, bytes: 0x1122_3344_5566_7788u64.to_le_bytes().to_vec() }],
    ));
    assert_eq!(call_with(&d, "addr", &[], &mut []) as usize, d.arena_ptr() as usize);
    let arena = d.arena();
    assert_eq!(arena.len(), 64);
    assert_eq!(arena[8..16], 0x1122_3344_5566_7788u64.to_le_bytes());
    assert!(arena[..8].iter().chain(&arena[16..]).all(|&b| b == 0));
}

/// What one call leaves in the arena, the next call, of any function, reads.
#[test]
fn the_arena_persists_across_calls() {
    let d = driver(artifact(
        functions(vec![
            function().named("add").entry(vec![load64(v(1), v(0), 0), load64(v(2), input(), 0), iadd(v(3), v(1), v(2)), store(v(3), v(0), 0), ret()]),
            function().named("double").entry(vec![load64(v(1), v(0), 0), iadd(v(2), v(1), v(1)), ret_status(v(2))]),
        ]),
        64,
        vec![],
    ));
    for k in [10i64, 25, 5] {
        call_with(&d, "add", &k.to_le_bytes(), &mut []);
    }
    assert_eq!(read_i64(&d, 0), 40);
    assert_eq!(call_with(&d, "double", &[], &mut []), 80);
}

#[test]
fn a_function_calls_the_others() {
    let d = driver(artifact(
        functions(vec![
            function().block(0, &[(v(0), I64)], vec![iconst64(v(1), 10), imul(v(2), v(0), v(1)), ret_status(v(2))]),
            function().named("main").entry(vec![iconst64(v(1), 4), call(Some(v(2)), local(0), &[v(1)]), ret_status(v(2))]),
        ]),
        64,
        vec![],
    ));
    assert_eq!(call_with(&d, "main", &[], &mut []), 40);
}

/// A host finds a function by name only: an unnamed one, or the position a host
/// might guess, reaches nothing.
#[test]
fn functions_are_found_by_name_only() {
    let d = driver(artifact(functions(vec![answering(1), answering(2).named("two")]), 64, vec![]));
    assert_eq!(call_with(&d, "two", &[], &mut []), 2);
    for name in ["u0:0", "one", ""] {
        let Err(base::Error::NoSuchFunction(msg)) = d.function(name) else { panic!("{name:?} should not be found") };
        assert!(msg.contains(&format!("{name:?}")), "{msg}");
    }
}

/// The functions moved, each local call following its callee: a host calling
/// by name sees no difference.
#[test]
fn names_survive_a_reordering() {
    let times_ten = function().block(0, &[(v(0), I64)], vec![iconst64(v(1), 10), imul(v(2), v(0), v(1)), ret_status(v(2))]);
    let tens = function().named("tens").entry(vec![iconst64(v(1), 4), call(Some(v(2)), local(1), &[v(1)]), ret_status(v(2))]);
    let before = functions(vec![function().entry(vec![ret()]), times_ten, tens, answering(7).named("seven")]);
    let order = [3u32, 2, 0, 1];
    let after: Vec<Function> = order
        .iter()
        .map(|&old| {
            let mut f = before[old as usize].clone();
            for i in f.blocks.iter_mut().flat_map(|b| &mut b.insts) {
                if let Inst::Call(_, _, _, Callee::Function(c), _) = i {
                    *c = order.iter().position(|&o| o == *c).unwrap() as u32;
                }
            }
            f
        })
        .collect();
    for fs in [before, after] {
        let d = driver(artifact(fs, 64, vec![]));
        assert_eq!(call_with(&d, "tens", &[], &mut []), 40);
        assert_eq!(call_with(&d, "seven", &[], &mut []), 7);
    }
}

/// Many threads call one driver at once, each call on its own thread against
/// the one arena: no count is lost.
#[test]
fn threads_call_one_driver_at_once() {
    let d = driver(main_of(vec![iconst64(v(1), 1), atomic_add(v(2), v(0), v(1)), ret()]));
    let f = d.function("main").unwrap();
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                for _ in 0..10_000 {
                    unsafe { f.call(&[], &mut []) };
                }
            });
        }
    });
    assert_eq!(read_i64(&d, 0), 80_000);
}

#[test]
fn a_driver_is_shared_between_threads() {
    fn shared<T: Send + Sync>() {}
    shared::<Driver>();
}
