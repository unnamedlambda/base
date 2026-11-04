//! What a load accepts, and what it refuses and says.

use base::Driver;
use base_types::{Artifact, Segment};

use crate::common::*;

fn named(fs: Vec<Func>) -> Artifact {
    artifact(functions(fs), 256, vec![])
}

fn answering(k: i64) -> Func {
    function().entry(vec![iconst64(v(1), k), ret_status(v(1))])
}

fn body(insts: Vec<Inst>) -> Artifact {
    named(vec![function().named("main").entry(insts)])
}

fn arena_of(size: u64, data: Vec<Segment>) -> Artifact {
    artifact(vec![], size, data)
}

fn load_with(flags: &[&str]) -> Inst {
    Inst::Load(vec![v(1)], "load".into(), Some(I64), flags.iter().map(|f| f.to_string()).collect(), v(0), 0)
}

#[test]
fn an_empty_artifact_loads() {
    assert!(Driver::load(&arena_of(256, vec![]).to_bytes()).is_ok());
}

#[test]
fn every_generated_artifact_loads() {
    for &name in lean_artifacts::NAMES {
        Driver::load(lean_artifacts::by_name(name).unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn each_malformed_artifact_is_refused_with_what_is_wrong() {
    let disagreeing = |params: &[(Val, ClifTy)]| {
        function()
            .block(0, params, vec![iconst64(v(1), 1), brif(v(1), 1, &[], 2, &[])])
            .block(1, &[], vec![ret_status(v(1))])
            .block(2, &[], vec![ret()])
    };
    // Four i64 parameters, numbered apart from what `functions` rewrites.
    let host = [(v(5), I64), (v(6), I64), (v(7), I64), (v(8), I64)];
    let normal = Signature { convention: Convention::Host, params: vec![AbiParam(I64, vec!["normal".into()])], returns: vec![] };
    let mut twice = body(vec![call(None, base_fn("cl_twice", &[], None), &[]), ret()]);
    twice.primitives.push(twice.primitives[0].clone());
    let mut no_blocks = body(vec![ret()]);
    no_blocks.functions[0].blocks.clear();
    let mut wrong_signature = named(vec![answering(7).named("seven")]);
    wrong_signature.functions[0].signature.params.pop();

    let cases: Vec<(&str, Artifact, &[&str])> = vec![
        ("a segment past the arena", arena_of(256, vec![Segment { offset: 200, bytes: vec![1; 100] }]), &["ends past the 256 bytes"]),
        ("a segment past every address", arena_of(8, vec![Segment { offset: u64::MAX, bytes: vec![1] }]), &["ends past the 8 bytes"]),
        ("segments out of order", arena_of(8, vec![Segment { offset: 4, bytes: vec![1] }, Segment { offset: 0, bytes: vec![2] }]), &["out of order or overlaps"]),
        ("overlapping segments", arena_of(8, vec![Segment { offset: 0, bytes: vec![1, 2] }, Segment { offset: 1, bytes: vec![3] }]), &["out of order or overlaps"]),
        ("an arena no host holds", arena_of(u64::MAX, vec![]), &["more than this host can hold"]),
        ("returns that disagree", named(vec![answering(0).named("main"), disagreeing(&[(v(5), I64)])]), &["return must match"]),
        ("a named function returning nothing", named(vec![disagreeing(&host).named("main")]), &["so a host calls it", "return"]),
        ("a named function of another shape", named(vec![function().named("pair").block(0, &[(v(0), I64), (v(1), I64)], vec![ret()])]), &["has a name, so a host calls it, and its signature is not BaseFunction's"]),
        ("a named function stating another signature", wrong_signature, &["not BaseFunction's"]),
        ("two functions of one name", named(vec![answering(1).named("x"), answering(2).named("x")]), &["u0:0", "u0:1", "\"x\""]),
        ("a function with no blocks", no_blocks, &["has no blocks"]),
        ("two blocks of one name", named(vec![function().named("main").entry(vec![jump(1, &[])]).block(1, &[], vec![ret()]).block(1, &[], vec![ret()])]), &["names two blocks block1"]),
        ("a value used before it is defined", body(vec![store(v(9), v(0), 0), ret()]), &["v9"]),
        ("a value defined twice", body(vec![iconst64(v(1), 1), iconst64(v(1), 2), ret()]), &["v1 is defined twice"]),
        ("a branch to no block", body(vec![jump(7, &[])]), &["block7"]),
        ("a call to no function", body(vec![call(None, local(3), &[v(0)]), ret()]), &["u0:3"]),
        ("a call to no primitive", body(vec![call(None, Callee::Primitive(1000), &[]), ret()]), &["primitive 1000, which the program does not declare"]),
        ("a primitive declared twice", twice, &["declared twice"]),
        ("a primitive called at another signature", body(vec![call(None, base_primitive("cl_native_arch"), &[v(0)]), ret()]), &["cl_native_arch", "takes"]),
        ("a function called at another signature", named(vec![answering(0), function().named("main").entry(vec![call(None, local(0), &[v(0)]), ret()])]), &["u0:0", "takes"]),
        ("an indirect call at another signature", body(vec![iconst64(v(1), 0), call_indirect(Some(v(2)), sig(Convention::Arch, &[I64; 4], Some(I64)), v(1), &[v(1)]), ret()]), &["indirect callee", "1 argument"]),
        ("no such opcode", body(vec![Inst::Binary(vec![v(1)], "iadd_frobnicate".into(), None, v(0), v(0)), ret()]), &["not a CLIF opcode"]),
        ("an opcode of another format", body(vec![Inst::Binary(vec![v(1)], "ineg".into(), None, v(0), v(0)), ret()]), &["Unary instruction"]),
        ("a needed type left out", body(vec![Inst::Unary(vec![v(1)], "uextend".into(), None, v(0)), ret()]), &["names none"]),
        ("a type CLIF takes from an operand, named", body(vec![Inst::Binary(vec![v(1)], "iadd".into(), Some(I64), v(0), v(0)), ret()]), &["takes its type from an operand"]),
        ("a type the opcode is not defined at", body(vec![Inst::Binary(vec![v(1)], "fadd".into(), None, v(0), v(0)), ret()]), &["not defined at i64"]),
        ("a float constant naming a type", body(vec![Inst::UnaryIeee64(vec![v(1)], "f64const".into(), Some(I64), 0), ret()]), &["f64const"]),
        ("a type CLIF does not spell so", body(vec![iconst(v(1), ClifTy("i32x01".into()), 0), ret()]), &["not a CLIF type"]),
        ("a result left unnamed", body(vec![Inst::Binary(vec![], "iadd".into(), None, v(0), v(0)), ret()]), &["answers 1 result"]),
        ("a result named for none", body(vec![Inst::Store(vec![v(1)], "store".into(), None, vec![], v(0), v(0), 0), ret()]), &["answers 0 results"]),
        ("a flag CLIF does not have", body(vec![load_with(&["fast"]), ret()]), &["not a CLIF memory flag"]),
        ("a flag that is the engine's", body(vec![load_with(&["notrap"]), ret()]), &["the engine's to say"]),
        ("flags out of CLIF's order", body(vec![load_with(&["readonly", "aligned"]), ret()]), &["CLIF writes as aligned readonly"]),
        ("a parameter attribute CLIF does not write", body(vec![call_indirect(None, normal, v(0), &[v(0)]), ret()]), &["not a parameter as CLIF writes one"]),
    ];
    for (what, a, says) in cases {
        let Err(e) = Driver::load(&a.to_bytes()) else { panic!("{what}: loaded") };
        let msg = e.to_string();
        for s in says {
            assert!(msg.contains(s), "{what}: {msg:?} does not say {s:?}");
        }
    }
}
