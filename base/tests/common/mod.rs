//! Constructing CLIF programs for tests.
//!
//! These tests are about FFI linkage — that the JIT resolves a symbol, calls it
//! with the arguments the program named, and gets the result back. The program
//! itself is scaffolding, so the point of these helpers is that the scaffolding
//! stays as legible as the CLIF text it replaces.

#![allow(dead_code)]

pub use base_types::clif::*;
use std::cell::RefCell;

pub const I8: ClifTy = ClifTy::I8;
pub const I16: ClifTy = ClifTy::I16;
pub const I32: ClifTy = ClifTy::I32;
pub const I64: ClifTy = ClifTy::I64;
pub const F32: ClifTy = ClifTy::F32;
pub const F64: ClifTy = ClifTy::F64;
pub const F32X4: ClifTy = ClifTy::F32X4;
pub const I8X16: ClifTy = ClifTy::I8X16;

/// `vN`
pub fn v(n: u32) -> Val {
    Val(n)
}

/// The caller's input buffer, as the entry block receives it.
pub fn data_ptr() -> Val {
    v(9500)
}

/// How many bytes of input the caller supplied.
pub fn data_len() -> Val {
    v(9501)
}

/// The caller's output buffer.
pub fn out_ptr() -> Val {
    v(9502)
}

/// How much room the caller left for the answer.
pub fn out_len() -> Val {
    v(9503)
}

/// `blockN`
pub fn b(n: u32) -> BlockRef {
    BlockRef(n)
}

/// Accumulates one function. Its `u0:N` is its position in the program.
pub struct Func {
    inner: Function,
}

pub fn function() -> Func {
    Func {
        inner: Function {
            name: None,
            signature: Signature { convention: Convention::Host, params: vec![], returns: vec![] },
            stack_slots: vec![],
            blocks: vec![],
        },
    }
}

impl Func {
    /// Exported as `name`, so a host can call it.
    pub fn export(mut self, name: &str) -> Self {
        self.inner.name = Some(name.to_string());
        self
    }

    /// `block0(v9500, v9501, v9502, v9503: i64): v0 = memory` — the caller's
    /// input buffer and its length and output buffer and its length, which is
    /// what an entry point is called with, then the address of the program's
    /// memory, which it takes itself.
    ///
    /// The four are numbered clear of everything a test body uses, so a body
    /// that wants one names it with [`data_ptr`] and the rest keep low ids.
    pub fn entry(self, insts: Vec<Inst>) -> Self {
        let mut body = vec![arena(v(0))];
        body.extend(insts);
        self.block(0, &[(data_ptr(), I64), (data_len(), I64), (out_ptr(), I64), (out_len(), I64)], body)
    }

    /// `block0(v0: i64):` — one pointer, and not the arena base: what
    /// `cl_thread_spawn` hands a worker is whatever the spawning program chose.
    /// A function reached that way is not an entry point and says so here.
    pub fn entry_spawned(self, insts: Vec<Inst>) -> Self {
        self.block(0, &[(v(0), I64)], insts)
    }

    pub fn block(mut self, n: u32, params: &[(Val, ClifTy)], insts: Vec<Inst>) -> Self {
        self.inner.blocks.push(Block {
            reference: BlockRef(n),
            params: params.to_vec(),
            insts,
        });
        self
    }
}

/// One function, the common case.
pub fn program(f: Func) -> Vec<Function> {
    programs(vec![f])
}

/// Several functions, in `u0:N` order, each at the signature its body has.
///
/// An entry point answers a status, as the C function a host calls does: one
/// whose body returns nothing returns `0`. Any other function takes what its
/// entry block does, and answers an `i64` when it returns one.
pub fn programs(fs: Vec<Func>) -> Vec<Function> {
    fs.into_iter()
        .map(|f| {
            let mut f = f.inner;
            let is_entry = f.blocks.first().is_some_and(|b| b.params.first() == Some(&(data_ptr(), I64)));
            if is_entry {
                f.signature = Signature::base_function();
                let mut fresh = 9600;
                for b in &mut f.blocks {
                    let insts = std::mem::take(&mut b.insts);
                    for i in insts {
                        if i == ret() {
                            b.insts.push(iconst64(v(fresh), 0));
                            b.insts.push(ret_status(v(fresh)));
                            fresh += 1;
                        } else {
                            b.insts.push(i);
                        }
                    }
                }
            } else {
                let answers = f
                    .blocks
                    .iter()
                    .flat_map(|b| &b.insts)
                    .any(|i| matches!(i, Inst::MultiAry(_, op, _, args) if op == "return" && !args.is_empty()));
                f.signature.params =
                    f.blocks.first().map(|b| b.params.iter().map(|(_, t)| t.clone().into()).collect()).unwrap_or_default();
                f.signature.returns = if answers { vec![I64.into()] } else { vec![] };
            }
            f
        })
        .collect()
}

thread_local! {
    /// The primitives the test running on this thread has named, in the order
    /// it first named them: what an artifact it builds declares.
    static PRIMITIVES: RefCell<Vec<Primitive>> = const { RefCell::new(Vec::new()) };
}

/// `p`, declared in the artifacts this test builds.
pub fn primitive(p: Primitive) -> Callee {
    PRIMITIVES.with_borrow_mut(|ps| {
        let i = ps.iter().position(|q| *q == p).unwrap_or_else(|| {
            ps.push(p);
            ps.len() - 1
        });
        Callee::Primitive(i as u32)
    })
}

/// The artifact running `functions` in an arena of `arena_size` bytes,
/// declaring every primitive this test has named.
pub fn artifact(functions: Vec<Function>, arena_size: u64, data: Vec<base_types::Segment>) -> base_types::Artifact {
    let primitives = PRIMITIVES.with_borrow(|ps| ps.clone());
    base_types::Artifact { primitives, functions, arena_size, data }
}

/// A function doing nothing — the slot generated artifacts reserve at `u0:0`
/// so that the interesting function is `u0:1`.
pub fn noop() -> Func {
    function().entry(vec![ret()])
}

// --- instructions -----------------------------------------------------------
//
// One name per instruction. Each is the constructor with a shorter name, so the
// tests read as CLIF and rustc still checks every operand.

// --- instructions ------------------------------------------------------------
//
// Each is a Cranelift format and an opcode as CLIF spells it, with the type
// named only where CLIF does not take it from an operand.

fn un(d: Val, op: &str, a: Val) -> Inst { Inst::Unary(vec![d], op.into(), None, a) }
fn un_at(d: Val, op: &str, ty: ClifTy, a: Val) -> Inst { Inst::Unary(vec![d], op.into(), Some(ty), a) }
fn bin(d: Val, op: &str, a: Val, b: Val) -> Inst { Inst::Binary(vec![d], op.into(), None, a, b) }
fn ter(d: Val, op: &str, a: Val, b: Val, c: Val) -> Inst { Inst::Ternary(vec![d], op.into(), None, a, b, c) }
fn aligned(yes: bool) -> Vec<String> { if yes { vec!["aligned".into()] } else { vec![] } }

/// `dst = symbol_value.i64 arena`
pub fn arena(d: Val) -> Inst { Inst::UnaryGlobalValue(vec![d], "symbol_value".into(), Some(I64), Global::Arena) }
pub fn iconst(d: Val, ty: ClifTy, k: i64) -> Inst { Inst::UnaryImm(vec![d], "iconst".into(), Some(ty), k) }
pub fn iconst64(d: Val, k: i64) -> Inst { iconst(d, I64, k) }
pub fn iconst32(d: Val, k: i64) -> Inst { iconst(d, I32, k) }
pub fn iadd(d: Val, a: Val, b: Val) -> Inst { bin(d, "iadd", a, b) }
pub fn iadd_imm(d: Val, a: Val, k: i64) -> Inst { Inst::BinaryImm64(vec![d], "iadd_imm".into(), None, a, k) }
pub fn isub(d: Val, a: Val, b: Val) -> Inst { bin(d, "isub", a, b) }
pub fn imul(d: Val, a: Val, b: Val) -> Inst { bin(d, "imul", a, b) }
pub fn udiv(d: Val, a: Val, b: Val) -> Inst { bin(d, "udiv", a, b) }
pub fn ineg(d: Val, a: Val) -> Inst { un(d, "ineg", a) }
pub fn ishl(d: Val, a: Val, b: Val) -> Inst { bin(d, "ishl", a, b) }
pub fn ushr(d: Val, a: Val, b: Val) -> Inst { bin(d, "ushr", a, b) }
pub fn band(d: Val, a: Val, b: Val) -> Inst { bin(d, "band", a, b) }
pub fn band_not(d: Val, a: Val, b: Val) -> Inst { bin(d, "band_not", a, b) }
pub fn bor(d: Val, a: Val, b: Val) -> Inst { bin(d, "bor", a, b) }
pub fn bxor(d: Val, a: Val, b: Val) -> Inst { bin(d, "bxor", a, b) }
pub fn ireduce32(d: Val, a: Val) -> Inst { un_at(d, "ireduce", I32, a) }
pub fn uextend64(d: Val, a: Val) -> Inst { un_at(d, "uextend", I64, a) }
pub fn sextend64(d: Val, a: Val) -> Inst { un_at(d, "sextend", I64, a) }

pub fn store(val: Val, addr: Val, off: i32) -> Inst { Inst::Store(vec![], "store".into(), None, vec![], val, addr, off) }
pub fn istore8(val: Val, addr: Val, off: i32) -> Inst { Inst::Store(vec![], "istore8".into(), None, vec![], val, addr, off) }
/// A store that states its address is aligned.
pub fn store_aligned(val: Val, addr: Val, off: i32) -> Inst {
    Inst::Store(vec![], "store".into(), None, aligned(true), val, addr, off)
}

/// `cc` as CLIF spells it: `"eq"`, `"ult"`, `"slt"`, ….
pub fn icmp(d: Val, cc: &str, a: Val, b: Val) -> Inst { Inst::IntCompare(vec![d], "icmp".into(), None, cc.into(), a, b) }
pub fn fcmp(d: Val, cc: &str, a: Val, b: Val) -> Inst { Inst::FloatCompare(vec![d], "fcmp".into(), None, cc.into(), a, b) }
pub fn select(d: Val, c: Val, a: Val, b: Val) -> Inst { ter(d, "select", c, a, b) }
pub fn bitselect(d: Val, c: Val, a: Val, b: Val) -> Inst { ter(d, "bitselect", c, a, b) }

/// A signature with no parameter attributes.
pub fn sig(convention: Convention, params: &[ClifTy], result: Option<ClifTy>) -> Signature {
    Signature {
        convention,
        params: params.iter().map(|t| t.clone().into()).collect(),
        returns: result.into_iter().map(Into::into).collect(),
    }
}

/// A function of the engine's own library `base`, at the signature given.
pub fn base_fn(symbol: &str, params: &[ClifTy], result: Option<ClifTy>) -> Callee {
    primitive(Primitive { lib: "base".into(), symbol: symbol.into(), signature: sig(Convention::Host, params, result) })
}
/// `%name` — one of `base`'s functions these tests call, at the signature
/// Lean's `Ffi.sig` states for it.
pub fn imp(name: &str) -> Callee {
    let (params, result): (&[ClifTy], _) = match name {
        "cl_file_read" | "cl_file_write" => (&[I64; 5], Some(I64)),
        "cl_file_read_to_ptr" | "cl_file_write_from_ptr" => (&[I64; 4], Some(I64)),
        "cl_native_load" => (&[I64; 2], Some(I64)),
        "cl_native_free" => (&[I64], Some(I32)),
        "cl_native_arch" => (&[], Some(I32)),
        _ => panic!("no signature here for {name}"),
    };
    base_fn(name, params, result)
}
/// `u0:N` — another function of this program.
pub fn loc(index: u32) -> Callee { Callee::Function(index) }

/// `dst = call fnN(args)`, or `call fnN(args)` when the callee returns nothing.
pub fn call(d: Option<Val>, c: Callee, args: &[Val]) -> Inst {
    Inst::Call(d.into_iter().collect(), "call".into(), None, c, args.to_vec())
}
/// `dst = call_indirect sig, addr(args)`: the function at `addr`.
pub fn call_indirect(d: Option<Val>, s: Signature, addr: Val, args: &[Val]) -> Inst {
    Inst::CallIndirect(d.into_iter().collect(), "call_indirect".into(), None, s, addr, args.to_vec())
}
/// `dst = func_addr.i64 fnN`
pub fn func_addr(d: Val, c: Callee) -> Inst { Inst::FuncAddr(vec![d], "func_addr".into(), Some(I64), c) }
pub fn jump(t: u32, args: &[Val]) -> Inst { Inst::Jump(vec![], "jump".into(), None, BlockCall(BlockRef(t), args.to_vec())) }
pub fn brif(c: Val, t: u32, ta: &[Val], e: u32, ea: &[Val]) -> Inst {
    Inst::Brif(vec![], "brif".into(), None, c, BlockCall(BlockRef(t), ta.to_vec()), BlockCall(BlockRef(e), ea.to_vec()))
}
pub fn ret() -> Inst { Inst::MultiAry(vec![], "return".into(), None, vec![]) }

/// `return v` — the status the caller reads.
pub fn ret_status(v: Val) -> Inst { Inst::MultiAry(vec![], "return".into(), None, vec![v]) }

pub fn fadd(d: Val, a: Val, b: Val) -> Inst { bin(d, "fadd", a, b) }
pub fn fsub(d: Val, a: Val, b: Val) -> Inst { bin(d, "fsub", a, b) }
pub fn fmul(d: Val, a: Val, b: Val) -> Inst { bin(d, "fmul", a, b) }
pub fn fmax(d: Val, a: Val, b: Val) -> Inst { bin(d, "fmax", a, b) }
pub fn fmin(d: Val, a: Val, b: Val) -> Inst { bin(d, "fmin", a, b) }
pub fn fneg(d: Val, a: Val) -> Inst { un(d, "fneg", a) }
pub fn fpromote(d: Val, a: Val) -> Inst { un_at(d, "fpromote", F64, a) }
pub fn fcvt_from_sint(d: Val, ty: ClifTy, s: Val) -> Inst { un_at(d, "fcvt_from_sint", ty, s) }
pub fn fcvt_to_uint(d: Val, ty: ClifTy, s: Val) -> Inst { un_at(d, "fcvt_to_uint_sat", ty, s) }
pub fn splat(d: Val, ty: ClifTy, s: Val) -> Inst { un_at(d, "splat", ty, s) }
pub fn extractlane(d: Val, s: Val, lane: u8) -> Inst { Inst::BinaryImm8(vec![d], "extractlane".into(), None, s, lane) }
pub fn bitcast(d: Val, ty: ClifTy, s: Val) -> Inst { Inst::LoadNoOffset(vec![d], "bitcast".into(), Some(ty), vec![], s) }
pub fn ctz(d: Val, a: Val) -> Inst { un(d, "ctz", a) }
pub fn popcnt(d: Val, a: Val) -> Inst { un(d, "popcnt", a) }
pub fn vhigh_bits(d: Val, a: Val) -> Inst { un_at(d, "vhigh_bits", I32, a) }

// --- loads -----------------------------------------------------------------

fn load_at(d: Val, op: &str, ty: Option<ClifTy>, is_aligned: bool, addr: Val, off: i32) -> Inst {
    Inst::Load(vec![d], op.into(), ty, aligned(is_aligned), addr, off)
}

/// `dst = load.ty addr+off`
pub fn load(d: Val, ty: ClifTy, addr: Val, off: i32) -> Inst { load_at(d, "load", Some(ty), false, addr, off) }
pub fn load64(d: Val, addr: Val, off: i32) -> Inst { load(d, I64, addr, off) }
pub fn load32(d: Val, addr: Val, off: i32) -> Inst { load(d, I32, addr, off) }
/// `dst = load.ty aligned addr+off`
pub fn load_trusted(d: Val, ty: ClifTy, addr: Val, off: i32) -> Inst { load_at(d, "load", Some(ty), true, addr, off) }
pub fn uload8(d: Val, addr: Val, off: i32) -> Inst { load_at(d, "uload8", Some(I64), false, addr, off) }
/// `uload32` answers an `i64` whatever: CLIF takes its type from the address.
pub fn uload32(d: Val, addr: Val, off: i32) -> Inst { load_at(d, "uload32", None, false, addr, off) }
pub fn sload8(d: Val, addr: Val, off: i32) -> Inst { load_at(d, "sload8", Some(I64), false, addr, off) }

/// Float constants carry bits, so a literal that cannot be spelled is a
/// compile error here rather than a JIT-time parse failure.
pub fn f32const(d: Val, x: f32) -> Inst { Inst::UnaryIeee32(vec![d], "f32const".into(), None, x.to_bits()) }
pub fn f64const(d: Val, x: f64) -> Inst { Inst::UnaryIeee64(vec![d], "f64const".into(), None, x.to_bits()) }

// --- reading a program ------------------------------------------------------

/// Prints the program as CLIF when `BASE_DUMP_CLIF` is set.
///
/// Nothing asserts on this text: it exists so a test can be read as the CLIF it
/// actually compiles. Because there is no expected value, a change in how
/// Cranelift renders IR changes what is printed and breaks nothing.
pub fn dump(artifact: &base_types::Artifact) {
    if std::env::var_os("BASE_DUMP_CLIF").is_none() || artifact.functions.is_empty() {
        return;
    }
    let test = std::thread::current().name().unwrap_or("?").to_string();
    match base::clif_text(artifact) {
        Ok(text) => println!("=== {test} ===\n{text}"),
        Err(e) => println!("=== {test} === does not decode: {e}"),
    }
}
